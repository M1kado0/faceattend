"""Run one consented local liveness trial without retaining camera images.

Example bona-fide trial:
  PYTHONPATH=src uv run python scripts/run_liveness_trial.py \
    --consent --split proposal --trial-id bf-normal-01 --presentation bona_fide \
    --lighting normal --distance normal --glasses no

Experimental single-action validation:
  add ``--challenge look_up`` or ``--challenge smile``. Explicit challenges
  never change the production default challenge pool.

For attack trials, point the camera at the declared presentation instrument.
Never identify the participant in ``--trial-id`` or notes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from faceattend.camera.worker import CameraEvidenceWorker  # noqa: E402
from faceattend.vision.active_liveness import (  # noqa: E402
    ActiveLivenessChallengeEvaluator,
)
from faceattend.vision.challenge_session import (  # noqa: E402
    ChallengeAction,
    ChallengeSession,
    ChallengeSessionConfig,
)
from faceattend.vision.evidence import EvidenceStreamGuard  # noqa: E402
from faceattend.vision.face_analyzer import (  # noqa: E402
    ModelArtifact,
    RuntimeModelManifest,
    create_face_analyzer,
)
from faceattend.vision.model_lifecycle import file_sha256  # noqa: E402
from faceattend.vision.passive_liveness import TemporalPassiveLivenessSession  # noqa: E402
from faceattend.vision.types import EvidenceDecision, FrameEvidence, ModelMetadata  # noqa: E402
from ml.liveness.mediapipe_active import (  # noqa: E402
    LivenessRuntimePhase,
    MediaPipeLivenessSession,
)


class _FixedChallengeSource:
    """Preserve an explicitly requested diagnostic challenge sequence."""

    def __init__(self, sequence: tuple[ChallengeAction, ...]) -> None:
        self.sequence = sequence

    def randint(self, start: int, stop: int) -> int:
        if not start <= len(self.sequence) <= stop:
            raise ValueError("explicit challenge count is outside configured limits")
        return len(self.sequence)

    def sample(self, population: list[ChallengeAction], k: int) -> list[ChallengeAction]:
        if k != len(self.sequence) or any(action not in population for action in self.sequence):
            raise ValueError("explicit challenge is outside the configured pool")
        return list(self.sequence)


def _artifact(path: Path, name: str, version: str) -> ModelArtifact:
    return ModelArtifact(path, ModelMetadata(name, version, file_sha256(path)))


def _manifest(models: Path) -> RuntimeModelManifest:
    return RuntimeModelManifest(
        detector=_artifact(models / "det_10g.onnx", "insightface-detector", "buffalo_l-det_10g"),
        embedding=_artifact(models / "w600k_r50.onnx", "buffalo_l_recognition", "buffalo_l"),
        passive=_artifact(models / "MiniFASNetV2.onnx", "MiniFASNetV2", "MiniFASNetV2"),
        landmarker=_artifact(
            models / "face_landmarker_v2_with_blendshapes.task",
            "mediapipe-face-landmarker",
            "face-landmarker-v2-with-blendshapes",
        ),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consent", action="store_true", help="Confirm consent for this trial.")
    parser.add_argument("--trial-id", required=True, help="Non-identifying trial ID.")
    parser.add_argument(
        "--presentation",
        required=True,
        choices=("bona_fide", "print", "phone", "prerecorded_video"),
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("proposal", "validation", "test"),
        help="Trial role. Tune only on proposal/validation; never tune on test trials.",
    )
    parser.add_argument("--lighting", required=True, choices=("dim", "normal", "bright"))
    parser.add_argument("--distance", required=True, choices=("near", "normal", "far"))
    parser.add_argument("--glasses", required=True, choices=("no", "clear", "dark"))
    parser.add_argument("--natural-movement", action="store_true")
    parser.add_argument(
        "--challenge",
        action="append",
        choices=tuple(action.value for action in ChallengeAction),
        help=(
            "Force an experimental challenge; repeat to specify an ordered sequence. "
            "Omit to use the production-default randomized pool."
        ),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--models", type=Path, default=repo_root / "models")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "local-biometric-recordings" / "liveness-trials.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.consent:
        print("ERROR: pass --consent only after the participant agrees to this local trial.")
        return 2
    if args.max_frames <= 0:
        print("ERROR: --max-frames must be positive")
        return 2
    manifest = _manifest(args.models)
    explicit_actions = tuple(ChallengeAction(value) for value in (args.challenge or ()))
    if explicit_actions:
        unique_actions = tuple(dict.fromkeys(explicit_actions))
        if len(unique_actions) != len(explicit_actions):
            print("ERROR: an explicit challenge sequence cannot contain duplicates")
            return 2
        challenge_config = ChallengeSessionConfig(
            min_challenges=len(explicit_actions),
            max_challenges=len(explicit_actions),
            challenge_pool=unique_actions,
        )
        challenge = ChallengeSession(
            challenge_config,
            random_source=_FixedChallengeSource(explicit_actions),
        )
    else:
        challenge = ChallengeSession(ChallengeSessionConfig())
    processor = create_face_analyzer(manifest)
    started_ns = time.monotonic_ns()
    challenge.start(started_ns // 1_000_000 - 1)
    evaluator = ActiveLivenessChallengeEvaluator(session=challenge)

    def extract_selected_embedding(evidence: FrameEvidence) -> Any:
        if evidence.face is None:
            raise ValueError("embedding candidate has no face")
        return processor.embedder.embed(processor.aligner.align(evidence.frame, evidence.face))

    runtime = MediaPipeLivenessSession(
        evaluator,
        TemporalPassiveLivenessSession(processor.passive),
        evidence_guard=EvidenceStreamGuard(),
        processor=processor,
        embedding_extractor=extract_selected_embedding,
    )
    worker = CameraEvidenceWorker(args.camera_index)
    samples: list[FrameEvidence] = []
    last_phase: str | None = None

    def receive(item: FrameEvidence) -> None:
        nonlocal last_phase
        samples.append(item)
        phase = runtime.instruction
        if phase != last_phase:
            print(f"ACTION: {phase}", flush=True)
            last_phase = phase
        if runtime.phase in {
            LivenessRuntimePhase.COMPLETED,
            LivenessRuntimePhase.FAILED,
            LivenessRuntimePhase.CANCELLED,
        }:
            worker.stop()

    try:
        print("No images will be saved. Keep exactly one presentation in view.")
        worker.run(runtime, max_frames=args.max_frames, result_capacity=0, on_evidence=receive)
        passive = runtime.finalize_passive()
    except (OSError, RuntimeError, ValueError) as exc:
        runtime.cancel()
        print(f"ERROR: {type(exc).__name__}: {exc}")
        passive = runtime.finalize_passive()
    finally:
        processor.close()

    qualities = [item.quality for item in samples if item.quality is not None]
    active = runtime.active.evaluator.result
    accepted = runtime.result.decision is EvidenceDecision.PASSED
    expected_live = args.presentation == "bona_fide"
    record: dict[str, Any] = {
        "schema_version": 1,
        "trial_id": args.trial_id,
        "recorded_at_unix_s": time.time(),
        "presentation": args.presentation,
        "split": args.split,
        "lighting_condition": args.lighting,
        "distance_condition": args.distance,
        "glasses_condition": args.glasses,
        "natural_movement": args.natural_movement,
        "camera_index": args.camera_index,
        "model_checksums": {
            role: getattr(manifest, role).metadata.checksum
            for role in ("detector", "embedding", "passive", "landmarker")
        },
        "challenge_sequence": [action.value for action in challenge.snapshot.sequence],
        "explicit_experimental_challenges": bool(explicit_actions),
        "completed_challenges": [action.value for action in challenge.snapshot.completed],
        "active_status": active.status.value,
        "active_reason": active.reason,
        "runtime_phase": runtime.phase.value,
        "passive_decision": passive.decision.value,
        "passive_reason": passive.reason,
        "passive_median": passive.median_score,
        "passive_minimum": passive.minimum_score,
        "passive_suspicious_frames": passive.suspicious_frame_count,
        "passive_processing_failures": passive.failure_to_process_count,
        "overall_accepted": accepted,
        "embedding_dimensions": int(runtime.embedding.size)
        if runtime.embedding is not None
        else None,
        "correct_decision": accepted == expected_live,
        "frames_observed": len(samples),
        "duration_ms": (time.monotonic_ns() - started_ns) / 1_000_000,
        "quality_medians": {
            "brightness": median(q.brightness for q in qualities) if qualities else None,
            "sharpness": median(q.sharpness for q in qualities) if qualities else None,
            "face_area_ratio": median(q.face_area_ratio for q in qualities) if qualities else None,
            "center_offset": median(q.center_offset for q in qualities) if qualities else None,
        },
        "failure_reasons": sorted(
            {item.failure_reason for item in samples if item.failure_reason is not None}
        ),
        "warning_counts": {
            warning: sum(warning in q.warnings for q in qualities)
            for warning in sorted({warning for q in qualities for warning in q.warnings})
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"Saved metrics only: {args.output}")
    return 0 if record["correct_decision"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
