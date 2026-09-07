# Frame quality and session policy

Date: 2026-09-06. Scope: Phase 2C; implementation plus synthetic verification.
No camera trials, calibrated quality thresholds, or new model selection.

## Reference review

The following are source-code observations, not independently verified accuracy claims.
No reference implementation was copied wholesale.

| Reference | Useful ideas | What FaceAttend does differently |
|---|---|---|
| [face-liveness-check quality](https://github.com/AbdulazeezAde/face-liveness-check/blob/e8ea992d09b5edd743dd2270278d619d2d070e6d/src/face_liveness_check/quality.py) and [session](https://github.com/AbdulazeezAde/face-liveness-check/blob/e8ea992d09b5edd743dd2270278d619d2d070e6d/src/face_liveness_check/session.py) | Face-crop exposure/contrast, Laplacian sharpness, fingerprinting; session-level quality and lighting minima. | Keep interpretable measurements rather than a weighted confidence probability. Use exact recent-pixel hashes, not perceptual dHash, for a strict frozen-frame signal. Missing/invalid evidence terminates the attempt. PAD stays mandatory. |
| [face-biometrics-api photo validator](https://github.com/amoghgg/face-biometrics-api/blob/1b856fe0adff45c8f8cdc7e16d2577d8e5aaca67/backend/photo_validator.py) and [liveness](https://github.com/amoghgg/face-biometrics-api/blob/1b856fe0adff45c8f8cdc7e16d2577d8e5aaca67/backend/liveness_engine.py) | Face area, centering, region-based quality signals, explicit rejection results. | Reject every multiple-face event, not only a competing large face. Do not label glasses from edge density, masks from skin hue, or AI images from symmetry/noise. Do not use landmark-depth variance or texture as a replacement for MiniFASNet. |
| [rule-based-liveness-detection signals](https://github.com/Anantu-Rajesh/rule-based-liveness-detection/blob/d76717b210fb344480d914ce4d3836b640eba288/src/liveness/rule_based/signal_extractor.py), [landmarks](https://github.com/Anantu-Rajesh/rule-based-liveness-detection/blob/d76717b210fb344480d914ce4d3836b640eba288/src/liveness/rule_based/landmark_detector.py), [orchestration](https://github.com/Anantu-Rajesh/rule-based-liveness-detection/blob/d76717b210fb344480d914ce4d3836b640eba288/src/liveness/rule_based/active_liveness.py) | Separate signal extraction from state updates. | Use capture monotonic time, not synthesized 33 ms increments or wall-clock time. Failed pose does not become zero/neutral; retain matrix pose rather than copying its solvePnP axis extraction or angle clipping. Its inspected signal path does not supply calibrated image-quality or occlusion evidence. |

The photo-validator documentation says 15% minimum area/middle 60%, while its
constants use 12% and a 0.25 center offset. This discrepancy reinforces using
measured local validation rather than inheriting prose or numerical defaults.

## Implemented policy

`quality.py` measures a clipped BGR face ROI, but rejects a box crossing the image
boundary. It reports normalized mean grayscale brightness, fractions at intensity
≤8 and ≥247, Laplacian variance after resizing to 128×128, visible area fraction,
and maximum per-axis center offset from frame center. Pixel size and area are both
checked. Invalid or out-of-face five-point landmarks fail explicitly.

`QualityConfig` controls thresholds. Initial defaults are provisional engineering
settings: brightness 0.12–0.90, clipping fraction at most 0.50, sharpness at least
20, shortest face side at least 60 pixels, area 0.02–0.85, center offset at most
0.40. They are not paper results or calibrated security settings. The composed
analyzer accepts an injected config. Enrollment head turns must be included in
validation: sharpness and visibility change with pose.

Eye darkness, eye glare, and low lower-face texture produce **advisory warnings**.
They can reflect shadows, glasses, skin appearance, masks, facial hair, or ordinary
image content. They do not classify glasses/masks; clear glasses are not prohibited.
No skin-colour rule is used. Reliable semantic occlusion detection would require
a separately validated method; warning absence is not proof of unobstructed eyes
or mouth. Active evidence and passive PAD remain required even without warnings.

`EvidenceStreamGuard` is owned by each `MediaPipeLivenessSession`. It validates
capture time and sequence, freshness (default 1000 ms using the same monotonic
clock), consistent quality/lighting, finite pose/landmarks, and upstream failures.
It then applies `FaceContinuityTracker` and exact duplicate detection. Any failure
is sticky for that attempt. No-face events fail immediately; a gap beyond the
tracker's limit also fails when the next face arrives. Known track changes and
implausible center/scale jumps fail. Geometric continuity without trustworthy
identity evidence cannot detect all same-position face substitutions.

An exact pixel repeat within the recent three-frame history fails with
`duplicate_frame`. It is a frozen-feed/repeated-input policy, **not full replay
protection**. Different frames from a replay video or injected/deepfake feed can
pass this check. A deterministic/noiseless stationary camera can also trigger it;
measure false failures before changing policy. No duplicate contributes dwell or
PAD evidence after the flag is raised.

The headless analyzer owns detection and matrix pose. Legacy evidence building
checks supplied time/count/pose against capture/MediaPipe output; conflicting
sources fail. Physical yaw inversion occurs once at the challenge adapter,
leaving the raw evidence unchanged. Invalid blendshape scores fail rather than
creating an action.

The runtime observes active evidence before buffering the validated frame for
PAD. Buffering performs no PAD inference. Only successful active completion permits
temporal PAD inference. Input/native inference failures produce terminal results;
buffers clear on abort/finalization. `session.result` is the overall decision:
an active result of COMPLETED alone is not an overall liveness pass.

## Runtime connection

Create `MediaPipeLivenessSession(evaluator, TemporalPassiveLivenessSession(processor.passive),
processor=processor)` and pass that callable to `CameraEvidenceWorker.run`.
Use `finalize_passive()` after active completion and inspect `session.result`.
Own one processor and one session per stream/attempt; close the processor afterward.
Warm models before a timed attempt so first-load latency does not produce stale
frames. A session's injected test clock is only for deterministic headless tests.
The camera diagnostic still accumulates evidence: use a bounded frame limit.

## Evidence and next experiments

Synthetic tests cover exposure/blur/size/centering, advisory regions, continuity,
timestamps, duplicates, failure propagation, and camera → actual adapter wrappers
→ session → temporal PAD. Model SDK outputs are synthetic, not model benchmarks.

Pending: consented repeated trials varying camera/lighting/distance/pose/glasses,
masked and unobstructed subjects, exposure clipping and blur. Select quality
thresholds on validation attempts; report false failures by condition on held-out
attempts. Print/display/video attack acceptance remains unmeasured. Minimum PAD
window/sampling/retention and application-owned cancellation are separate Phase 2E
gates. No attendance, audit persistence, GUI, raw-image saving, or legacy removal
was added here.
