# Local liveness trial protocol

**Status:** Deferred by ADR-003. The protocol and tools are preserved, but no
new trial collection is required in the current project scope.

> Do not present the existing exploratory records as calibrated accuracy,
> presentation-attack resistance, or a biometric-security evaluation. Reopen
> ADR-003 before collecting new trials or making such claims.

## Purpose

Measure FaceAttend's active-liveness completion, passive-PAD behavior, latency,
and attack acceptance without saving raw camera images. The current thresholds
are baselines, not validated security settings.

## Retention and consent

- Run a trial only after the participant explicitly agrees.
- Use non-identifying trial IDs. Do not put names, emails, or student IDs in them.
- Save only aggregate JSONL metrics under `local-biometric-recordings/`, which is
  ignored by Git. Ordinary frames and PAD windows stay in memory and are cleared
  after finalization, failure, or cancellation.
- Do not record or retain photos/video. A separate approved protocol is required
  if raw recordings later become necessary.
- Delete local metrics when they are no longer needed for the stated experiment.

## Split discipline

- `proposal`: exploratory trials used to propose threshold ranges.
- `validation`: fresh trials used to choose and freeze thresholds.
- `test`: held-out trials used only once for final reporting. Never tune on them.

Do not claim participant-independent performance from trials involving one person.
Where multiple participants are available, keep each participant in only one split.

## Minimum development matrix

Collect repeated trials rather than relying on one capture. Start with at least ten
normal bona-fide attempts and five attempts for each changed condition: dim light,
bright light, near distance, far distance, clear glasses, and natural movement.
Repeat the same matrix after changing thresholds. Record the camera, hardware,
commit, model checksums, and configuration version with the experiment report.
Use `--camera-placement level|above|below` to make camera placement an explicit
condition rather than an undocumented source of variation.

For attacks, perform at least ten declared attempts per instrument:

- printed photograph;
- phone displaying a still photograph;
- phone/display replay of a prerecorded challenge sequence;
- fixed challenge sequence that does not match the requested random order;
- face substitution during an otherwise valid session;
- frozen-frame/camera-feed failure simulation where feasible.

These small counts are engineering smoke evidence, not a security certification.
Masks, deepfakes, virtual-camera injection, and direct application tampering remain
unevaluated unless separately tested.

## Running one trial

```bash
PYTHONPATH=src uv run python scripts/run_liveness_trial.py \
  --consent --split proposal --trial-id bf-normal-01 \
  --presentation bona_fide --lighting normal --distance normal --glasses no \
  --camera-index 1 --camera-placement level
```

For an attack trial, point the camera at the declared presentation instrument and
change `--presentation` accordingly. Follow terminal challenge instructions; do
not help an attack complete a different random challenge than the displayed one.

## Experimental LOOK_UP and SMILE validation

These actions are intentionally absent from the production-default random pool.
Validate each independently before proposing a default-policy change:

```bash
PYTHONPATH=src uv run python scripts/run_liveness_trial.py \
  --consent --split proposal --trial-id lookup-normal-01 \
  --presentation bona_fide --lighting normal --distance normal --glasses no \
  --camera-index 1 --challenge look_up

PYTHONPATH=src uv run python scripts/run_liveness_trial.py \
  --consent --split proposal --trial-id smile-normal-01 \
  --presentation bona_fide --lighting normal --distance normal --glasses no \
  --camera-index 1 --challenge smile
```

Run at least ten fresh attempts per action under normal conditions, followed by
five attempts per relevant changed condition. For `LOOK_UP`, vary camera height
and near/far distance. For `SMILE`, include clear glasses, dim/bright lighting,
natural facial movement, and neutral non-smiling trials to check false triggers.

Record completion rate, timeout/failure reasons, completion time, action evidence,
quality measurements, and passive-PAD result. Do not enable an action by default
from a single success. A proposed enablement requires acceptable validation results,
documented accessibility limitations, frozen thresholds, and user approval of the
active-liveness policy change.

## Decision metrics

Report denominators with every rate:

- bona-fide completion and false-failure rate;
- timeout and per-failure-reason counts;
- attack acceptance rate per presentation type;
- passive minimum/median score and processing failures;
- completion-time median and upper percentiles;
- quality distributions by lighting, distance, and glasses condition.

Use proposal/validation score distributions to choose candidate quality and PAD
thresholds. Freeze the complete configuration before running test trials. A
threshold copied from another repository or inferred from one person is not a
validated project setting.

## Aggregate-only analysis

The runner records a local Git commit, Python/platform details, camera index and
placement, model hashes, active/PAD window configuration, quality aggregates,
and terminal outcome. It records neither raw frames nor templates.

Summarize one split without changing a policy:

```bash
PYTHONPATH=src uv run python scripts/summarize_liveness_trials.py \
  --input local-biometric-recordings/liveness-trials.jsonl \
  --split proposal --passive-threshold 0.85
```

Choose candidate passive thresholds using *only* the validation split:

```bash
PYTHONPATH=src uv run python scripts/calibrate_liveness.py \
  --input local-biometric-recordings/liveness-trials.jsonl --split validation
```

This command only prints candidate operating points. It cannot freeze or change
the desktop configuration. BPCER applies only to bona-fide trials that reached
PAD and yielded a score; active failures and PAD processing failures remain
separate false-failure/failure-to-process counts. ACER is therefore reported as
a protocol-compatible engineering estimate, not ISO certification.

## Recognition score analysis

Recognition evaluation uses a second ignored local JSONL file of *score-only*
records. Each line is limited to `split`, `score`, `is_genuine`, and a broad
condition label such as `normal`, `dim`, `glasses`, or `far`; it must never
contain an embedding, person identifier, template identifier, or image path.

For each comparison protocol, collect both genuine and impostor scores. Use
validation scores to propose an operating threshold and held-out test scores for
the final report:

```bash
PYTHONPATH=src uv run python scripts/summarize_recognition_scores.py \
  --input local-biometric-recordings/verification-scores.jsonl \
  --split validation --threshold 0.75
```

The summary reports FMR, FNMR, a discrete EER approximation, and condition
counts. Rank-1 identification, unknown rejection, ambiguity margin, and
template-aggregation experiments require a separate labelled protocol; they are
not implied by verification scores alone.
