---
type: "query"
date: "2026-09-06T19:32:18.968414+00:00"
question: "What happened in the first LOOK_UP camera validation trial?"
contributor: "graphify"
source_nodes: ["FaceContinuityTracker", "ChallengeAction"]
---

# Q: What happened in the first LOOK_UP camera validation trial?

## Answer

The proposal trial lookup-normal-01 was invalid rather than a LOOK_UP effectiveness failure. It processed only two valid-face frames in about 2.6 seconds, then emitted face_missing_too_long. The continuity tracker had conflated a slow valid-to-valid inference interval with an observed face disappearance. Continuity now applies max-gap failure only after an explicit missing-face observation; valid observations separated by slow CPU inference remain acceptable. Full regression is 239 passed and mypy src passes. LOOK_UP and SMILE remain disabled by default and require fresh repeated trials.

## Source Nodes

- FaceContinuityTracker
- ChallengeAction