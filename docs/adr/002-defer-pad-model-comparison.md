# ADR-002: Defer passive-PAD model comparison

**Status:** Accepted
**Date:** 2026-09-09
**Deciders:** Project owner

## Context

ADR-001 identified MiniFASNetV2 as a baseline and proposed comparison with a
second lightweight PAD model. That experiment requires additional model
artifacts, preprocessing validation, target-camera attack trials, and CPU
benchmarking. The project owner has chosen to keep the existing MiniFASNetV2
baseline and not spend current project scope on comparative PAD model selection.

## Options Considered

1. Compare MiniFASNetV2 with another lightweight PAD model now.
2. Retain MiniFASNetV2 and defer model comparison.

## Decision

Choose option 2. FaceAttend retains MiniFASNetV2 as the only passive-PAD model
in the current local prototype. No claim is made that it is optimal, more
accurate, or secure against a particular attack class.

## Consequences

- No alternative PAD model is downloaded, integrated, or benchmarked.
- The model-comparison task is marked deferred rather than completed.
- Existing liveness-before-attendance policy remains unchanged.
- If model selection or stronger PAD claims become important, reopen this ADR
  and perform a documented target-camera comparison before changing the model.

## References

- [ADR-001](001-local-desktop-cv-architecture.md)
- [PAD alternative research note](../research/pad-alternative-comparison.md)
