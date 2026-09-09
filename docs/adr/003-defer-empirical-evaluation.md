# ADR-003: Defer Phase 7 empirical evaluation

**Status:** Accepted
**Date:** 2026-09-09
**Deciders:** Project owner

## Context

FaceAttend has reproducible local evaluation tooling and a small exploratory
bona-fide trial ledger, but completing Phase 7 requires repeated consented
camera trials, presentation-attack instruments, validation/test split
discipline, recognition-score collection, performance profiling, and PAD model
comparison. The owner has chosen not to spend current project scope on this
empirical evaluation.

## Options Considered

1. Complete the full Phase 7 experimental protocol.
2. Preserve the tooling and defer remaining Phase 7 collection and analysis.

## Decision

Choose option 2. Defer all uncompleted Phase 7 evaluation tasks. Retain
MiniFASNetV2 as the existing PAD baseline under ADR-002. Preserve the protocol,
scripts, exploratory metrics, and privacy safeguards without treating them as
final evaluation evidence.

## Consequences

- The desktop application remains a functional local CV prototype.
- No accuracy, PAD, attack-resistance, calibration, latency/FPS, CPU/memory, or
  state-of-the-art claim may be made from the deferred work.
- Configured matching and PAD thresholds remain unvalidated baseline settings.
- Reopen this ADR before new formal trial collection, threshold promotion, or
  security/performance claims.

## References

- [Migration plan](../research/local-first-migration-plan.md)
- [Local liveness trial protocol](../research/liveness-trial-protocol.md)
- [ADR-002](002-defer-pad-model-comparison.md)
