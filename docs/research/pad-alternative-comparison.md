# PAD alternative comparison — Phase 7 candidate research

**Status:** Deferred by ADR-002. No new model has been downloaded, run, or
adopted by FaceAttend.

## Decision boundary

This note preserves an investigated alternative in case the project reopens PAD
model selection. The current scope retains MiniFASNetV2 and does not compare it
with another model. Reported benchmark numbers from different datasets and
capture protocols are not directly comparable to FaceAttend results.

## Candidate: OpenVINO anti-spoof-mn3

The Open Model Zoo publishes `anti-spoof-mn3`, a MobileNetV3 PAD model with
documented conversion/preprocessing and permissive source licensing. Its model
card describes a 128×128 BGR input, CelebA-Spoof training, roughly 3.02M
parameters and 0.15 GFLOPs. The card's ACER is a result under its own benchmark,
not an expected FaceAttend result.

References:

- [Open Model Zoo model card](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/anti-spoof-mn3/README.md)
- [OpenVINO model documentation](https://docs.openvino.ai/2023.3/omz_models_model_anti_spoof_mn3.html)

## Current baseline: MiniFASNetV2

FaceAttend currently runs the existing MiniFASNetV2 ONNX wrapper. The upstream
Silent-Face-Anti-Spoofing repository lists MiniFASNetV2 as a lightweight model
and warns that target-camera collection conditions affect results. Its current
threshold and preprocessing are project baselines, not calibrated security
settings.

Reference: [Silent Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing).

## Required fair comparison if reopened

Before any model selection:

1. Obtain the alternative model from its official source and verify license,
   checksum, input colour order, crop, size, normalisation, and output meaning.
2. Run both models using the identical consented validation protocol, camera,
   PAD window policy, and attack instruments.
3. Report APCER by attack instrument, BPCER, compatible ACER, FTO, p50/p95
   inference latency, warm/cold start, CPU, and memory.
4. Freeze a model and threshold only after validation; evaluate the selected
   configuration once on held-out trials.

Adding or changing a biometric model is an ADR-level/user-approved decision.
