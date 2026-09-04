# ml/CLAUDE.md — Machine Learning & Inference

> Read the root [`CLAUDE.md`](../CLAUDE.md) first for project-wide rules.

This directory contains all ML models and inference pipelines: face detection, alignment, embedding, liveness/antispoofing, clustering, and perceptual hashing.

---

## Layout

```
ml/
├── detection/
│   ├── scrfd.py            # SCRFD face detector
│   ├── retinaface.py       # RetinaFace alternative
│   └── base.py             # FaceDetector interface
├── alignment/
│   └── align.py            # 5-point landmark alignment
├── embedding/
│   ├── arcface.py          # ArcFace embedder
│   ├── adaface.py          # AdaFace alternative
│   └── base.py             # FaceEmbedder interface
├── liveness/
│   ├── passive.py          # Passive liveness (single image)
│   ├── active.py           # Active liveness (challenge-response)
│   ├── minifasnet.py       # MiniFASNet model wrapper
│   ├── deepfake_detect.py  # Deepfake detection
│   └── base.py             # LivenessChecker interface
├── clustering/
│   ├── cluster.py          # Online clustering of embeddings
│   └── merge.py            # Cluster merge/split logic
├── phash/
│   └── hash.py             # Perceptual image hashing
├── pipeline/
│   ├── ingest.py           # End-to-end: image → [face_embeddings]
│   └── query.py            # End-to-end: query image → search
├── serving/
│   ├── triton/             # Triton model configs
│   ├── ray_serve/          # Ray Serve deployments
│   └── api.py              # Thin HTTP wrapper for non-Triton path
├── models/                 # Downloaded model weights (gitignored)
├── benchmarks/
└── tests/
```

---

## The Inference Pipeline

### Enrollment & search path (per query image)

```
Image Bytes
   │
   ▼
[Liveness Check]──▶ FAIL ──▶ Reject
   │ PASS
   ▼
[Face Detection]──▶ NO FACE ──▶ Reject
   │
   ▼ (one or more bboxes + landmarks)
[Alignment]
   │
   ▼ (aligned 112x112 face crops)
[Embedding]
   │
   ▼ (one 512-dim L2-normalized vector per face)
[Output]
```

### Crawler ingest path (per crawled image)

```
Image Bytes
   │
   ▼
[Perceptual Hash]──▶ DUPLICATE ──▶ Skip
   │
   ▼
[Face Detection]──▶ NO FACE ──▶ Mark as indexed-no-face
   │
   ▼
[Alignment]
   │
   ▼
[Embedding]
   │
   ▼ (per-face embeddings)
[Cluster Assignment]
   │
   ▼
[Index Write] → Vector DB
```

**Note**: liveness is **not** run on crawled images — it only applies to user-submitted face registration/check-in images.

---

## Model Choices (Defaults)

| Stage | Model | Why |
|---|---|---|
| Detection | **SCRFD-10G** | Fast, accurate, ONNX-friendly |
| Alignment | 5-point similarity transform | Standard, deterministic |
| Embedding | **ArcFace (r100)** | Industry standard, well-benchmarked |
| Passive Liveness | **MiniFASNet v2** | Lightweight, good accuracy |
| Active Liveness | Custom blink/turn detector | Built on MediaPipe landmarks |
| Deepfake | **EfficientNet-based detector** | TBD — evaluate during ADR |
| Perceptual Hash | **pHash** (DCT-based) | Simple, robust to compression |

Models can be swapped — interfaces are defined in `base.py` files. Document any swap as an ADR.

---

## Serving Strategy

Two serving paths supported, configurable per environment:

### Option A: Triton Inference Server (preferred for production)
- Each model packaged in `ml/serving/triton/<model>/` with `config.pbtxt`
- TensorRT engines for GPU; ONNX runtime fallback for CPU
- Dynamic batching enabled (batch size 1-32, max latency 50ms)
- Models hot-reload on file change

### Option B: Ray Serve (simpler for dev / smaller scale)
- Each model is a Ray Serve deployment
- Horizontal autoscaling based on queue depth
- Easier to debug, lower ops overhead

Choose with `INFERENCE_BACKEND=triton|ray` env var.

---

## Liveness in Detail

Liveness has two modes; backend chooses based on confidence.

### Passive (default — single image or short clip)
1. Frontend captures a single image or 1-second clip via webcam
2. Backend forwards to `/v1/liveness/passive` (ML service)
3. Model returns `{passed: bool, score: float, reason: str | None}`
4. If `score >= LIVENESS_THRESHOLD_PASSIVE` (default 0.85) → pass
5. If `score < threshold` → fall back to active

### Active (fallback — challenge-response)
1. Backend issues a challenge: `"blink twice"`, `"turn head left"`, `"smile"`
2. Frontend records a short video while user performs the action
3. Backend forwards to `/v1/liveness/active`
4. Model verifies the action was performed and the same face is present throughout
5. Returns `{passed: bool, score: float, challenge_completed: bool}`

### Deepfake check
Run on both passive and active inputs. Lightweight EfficientNet-based binary classifier — rejects obvious AI-generated faces.

### Calibration
Thresholds are conservative defaults; recalibrate against your own FAR/FRR targets:
- **FAR (False Accept Rate)** = % spoofs that pass — should be <0.1%
- **FRR (False Reject Rate)** = % real users rejected — should be <5%

Run `scripts/calibrate_liveness.py` against the evaluation set after any model swap.

---

## Embeddings

### Format
- **512-dim float32**
- **L2-normalized** at inference output (so cosine = dot product)
- Embeddings are biometric data — treat as PII

### Versioning
Every embedding stored with `embedding_model_version` metadata, e.g. `"arcface-r100-v2.1"`. Required because:
1. Embeddings from different models are **not comparable**
2. When upgrading models, you must re-embed the entire index
3. Search must filter by current model version

### Re-embedding pipeline
When promoting a new model:
1. Deploy new model alongside old one (dual-write period)
2. Re-embed all images in the background (`indexer/reembed.py`)
3. Once index parity is reached, flip the read path to new version
4. Deprecate old embeddings after grace period

This is heavy — plan capacity for it (a 100M image index re-embed at 1000 img/sec = ~28 hours of GPU time).

---

## Clustering

Faces of the same person get clustered into identity groups. Two-phase approach:

1. **Online clustering** at index time: each new embedding is compared against cluster centroids; if cosine similarity > threshold, join the cluster; otherwise start a new one
2. **Offline refinement**: periodic batch job (DBSCAN or HDBSCAN) consolidates fragmented clusters

Why clustering matters:
- UX: "this person appears in 47 sources" instead of 47 raw hits
- Performance: search can short-circuit at cluster level
- Moderation: admins can review identity clusters, manually merge/split

Threshold tuning matters — too low → over-merging (false identities), too high → fragmentation. Default cosine threshold: **0.65** (calibrate with `scripts/cluster_eval.py`).

---

## Perceptual Hashing

Sometimes the same image appears across sources without a visible face (e.g. cropped, watermarked). pHash catches these.

- Computed on every crawled image (in crawler pipeline)
- Stored alongside the face embedding for the image
- Used for image-level dedup AND as a "same image, different face/no face" matching signal

---

## Metrics & Monitoring

Each inference service emits:
- `ml_inference_latency_ms{model,stage}` — histogram, with p50/p95/p99
- `ml_inference_requests_total{model,status}` — counter
- `ml_gpu_utilization{device}` — gauge
- `ml_batch_size{model}` — histogram
- `ml_liveness_failures_total{reason}` — counter (passive_lowscore, active_failed, deepfake_detected)
- `ml_model_version_info{model,version}` — gauge with metadata

---

## Benchmarks

Before promoting any model, run `ml/benchmarks/`:

```bash
uv run python ml/benchmarks/detection.py --model scrfd-10g
uv run python ml/benchmarks/embedding.py --model arcface-r100
uv run python ml/benchmarks/liveness.py --model minifasnet-v2
```

Outputs latency, throughput, accuracy on the eval set.

---

## Testing

```bash
# Unit tests (no GPU required, uses small dummy models)
uv run pytest ml/tests/

# Full inference tests (requires ONNX runtime or GPU)
uv run pytest ml/tests/ --slow
```

Fixture face images live in `ml/tests/fixtures/` — synthetic only, no real biometric data in the repo.

---

## Common Tasks

### Swap an embedding model
1. Add new model wrapper in `ml/embedding/<model>.py` implementing `FaceEmbedder`
2. Convert weights: PyTorch → ONNX → TensorRT
3. Add Triton config in `ml/serving/triton/<model>/`
4. Run benchmark suite, document results
5. Write ADR
6. Deploy alongside old model (dual-write)
7. Run re-embedding pipeline
8. Flip read path

### Add a new liveness check
1. Implement `LivenessChecker` interface in `ml/liveness/<check>.py`
2. Wire into the liveness orchestrator (`ml/liveness/__init__.py`)
3. Add unit + integration tests
4. Calibrate thresholds against eval set

### Lower inference latency
1. Profile first — use `nvidia-nsight` or `pyinstrument`
2. Check dynamic batching is working (look at `ml_batch_size` p50)
3. Try fp16 / int8 quantization (validate accuracy drop)
4. Consider TensorRT optimization profiles for variable input sizes

---

## What NOT to Do (ML-Specific)

- ❌ Do **not** check in model weights — store in object storage, download at startup
- ❌ Do **not** ship a model without benchmark results
- ❌ Do **not** mix embeddings from different model versions without filtering
- ❌ Do **not** skip L2 normalization on embeddings
- ❌ Do **not** log raw face crops or embeddings in production
- ❌ Do **not** call models per-image in a Python for-loop — always batch
- ❌ Do **not** use fp16/int8 without validating accuracy on the eval set
- ❌ Do **not** train on user data without an explicit opt-in flow (and ADR + legal review)