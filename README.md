# Reverse Image Search

Find similar images in a local collection using ImageNet ResNet50 embeddings,
FAISS nearest-neighbor retrieval, and a FastAPI browser gallery.

## Methods and fixes

Images share one decode/EXIF orientation/RGB/224x224 bilinear/ResNet50 preprocessing
pipeline. Global-average-pooled embeddings are L2 normalized; IndexFlatL2 ranks
by squared Euclidean distance (lower is closer). Index rows and JSON metadata are
appended together after successful extraction, fixing the original enumeration-ID
mismatch when non-images or corrupt files occur. Offline building is separate from
serving. Safe IDs route to sanitized RGB PNG copies; API responses contain no local paths.
No pickle is used. Artifact loading checks format, dimensions, counts and index checksum.

## Setup and use

Use a Python version supported by your TensorFlow/FAISS wheels (Python 3.11 or 3.12
is a practical starting point). Reuse a compatible environment when available.
Install dependencies only if needed; the following command includes TensorFlow.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python build_index.py --images "<local-image-directory>" --output artifacts/index
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, upload an image, and select 1–50 neighbors.
The API accepts multipart `file` at `POST /search?top_k=10` and returns `results`
with `id`, `distance`, and `image_url`. OpenAPI documentation is at `/docs`.
For a different index, set `SEARCH_INDEX_DIR` before starting uvicorn.
Run commands from this project folder. First use of ResNet50 can download ImageNet
weights into the normal Keras cache; no weights are bundled.

The builder recursively reads JPEG/PNG/WebP/BMP files; unsupported and corrupt
images are ignored. Oversized images are skipped. It rejects an empty collection,
an existing output, or output inside the input folder. Choose a new output and
restart the server to switch indexes. `build-report.json` records indexed/skipped
counts. Gallery copies take additional disk space but discard EXIF and local paths.

## Tests and results

```powershell
python -m unittest discover -s tests -v
```

Tests use real FAISS with deterministic synthetic vectors and generated tiny images;
they need no TensorFlow import, ImageNet weights, dataset or GPU. Coverage includes
row mapping after skips, artifact round-trip/corruption, nearest-neighbor order,
top-k capping, RGB conversion, empty indexes, upload validation and image serving.
See VERIFICATION.md for the actual execution evidence. No retrieval-quality metric
has been measured on a labeled real-image benchmark.

## Limitations

This is a local portfolio demo. It has no authentication, rate limiting or relevance
labels. Inference is serialized to limit model concurrency. Uploads are limited to
8 MiB and 20 million decoded pixels; for public deployment also enforce request-body
limits at a reverse proxy (multipart parsing happens before endpoint validation).
Load only locally built, trusted FAISS artifacts: checksums detect inconsistency,
not malicious artifact replacement. Images must be ones you have permission to use.
ImageNet similarity may miss domain-specific semantics. The real pretrained path
still requires a smoke test with user images and cached/downloadable weights.

## Attribution

Adapted from `Checkpoint2.ipynb`, preserving the original ResNet50 + FAISS + FastAPI
design. See ATTRIBUTION.md for source provenance and unresolved author/license details.
Datasets, indexes, weights and private paths are excluded from the publication copy.
