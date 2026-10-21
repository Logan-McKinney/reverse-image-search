# Reverse Image Search

A group project that finds visually similar images in a local collection using **ResNet50 embeddings and FAISS nearest-neighbor search**. A FastAPI backend serves a browser gallery where you can upload a query image and explore its closest matches. The original project was developed by **Logan McKinney, Patrick Conley, and Greg Pologruto**.

## How it works

1. Decode images, apply EXIF orientation, convert to RGB, and resize to 224×224 with consistent ResNet50 preprocessing.
2. Extract global-average-pooled ResNet50 features and L2-normalize each embedding.
3. Store vectors in a FAISS `IndexFlatL2` index and keep matching image metadata alongside it.
4. Embed an uploaded image, retrieve its nearest neighbors, and display gallery copies. Lower squared Euclidean distance means a closer embedding.

The repository adapts our original `Checkpoint2.ipynb` into separate index-building and API modules. Later Codex-assisted cleanup added consistent preprocessing, corrected metadata alignment when files are skipped, safe image IDs, and functional tests. See [attribution](ATTRIBUTION.md).

## Setup and use

Uses Python, TensorFlow/Keras, FAISS, FastAPI, Uvicorn, and Pillow. Choose a Python version supported by the TensorFlow and FAISS wheels you install. From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python build_index.py --images "<local-image-directory>" --output artifacts/index
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [the gallery](http://127.0.0.1:8000), upload an image, and select 1–50 neighbors. The API accepts a multipart `file` at `POST /search?top_k=10`; [interactive API docs](http://127.0.0.1:8000/docs) describe the response. Set `SEARCH_INDEX_DIR` to serve a different index.

The builder recursively reads JPEG, PNG, WebP, and BMP files, skipping unsupported, corrupt, or oversized images. Use a new output directory outside the input folder; `build-report.json` records indexed/skipped counts. First use can download ImageNet weights to the Keras cache. Supply images you have permission to use; datasets, indexes, gallery copies, and weights are excluded from Git.

## Tests and limitations

```sh
python -m unittest discover -s tests -v
```

The saved verification run passed **6 functional tests**, covering metadata alignment, index persistence and corruption checks, neighbor order, image conversion, upload validation, and gallery responses. Tests use real FAISS with synthetic vectors and tiny generated images, without loading ResNet50 weights. See [testing notes](TESTING.md).

No quantitative retrieval-accuracy benchmark has been established, and the pretrained real-image pipeline was not executed during cleanup. ImageNet features can miss domain-specific similarity. This is a local demo with no authentication or rate limiting; load only trusted, locally built indexes. Uploads are limited to 8 MiB and 20 million decoded pixels.
