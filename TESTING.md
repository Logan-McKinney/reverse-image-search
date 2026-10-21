# Testing

```sh
python -m unittest discover -s tests -v
```

The saved October 21, 2026 verification run passed **6 tests** using real FAISS and FastAPI with deterministic synthetic 2048-dimensional embeddings and generated tiny images. It covered metadata alignment after skipped files, index/JSON round-trips and checksum checks, neighbor order and top-k limits, RGB conversion, upload validation, safe image IDs, and gallery responses.

A saved loopback Uvicorn smoke test also exercised the gallery, search, returned image URL, and API docs with synthetic embeddings. No pretrained ResNet50 retrieval run or relevance benchmark was performed.

The functional tests need FAISS, FastAPI, Pillow, NumPy, python-multipart, and httpx, but do not import TensorFlow or download weights. `requirements-dev.txt` includes the application and test dependencies.
