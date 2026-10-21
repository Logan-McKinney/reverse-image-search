# Verification

`python -m unittest discover -s tests -v`: **6 tests passed**, 2.491 seconds.
The tests used real FAISS IndexFlatL2 and real FastAPI multipart parsing with a
deterministic synthetic 2048-dimensional embedder. Checks covered metadata alignment
after ignored/corrupt files, persisted index/JSON loading, checksum corruption,
nearest-neighbor order, capped top-k, RGB conversion, invalid/empty/unsupported/oversized
uploads, invalid top-k, missing/empty indexes, safe image IDs and gallery responses.

A Uvicorn server was started on an ephemeral loopback port and shut down after a
successful HTTP smoke test: GET /, POST /search, GET the returned PNG image URL,
and GET /docs. Generated tiny images and synthetic embeddings were used. No manual
browser visual inspection or pretrained ResNet50 retrieval run was performed.

Remaining: run a real-image pretrained smoke test when weights and a user image
directory are available; no relevance benchmark is claimed. Confirm original
author/team credit and redistribution permission before public release.

Verified on 2026-10-21, Windows, Python 3.13.3.
The preinstalled TensorFlow was reused. Only FAISS, python-multipart and Uvicorn
were downloaded into a temporary test dependency directory; no ML framework was installed.
Initial sandbox attempts failed on Windows temporary-folder permissions. The suites
were rerun outside that sandbox and completed with exit code 0.

Dependency versions actually used are in `tested-environment.json`. Requirements
specify compatible ranges, not a complete cross-platform lockfile. Existing package
deprecation warnings were observed (Starlette/httpx, Keras/NumPy and TensorFlow/gast);
they did not fail the tests. No dataset was downloaded or copied. No network was trained.
