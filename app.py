"""Local gallery API; imports do not load weights or build an index."""
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import os
from threading import Lock
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from search_engine import MAX_BYTES, ResNetEmbedder, SearchStore, decode_image, safe_image_path

def create_app(store=None, embedder=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store, app.state.embedder = store, embedder
        if app.state.store is None:
            try:
                app.state.store = SearchStore.load(os.environ.get('SEARCH_INDEX_DIR', 'artifacts/index'))
            except Exception:
                logging.getLogger(__name__).exception('Index unavailable; build a local trusted index first')
        yield

    app = FastAPI(title='Reverse Image Search', lifespan=lifespan)
    inference_lock = Lock()

    @app.get('/', response_class=HTMLResponse)
    def gallery():
        return (Path(__file__).parent / 'static' / 'index.html').read_text(encoding='utf-8')

    @app.post('/search')
    def search(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=50)):
        try:
            if file.content_type not in {'image/jpeg', 'image/png', 'image/webp', 'image/bmp'}:
                raise HTTPException(415, 'Use JPEG, PNG, WebP, or BMP')
            contents = file.file.read(MAX_BYTES + 1)
            if len(contents) > MAX_BYTES:
                raise HTTPException(413, 'Upload exceeds 8 MiB')
            try:
                image = decode_image(contents)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            current = app.state.store
            if current is None or not current.rows:
                raise HTTPException(503, 'Index unavailable or empty; build it first')
            with inference_lock:
                if app.state.embedder is None:
                    try:
                        app.state.embedder = ResNetEmbedder()
                    except Exception as exc:
                        logging.getLogger(__name__).exception('ResNet50 unavailable')
                        raise HTTPException(503, 'Embedding model unavailable') from exc
                return {'results': current.search(app.state.embedder(image), top_k)}
        finally:
            file.file.close()

    @app.get('/images/{image_id}')
    def image(image_id: str):
        current = app.state.store
        if current is None:
            raise HTTPException(503, 'Index unavailable')
        try:
            path = safe_image_path(current.root, image_id)
        except ValueError as exc:
            raise HTTPException(404, 'Image not found') from exc
        if image_id not in {row['id'] for row in current.rows} or not path.is_file():
            raise HTTPException(404, 'Image not found')
        return FileResponse(path, media_type='image/png', headers={'X-Content-Type-Options': 'nosniff'})

    @app.get('/favicon.ico', include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    return app

app = create_app()
