"""Offline index builder. Run separately from uvicorn."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from search_engine import MAX_BYTES, SPEC, ResNetEmbedder, decode_image, unit_vector

def build(image_dir, output, embedder=None):
    import faiss
    source, output = Path(image_dir).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError("Image directory does not exist")
    if output.exists():
        raise ValueError("Output exists; choose a fresh output directory")
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the image directory")
    embedder = embedder or ResNetEmbedder()
    rows, vectors, skipped = [], [], 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "index"
        (stage / "images").mkdir(parents=True)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                continue
            if not path.resolve().is_relative_to(source):
                continue
            try:
                if path.stat().st_size > MAX_BYTES:
                    raise ValueError("Oversized image")
                image = decode_image(path.read_bytes())
            except (ValueError, OSError):
                skipped += 1
                continue
            vector = unit_vector(embedder(image))
            if vector.shape != (embedder.dimension,):
                raise ValueError("Unexpected embedding dimension")
            image_id = hashlib.sha256(path.relative_to(source).as_posix().encode()).hexdigest()[:32]
            image.save(stage / "images" / (image_id + '.png'))
            # Append metadata only after successful extraction: row i always means rows[i].
            rows.append({'id': image_id})
            vectors.append(vector)
        if not rows:
            raise ValueError("No valid images found; index was not created")
        index = faiss.IndexFlatL2(embedder.dimension)
        index.add(np.ascontiguousarray(vectors, dtype='float32'))
        faiss.write_index(index, str(stage / 'vectors.faiss'))
        metadata = dict(schema=1, embedding=SPEC, dimension=embedder.dimension, rows=rows,
                        index_sha256=hashlib.sha256((stage / 'vectors.faiss').read_bytes()).hexdigest())
        (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        (stage / 'build-report.json').write_text(json.dumps(dict(indexed=len(rows), skipped=skipped)), encoding='utf-8')
        stage.rename(output)
    return dict(indexed=len(rows), skipped=skipped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', default='artifacts/index')
    args = parser.parse_args()
    print(build(args.images, args.output))
