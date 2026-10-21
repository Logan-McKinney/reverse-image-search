"""Shared RGB decoding, ResNet50 embeddings and FAISS row mapping."""
from pathlib import Path
import hashlib
import io
import json
import re
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000
SPEC = "resnet50-imagenet-avg-rgb224-bilinear-l2-v1"
SAFE_ID = re.compile(r"[0-9a-f]{32}")

def decode_image(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image must contain 1 byte to 8 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in {'JPEG', 'PNG', 'WEBP', 'BMP'}:
                    raise ValueError("Use JPEG, PNG, WebP, or BMP")
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError("Image exceeds 20 million pixels")
                im.load()
                return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or unsafe image") from exc

def unit_vector(vector):
    value = np.asarray(vector, dtype="float32").reshape(-1)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must be finite and nonzero")
    return np.ascontiguousarray(value / norm)

class ResNetEmbedder:
    dimension = 2048

    def __init__(self):
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        self.preprocess = preprocess_input
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def __call__(self, image):
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        batch = np.asarray(image, dtype="float32")[None, ...]
        return unit_vector(self.model(self.preprocess(batch), training=False).numpy()[0])

def safe_image_path(root, image_id):
    if not SAFE_ID.fullmatch(image_id):
        raise ValueError("Invalid result identifier")
    root = Path(root).resolve()
    path = (root / "images" / (image_id + ".png")).resolve()
    if not path.is_relative_to(root / "images"):
        raise ValueError("Image outside gallery")
    return path

class SearchStore:
    def __init__(self, index, rows, root, dimension):
        if index.ntotal != len(rows) or index.d != dimension:
            raise ValueError("FAISS index/metadata mismatch")
        ids = [row['id'] for row in rows]
        if len(set(ids)) != len(ids) or any(not SAFE_ID.fullmatch(i) for i in ids):
            raise ValueError("Invalid/duplicate metadata IDs")
        self.index, self.rows, self.root = index, rows, Path(root)

    @classmethod
    def load(cls, root):
        # Only load artifacts built locally: FAISS binary readers are not a sandbox.
        import faiss
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata['schema'] != 1 or metadata['embedding'] != SPEC:
            raise ValueError("Unsupported index format or embedding pipeline")
        if metadata['dimension'] != 2048:
            raise ValueError("Unexpected ResNet50 dimension")
        raw = (root / "vectors.faiss").read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata['index_sha256']:
            raise ValueError("Index checksum mismatch")
        index = faiss.deserialize_index(np.frombuffer(raw, dtype="uint8"))
        if not isinstance(index, faiss.IndexFlatL2):
            raise ValueError("Expected IndexFlatL2")
        store = cls(index, metadata['rows'], root, metadata['dimension'])
        for row in store.rows:
            if not safe_image_path(root, row['id']).is_file():
                raise ValueError("Missing gallery image")
        return store

    def search(self, vector, top_k):
        if not self.rows:
            raise ValueError("Index is empty")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        vector = unit_vector(vector)
        if len(vector) != self.index.d:
            raise ValueError("Query dimension differs from index")
        distances, indices = self.index.search(vector[None, :], min(top_k, len(self.rows)))
        return [dict(id=self.rows[int(i)]['id'], distance=float(d),
                     image_url="/images/" + self.rows[int(i)]['id'])
                for d, i in zip(distances[0], indices[0]) if 0 <= i < len(self.rows)]
