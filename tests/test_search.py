import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import faiss
from fastapi.testclient import TestClient
from app import create_app
from build_index import build
from search_engine import MAX_BYTES, SearchStore, decode_image, safe_image_path, unit_vector

class SyntheticEmbedder:
    dimension = 2048
    def __call__(self, image):
        vector = np.zeros(self.dimension, dtype='float32')
        vector[:3] = np.asarray(image.convert('RGB'), dtype='float32').mean(axis=(0, 1)) + 1
        return unit_vector(vector)

def image_bytes(mode='RGB', color='red'):
    stream = io.BytesIO()
    Image.new(mode, (8, 8), color).save(stream, format='PNG')
    return stream.getvalue()

class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / 'input'
        self.images.mkdir()
        (self.images / '00-not-image.txt').write_text('ignore')
        (self.images / '01-broken.jpg').write_bytes(b'broken')
        (self.images / '02-red.png').write_bytes(image_bytes())
        (self.images / '03-blue.png').write_bytes(image_bytes(color='blue'))
        self.output = self.root / 'index'
        self.embedder = SyntheticEmbedder()
        self.report = build(self.images, self.output, self.embedder)
        self.store = SearchStore.load(self.output)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_mapping_after_skipped_files_and_roundtrip(self):
        self.assertEqual(self.report, dict(indexed=2, skipped=1))
        expected = hashlib.sha256(b'02-red.png').hexdigest()[:32]
        results = self.store.search(self.embedder(decode_image(image_bytes())), 50)
        self.assertEqual(len(results), 2)  # no FAISS -1 sentinel returned
        self.assertEqual(results[0]['id'], expected)
        self.assertAlmostEqual(results[0]['distance'], 0, places=6)
        self.assertNotIn(str(self.images), json.dumps(results))

    def test_bad_artifacts_and_safe_path(self):
        with self.assertRaises(ValueError):
            SearchStore(faiss.IndexFlatL2(2048), self.store.rows, self.output, 2048)
        with self.assertRaises(ValueError):
            safe_image_path(self.output, '../secret')
        with (self.output / 'vectors.faiss').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            SearchStore.load(self.output)

    def test_rgb_and_invalid_decoding(self):
        for mode, color in [('L', 128), ('RGBA', (255, 0, 0, 100))]:
            self.assertEqual(decode_image(image_bytes(mode, color)).mode, 'RGB')
        for invalid in (b'', b'not an image'):
            with self.assertRaises(ValueError):
                decode_image(invalid)
        with self.assertRaises(ValueError):
            unit_vector([0, 0, 0])
        with self.assertRaises(ValueError):
            unit_vector([float('nan'), 1])
        gif = io.BytesIO()
        Image.new('RGB', (2, 2)).save(gif, format='GIF')
        with self.assertRaisesRegex(ValueError, 'Use JPEG'):
            decode_image(gif.getvalue())

    def test_missing_index_returns_service_unavailable(self):
        from unittest.mock import patch
        with patch('app.SearchStore.load', side_effect=FileNotFoundError), patch('app.logging.getLogger'):
            with TestClient(create_app(embedder=self.embedder)) as client:
                self.assertEqual(client.get('/').status_code, 200)
                response = client.post('/search', files={'file': ('a.png', image_bytes(), 'image/png')})
                self.assertEqual(response.status_code, 503)

    def test_empty_index_and_builder(self):
        empty = SearchStore(faiss.IndexFlatL2(2048), [], self.output, 2048)
        with TestClient(create_app(empty, self.embedder)) as client:
            response = client.post('/search', files={'file': ('a.png', image_bytes(), 'image/png')})
            self.assertEqual(response.status_code, 503)
        folder = self.root / 'empty'
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, 'No valid images'):
            build(folder, self.root / 'empty-output', self.embedder)
        self.assertFalse((self.root / 'empty-output').exists())

    def test_search_api_gallery_and_validation(self):
        with TestClient(create_app(self.store, self.embedder)) as client:
            self.assertIn('fetch(', client.get('/').text)
            valid = {'file': ('query.png', image_bytes(), 'image/png')}
            response = client.post('/search?top_k=1', files=valid)
            self.assertEqual(response.status_code, 200)
            result = response.json()['results'][0]
            self.assertEqual(client.get(result['image_url']).status_code, 200)
            self.assertEqual(client.get('/images/' + '0' * 32).status_code, 404)
            self.assertEqual(client.get('/images/secret').status_code, 404)
            self.assertEqual(client.post('/search', files={}).status_code, 422)
            for k in (0, 51, 'bad'):
                self.assertEqual(client.post('/search?top_k=' + str(k), files=valid).status_code, 422)
            cases = [(b'bad', 'image/png', 400), (b'', 'image/png', 400),
                     (b'bad', 'text/plain', 415), (b'x' * (MAX_BYTES + 1), 'image/png', 413)]
            for data, mime, status in cases:
                with self.subTest(status=status, size=len(data)):
                    reply = client.post('/search', files={'file': ('query', data, mime)})
                    self.assertEqual(reply.status_code, status)

if __name__ == '__main__':
    unittest.main()
