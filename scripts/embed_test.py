import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import itertools

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from backend.indexer.store import get_store


def load_model() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def get_face_embedding(app: FaceAnalysis, image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")

    if len(faces) > 1:
        print(f"Multiple faces in {image_path.name}. Using the first detected face")

    return faces[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main(image_dir: str) -> None:
    app = load_model()
    path = Path(image_dir)
    im2emb = {}
    for img in path.iterdir():
        emb = get_face_embedding(app, img)
        im2emb[img.name] = emb

    for combo in itertools.combinations(im2emb.keys(), 2):
        sim = cosine_similarity(im2emb[combo[0]], im2emb[combo[1]])
        print(f"Similarity between {combo[0]} and {combo[1]}: {sim}")

    index = get_store()
    for img, emb in im2emb.items():
        asyncio.run(index.add(embedding_id=img, embedding=emb, metadata={}))

    im = Path("images/1.jpg")
    emb = get_face_embedding(app, im)
    results = asyncio.run(index.search(embedding=emb, top_k=3))
    for match in results:
        print(match.embedding_id, match.score)


if __name__ == "__main__":
    main(sys.argv[1])
