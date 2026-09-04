"""Tests for the injectable InsightFace recognition adapter."""

import numpy as np
import pytest

from faceattend.vision.embeddings import EmbeddingError, InsightFaceEmbedder


def _embedder(output: np.ndarray) -> InsightFaceEmbedder:
    return InsightFaceEmbedder(
        model_path="unused-in-test.onnx",
        model_loader=lambda: type("FakeModel", (), {"get_feat": lambda self, _image: output})(),
        model_checksum="abc123",
    )


def test_embedder_normalizes_recognition_output_and_exposes_metadata() -> None:
    embedder = _embedder(np.array([[3.0, 4.0]], dtype=np.float32))

    result = embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))

    assert result.dtype == np.float32
    assert result.shape == (2,)
    np.testing.assert_allclose(result, np.array([0.6, 0.8], dtype=np.float32))
    assert embedder.model_name == "buffalo_l_recognition"
    assert embedder.model_version == "buffalo_l"
    assert embedder.model_checksum == "abc123"


def test_embedder_loads_model_once() -> None:
    loads: list[int] = []

    def load() -> object:
        loads.append(1)
        return type(
            "FakeModel",
            (),
            {"get_feat": lambda self, _image: np.array([1.0, 0.0], dtype=np.float32)},
        )()

    embedder = InsightFaceEmbedder(model_path="unused", model_loader=load)
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    embedder.embed(image)
    embedder.embed(image)

    assert loads == [1]


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((112, 112), dtype=np.uint8),
        np.zeros((112, 112, 4), dtype=np.uint8),
        np.zeros((112, 112, 3), dtype=np.float32),
    ],
)
def test_embedder_rejects_invalid_aligned_image(image: np.ndarray) -> None:
    embedder = _embedder(np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(EmbeddingError, match="HWC uint8 BGR"):
        embedder.embed(image)


def test_embedder_rejects_zero_or_nonfinite_output() -> None:
    image = np.zeros((112, 112, 3), dtype=np.uint8)

    with pytest.raises(EmbeddingError, match="zero"):
        _embedder(np.zeros(2, dtype=np.float32)).embed(image)
    with pytest.raises(EmbeddingError, match="invalid"):
        _embedder(np.array([1.0, np.nan], dtype=np.float32)).embed(image)
