from types import SimpleNamespace

import torch

from steering_geometry.config import ExtractionConfig
from steering_geometry.extraction import (
    extract_steering_vector,
    mean_aggregator,
    pca_aggregator,
)
from steering_geometry.types import ContrastPair


class DummyModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_name="dummy-model")

    def resolve_layers(self, relative_layers: list[float]) -> list[int]:
        return [int(value * 10) for value in relative_layers]

    def get_activations(self, texts: list[str], layers: list[int]) -> dict[int, torch.Tensor]:
        max_length = max(len(text) for text in texts)
        hidden_dim = 4
        activations: dict[int, torch.Tensor] = {}

        for layer in layers:
            layer_tensor = torch.zeros((len(texts), max_length, hidden_dim), dtype=torch.float32)
            for batch_index, text in enumerate(texts):
                length = len(text)
                layer_tensor[batch_index, :length, :] = 1.0
                if text.startswith("pos"):
                    layer_tensor[batch_index, length - 1, :] += float(layer + 1)
            activations[layer] = layer_tensor

        return activations


def test_mean_aggregator() -> None:
    pos = torch.ones((10, 8), dtype=torch.float32)
    neg = torch.zeros((10, 8), dtype=torch.float32)

    vector = mean_aggregator(pos, neg)

    assert torch.allclose(vector, torch.ones((8,), dtype=torch.float32))


def test_pca_aggregator_returns_first_principal_component() -> None:
    pos = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    neg = torch.zeros_like(pos)

    vector = pca_aggregator(pos, neg)

    assert vector.shape == (3,)
    assert torch.isclose(vector.abs()[0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(vector[1:], torch.zeros(2), atol=1e-6)


def test_extract_steering_vector_batches_and_uses_last_non_padding_token() -> None:
    model = DummyModel()
    pairs = [
        ContrastPair(positive="pos-long", negative="neg", metadata={"concept": "honesty"}),
        ContrastPair(positive="pos", negative="neg-long", metadata={"concept": "honesty"}),
        ContrastPair(positive="pos-mid", negative="neg", metadata={"concept": "honesty"}),
    ]
    config = ExtractionConfig(
        layers=[0.4, 0.6],
        method="mean",
        batch_size=2,
        read_token_index=-1,
    )

    vector = extract_steering_vector(model=model, pairs=pairs, config=config)

    assert vector.model_name == "dummy-model"
    assert vector.concept == "honesty"
    assert vector.method == "mean"
    assert set(vector.layer_activations.keys()) == {4, 6}
    assert torch.allclose(vector.layer_activations[4], torch.full((4,), 5.0))
    assert torch.allclose(vector.layer_activations[6], torch.full((4,), 7.0))
