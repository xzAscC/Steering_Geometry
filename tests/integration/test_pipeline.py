from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch
from torch import Tensor

from steering_geometry.config import ExtractionConfig
from steering_geometry.extraction import extract_steering_vector
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair, SteeringVector


class SerializedSteeringVector(TypedDict):
    layer_activations: dict[int, Tensor]
    model_name: str
    concept: str
    method: str


@pytest.mark.slow
def test_pipeline_extract_save_load_and_apply_vector(
    mock_hooked_model: HookedModel,
    sample_contrast_pairs: list[ContrastPair],
    tmp_path: Path,
) -> None:
    config = ExtractionConfig(
        layers=[0.5],
        method="mean",
        batch_size=2,
        read_token_index=-1,
    )

    vector = extract_steering_vector(
        model=mock_hooked_model,
        pairs=sample_contrast_pairs[:5],
        config=config,
    )

    assert vector.model_name == "sshleifer/tiny-gpt2"
    assert vector.concept == "test_concept"
    assert vector.method == "mean"
    assert len(vector.layer_activations) == 1

    vector_path = tmp_path / "steering_vector.pt"
    payload = {
        "layer_activations": {k: v.cpu() for k, v in vector.layer_activations.items()},
        "model_name": vector.model_name,
        "concept": vector.concept,
        "method": vector.method,
    }
    torch.save(payload, vector_path)

    loaded_payload = cast(SerializedSteeringVector, torch.load(vector_path))
    loaded_vector = SteeringVector(
        layer_activations=loaded_payload["layer_activations"],
        model_name=loaded_payload["model_name"],
        concept=loaded_payload["concept"],
        method=loaded_payload["method"],
    )

    assert loaded_vector.model_name == vector.model_name
    assert loaded_vector.concept == vector.concept
    assert loaded_vector.method == vector.method
    for layer_idx, layer_activation in vector.layer_activations.items():
        assert torch.allclose(loaded_vector.layer_activations[layer_idx], layer_activation.cpu())

    baseline = mock_hooked_model.generate(prompt="hello", max_new_tokens=1)
    steered = mock_hooked_model.generate(
        prompt="hello",
        max_new_tokens=1,
        steering_vector=loaded_vector,
    )

    assert isinstance(baseline, str)
    assert isinstance(steered, str)
    assert steered
