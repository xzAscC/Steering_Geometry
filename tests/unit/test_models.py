"""Unit tests for HookedModel class."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from steering_geometry.config import ModelConfig
from steering_geometry.models import HookedModel


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock HuggingFace model with 32 layers."""
    mock = MagicMock()
    mock_layers = [MagicMock() for _ in range(32)]

    mock.model.layers = mock_layers
    inner_model = MagicMock()
    inner_model.layers = mock_layers
    mock.model = inner_model
    mock.parameters.return_value = iter([torch.zeros(1)])

    return mock


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer."""
    mock = MagicMock()
    mock.pad_token = None
    mock.eos_token = "<eos>"
    mock.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
    return mock


class TestResolveLayers:
    """Tests for the resolve_layers method."""

    def test_resolve_layers_middle(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test resolving a middle layer (0.5 -> 16 on 32-layer model)."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)
            result = hooked.resolve_layers([0.5])

        assert result == [16]

    def test_resolve_layers_first(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test resolving the first layer (0.0 -> 0)."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)
            result = hooked.resolve_layers([0.0])

        assert result == [0]

    def test_resolve_layers_last(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test resolving the last layer (1.0 -> 31 on 32-layer model)."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)
            result = hooked.resolve_layers([1.0])

        assert result == [31]

    def test_resolve_layers_multiple(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test resolving multiple layers."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)
            result = hooked.resolve_layers([0.25, 0.5, 0.75])

        assert result == [8, 16, 23]

    def test_resolve_layers_clamp_negative(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test that negative values are clamped to 0."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)
            result = hooked.resolve_layers([-0.5])

        assert result == [0]

    def test_resolve_layers_clamp_over_one(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test that values > 1.0 are clamped to 1.0."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)
            result = hooked.resolve_layers([1.5])

        assert result == [31]


class TestNumLayers:
    """Tests for the num_layers property."""

    def test_num_layers(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test that num_layers returns correct count."""
        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            hooked = HookedModel(config)

        assert hooked.num_layers == 32


class TestInit:
    """Tests for HookedModel initialization."""

    def test_init_sets_pad_token(self, mock_model: MagicMock) -> None:
        """Test that pad_token is set to eos_token if not present."""
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"

        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tok,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            HookedModel(config)

        assert mock_tok.pad_token == "<eos>"

    def test_init_preserves_existing_pad_token(self, mock_model: MagicMock) -> None:
        """Test that existing pad_token is not overwritten."""
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token = "<eos>"

        with (
            patch(
                "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tok,
            ),
        ):
            config = ModelConfig(model_name="test-model")
            HookedModel(config)

        assert mock_tok.pad_token == "<pad>"

    def test_init_dtype_mapping(self, mock_model: MagicMock, mock_tokenizer: MagicMock) -> None:
        """Test that dtype strings are correctly mapped."""
        with (
            patch("steering_geometry.models.AutoModelForCausalLM.from_pretrained") as mock_load,
            patch(
                "steering_geometry.models.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            mock_load.return_value = mock_model
            config = ModelConfig(model_name="test-model", dtype="bfloat16")
            HookedModel(config)

        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.bfloat16
