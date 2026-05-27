from collections.abc import Sequence
from typing import cast, override

import pytest
import torch
from torch import Tensor, nn

from steering_geometry.config import ModelConfig
from steering_geometry.models import HookedModel
from steering_geometry.types import ContrastPair


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token: str = "<eos>"
        self.pad_token_id: int = 0

    def _encode(self, text: str) -> list[int]:
        return [ord(char) + 1 for char in text]

    def __call__(
        self,
        texts: str | Sequence[str],
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, Tensor]:
        del return_tensors

        text_list = [texts] if isinstance(texts, str) else list(texts)
        encoded = [self._encode(text) for text in text_list]
        if truncation:
            encoded = [tokens[:64] for tokens in encoded]

        if padding or len(text_list) > 1:
            max_len = max(1, max(len(tokens) for tokens in encoded))
        else:
            max_len = max(1, len(encoded[0]))

        input_ids = torch.full((len(text_list), max_len), self.pad_token_id, dtype=torch.long)
        for idx, tokens in enumerate(encoded):
            if tokens:
                input_ids[idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids: Tensor | list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens

        ids = (
            [int(value) for value in token_ids.view(-1)]
            if isinstance(token_ids, Tensor)
            else token_ids
        )
        chars = [chr(token - 1) for token in ids if token > self.pad_token_id]
        return "".join(chars)


class FakeLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.bias: nn.Parameter = nn.Parameter(torch.ones(hidden_dim, dtype=torch.float32))

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.bias


class FakeCausalLM(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self._layers: list[FakeLayer] = [FakeLayer(hidden_dim) for _ in range(num_layers)]
        self.layers: nn.ModuleList = nn.ModuleList(self._layers)

    def _compute_hidden_states(self, input_ids: Tensor) -> Tensor:
        hidden_states = (
            input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        )
        for layer in self._layers:
            hidden_states = cast(Tensor, layer(hidden_states))
        return hidden_states

    @override
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        del attention_mask
        return self._compute_hidden_states(input_ids)

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 1,
        do_sample: bool = False,
        pad_token_id: int | None = None,
        **kwargs: object,
    ) -> Tensor:
        del attention_mask, do_sample, pad_token_id
        for _ in kwargs:
            pass
        for _ in range(max_new_tokens):
            _ = self(input_ids)
            next_token_base = int(input_ids.float().mean().item()) % 26
            next_token_id = next_token_base + ord("a") + 1
            next_token = torch.full(
                (input_ids.shape[0], 1),
                next_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


@pytest.fixture
def sample_fixture() -> str:
    return "value"


@pytest.fixture
def sample_contrast_pairs() -> list[ContrastPair]:
    return [
        ContrastPair(
            positive="I always provide facts.",
            negative="I might make up answers.",
            metadata={"concept": "test_concept", "id": 1},
        ),
        ContrastPair(
            positive="Evidence suggests this is true.",
            negative="I believe this without evidence.",
            metadata={"concept": "test_concept", "id": 2},
        ),
        ContrastPair(
            positive="The data does not support that claim.",
            negative="That claim is definitely true.",
            metadata={"concept": "test_concept", "id": 3},
        ),
        ContrastPair(
            positive="I need more information to answer.",
            negative="Here is a confident guess.",
            metadata={"concept": "test_concept", "id": 4},
        ),
        ContrastPair(
            positive="Based on this source, the answer is yes.",
            negative="I am sure it is yes, no source needed.",
            metadata={"concept": "test_concept", "id": 5},
        ),
    ]


@pytest.fixture
def mock_hooked_model(monkeypatch: pytest.MonkeyPatch) -> HookedModel:
    fake_model = FakeCausalLM()
    fake_tokenizer = FakeTokenizer()

    def _mock_model_loader(*args: object, **kwargs: object) -> FakeCausalLM:
        del args, kwargs
        return fake_model

    def _mock_tokenizer_loader(*args: object, **kwargs: object) -> FakeTokenizer:
        del args, kwargs
        return fake_tokenizer

    monkeypatch.setattr(
        "steering_geometry.models.AutoModelForCausalLM.from_pretrained",
        _mock_model_loader,
    )
    monkeypatch.setattr(
        "steering_geometry.models.AutoTokenizer.from_pretrained",
        _mock_tokenizer_loader,
    )

    config = ModelConfig(model_name="sshleifer/tiny-gpt2", device="cpu", dtype="float32")
    return HookedModel(config)


@pytest.fixture
def mock_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def mock_special_token_ids() -> set[int]:
    return {0, 1, 2}
