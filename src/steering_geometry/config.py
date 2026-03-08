"""Configuration dataclasses for steering geometry package."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model loading and inference.

    Attributes:
        model_name: Name or path of the model to load.
        device: Device to run the model on ("auto", "cuda", "cpu", "mps").
        dtype: Data type for model weights ("float16", "bfloat16", "float32").
        trust_remote_code: Whether to trust remote code when loading the model.
    """

    model_name: str
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False


@dataclass
class ExtractionConfig:
    """Configuration for steering vector extraction.

    Attributes:
        layers: Relative layer positions (0.0-1.0) to extract activations from.
            For example, [0.4, 0.5, 0.6] extracts from 40%, 50%, 60% of model depth.
        method: Extraction method ("mean", "pca").
        batch_size: Batch size for processing contrast pairs.
        read_token_index: Token position to read activations from (-1 for last token).
    """

    layers: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    method: str = "mean"
    batch_size: int = 8
    read_token_index: int = -1


@dataclass
class ConceptConfig:
    """Configuration for a behavioral concept.

    Attributes:
        concept_name: Name of the concept (e.g., "honesty", "sycophancy").
        dataset_name: Name or path of the dataset containing contrast pairs.
        num_pairs: Number of contrast pairs to use for extraction.
    """

    concept_name: str
    dataset_name: str
    num_pairs: int = 500


@dataclass
class SteeringConfig:
    """Configuration for applying steering vectors.

    Attributes:
        multipliers: Scale factors multiplied by avg activation norm.
        num_samples: Number of negative samples to steer.
        seed: Random seed for reproducible sample selection.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for greedy decoding).
    """

    multipliers: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    num_samples: int = 10
    seed: int = 42
    max_new_tokens: int = 100
    temperature: float = 0.0


__all__ = [
    "ModelConfig",
    "ExtractionConfig",
    "ConceptConfig",
    "SteeringConfig",
]
