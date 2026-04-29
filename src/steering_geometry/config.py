"""Configuration dataclasses for steering geometry package."""

from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# Canonical Constants
# =============================================================================

SUPPORTED_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "allenai/OLMo-2-1124-7B",
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-1125-32B",
)

SUPPORTED_CONCEPTS: tuple[str, ...] = (
    "refusal",
    "polite",
    "sentiment",
)

DEFAULT_MODEL: str = "Qwen/Qwen3-1.7B"


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

    def __post_init__(self) -> None:
        """Auto-detect trust_remote_code for models that require it."""
        if self.model_name.startswith("allenai/"):
            self.trust_remote_code = True


@dataclass
class ExtractionConfig:
    """Configuration for steering vector extraction.

    Attributes:
        layers: Relative layer positions (0.0-1.0) to extract activations from.
            For example, [0.4, 0.5, 0.6] extracts from 40%, 50%, 60% of model depth.
        method: Extraction method ("mean", "pca", "weighted_mean", "discriminative").
        batch_size: Batch size for processing contrast pairs.
        read_token_index: Token position to read activations from (-1 for last token).
        top_k: Number of top tokens to select for discriminative method.
            Only used when method="discriminative". Default is None (uses 100 internally).
        data_mode: How to format contrast pair data ("prompt_only", "prompt_response").
        token_select: Token selection strategy ("all", "last_n").
        last_n: Number of trailing tokens for "last_n" token_select mode.
        seed: Deterministic subsampling seed.
    """

    layers: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8])
    method: str = "mean"
    batch_size: int = 8
    read_token_index: int = -1
    top_k: int | None = None
    data_mode: str = "prompt_only"
    token_select: str = "default"
    last_n: int = 1
    seed: int = 42


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
        steer_tokens: Number of generation steps to apply steering (None = all steps).
    """

    multipliers: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    num_samples: int = 10
    seed: int = 42
    max_new_tokens: int = 100
    temperature: float = 0.0
    steer_tokens: int | None = None


@dataclass
class JudgeConfig:
    """Configuration for judge model in evaluation.

    Attributes:
        model: Model identifier for the judge (e.g., "google/gemini-3.1-flash-lite-preview").
        api_base: Base URL for the API endpoint.
        temperature: Sampling temperature for judge responses (0.0 for deterministic).
        max_retries: Maximum number of retry attempts on API failure.
    """

    model: str = "google/gemini-3.1-flash-lite-preview"
    api_base: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_retries: int = 3


@dataclass
class MMLUConfig:
    """Configuration for MMLU benchmark evaluation.

    Attributes:
        num_questions: Number of MMLU questions to sample for evaluation.
        seed: Random seed for reproducible question selection.
        use_cot: Whether to use chain-of-thought prompting for evaluation.
    """

    num_questions: int = 10
    seed: int = 42
    use_cot: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for steering vector evaluation.

    Attributes:
        judge: Judge model configuration for response evaluation.
        mmlu: MMLU benchmark configuration.
        output_dir: Directory to save evaluation results.
    """

    judge: JudgeConfig = field(default_factory=JudgeConfig)
    mmlu: MMLUConfig = field(default_factory=MMLUConfig)
    output_dir: str = "data/eval"


@dataclass
class TDNVConfig:
    """Configuration for TDNV metrics computation.

    Attributes:
        num_pairs: Number of contrast pairs to use.
        batch_size: Batch size for processing activations.
        output_dir: Directory to save JSON results.
        plot_dir: Directory to save visualization plots.
        read_token_index: Token position to read activations from (-1 for last token).
    """

    num_pairs: int = 500
    batch_size: int = 8
    output_dir: str = "data/tdnv/"
    plot_dir: str = "plot/tdnv/"
    read_token_index: int = -1


@dataclass
class TokenAnalysisConfig:
    """Configuration for discriminative token analysis experiments.

    Attributes:
        top_k: Number of top tokens to select per layer for analysis.
        tokens_per_class: Number of tokens to sample per class (positive/negative).
        test_size: Fraction of tokens to use for testing (0.0-1.0).
        layers: Relative layer positions (0.0-1.0) to analyze.
        batch_size: Batch size for processing activations.
        random_seed: Random seed for reproducible token sampling.
        last_n: If set, only use the last N tokens per sequence for scoring.
            None means use all tokens (default behavior).
    """

    top_k: int = 50
    tokens_per_class: int = 10000
    test_size: float = 0.2
    layers: list[float] = field(default_factory=lambda: [i / 9 for i in range(10)])
    batch_size: int = 8
    random_seed: int = 42
    last_n: int | None = None


@dataclass
class StabilityComparisonConfig:
    """Configuration for steering vector stability comparison experiments.

    Attributes:
        concept: Name of the concept to analyze (e.g., "sentiment", "honesty").
        num_tokens: Number of tokens to use for extraction.
        num_runs: Number of extraction runs for comparison (must be >= 2).
        layers: Relative layer positions (0.0-1.0) to extract activations from.
        top_k: Number of top tokens for discriminative method.
        model_name: Name or path of the model to load.
        output_dir: Directory to save results.
    """

    concept: str = "sentiment"
    num_tokens: int = 10000
    num_runs: int = 3
    layers: list[float] = field(
        default_factory=lambda: [i / 10 for i in range(10)]
    )  # 0.0, 0.1, ..., 0.9
    top_k: int = 30
    model_name: str = "Qwen/Qwen3-1.7B"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_runs < 2:
            raise ValueError(f"num_runs must be at least 2 for comparison, got {self.num_runs}")


# Concept name mappings for paper display
_CONCEPT_DISPLAY_NAMES: dict[str, str] = {
    "refusal": "Safety",
    "polite": "Politeness",
    "sentiment": "Sentiment",
}

_PAPER_TO_CANONICAL: dict[str, str] = {
    "safety": "refusal",
    "politeness": "polite",
    "sentiment": "sentiment",
    "refusal": "refusal",
    "polite": "polite",
}


@dataclass
class StabilitySweepConfig:
    """Configuration for stability sweep experiments.

    Measures DiM steering vector stability via pairwise cosine similarity
    across multiple independent runs at varying sample sizes (N).

    Attributes:
        model_name: HuggingFace model identifier.
        concept: Concept name (canonical: refusal, polite, sentiment).
        n_values: Sample sizes to sweep.
        layers: Layer fractions to test (relative positions 0.0-1.0).
        num_runs: Number of independent runs per (N, layer) setting.
        seed: Base random seed for reproducibility.
        output_dir: Directory to save results and vectors.
        device: Torch device for model inference.
        dtype: Model weight data type.
    """

    model_name: str
    concept: str
    n_values: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])
    layers: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    num_runs: int = 5
    seed: int = 42
    output_dir: Path | str = field(default_factory=lambda: "outputs/stability_sweep")
    device: str = "auto"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Resolve paper names to canonical names
        canonical = _PAPER_TO_CANONICAL.get(self.concept)
        if canonical is not None:
            self.concept = canonical
        if self.concept not in SUPPORTED_CONCEPTS:
            raise ValueError(
                f"Unsupported concept '{self.concept}'. Must be one of {SUPPORTED_CONCEPTS}"
            )
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. Must be one of {SUPPORTED_MODELS}"
            )
        if self.num_runs < 2:
            raise ValueError(
                f"num_runs must be at least 2 for pairwise comparison, got {self.num_runs}"
            )

    @property
    def display_concept(self) -> str:
        """Return the paper display name for the concept."""
        return _CONCEPT_DISPLAY_NAMES.get(self.concept, self.concept.title())

    @property
    def canonical_concept(self) -> str:
        """Return the canonical codebase concept name."""
        return self.concept


@dataclass
class StabilitySweepBatchConfig:
    """Configuration for batched stability sweep experiments.

    Runs stability sweeps for multiple concepts under a single model load,
    avoiding redundant model loading when processing multiple concepts for
    the same model.

    Attributes:
        model_name: HuggingFace model identifier.
        concepts: List of concept names to sweep.
        n_values: Sample sizes to sweep.
        layers: Layer fractions to test (relative positions 0.0-1.0).
        num_runs: Number of independent runs per (N, layer) setting.
        seed: Base random seed for reproducibility.
        output_dir: Directory to save results and vectors.
        device: Torch device for model inference.
        dtype: Model weight data type.
    """

    model_name: str
    concepts: list[str]
    n_values: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])
    layers: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    num_runs: int = 5
    seed: int = 42
    output_dir: Path | str = field(default_factory=lambda: "outputs/stability_sweep")
    device: str = "auto"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.concepts:
            msg = "concepts list cannot be empty"
            raise ValueError(msg)
        for concept in self.concepts:
            canonical = _PAPER_TO_CANONICAL.get(concept)
            if canonical is not None:
                continue  # Will be canonicalized per-concept during sweep
            if concept not in SUPPORTED_CONCEPTS:
                raise ValueError(
                    f"Unsupported concept '{concept}'. Must be one of {SUPPORTED_CONCEPTS}"
                )
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. Must be one of {SUPPORTED_MODELS}"
            )
        if self.num_runs < 2:
            raise ValueError(
                f"num_runs must be at least 2 for pairwise comparison, got {self.num_runs}"
            )


@dataclass
class HarmBenchConfig:
    """Configuration for HarmBench evaluation.

    Attributes:
        classifier_model: Model to use as HarmBench classifier.
        classifier_api_base: Base URL for the classifier API endpoint.
        classifier_api_key: API key for the classifier service.
        behaviors_file: Path to harmbench CSV file, or empty for default download.
        max_completion_tokens: Maximum tokens for classifier completion analysis.
        max_retries: Maximum number of retry attempts on API failure.
    """

    classifier_model: str = "google/gemma-4-31B"
    classifier_api_base: str = "http://localhost:8000/v1"
    classifier_api_key: str = "EMPTY"
    behaviors_file: str = ""
    max_completion_tokens: int = 512
    max_retries: int = 3


@dataclass
class ORBenchConfig:
    """Configuration for OR-Bench over-refusal evaluation.

    Attributes:
        split: HuggingFace dataset split name for OR-Bench prompts.
        num_samples: Number of samples to use (0 = all prompts in split).
        seed: Random seed for reproducible sample selection.
    """

    split: str = "or-bench-hard-1k"
    num_samples: int = 0
    seed: int = 42


@dataclass
class MMLUProConfig:
    """Configuration for MMLU-Pro benchmark evaluation.

    Attributes:
        num_questions: Number of test questions (0 = all 12,032).
        n_shot: Number of few-shot examples from validation set.
        use_cot: Whether to enable chain-of-thought reasoning.
        seed: Random seed for reproducible question selection.
        categories: List of categories to evaluate (None = all 14 categories).
        max_new_tokens: Maximum tokens for generation (CoT needs more).
    """

    num_questions: int = 0
    n_shot: int = 5
    use_cot: bool = True
    seed: int = 42
    categories: list[str] | None = None
    max_new_tokens: int = 2048


__all__ = [
    "SUPPORTED_MODELS",
    "SUPPORTED_CONCEPTS",
    "DEFAULT_MODEL",
    "ModelConfig",
    "ExtractionConfig",
    "ConceptConfig",
    "SteeringConfig",
    "JudgeConfig",
    "MMLUConfig",
    "EvaluationConfig",
    "TDNVConfig",
    "TokenAnalysisConfig",
    "StabilityComparisonConfig",
    "StabilitySweepConfig",
    "StabilitySweepBatchConfig",
    "HarmBenchConfig",
    "ORBenchConfig",
    "MMLUProConfig",
]
