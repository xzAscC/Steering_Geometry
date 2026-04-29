"""Token-level analysis for steering vector investigation.

This module provides tools for analyzing token-level activations and
their contribution to steering vector effectiveness.

Usage:
    # Visualize discriminative tokens
    uv run python -m steering_geometry.token_analysis visualize --concept honesty

    # Probe token-level separability
    uv run python -m steering_geometry.token_analysis probe --concept toxicity
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerBase

from .config import SUPPORTED_CONCEPTS, ModelConfig, TokenAnalysisConfig
from .extract import load_contrast_pairs
from .models import HookedModel
from .types import (
    DiscriminativeTokenResult,
    ProbeExperimentResult,
    ProbeLayerResult,
    TokenRecord,
)
from .utils import DISCRIMINATIVE_EPS, configure_logging, ensure_dir, safe_model_name

logger = logging.getLogger(__name__)


def _detokenize_token(
    tokenizer: PreTrainedTokenizerBase,
    token_id: int,
    token_ids_context: list[int],
) -> str:
    """Convert a single token_id to readable text, handling subword merging.

    Args:
        tokenizer: The tokenizer used to encode the text.
        token_id: The single token ID to decode.
        token_ids_context: The full sequence of token IDs for context.

    Returns:
        The decoded text for the token, with subword prefixes handled.
    """
    raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    decoded = tokenizer.convert_tokens_to_string([raw_token])
    return decoded


def extract_all_token_activations(
    model: HookedModel,
    texts: list[str],
    layers: list[int],
    label: str,
    config: TokenAnalysisConfig,
) -> dict[int, list[TokenRecord]]:
    """Extract activations for tokens in texts.

    Processes texts in batches, extracts activations for each layer, and creates
    TokenRecord objects for each token position (excluding padding tokens).
    Stops collecting tokens when tokens_per_class limit is reached.
    When config.last_n is set, only the last N non-padding tokens per sequence
    are collected; otherwise all tokens are collected.

    Args:
        model: The hooked model to extract activations from.
        texts: List of input texts to process.
        layers: List of absolute layer indices to extract activations from.
        label: Label for all tokens ("positive" or "negative").
        config: Configuration controlling batch size, token limits, and last_n.

    Returns:
        Dictionary mapping layer index to list of TokenRecord objects.
        Each list is flattened across all sequences, stopping at tokens_per_class.
    """
    token_records: dict[int, list[TokenRecord]] = {layer: [] for layer in layers}
    tokens_collected = {layer: 0 for layer in layers}

    for start in range(0, len(texts), config.batch_size):
        if all(tokens_collected[layer] >= config.tokens_per_class for layer in layers):
            break

        batch_texts = texts[start : start + config.batch_size]

        inputs = model.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        device = next(model.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        activations = model.get_activations(batch_texts, layers)
        batch_size, seq_len = input_ids.shape

        for batch_idx in range(batch_size):
            seq_input_ids = input_ids[batch_idx]
            seq_attention_mask = attention_mask[batch_idx]
            actual_len = int(seq_attention_mask.sum().item())
            token_ids_list = seq_input_ids[:actual_len].tolist()

            for pos in range(actual_len):
                if config.last_n is not None:
                    if config.last_n <= 0:
                        raise ValueError("last_n must be a positive integer")
                    if pos < actual_len - config.last_n:
                        continue
                token_id = int(seq_input_ids[pos].item())
                token_text = _detokenize_token(model.tokenizer, token_id, token_ids_list)

                for layer in layers:
                    if tokens_collected[layer] >= config.tokens_per_class:
                        continue

                    # activations[layer] shape: (batch_size, seq_len, hidden_dim)
                    activation = activations[layer][batch_idx, pos].detach().cpu()

                    record = TokenRecord(
                        token_id=token_id,
                        token_text=token_text,
                        activation=activation,
                        contrast_pair_idx=start + batch_idx,
                        position_in_sequence=pos,
                        label=label,
                    )
                    token_records[layer].append(record)
                    tokens_collected[layer] += 1

    return token_records


def compute_discriminative_scores(
    pos_records: list[TokenRecord],
    neg_records: list[TokenRecord],
) -> tuple[list[TokenRecord], list[TokenRecord]]:
    """Compute discriminative scores for tokens using the formula:
    s_i = (||h_i - μ_other||² - ||h_i - μ_own||²) / (||h_i - μ_own||² + ||h_i - μ_other||² + ε)

    Higher score = more discriminative (closer to own class, farther from other).

    Args:
        pos_records: List of TokenRecord objects for positive class.
        neg_records: List of TokenRecord objects for negative class.

    Returns:
        Tuple of (pos_records_with_scores, neg_records_with_scores) sorted by
        score descending (most discriminative first).

    Raises:
        ValueError: If either input list is empty.
    """
    if len(pos_records) == 0 or len(neg_records) == 0:
        msg = "Cannot compute discriminative scores with empty records"
        raise ValueError(msg)

    # Convert to float32 to avoid overflow in squared distance computation
    # (activations can have large values that overflow float16 when squared)
    pos_activations = torch.stack([r.activation for r in pos_records]).float()
    neg_activations = torch.stack([r.activation for r in neg_records]).float()

    pos_center = pos_activations.mean(dim=0)
    neg_center = neg_activations.mean(dim=0)

    pos_dist_other = ((pos_activations - neg_center) ** 2).sum(dim=1)
    pos_dist_own = ((pos_activations - pos_center) ** 2).sum(dim=1)
    pos_scores = (pos_dist_other - pos_dist_own) / (
        pos_dist_own + pos_dist_other + DISCRIMINATIVE_EPS
    )

    neg_dist_other = ((neg_activations - pos_center) ** 2).sum(dim=1)
    neg_dist_own = ((neg_activations - neg_center) ** 2).sum(dim=1)
    neg_scores = (neg_dist_other - neg_dist_own) / (
        neg_dist_own + neg_dist_other + DISCRIMINATIVE_EPS
    )

    for i, record in enumerate(pos_records):
        record.score = float(pos_scores[i].item())

    for i, record in enumerate(neg_records):
        record.score = float(neg_scores[i].item())

    pos_sorted = sorted(pos_records, key=lambda r: r.score, reverse=True)
    neg_sorted = sorted(neg_records, key=lambda r: r.score, reverse=True)

    return pos_sorted, neg_sorted


def select_top_k_tokens(
    pos_records: list[TokenRecord],
    neg_records: list[TokenRecord],
    top_k: int,
    concept: str = "",
    layer: int = 0,
) -> DiscriminativeTokenResult:
    """Select top-k most discriminative tokens for each class.

    Args:
        pos_records: List of TokenRecord objects for positive class (should have
            scores already computed).
        neg_records: List of TokenRecord objects for negative class (should have
            scores already computed).
        top_k: Number of top tokens to select per class.
        concept: Concept name for the result (default: empty string).
        layer: Layer index for the result (default: 0).

    Returns:
        DiscriminativeTokenResult containing the top-k tokens for each class.
    """
    k_pos = min(top_k, len(pos_records))
    k_neg = min(top_k, len(neg_records))

    top_positive = sorted(pos_records, key=lambda r: r.score, reverse=True)[:k_pos]
    top_negative = sorted(neg_records, key=lambda r: r.score, reverse=True)[:k_neg]

    return DiscriminativeTokenResult(
        concept=concept,
        layer=layer,
        top_positive=top_positive,
        top_negative=top_negative,
    )


def train_linear_probe(
    train_activations: Tensor,
    train_labels: Tensor,
    hidden_dim: int,
    epochs: int = 100,
    lr: float = 0.01,
) -> nn.Linear:
    """Train a linear probe using PyTorch nn.Linear + CrossEntropyLoss + Adam.

    Args:
        train_activations: Training activations with shape (n_train, hidden_dim).
        train_labels: Training labels with shape (n_train,), 0=negative, 1=positive.
        hidden_dim: Dimension of the hidden layer activations.
        epochs: Number of training epochs (default: 100).
        lr: Learning rate for Adam optimizer (default: 0.01).

    Returns:
        Trained nn.Linear probe for binary classification.
    """
    probe = nn.Linear(hidden_dim, 2)
    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    dataset = TensorDataset(train_activations, train_labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for batch_acts, batch_labels in dataloader:
            optimizer.zero_grad()
            logits = probe(batch_acts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    return probe


def evaluate_probe(
    probe: nn.Linear,
    test_activations: Tensor,
    test_labels: Tensor,
) -> ProbeLayerResult:
    """Evaluate probe and return metrics: accuracy, auc_score.

    Args:
        probe: Trained linear probe.
        test_activations: Test activations with shape (n_test, hidden_dim).
        test_labels: Test labels with shape (n_test,), 0=negative, 1=positive.

    Returns:
        ProbeLayerResult with layer_idx=-1 (caller should set correct index),
        train_accuracy=-1 (not computed here), test_accuracy, and auc_score.
    """
    probe.eval()
    with torch.no_grad():
        logits = probe(test_activations)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        test_accuracy = float(accuracy_score(test_labels.numpy(), predictions.numpy()))
        pos_probs = probs[:, 1].numpy()
        auc = float(roc_auc_score(test_labels.numpy(), pos_probs))

    return ProbeLayerResult(
        layer_idx=-1,
        train_accuracy=-1.0,
        test_accuracy=test_accuracy,
        auc_score=auc,
    )


class _Args(Protocol):
    """Protocol defining CLI arguments for token analysis."""

    command: str
    concept: str
    model: str
    output: str
    top_k: int
    tokens_per_class: int
    last_n: int | None
    log_level: str


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for token analysis CLI."""
    parser = argparse.ArgumentParser(
        prog="steering_geometry.token_analysis",
        description="Analyze token-level contributions to steering vectors",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Visualize top discriminative tokens for a concept",
    )
    visualize_parser.add_argument(
        "--concept",
        required=True,
        choices=SUPPORTED_CONCEPTS,
        help="Concept to analyze",
    )
    visualize_parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model name (default: sshleifer/tiny-gpt2)",
    )
    visualize_parser.add_argument(
        "--output",
        default="outputs/token_analysis/",
        help="Output directory for visualizations (default: outputs/token_analysis/)",
    )
    visualize_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top tokens to visualize (default: 50)",
    )
    visualize_parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Only use the last N tokens per sequence for scoring (default: all tokens)",
    )

    probe_parser = subparsers.add_parser(
        "probe",
        help="Probe token-level separability across layers",
    )
    probe_parser.add_argument(
        "--concept",
        required=True,
        choices=SUPPORTED_CONCEPTS,
        help="Concept to probe",
    )
    probe_parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model name (default: sshleifer/tiny-gpt2)",
    )
    probe_parser.add_argument(
        "--output",
        default="outputs/token_analysis/",
        help="Output directory for probe results (default: outputs/token_analysis/)",
    )
    probe_parser.add_argument(
        "--tokens-per-class",
        type=int,
        default=10000,
        help="Number of tokens to sample per class (default: 10000)",
    )

    return parser


def run_visualize(args: _Args) -> None:
    """Run token visualization subcommand.

    Loads contrast pairs, extracts token activations, computes discriminative
    scores, and outputs top-k tokens per layer to console and JSON file.

    Args:
        args: Parsed CLI arguments with concept, model, output, and top_k.
    """
    logger.info("Loading contrast pairs for concept: %s", args.concept)
    pairs = load_contrast_pairs(args.concept, num_pairs=500)

    positive_texts = [p.positive for p in pairs]
    negative_texts = [p.negative for p in pairs]

    logger.info("Loading model: %s", args.model)
    model_config = ModelConfig(model_name=args.model)
    model = HookedModel(model_config)

    # Relative layers 0.0-0.9 mapped to absolute indices
    relative_layers = [i / 9 for i in range(10)]
    absolute_layers = model.resolve_layers(relative_layers)
    logger.info("Analyzing layers: %s", absolute_layers)

    config = TokenAnalysisConfig(
        top_k=args.top_k,
        tokens_per_class=10000,
        batch_size=8,
        last_n=args.last_n,
    )

    logger.info("Extracting token activations for positive texts...")
    pos_records = extract_all_token_activations(
        model=model,
        texts=positive_texts,
        layers=absolute_layers,
        label="positive",
        config=config,
    )

    logger.info("Extracting token activations for negative texts...")
    neg_records = extract_all_token_activations(
        model=model,
        texts=negative_texts,
        layers=absolute_layers,
        label="negative",
        config=config,
    )

    # Process each layer sequentially for memory safety
    results: list[dict[str, object]] = []

    for layer in absolute_layers:
        logger.info("Processing layer %d...", layer)

        layer_pos = pos_records[layer]
        layer_neg = neg_records[layer]

        pos_scored, neg_scored = compute_discriminative_scores(layer_pos, layer_neg)

        result = select_top_k_tokens(
            pos_scored,
            neg_scored,
            top_k=args.top_k,
            concept=args.concept,
            layer=layer,
        )

        print(f"\n=== Layer {layer} ===")
        print(f"\nTop {args.top_k} positive tokens (discriminate toward {args.concept}):")
        for i, token in enumerate(result.top_positive, 1):
            print(f"  {i:3d}. '{token.token_text}' (score: {token.score:.4f})")

        print(f"\nTop {args.top_k} negative tokens (discriminate away from {args.concept}):")
        for i, token in enumerate(result.top_negative, 1):
            print(f"  {i:3d}. '{token.token_text}' (score: {token.score:.4f})")

        # Store for JSON output (without activation tensors)
        layer_data: dict[str, object] = {
            "layer": layer,
            "concept": args.concept,
            "top_positive": [
                {
                    "token_id": t.token_id,
                    "token_text": t.token_text,
                    "score": t.score,
                    "contrast_pair_idx": t.contrast_pair_idx,
                    "position_in_sequence": t.position_in_sequence,
                }
                for t in result.top_positive
            ],
            "top_negative": [
                {
                    "token_id": t.token_id,
                    "token_text": t.token_text,
                    "score": t.score,
                    "contrast_pair_idx": t.contrast_pair_idx,
                    "position_in_sequence": t.position_in_sequence,
                }
                for t in result.top_negative
            ],
        }
        results.append(layer_data)

        # Clear GPU memory
        torch.cuda.empty_cache()

    output_dir = ensure_dir(Path(args.output))
    model_slug = safe_model_name(args.model)
    output_file = output_dir / f"{args.concept}_{model_slug}.json"

    output_data = {
        "concept": args.concept,
        "model": args.model,
        "top_k": args.top_k,
        "layers": absolute_layers,
        "results": results,
    }

    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved results to %s", output_file)


def run_probe(args: _Args) -> None:
    """Run token probe subcommand.

    Loads contrast pairs, extracts token activations, trains linear probes
    on each layer using 80/20 stratified split, and saves metrics to JSON.

    Args:
        args: Parsed CLI arguments with concept, model, output, tokens_per_class.
    """
    logger.info("Loading contrast pairs for concept: %s", args.concept)
    pairs = load_contrast_pairs(args.concept, num_pairs=500)

    positive_texts = [p.positive for p in pairs]
    negative_texts = [p.negative for p in pairs]

    logger.info("Loading model: %s", args.model)
    model_config = ModelConfig(model_name=args.model)
    model = HookedModel(model_config)

    relative_layers = [i / 9 for i in range(10)]
    absolute_layers = model.resolve_layers(relative_layers)
    logger.info("Analyzing layers: %s", absolute_layers)

    config = TokenAnalysisConfig(
        tokens_per_class=args.tokens_per_class,
        batch_size=8,
    )

    logger.info("Extracting token activations for positive texts...")
    pos_records = extract_all_token_activations(
        model=model,
        texts=positive_texts,
        layers=absolute_layers,
        label="positive",
        config=config,
    )

    logger.info("Extracting token activations for negative texts...")
    neg_records = extract_all_token_activations(
        model=model,
        texts=negative_texts,
        layers=absolute_layers,
        label="negative",
        config=config,
    )

    experiment_result = ProbeExperimentResult(
        concept=args.concept,
        model_name=args.model,
        tokens_per_class=args.tokens_per_class,
    )

    for layer in absolute_layers:
        logger.info("Training probe for layer %d...", layer)

        layer_pos = pos_records[layer]
        layer_neg = neg_records[layer]

        pos_activations = torch.stack([r.activation for r in layer_pos]).float()
        neg_activations = torch.stack([r.activation for r in layer_neg]).float()

        pos_labels = torch.ones(len(layer_pos), dtype=torch.long)
        neg_labels = torch.zeros(len(layer_neg), dtype=torch.long)

        all_activations = torch.cat([pos_activations, neg_activations], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)

        train_acts, test_acts, train_labels, test_labels = train_test_split(
            all_activations.numpy(),
            all_labels.numpy(),
            test_size=config.test_size,
            stratify=all_labels.numpy(),
            random_state=config.random_seed,
        )

        train_acts = torch.from_numpy(train_acts)
        test_acts = torch.from_numpy(test_acts)
        train_labels = torch.from_numpy(train_labels)
        test_labels_tensor = torch.from_numpy(test_labels)

        hidden_dim = train_acts.shape[1]
        probe = train_linear_probe(train_acts, train_labels, hidden_dim)

        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_acts)
            train_preds = torch.argmax(train_logits, dim=1)
            train_accuracy = float(accuracy_score(train_labels.numpy(), train_preds.numpy()))

        test_result = evaluate_probe(probe, test_acts, test_labels_tensor)

        layer_result = ProbeLayerResult(
            layer_idx=layer,
            train_accuracy=train_accuracy,
            test_accuracy=test_result.test_accuracy,
            auc_score=test_result.auc_score,
        )
        experiment_result.layer_results.append(layer_result)

        print(f"\n=== Layer {layer} ===")
        print(f"  Train accuracy: {train_accuracy:.4f}")
        print(f"  Test accuracy:  {test_result.test_accuracy:.4f}")
        print(f"  AUC score:      {test_result.auc_score:.4f}")

        del probe, train_acts, test_acts, train_labels, test_labels_tensor
        del pos_activations, neg_activations, all_activations, all_labels
        torch.cuda.empty_cache()

    output_dir = ensure_dir(Path(args.output))
    model_slug = safe_model_name(args.model)
    output_file = output_dir / f"{args.concept}_{model_slug}_probe.json"

    output_data = {
        "concept": experiment_result.concept,
        "model_name": experiment_result.model_name,
        "tokens_per_class": experiment_result.tokens_per_class,
        "layer_results": [
            {
                "layer_idx": r.layer_idx,
                "train_accuracy": r.train_accuracy,
                "test_accuracy": r.test_accuracy,
                "auc_score": r.auc_score,
            }
            for r in experiment_result.layer_results
        ],
    }

    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved probe results to %s", output_file)


def main() -> None:
    """CLI entry point for token analysis."""
    args = cast(_Args, cast(object, _build_parser().parse_args()))
    configure_logging(level=args.log_level)

    if args.command == "visualize":
        run_visualize(args)
    elif args.command == "probe":
        run_probe(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()


__all__ = [
    "extract_all_token_activations",
    "_detokenize_token",
    "compute_discriminative_scores",
    "select_top_k_tokens",
    "train_linear_probe",
    "evaluate_probe",
    "run_visualize",
    "run_probe",
]
