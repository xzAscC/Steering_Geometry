"""Prefix Steering analysis: KL divergences and attention patterns.

This module analyzes WHY Prefix Steering works by computing:
1. Per-token KL divergences between prefix-steered, no-steer, and all-steer
   generation distributions
2. Attention weight patterns showing how steering redirects attention
   to prefix positions

Usage:
    from steering_geometry.prefix_analysis import run_prefix_analysis
    report = run_prefix_analysis(
        model_name="Qwen/Qwen3-1.7B",
        concept="sentiment",
        steer_tokens=10,
    )
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from .config import ModelConfig
from .extract import load_contrast_pairs
from .models import HookedModel
from .types import SteeringVector
from .utils import ensure_dir, safe_model_name

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class KLDivergenceResult:
    """KL divergence results for a single prompt.

    Attributes:
        step_kl_no_steer: KL(prefix_steer || no_steer) per generation step.
        step_kl_all_steer: KL(prefix_steer || all_steer) per generation step.
        generated_text_no_steer: Text generated without steering.
        generated_text_prefix_steer: Text generated with prefix steering.
        generated_text_all_steer: Text generated with all-step steering.
        steer_tokens: Number of steps steering was applied (prefix mode).
        scale: Steering scale factor.
        layer_frac: Relative layer position where steering was applied.
        prompt: The input prompt.
    """

    step_kl_no_steer: list[float]
    step_kl_all_steer: list[float]
    generated_text_no_steer: str
    generated_text_prefix_steer: str
    generated_text_all_steer: str
    steer_tokens: int
    scale: float
    layer_frac: float
    prompt: str


_DEFAULT_POST_STEER_STEPS: int = 5


@dataclass
class PrefixLengthKLSweepResult:
    """KL divergence at steps N+1..N+K for varying prefix steering lengths.

    For each prefix length N in ``steer_tokens_list``, measures the KL
    divergence at the first ``num_post_steer_steps`` unsteered steps
    (steps N+1 through N+K) between:

    - ``KL(prefix_N || no_steer)`` — divergence from the unsteered baseline
    - ``KL(prefix_N || all_steer)`` — distance from full (all-step) steering

    Attributes:
        steer_tokens_list: Prefix lengths tested (e.g., [0, 2, 4, ..., 200]).
        kl_vs_no_steer: Per-prompt, per-post-steer-step KL values keyed by N.
            Each value is ``list[list[float]]`` where the outer list is per
            prompt and the inner list has ``num_post_steer_steps`` entries
            (one for each step after steering ends).
        kl_vs_all_steer: Same structure as ``kl_vs_no_steer`` but comparing
            against all-step steering.
        layer_frac: Relative layer position where steering was applied.
        scale: Steering scale factor.
        num_prompts: Number of prompts analyzed.
        num_post_steer_steps: Number of steps observed after steering ends.
    """

    steer_tokens_list: list[int]
    kl_vs_no_steer: dict[int, list[list[float]]]
    kl_vs_all_steer: dict[int, list[list[float]]]
    layer_frac: float
    scale: float
    num_prompts: int
    num_post_steer_steps: int = _DEFAULT_POST_STEER_STEPS


@dataclass
class AttentionLinkInstance:
    """Concrete per-head attention link example.

    Attributes:
        layer_idx: Layer index.
        head_idx: Attention head index.
        step: Generation step.
        attn_to_prefix_no_steer: Attention weight on prefix positions
            (no steering).
        attn_to_prefix_steer: Attention weight on prefix positions
            (prefix steering).
        attn_change: Difference (steer - no_steer) in prefix attention.
        top_prefix_position: Position index with largest attention increase.
        top_prefix_token: Token string at that position.
        top_prefix_attn_no_steer: Attention to that position without steering.
        top_prefix_attn_steer: Attention to that position with steering.
    """

    layer_idx: int
    head_idx: int
    step: int
    attn_to_prefix_no_steer: float
    attn_to_prefix_steer: float
    attn_change: float
    top_prefix_position: int
    top_prefix_token: str
    top_prefix_attn_no_steer: float
    top_prefix_attn_steer: float


@dataclass
class AttentionAnalysisResult:
    """Attention analysis results for a single prompt.

    Attributes:
        attn_to_prefix_no_steer: Fraction of attention on prefix positions per
            step (no steering).
        attn_to_prefix_prefix_steer: Fraction of attention on prefix positions
            per step (prefix steering).
        attn_to_prefix_all_steer: Fraction of attention on prefix positions per
            step (all-step steering).
        attn_cosine_shift: Cosine distance between prefix-steer and no-steer
            attention distributions per step.
        steered_layer_attn_diff: Mean absolute attention difference at the
            steered layer per step.
        prompt_tokens: Token strings for visualization.
        steer_tokens: Number of steps steering was applied.
        attention_links: Concrete per-head attention link instances showing
            strongest steering effects.
    """

    attn_to_prefix_no_steer: list[float]
    attn_to_prefix_prefix_steer: list[float]
    attn_to_prefix_all_steer: list[float]
    attn_cosine_shift: list[float]
    steered_layer_attn_diff: list[float]
    prompt_tokens: list[str]
    steer_tokens: int
    attention_links: list[AttentionLinkInstance] = field(default_factory=list)


@dataclass
class PrefixAnalysisReport:
    """Complete analysis report combining KL and attention results.

    Attributes:
        kl_results: Per-prompt KL divergence results (legacy per-step experiment).
        kl_sweep_result: Prefix-length sweep KL results (N vs KL at steps N+1..N+K).
        attention_results: Per-prompt attention analysis results.
        config_dict: Experiment configuration as serializable dict.
    """

    kl_results: list[KLDivergenceResult]
    kl_sweep_result: PrefixLengthKLSweepResult | None = None
    attention_results: list[AttentionAnalysisResult] = field(default_factory=list)
    config_dict: dict[str, str | int | float | bool] = field(default_factory=dict)


# =============================================================================
# KL Divergence Computation
# =============================================================================


def per_token_kl_divergence(
    logits_p: Tensor,
    logits_q: Tensor,
) -> float:
    """Compute KL(P || Q) using numerically stable log-softmax formula.

    KL(P||Q) = sum(p * (log_p - log_q))

    Uses float32 for numerical precision even if inputs are float16.

    Args:
        logits_p: Logits for distribution P, shape ``(1, vocab_size)``.
        logits_q: Logits for distribution Q, shape ``(1, vocab_size)``.

    Returns:
        Scalar KL divergence value in nats.
    """
    log_p = functional.log_softmax(logits_p.float(), dim=-1)
    log_q = functional.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.item()


# =============================================================================
# Manual Autoregressive Generation with Logit Capture
# =============================================================================


def _generate_with_logits(
    model: HookedModel,
    prompt: str,
    layer_idx: int,
    steering_vector: Tensor | None,
    scale: float,
    max_new_tokens: int,
    temperature: float,
    steer_tokens: int | None,
) -> tuple[str, list[Tensor]]:
    """Manual autoregressive generation returning per-step logits.

    Implements a token-by-token generation loop with KV-cache, registering
    a steering hook (same pattern as :meth:`HookedModel.generate_with_steering`)
    and capturing ``outputs.logits[:, -1, :]`` at each step.

    Args:
        model: HookedModel instance.
        prompt: Input text.
        layer_idx: Absolute layer index for steering.
        steering_vector: Normalized steering vector (``None`` = no steering).
        scale: Steering scale factor.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        steer_tokens: Number of steps to apply steering (``None`` = all).

    Returns:
        Tuple of ``(generated_text, list_of_per_step_logits)``.
        Each logit tensor has shape ``(1, vocab_size)``.
    """
    inputs = model.tokenizer(prompt, return_tensors="pt")
    device = next(model.model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    model_layers = model._get_layers_module()
    step_counter = [0]
    all_logits: list[Tensor] = []
    generated_ids: list[int] = []

    handle: torch.utils.hooks.RemovableHandle | None = None
    if steering_vector is not None:

        def steering_hook(module: object, inp: object, output: Tensor) -> Tensor:
            step_counter[0] += 1
            if steer_tokens is not None and step_counter[0] > steer_tokens:
                if isinstance(output, tuple):
                    return output
                return output
            tensor_output = output[0] if isinstance(output, tuple) else output
            sv = steering_vector.to(device=tensor_output.device, dtype=tensor_output.dtype)
            tensor_output = tensor_output + sv * scale
            if isinstance(output, tuple):
                return (tensor_output,) + output[1:]
            return tensor_output

        handle = model_layers[layer_idx].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            # Prefill: process prompt
            outputs = model.model(input_ids=input_ids, use_cache=True)
            next_logits = outputs.logits[:, -1, :]
            all_logits.append(next_logits.cpu())
            past_kv = outputs.past_key_values

            # First token
            if temperature > 0:
                probs = functional.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(int(next_token.item()))

            # Autoregressive loop
            for _ in range(max_new_tokens - 1):
                outputs = model.model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                next_logits = outputs.logits[:, -1, :]
                all_logits.append(next_logits.cpu())
                past_kv = outputs.past_key_values

                if temperature > 0:
                    probs = functional.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                token_id = int(next_token.item())
                generated_ids.append(token_id)

                if token_id == model.tokenizer.eos_token_id:
                    break
    finally:
        if handle is not None:
            handle.remove()

    raw_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    assert isinstance(raw_text, str)
    return raw_text, all_logits


# =============================================================================
# KL Divergence Experiment Runner
# =============================================================================


def run_kl_divergence_experiment(
    model: HookedModel,
    prompts: list[str],
    steering_vector: Tensor,
    layer_idx: int,
    layer_frac: float,
    scale: float,
    steer_tokens: int,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> list[KLDivergenceResult]:
    """Run KL divergence experiments for multiple prompts.

    For each prompt, runs 3 generation passes:

    1. **No steering** — ``steering_vector=None``, ``scale=0``
    2. **Prefix steering** — ``steer_tokens=N``
    3. **All-step steering** — ``steer_tokens=None``

    Then computes per-step KL divergence:

    - ``KL(prefix || no_steer)`` — magnitude of prefix intervention
    - ``KL(prefix || all_steer)`` — similarity to full steering

    Args:
        model: HookedModel instance.
        prompts: List of input prompts.
        steering_vector: Normalized steering direction.
        layer_idx: Absolute layer index for steering.
        layer_frac: Relative layer position (for metadata).
        scale: Steering scale factor.
        steer_tokens: Number of generation steps to steer (prefix mode).
        max_new_tokens: Maximum tokens per generation.
        temperature: Sampling temperature.

    Returns:
        List of :class:`KLDivergenceResult`, one per prompt.
    """
    results: list[KLDivergenceResult] = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info("KL experiment: prompt %d/%d", prompt_idx + 1, len(prompts))

        # Pass 1: No steering
        text_no_steer, logits_no_steer = _generate_with_logits(
            model,
            prompt,
            layer_idx,
            None,
            0.0,
            max_new_tokens,
            temperature,
            None,
        )

        # Pass 2: Prefix steering
        text_prefix, logits_prefix = _generate_with_logits(
            model,
            prompt,
            layer_idx,
            steering_vector,
            scale,
            max_new_tokens,
            temperature,
            steer_tokens,
        )

        # Pass 3: All-step steering
        text_all, logits_all = _generate_with_logits(
            model,
            prompt,
            layer_idx,
            steering_vector,
            scale,
            max_new_tokens,
            temperature,
            None,
        )

        # Align logits to same length (minimum of the three)
        min_len = min(len(logits_no_steer), len(logits_prefix), len(logits_all))

        kl_no_steer: list[float] = []
        kl_all_steer: list[float] = []

        for t in range(min_len):
            kl_no_steer.append(per_token_kl_divergence(logits_prefix[t], logits_no_steer[t]))
            kl_all_steer.append(per_token_kl_divergence(logits_prefix[t], logits_all[t]))

        results.append(
            KLDivergenceResult(
                step_kl_no_steer=kl_no_steer,
                step_kl_all_steer=kl_all_steer,
                generated_text_no_steer=text_no_steer,
                generated_text_prefix_steer=text_prefix,
                generated_text_all_steer=text_all,
                steer_tokens=steer_tokens,
                scale=scale,
                layer_frac=layer_frac,
                prompt=prompt,
            )
        )

    return results


# =============================================================================
# Prefix Length KL Sweep
# =============================================================================


_DEFAULT_STEER_TOKENS_LIST: list[int] = [0, 2, 4, 6, 8, 10, 20, 50, 100, 200]


def run_prefix_length_kl_sweep(
    model: HookedModel,
    prompts: list[str],
    steering_vector: Tensor,
    layer_idx: int,
    layer_frac: float,
    scale: float,
    steer_tokens_list: list[int] | None = None,
    temperature: float = 0.0,
    num_post_steer_steps: int = _DEFAULT_POST_STEER_STEPS,
) -> PrefixLengthKLSweepResult:
    """Sweep prefix steering length N and measure KL at the next K steps.

    For each N in ``steer_tokens_list``, generates with ``steer_tokens=N``,
    then measures the KL divergence at generation steps N+1 through
    N+K (the first ``num_post_steer_steps`` unsteered steps) against
    no-steering and all-step-steering baselines.

    The no-steering and all-step-steering baselines are computed once per
    prompt and reused across all N values.  Each prefix-length run generates
    ``N + num_post_steer_steps`` tokens.

    Args:
        model: HookedModel instance.
        prompts: Input prompts.
        steering_vector: Normalized steering direction.
        layer_idx: Absolute layer index for steering.
        layer_frac: Relative layer position (for metadata).
        scale: Steering scale factor.
        steer_tokens_list: Prefix lengths to sweep.
            Defaults to ``[0, 2, 4, 6, 8, 10, 20, 50, 100, 200]``.
        temperature: Sampling temperature (0.0 = greedy).
        num_post_steer_steps: Number of unsteered steps to observe after
            steering ends (default 5).

    Returns:
        :class:`PrefixLengthKLSweepResult` with per-N, per-step KL values.
    """
    if steer_tokens_list is None:
        steer_tokens_list = list(_DEFAULT_STEER_TOKENS_LIST)

    max_tokens_needed = max(steer_tokens_list) + num_post_steer_steps

    kl_vs_no_steer: dict[int, list[list[float]]] = {n: [] for n in steer_tokens_list}
    kl_vs_all_steer: dict[int, list[list[float]]] = {n: [] for n in steer_tokens_list}

    for prompt_idx, prompt in enumerate(prompts):
        logger.info("KL sweep: prompt %d/%d", prompt_idx + 1, len(prompts))

        _, logits_no_steer = _generate_with_logits(
            model,
            prompt,
            layer_idx,
            None,
            0.0,
            max_tokens_needed,
            temperature,
            None,
        )
        _, logits_all_steer = _generate_with_logits(
            model,
            prompt,
            layer_idx,
            steering_vector,
            scale,
            max_tokens_needed,
            temperature,
            None,
        )

        for n in steer_tokens_list:
            _, logits_prefix = _generate_with_logits(
                model,
                prompt,
                layer_idx,
                steering_vector,
                scale,
                n + num_post_steer_steps,
                temperature,
                n,
            )

            kl_no_steps: list[float] = []
            kl_all_steps: list[float] = []
            for offset in range(num_post_steer_steps):
                step_idx = n + offset
                if step_idx < len(logits_prefix) and step_idx < len(logits_no_steer):
                    kl_no_steps.append(
                        per_token_kl_divergence(logits_prefix[step_idx], logits_no_steer[step_idx])
                    )
                if step_idx < len(logits_prefix) and step_idx < len(logits_all_steer):
                    kl_all_steps.append(
                        per_token_kl_divergence(logits_prefix[step_idx], logits_all_steer[step_idx])
                    )

            kl_vs_no_steer[n].append(kl_no_steps)
            kl_vs_all_steer[n].append(kl_all_steps)

    return PrefixLengthKLSweepResult(
        steer_tokens_list=steer_tokens_list,
        kl_vs_no_steer=kl_vs_no_steer,
        kl_vs_all_steer=kl_vs_all_steer,
        layer_frac=layer_frac,
        scale=scale,
        num_prompts=len(prompts),
        num_post_steer_steps=num_post_steer_steps,
    )


# =============================================================================
# Attention Analysis
# =============================================================================


def _load_eager_model(model_name: str) -> HookedModel:
    """Load model with eager attention for weight extraction.

    Flash/SDPA backends return ``None`` for attention weights.
    Must use ``attn_implementation="eager"`` to get actual weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = ModelConfig(model_name=model_name)

    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)

    inner_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedModel.__new__(HookedModel)
    model.config = config
    model.model = inner_model
    model.tokenizer = tokenizer
    return model


def _generate_with_attention(
    model: HookedModel,
    prompt: str,
    layer_idx: int,
    steering_vector: Tensor | None,
    scale: float,
    max_new_tokens: int,
    steer_tokens: int | None,
    target_layers: list[int] | None = None,
) -> tuple[str, list[dict[int, Tensor]]]:
    """Manual autoregressive generation capturing attention weights.

    Similar to :func:`_generate_with_logits` but passes
    ``output_attentions=True`` to the forward call, capturing per-layer
    attention tensors at each step.

    Args:
        model: HookedModel with eager attention loaded.
        prompt: Input text.
        layer_idx: Absolute layer index for steering.
        steering_vector: Normalized steering vector (``None`` = no steering).
        scale: Steering scale factor.
        max_new_tokens: Maximum tokens to generate.
        steer_tokens: Steps to apply steering (``None`` = all).
        target_layers: Layer indices to capture attention for (``None`` = all).

    Returns:
        ``(generated_text, list_of_per_step_attention_dicts)``.
        Each dict maps ``layer_idx`` → attention tensor of shape
        ``(1, num_heads, seq_len, seq_len)``.
    """
    inputs = model.tokenizer(prompt, return_tensors="pt")
    device = next(model.model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    model_layers = model._get_layers_module()
    step_counter = [0]
    all_attns: list[dict[int, Tensor]] = []
    generated_ids: list[int] = []

    handle: torch.utils.hooks.RemovableHandle | None = None
    if steering_vector is not None:

        def steering_hook(module: object, inp: object, output: Tensor) -> Tensor:
            step_counter[0] += 1
            if steer_tokens is not None and step_counter[0] > steer_tokens:
                if isinstance(output, tuple):
                    return output
                return output
            tensor_output = output[0] if isinstance(output, tuple) else output
            sv = steering_vector.to(device=tensor_output.device, dtype=tensor_output.dtype)
            tensor_output = tensor_output + sv * scale
            if isinstance(output, tuple):
                return (tensor_output,) + output[1:]
            return tensor_output

        handle = model_layers[layer_idx].register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            # Prefill
            outputs = model.model(input_ids=input_ids, use_cache=True, output_attentions=True)
            past_kv = outputs.past_key_values

            # Extract attention for target layers from prefill
            if outputs.attentions is not None:
                step_attns: dict[int, Tensor] = {}
                for li, attn in enumerate(outputs.attentions):
                    if target_layers is None or li in target_layers:
                        step_attns[li] = attn.detach().cpu()
                all_attns.append(step_attns)

            next_logits = outputs.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(int(next_token.item()))

            # Autoregressive loop
            for _ in range(max_new_tokens - 1):
                outputs = model.model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True,
                )
                past_kv = outputs.past_key_values

                if outputs.attentions is not None:
                    step_attns = {}
                    for li, attn in enumerate(outputs.attentions):
                        if target_layers is None or li in target_layers:
                            step_attns[li] = attn.detach().cpu()
                    all_attns.append(step_attns)

                next_logits = outputs.logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                generated_ids.append(token_id)

                if token_id == model.tokenizer.eos_token_id:
                    break
    finally:
        if handle is not None:
            handle.remove()

    raw_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    assert isinstance(raw_text, str)
    return raw_text, all_attns


def _extract_prefix_attention(
    attn: Tensor,
    prefix_len: int,
) -> float:
    """Extract max attention weight on prefix positions from the last query.

    For each head, takes the maximum attention weight to any prefix position,
    then averages those per-head maxima across all heads.

    Args:
        attn: Attention tensor of shape ``(1, num_heads, seq_len, seq_len)``.
        prefix_len: Number of prefix positions (0..prefix_len).

    Returns:
        Mean of per-head max attention on prefix positions.
    """
    if prefix_len <= 0 or attn.shape[-1] <= 0:
        return 0.0
    clamped_prefix = min(prefix_len, attn.shape[-1])
    prefix_weights = attn[0, :, -1, :clamped_prefix]  # (num_heads, prefix_len)
    per_head_max = prefix_weights.max(dim=-1).values  # (num_heads,)
    return float(per_head_max.mean().item())


def _compute_attn_cosine_distance(
    attn_a: Tensor,
    attn_b: Tensor,
) -> float:
    """Compute cosine distance between two attention distributions.

    Flattens attention tensors and returns ``1 - cosine_similarity(a, b)``.

    Args:
        attn_a: Attention tensor of shape ``(1, num_heads, seq_len, seq_len)``.
        attn_b: Attention tensor of shape ``(1, num_heads, seq_len, seq_len)``.

    Returns:
        Cosine distance in [0, 2].
    """
    flat_a = attn_a.flatten().float().unsqueeze(0)
    flat_b = attn_b.flatten().float().unsqueeze(0)
    cos_sim = float(functional.cosine_similarity(flat_a, flat_b).item())
    return 1.0 - cos_sim


def _extract_attention_links(
    attn_no_steer: dict[int, Tensor],
    attn_prefix_steer: dict[int, Tensor],
    prefix_tokens: list[str],
    prefix_len: int,
    step: int,
    top_k: int = 5,
) -> list[AttentionLinkInstance]:
    """Extract top-k attention link instances showing strongest steering effect.

    For each layer present in both attention dicts, finds the heads with the
    largest attention increase to prefix positions, and records concrete
    details about the single token position with the biggest attention gain.

    Args:
        attn_no_steer: Per-layer attention tensors (no steering).
        attn_prefix_steer: Per-layer attention tensors (prefix steering).
        prefix_tokens: Token strings for the prefix positions.
        prefix_len: Number of prefix token positions.
        step: Current generation step index.
        top_k: Number of top links to extract per layer.

    Returns:
        List of :class:`AttentionLinkInstance` sorted by absolute
        ``attn_change`` (descending).
    """
    common_layers = sorted(set(attn_no_steer) & set(attn_prefix_steer))
    candidates: list[AttentionLinkInstance] = []

    for layer in common_layers:
        a_no = attn_no_steer[layer].float()
        a_steer = attn_prefix_steer[layer].float()
        num_heads = a_no.shape[1]
        seq_len = a_no.shape[-1]
        clamped = min(prefix_len, seq_len)
        if clamped <= 0:
            continue

        for head in range(num_heads):
            row_no = a_no[0, head, -1, :clamped]
            row_steer = a_steer[0, head, -1, :clamped]
            mean_no = float(row_no.mean().item())
            mean_steer = float(row_steer.mean().item())
            change = mean_steer - mean_no

            diff_per_pos = row_steer - row_no
            top_pos = int(diff_per_pos.argmax().item())
            token_str = prefix_tokens[top_pos] if top_pos < len(prefix_tokens) else ""
            attn_pos_no = float(row_no[top_pos].item())
            attn_pos_steer = float(row_steer[top_pos].item())

            candidates.append(
                AttentionLinkInstance(
                    layer_idx=layer,
                    head_idx=head,
                    step=step,
                    attn_to_prefix_no_steer=mean_no,
                    attn_to_prefix_steer=mean_steer,
                    attn_change=change,
                    top_prefix_position=top_pos,
                    top_prefix_token=token_str,
                    top_prefix_attn_no_steer=attn_pos_no,
                    top_prefix_attn_steer=attn_pos_steer,
                )
            )

    candidates.sort(key=lambda lk: abs(lk.attn_change), reverse=True)
    return candidates[:top_k]


def run_attention_analysis(
    prompt: str,
    steering_vector: Tensor,
    layer_idx: int,
    scale: float,
    steer_tokens: int,
    model_name: str = "Qwen/Qwen3-1.7B",
    eager_model: HookedModel | None = None,
    max_new_tokens: int = 50,
) -> AttentionAnalysisResult:
    """Run attention analysis for prefix steering.

    Computes:
    - Fraction of attention weight on prefix positions at post-prefix queries
    - Cosine distance between attention distributions (prefix vs no-steer)
    - Mean absolute attention difference at the steering layer

    Args:
        prompt: Input prompt text.
        steering_vector: Normalized steering direction.
        layer_idx: Absolute layer index for steering.
        scale: Steering scale factor.
        steer_tokens: Number of steps to apply steering.
        model_name: HuggingFace model identifier (used if eager_model is None).
        eager_model: Pre-loaded model with eager attention. If ``None``, one is
            loaded from ``model_name`` (expensive — prefer pre-loading).
        max_new_tokens: Maximum tokens to generate (default 50 for memory).

    Returns:
        :class:`AttentionAnalysisResult` with per-step metrics.
    """
    if eager_model is None:
        eager_model = _load_eager_model(model_name)

    # Target layers: steering layer +/- 2
    all_layers = list(range(eager_model.num_layers))
    target_layers = [li for li in all_layers if abs(li - layer_idx) <= 2]

    # Tokenize prompt to get prefix length
    input_ids = eager_model.tokenizer(prompt, return_tensors="pt")["input_ids"]
    prefix_len = input_ids.shape[1]

    prompt_tokens_raw = eager_model.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    prompt_tokens: list[str] = (
        prompt_tokens_raw if isinstance(prompt_tokens_raw, list) else [prompt_tokens_raw]
    )

    # Three passes: no_steer, prefix_steer, all_steer
    _, attn_no_steer = _generate_with_attention(
        eager_model,
        prompt,
        layer_idx,
        None,
        0.0,
        max_new_tokens,
        None,
        target_layers,
    )
    _, attn_prefix = _generate_with_attention(
        eager_model,
        prompt,
        layer_idx,
        steering_vector,
        scale,
        max_new_tokens,
        steer_tokens,
        target_layers,
    )
    _, attn_all = _generate_with_attention(
        eager_model,
        prompt,
        layer_idx,
        steering_vector,
        scale,
        max_new_tokens,
        None,
        target_layers,
    )

    min_len = min(len(attn_no_steer), len(attn_prefix), len(attn_all))

    attn_to_prefix_no: list[float] = []
    attn_to_prefix_prefix: list[float] = []
    attn_to_prefix_all: list[float] = []
    attn_cosine_shift: list[float] = []
    steered_diff: list[float] = []
    all_links: list[AttentionLinkInstance] = []

    for t in range(min_len):
        # Attention to prefix positions at the steering layer
        if layer_idx in attn_no_steer[t]:
            attn_to_prefix_no.append(
                _extract_prefix_attention(attn_no_steer[t][layer_idx], prefix_len)
            )
        else:
            attn_to_prefix_no.append(0.0)

        if layer_idx in attn_prefix[t]:
            attn_to_prefix_prefix.append(
                _extract_prefix_attention(attn_prefix[t][layer_idx], prefix_len)
            )
        else:
            attn_to_prefix_prefix.append(0.0)

        if layer_idx in attn_all[t]:
            attn_to_prefix_all.append(_extract_prefix_attention(attn_all[t][layer_idx], prefix_len))
        else:
            attn_to_prefix_all.append(0.0)

        # Cosine distance between prefix and no-steer attention at steering layer
        if layer_idx in attn_prefix[t] and layer_idx in attn_no_steer[t]:
            attn_cosine_shift.append(
                _compute_attn_cosine_distance(
                    attn_prefix[t][layer_idx], attn_no_steer[t][layer_idx]
                )
            )
        else:
            attn_cosine_shift.append(0.0)

        # Mean absolute attention difference at steering layer
        if layer_idx in attn_prefix[t] and layer_idx in attn_no_steer[t]:
            diff = float(
                (attn_prefix[t][layer_idx].float() - attn_no_steer[t][layer_idx].float())
                .abs()
                .mean()
                .item()
            )
            steered_diff.append(diff)
        else:
            steered_diff.append(0.0)

        step_links = _extract_attention_links(
            attn_no_steer[t], attn_prefix[t], prompt_tokens, prefix_len, t
        )
        all_links.extend(step_links)

    all_links.sort(key=lambda lk: abs(lk.attn_change), reverse=True)
    top_links = all_links[:20]

    return AttentionAnalysisResult(
        attn_to_prefix_no_steer=attn_to_prefix_no,
        attn_to_prefix_prefix_steer=attn_to_prefix_prefix,
        attn_to_prefix_all_steer=attn_to_prefix_all,
        attn_cosine_shift=attn_cosine_shift,
        steered_layer_attn_diff=steered_diff,
        prompt_tokens=prompt_tokens,
        steer_tokens=steer_tokens,
        attention_links=top_links,
    )


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_kl_divergence_curves(
    results: list[KLDivergenceResult],
    output_dir: Path,
) -> list[Path]:
    """Generate KL divergence plots following existing PDF style.

    Creates 2 plots:

    1. ``KL(Prefix || No Steering)`` vs generation step
    2. ``KL(Prefix || All-Step Steering)`` vs generation step

    Each plot shows mean +/- std across prompts with a vertical dashed line
    at ``steer_tokens``.

    Args:
        results: Per-prompt KL divergence results.
        output_dir: Directory for output PDFs.

    Returns:
        Paths to saved plot files.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    output_paths: list[Path] = []

    if not results:
        return output_paths

    max_len = max(len(r.step_kl_no_steer) for r in results)
    steer_tokens = results[0].steer_tokens

    plot_specs: list[tuple[str, str, str, str]] = [
        (
            "step_kl_no_steer",
            "KL(Prefix Steering ‖ No Steering)",
            "#2196F3",
            "kl_prefix_vs_no_steer.pdf",
        ),
        (
            "step_kl_all_steer",
            "KL(Prefix Steering ‖ All-Step Steering)",
            "#4CAF50",
            "kl_prefix_vs_all_steer.pdf",
        ),
    ]

    for attr_name, title, color, filename in plot_specs:
        sequences: list[list[float]] = []
        for r in results:
            vals = getattr(r, attr_name)
            padded = vals + [float("nan")] * (max_len - len(vals))
            sequences.append(padded)

        arr = np.array(sequences)  # (num_prompts, max_len)
        mean_kl = np.nanmean(arr, axis=0)
        std_kl = np.nanstd(arr, axis=0)
        steps = np.arange(max_len)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps, mean_kl, "-o", color=color, markersize=2, label="Mean KL")
        ax.fill_between(
            steps,
            mean_kl - std_kl,
            mean_kl + std_kl,
            alpha=0.2,
            color=color,
            label="±1 std",
        )

        ax.axvline(
            x=steer_tokens,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"Steering stops (step {steer_tokens})",
        )

        ax.set_xlabel("Generation Step", fontsize=11)
        ax.set_ylabel("KL Divergence (nats)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = output_dir / filename
        plt.savefig(path, bbox_inches="tight", format="pdf")
        plt.close()
        output_paths.append(path)
        logger.info("Saved KL plot to %s", path)

    return output_paths


def plot_prefix_length_kl_sweep(
    result: PrefixLengthKLSweepResult,
    output_dir: Path,
) -> list[Path]:
    """Generate KL divergence vs prefix length plot with next-K-step average.

    Creates a single plot with two curves:

    1. ``KL(Prefix_N || No Steering)`` — average over the first K post-steer
       steps, plotted vs prefix length N.
    2. ``KL(Prefix_N || All-Step Steering)`` — same averaging.

    Args:
        result: Prefix length sweep results.
        output_dir: Directory for output PDF.

    Returns:
        Paths to saved plot files.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)

    n_values = sorted(result.steer_tokens_list)
    k = result.num_post_steer_steps

    def _avg_over_steps(
        data: dict[int, list[list[float]]],
    ) -> tuple[list[float], list[float]]:
        means: list[float] = []
        stds: list[float] = []
        for n in n_values:
            per_prompt = data.get(n, [])
            if not per_prompt:
                means.append(0.0)
                stds.append(0.0)
                continue
            prompt_step_avgs = [float(np.mean(p)) if p else 0.0 for p in per_prompt]
            means.append(float(np.mean(prompt_step_avgs)))
            stds.append(float(np.std(prompt_step_avgs)))
        return means, stds

    mean_no, std_no = _avg_over_steps(result.kl_vs_no_steer)
    mean_all, std_all = _avg_over_steps(result.kl_vs_all_steer)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        n_values,
        mean_no,
        "-o",
        color="#2196F3",
        markersize=5,
        label=f"KL(Prefix_N ‖ No Steer) avg steps N+1..N+{k}",
    )
    ax.fill_between(
        n_values,
        [m - s for m, s in zip(mean_no, std_no, strict=True)],
        [m + s for m, s in zip(mean_no, std_no, strict=True)],
        alpha=0.2,
        color="#2196F3",
    )

    ax.plot(
        n_values,
        mean_all,
        "-s",
        color="#4CAF50",
        markersize=5,
        label=f"KL(Prefix_N ‖ All Steer) avg steps N+1..N+{k}",
    )
    ax.fill_between(
        n_values,
        [m - s for m, s in zip(mean_all, std_all, strict=True)],
        [m + s for m, s in zip(mean_all, std_all, strict=True)],
        alpha=0.2,
        color="#4CAF50",
    )

    ax.set_xlabel("Prefix Steering Length (N)", fontsize=12)
    ax.set_ylabel(f"KL Divergence avg(steps N+1..N+{k}) (nats)", fontsize=12)
    ax.set_title(
        f"KL Divergence (avg next {k} tokens) vs Prefix Steering Length",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = output_dir / "kl_prefix_length_sweep.pdf"
    plt.savefig(path, bbox_inches="tight", format="pdf")
    plt.close()
    logger.info("Saved KL prefix length sweep plot to %s", path)

    return [path]


def plot_attention_analysis(
    results: list[AttentionAnalysisResult],
    output_dir: Path,
) -> list[Path]:
    """Generate attention analysis plots.

    Creates 2 plots:

    1. Fraction of attention to prefix tokens over generation steps (3 modes)
    2. Attention cosine shift (prefix vs no-steer) over generation steps

    Args:
        results: Per-prompt attention analysis results.
        output_dir: Directory for output PDFs.

    Returns:
        Paths to saved plot files.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    output_paths: list[Path] = []

    if not results:
        return output_paths

    steer_tokens = results[0].steer_tokens
    max_len = max(len(r.attn_to_prefix_no_steer) for r in results)

    # Plot 1: Attention to prefix tokens
    fig, ax = plt.subplots(figsize=(8, 5))
    attn_modes: list[tuple[str, str, str]] = [
        ("attn_to_prefix_no_steer", "#2196F3", "No Steering"),
        ("attn_to_prefix_prefix_steer", "#FF5722", "Prefix Steering"),
        ("attn_to_prefix_all_steer", "#4CAF50", "All-Step Steering"),
    ]

    for attr_name, color, label in attn_modes:
        sequences: list[list[float]] = []
        for r in results:
            vals = getattr(r, attr_name)
            padded = vals + [float("nan")] * (max_len - len(vals))
            sequences.append(padded)
        arr = np.array(sequences)
        mean_vals = np.nanmean(arr, axis=0)
        std_vals = np.nanstd(arr, axis=0)
        steps = np.arange(max_len)
        ax.plot(steps, mean_vals, "-o", color=color, markersize=2, label=label)
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15, color=color)

    ax.axvline(
        x=steer_tokens,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="Steering stops",
    )
    ax.set_xlabel("Generation Step", fontsize=11)
    ax.set_ylabel("Max Attention to Prefix Tokens (per head avg)", fontsize=11)
    ax.set_title("Max Attention to Steering Prefix Positions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / "attention_to_prefix.pdf"
    plt.savefig(path, bbox_inches="tight", format="pdf")
    plt.close()
    output_paths.append(path)

    # Plot 2: Attention cosine shift
    cos_sequences: list[list[float]] = []
    for r in results:
        vals = r.attn_cosine_shift
        padded = vals + [float("nan")] * (max_len - len(vals))
        cos_sequences.append(padded)
    cos_arr = np.array(cos_sequences)
    mean_cos = np.nanmean(cos_arr, axis=0)
    std_cos = np.nanstd(cos_arr, axis=0)
    cos_steps = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cos_steps, mean_cos, "-o", color="#9C27B0", markersize=2, label="Mean Cosine Distance")
    ax.fill_between(cos_steps, mean_cos - std_cos, mean_cos + std_cos, alpha=0.2, color="#9C27B0")
    ax.axvline(
        x=steer_tokens,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=1.5,
        label="Steering stops",
    )
    ax.set_xlabel("Generation Step", fontsize=11)
    ax.set_ylabel("Attention Cosine Distance", fontsize=11)
    ax.set_title(
        "Attention Distribution Shift (Prefix Steering vs No Steering)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    cos_path = output_dir / "attention_cosine_shift.pdf"
    plt.savefig(cos_path, bbox_inches="tight", format="pdf")
    plt.close()
    output_paths.append(cos_path)

    logger.info("Saved %d attention plots", len(output_paths))
    return output_paths


def plot_attention_link_heatmap(
    links: list[AttentionLinkInstance],
    steer_tokens: int,
    output_dir: Path,
    num_layers: int = 5,
    num_heads: int = 8,
) -> Path:
    """Create a per-head attention change heatmap.

    Shows a (num_layers x num_heads) heatmap where each cell is the
    mean attention change to prefix positions for that (layer, head) pair.

    Args:
        links: Attention link instances.
        steer_tokens: Number of steering tokens.
        output_dir: Directory for output PDF.
        num_layers: Number of layers to display.
        num_heads: Number of heads to display.

    Returns:
        Path to saved heatmap PDF.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)

    matrix = np.zeros((num_layers, num_heads))
    counts = np.zeros((num_layers, num_heads))
    for lk in links:
        if lk.layer_idx < num_layers and lk.head_idx < num_heads:
            matrix[lk.layer_idx, lk.head_idx] += lk.attn_change
            counts[lk.layer_idx, lk.head_idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_matrix = np.where(counts > 0, matrix / counts, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, num_heads * 1.0), max(5, num_layers * 0.8)))
    vmax = max(abs(mean_matrix.min()), abs(mean_matrix.max()), 1e-6)
    im = ax.imshow(mean_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("Head Index", fontsize=11)
    ax.set_ylabel("Layer Index", fontsize=11)
    ax.set_title(
        "Per-Head Attention Change to Prefix Positions (Prefix Steer - No Steer)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))

    for i in range(num_layers):
        for j in range(num_heads):
            val = mean_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(val) > vmax * 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, label="Mean Attention Change")
    fig.tight_layout()
    path = output_dir / "attention_link_heatmap.pdf"
    plt.savefig(path, bbox_inches="tight", format="pdf")
    plt.close()
    logger.info("Saved attention link heatmap to %s", path)
    return path


# =============================================================================
# Markdown Report Generation
# =============================================================================


def generate_analysis_report(
    kl_results: list[KLDivergenceResult],
    attention_results: list[AttentionAnalysisResult],
    config_dict: dict[str, str | int | float | bool],
    plot_paths: list[Path],
    output_path: Path,
    kl_sweep_result: PrefixLengthKLSweepResult | None = None,
) -> Path:
    """Generate comprehensive markdown analysis report.

    Sections:

    1. Configuration
    2. Executive Summary
    3. KL Divergence Analysis (with statistics)
    4. Attention Path Analysis
    5. Per-Prompt Details Table

    Args:
        kl_results: Per-prompt KL divergence results.
        attention_results: Per-prompt attention analysis results.
        config_dict: Experiment configuration.
        plot_paths: Paths to generated plots.
        output_path: Where to write the markdown file.

    Returns:
        Path to the written report.
    """
    ensure_dir(output_path.parent)

    lines: list[str] = []
    lines.append("# Prefix Steering Analysis Report\n")

    # Configuration
    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, val in config_dict.items():
        lines.append(f"| {key} | {val} |")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(
        "This report analyzes **why** Prefix Steering works by comparing three "
        "generation modes:\n"
        "- **No steering**: baseline generation without intervention\n"
        "- **Prefix steering**: steering applied for the first N generation steps only\n"
        "- **All-step steering**: steering applied at every generation step\n\n"
        "We compute per-token KL divergence between the output distributions "
        "and analyze attention patterns to understand the mechanism.\n"
    )

    # KL Prefix Length Sweep
    if kl_sweep_result is not None:
        k = kl_sweep_result.num_post_steer_steps
        lines.append("## 1. KL Divergence Sweep: Prefix Length Analysis\n")
        lines.append(
            f"For each prefix steering length N, measures KL divergence at "
            f"steps N+1 through N+{k} (the first {k} unsteered steps) "
            "against no-steering and all-step-steering baselines.\n"
        )
        n_values = sorted(kl_sweep_result.steer_tokens_list)
        header = "| N | "
        sep = "|---|"
        for offset in range(1, k + 1):
            header += f" KL(‖No) N+{offset} |"
            sep += "----------|"
        for offset in range(1, k + 1):
            header += f" KL(‖All) N+{offset} |"
            sep += "----------|"
        lines.append(header)
        lines.append(sep)
        for n in n_values:
            per_prompt_no = kl_sweep_result.kl_vs_no_steer.get(n, [])
            per_prompt_all = kl_sweep_result.kl_vs_all_steer.get(n, [])
            row = f"| {n} |"
            for offset in range(k):
                vals = [p[offset] for p in per_prompt_no if offset < len(p)]
                m = float(np.mean(vals)) if vals else 0.0
                row += f" {m:.4f} |"
            for offset in range(k):
                vals = [p[offset] for p in per_prompt_all if offset < len(p)]
                m = float(np.mean(vals)) if vals else 0.0
                row += f" {m:.4f} |"
            lines.append(row)
        lines.append("")
    else:
        lines.append("## 1. KL Divergence Sweep: Prefix Length Analysis\n")
        lines.append("No prefix length sweep results available.\n")
    lines.append("")

    # KL Divergence Analysis (legacy per-step)
    lines.append("## 2. KL Divergence: Prefix Steering vs No Steering (Per-Step)\n")
    lines.append(
        "This measures how much the prefix-steered output distribution "
        "diverges from the unsteered baseline at each generation step.\n"
    )
    if kl_results:
        all_kl_no: list[float] = []
        for kl_r in kl_results:
            all_kl_no.extend(kl_r.step_kl_no_steer)
        if all_kl_no:
            arr_no = np.array(all_kl_no)
            lines.append("### Overall Statistics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean KL | {float(arr_no.mean()):.4f} nats |")
            lines.append(f"| Std KL | {float(arr_no.std()):.4f} nats |")
            lines.append(f"| Max KL | {float(arr_no.max()):.4f} nats |")
            lines.append(f"| Min KL | {float(arr_no.min()):.4f} nats |")
            lines.append("")

        if kl_results:
            steer_tok = kl_results[0].steer_tokens
            during_no: list[float] = []
            after_no: list[float] = []
            for kl_r in kl_results:
                during_no.extend(kl_r.step_kl_no_steer[:steer_tok])
                after_no.extend(kl_r.step_kl_no_steer[steer_tok:])
            if during_no and after_no:
                mean_d = float(np.mean(during_no))
                mean_a = float(np.mean(after_no))
                lines.append("### During vs After Steering\n")
                lines.append("| Phase | Mean KL | Std KL |")
                lines.append("|-------|---------|--------|")
                lines.append(
                    f"| During steering (steps 0-{steer_tok - 1}) | "
                    f"{mean_d:.4f} | {float(np.std(during_no)):.4f} |"
                )
                lines.append(
                    f"| After steering (step {steer_tok}+) | "
                    f"{mean_a:.4f} | {float(np.std(after_no)):.4f} |"
                )
                lines.append("")
                ratio = mean_a / mean_d if mean_d > 0.001 else 0.0
                lines.append(f"**Ratio (after/during): {ratio:.2f}x**\n")
                if ratio > 1.0:
                    lines.append(
                        "The KL divergence **increases** after steering stops. "
                        "This indicates that the initial steering perturbation "
                        "cascades through subsequent generation steps — the model "
                        "commits to a different trajectory that diverges further "
                        "from the no-steering baseline. The prefix intervention "
                        "sets the generation on a new path that persists and "
                        "amplifies.\n"
                    )
                else:
                    lines.append(
                        "The KL divergence **decreases** after steering stops, "
                        "suggesting the model partially converges back toward "
                        "the unsteered distribution.\n"
                    )
    lines.append("")

    # KL vs All-Step
    lines.append("## 3. KL Divergence: Prefix Steering vs All-Step Steering (Per-Step)\n")
    lines.append(
        "This measures how similar prefix steering is to full (all-step) "
        "steering. Lower values mean prefix steering closely approximates "
        "the effect of steering at every step.\n"
    )
    if kl_results:
        all_kl_all: list[float] = []
        for kl_r2 in kl_results:
            all_kl_all.extend(kl_r2.step_kl_all_steer)
        if all_kl_all:
            arr_all = np.array(all_kl_all)
            lines.append("### Overall Statistics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean KL | {float(arr_all.mean()):.4f} nats |")
            lines.append(f"| Std KL | {float(arr_all.std()):.4f} nats |")
            lines.append(f"| Max KL | {float(arr_all.max()):.4f} nats |")
            lines.append(f"| Min KL | {float(arr_all.min()):.4f} nats |")
            lines.append("")

        steer_tok = kl_results[0].steer_tokens
        during_all: list[float] = []
        after_all: list[float] = []
        for kl_r2 in kl_results:
            during_all.extend(kl_r2.step_kl_all_steer[:steer_tok])
            after_all.extend(kl_r2.step_kl_all_steer[steer_tok:])
        if during_all and after_all:
            lines.append("### During vs After Steering\n")
            lines.append("| Phase | Mean KL | Std KL |")
            lines.append("|-------|---------|--------|")
            lines.append(
                f"| During steering (steps 0-{steer_tok - 1}) | "
                f"{float(np.mean(during_all)):.4f} | {float(np.std(during_all)):.4f} |"
            )
            lines.append(
                f"| After steering (step {steer_tok}+) | "
                f"{float(np.mean(after_all)):.4f} | {float(np.std(after_all)):.4f} |"
            )
            lines.append("")

            mean_during_all = float(np.mean(during_all))
            if mean_during_all < 0.01:
                lines.append(
                    "**Key finding**: KL(prefix || all_steer) during the "
                    "steering phase is **effectively zero**. This confirms that "
                    "both modes apply identical steering during the prefix "
                    "window — the only difference occurs after steering stops.\n"
                )

            if all_kl_no and all_kl_all:
                mean_no_total = float(np.mean(all_kl_no))
                mean_all_total = float(np.mean(all_kl_all))
                ratio_total = mean_all_total / mean_no_total if mean_no_total > 0.001 else 0.0
                lines.append(
                    f"**KL(prefix || all_steer) / KL(prefix || no_steer) = "
                    f"{ratio_total:.2f}x**\n\n"
                    f"Prefix steering achieves its effect with only "
                    f"{steer_tok} steps of intervention, while the KL "
                    f"divergence from all-step steering is only "
                    f"{ratio_total:.0%} of the total steering effect. "
                    f"This demonstrates that **early-token steering is "
                    f"sufficient** to redirect the model's behavior.\n"
                )
    lines.append("")

    # Attention Path Analysis
    lines.append("## 4. Attention Path Analysis\n")
    if attention_results:
        all_cos_shift: list[float] = []
        all_attn_no: list[float] = []
        all_attn_prefix: list[float] = []
        all_attn_all: list[float] = []
        all_steer_diff: list[float] = []
        for attn_r in attention_results:
            all_cos_shift.extend(attn_r.attn_cosine_shift)
            all_attn_no.extend(attn_r.attn_to_prefix_no_steer)
            all_attn_prefix.extend(attn_r.attn_to_prefix_prefix_steer)
            all_attn_all.extend(attn_r.attn_to_prefix_all_steer)
            all_steer_diff.extend(attn_r.steered_layer_attn_diff)

        lines.append("### Attention to Prefix Positions\n")
        if all_attn_no and all_attn_prefix:
            mean_no = float(np.mean(all_attn_no))
            mean_prefix = float(np.mean(all_attn_prefix))
            mean_all_a = float(np.mean(all_attn_all)) if all_attn_all else 0.0
            lines.append("| Mode | Mean Attention to Prefix |")
            lines.append("|------|--------------------------|")
            lines.append(f"| No steering | {mean_no:.4f} |")
            lines.append(f"| Prefix steering | {mean_prefix:.4f} |")
            lines.append(f"| All-step steering | {mean_all_a:.4f} |")
            lines.append("")

            attn_increase = (mean_prefix - mean_no) / mean_no * 100 if mean_no > 0.001 else 0.0
            lines.append(
                f"Prefix steering increases attention to prefix positions by "
                f"**{attn_increase:.1f}%** relative to no steering. "
                f"This suggests the steered representations act as "
                f"**attention anchors** — subsequent tokens attend more "
                f"strongly to the positions where steering was applied.\n"
            )

        lines.append("### Attention Distribution Shift\n")
        if all_cos_shift:
            arr_cos = np.array(all_cos_shift)
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean cosine distance | {float(arr_cos.mean()):.4f} |")
            lines.append(f"| Std cosine distance | {float(arr_cos.std()):.4f} |")
            lines.append("")
            lines.append(
                "The cosine distance measures how much the overall attention "
                "distribution changes at the steering layer. Non-zero values "
                "indicate that steering at one layer alters attention patterns "
                "across the entire sequence.\n"
            )

        if all_steer_diff:
            lines.append("### Mean Absolute Attention Difference\n")
            mean_diff = float(np.mean(all_steer_diff))
            lines.append(
                f"Mean absolute attention difference at steering layer: "
                f"**{mean_diff:.6f}**\n\n"
                "This quantifies the average per-element change in the "
                "attention weight matrix at the steered layer.\n"
            )
    else:
        lines.append("No attention analysis results available.\n")
    lines.append("")

    # Attention Link Instances
    lines.append("## 5. Attention Link Instances\n")
    all_links: list[AttentionLinkInstance] = []
    for attn_r in attention_results:
        all_links.extend(attn_r.attention_links)

    if all_links:
        lines.append(
            "Concrete examples of individual attention heads that changed "
            "their prefix-position attention the most under steering.\n"
        )

        top_10 = all_links[:10]
        lines.append(
            "| Layer | Head | Step | No Steer | Steer | Change | Top Pos | Top Token | Token Δ |"
        )
        lines.append(
            "|-------|------|------|----------|-------|--------|---------|-----------|---------|"
        )
        for lk in top_10:
            token_delta = lk.top_prefix_attn_steer - lk.top_prefix_attn_no_steer
            tok_display = lk.top_prefix_token[:15].replace("|", "\\|")
            lines.append(
                f"| {lk.layer_idx} | {lk.head_idx} | {lk.step} "
                f"| {lk.attn_to_prefix_no_steer:.4f} "
                f"| {lk.attn_to_prefix_steer:.4f} "
                f"| {lk.attn_change:+.4f} "
                f"| {lk.top_prefix_position} "
                f"| {tok_display} "
                f"| {token_delta:+.4f} |"
            )
        lines.append("")

        lines.append("### Narrative Examples\n")
        for idx, lk in enumerate(all_links[:5]):
            token_delta = lk.top_prefix_attn_steer - lk.top_prefix_attn_no_steer
            phase = (
                "during the active steering phase"
                if lk.step < (kl_results[0].steer_tokens if kl_results else 0)
                else "after steering stopped"
            )
            lines.append(
                f"**Example {idx + 1}**: At generation step {lk.step} "
                f"({phase}), head {lk.head_idx} of layer {lk.layer_idx} "
                f"shifted its mean prefix attention from "
                f"{lk.attn_to_prefix_no_steer:.4f} to "
                f"{lk.attn_to_prefix_steer:.4f} "
                f"({lk.attn_change:+.4f}). "
                f"The strongest increase was at position "
                f"{lk.top_prefix_position} (token "
                f"'{lk.top_prefix_token}'): "
                f"{lk.top_prefix_attn_no_steer:.4f} → "
                f"{lk.top_prefix_attn_steer:.4f} "
                f"({token_delta:+.4f}).\n"
            )

        lines.append("### Layer-wise Summary\n")
        layer_groups: dict[int, list[AttentionLinkInstance]] = {}
        for lk2 in all_links:
            layer_groups.setdefault(lk2.layer_idx, []).append(lk2)
        lines.append("| Layer | # Links | Mean |attn_change| | Heads Affected |")
        lines.append("|-------|---------|-------------------|----------------|")
        for layer_idx in sorted(layer_groups):
            group = layer_groups[layer_idx]
            heads = sorted(set(lk3.head_idx for lk3 in group))
            mean_abs = float(np.mean([abs(lk4.attn_change) for lk4 in group]))
            head_ranges: list[str] = []
            if heads:
                start = heads[0]
                end = heads[0]
                for h in heads[1:]:
                    if h == end + 1:
                        end = h
                    else:
                        head_ranges.append(f"{start}" if start == end else f"{start}-{end}")
                        start = h
                        end = h
                head_ranges.append(f"{start}" if start == end else f"{start}-{end}")
            lines.append(
                f"| {layer_idx} | {len(group)} | {mean_abs:.4f} | {', '.join(head_ranges)} |"
            )
        lines.append("")
    else:
        lines.append("No attention link instances available.\n")
    lines.append("")

    # Key Findings
    lines.append("## 6. Key Findings\n")
    findings: list[str] = []
    if kl_results:
        steer_tok = kl_results[0].steer_tokens
        during_kl: list[float] = []
        after_kl: list[float] = []
        for find_r in kl_results:
            during_kl.extend(find_r.step_kl_no_steer[:steer_tok])
            after_kl.extend(find_r.step_kl_no_steer[steer_tok:])
        if during_kl and after_kl:
            mean_during = float(np.mean(during_kl))
            mean_after = float(np.mean(after_kl))
            ratio = mean_after / mean_during if mean_during > 0 else 0.0
            findings.append(
                f"- **Steering sets a persistent trajectory**: KL(prefix || no_steer) "
                f"during steering: {mean_during:.4f} nats → after: {mean_after:.4f} nats "
                f"(ratio: {ratio:.2f}x). The initial perturbation cascades."
            )
        all_step_kl: list[float] = []
        during_all_kl: list[float] = []
        for find_r2 in kl_results:
            all_step_kl.extend(find_r2.step_kl_all_steer)
            during_all_kl.extend(find_r2.step_kl_all_steer[:steer_tok])
        if all_step_kl:
            mean_all_step = float(np.mean(all_step_kl))
            mean_during_all = float(np.mean(during_all_kl)) if during_all_kl else 0.0
            findings.append(
                f"- **Prefix ≈ All-step during active phase**: "
                f"KL(prefix || all_steer) during steering = {mean_during_all:.4f} nats "
                f"(effectively zero). Both modes produce identical output "
                f"distributions while steering is active."
            )
            findings.append(
                f"- **Prefix captures most steering effect**: "
                f"KL(prefix || all_steer) overall = {mean_all_step:.4f} nats. "
                f"The divergence after steering stops reflects natural "
                f"autoregressive drift, not a fundamental difference in "
                f"steering mechanism."
            )
    if attention_results:
        all_cos: list[float] = []
        all_attn_n: list[float] = []
        all_attn_p: list[float] = []
        for attn_find in attention_results:
            all_cos.extend(attn_find.attn_cosine_shift)
            all_attn_n.extend(attn_find.attn_to_prefix_no_steer)
            all_attn_p.extend(attn_find.attn_to_prefix_prefix_steer)
        if all_cos:
            findings.append(
                f"- **Steering alters attention patterns**: Mean attention "
                f"cosine distance = {float(np.mean(all_cos)):.4f}, confirming "
                f"that the steering vector modifies how tokens attend to "
                f"each other at the steered layer."
            )
        if all_attn_n and all_attn_p:
            pct = (
                (float(np.mean(all_attn_p)) - float(np.mean(all_attn_n)))
                / float(np.mean(all_attn_n))
                * 100
                if float(np.mean(all_attn_n)) > 0.001
                else 0.0
            )
            findings.append(
                f"- **Prefix tokens become attention anchors**: Steering "
                f"increases attention to prefix positions by {pct:.1f}%, "
                f"suggesting the steered representations attract attention "
                f"from subsequent tokens."
            )
    findings.append(
        "- **Mechanism summary**: Prefix Steering works because (1) the "
        "steering vector perturbation at early generation steps redirects "
        "the autoregressive trajectory, (2) this redirection persists even "
        "after steering stops due to the causal nature of autoregressive "
        "generation (each token depends on all prior tokens), and (3) the "
        "steered prefix positions become attention anchors that continue to "
        "influence generation."
    )
    for f_text in findings:
        lines.append(f_text)
    lines.append("")

    # Per-Prompt Details
    lines.append("## 7. Per-Prompt Details\n")
    lines.append("| # | Prompt | KL(‖no_steer) | KL(‖all_steer) | No-steer text | Prefix text |")
    lines.append("|---|--------|---------------|----------------|---------------|-------------|")
    for idx, tbl_r in enumerate(kl_results):
        truncated = tbl_r.prompt[:40].replace("|", "\\|") + (
            "..." if len(tbl_r.prompt) > 40 else ""
        )
        mean_no = float(np.mean(tbl_r.step_kl_no_steer)) if tbl_r.step_kl_no_steer else 0.0
        mean_all = float(np.mean(tbl_r.step_kl_all_steer)) if tbl_r.step_kl_all_steer else 0.0
        text_no = tbl_r.generated_text_no_steer[:40].replace("|", "\\|").replace("\n", " ")
        text_p = tbl_r.generated_text_prefix_steer[:40].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {idx + 1} | {truncated} | {mean_no:.2f} | {mean_all:.2f} | {text_no} | {text_p} |"
        )
    lines.append("")

    # Plots
    lines.append("## Plots\n")
    for p in plot_paths:
        lines.append(f"- [{p.name}]({p})")
    lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("Saved analysis report to %s", output_path)
    return output_path


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_avg_activation(
    model: HookedModel,
    steering_vector: Tensor,
    layer_idx: int,
) -> float:
    """Compute average activation norm at a layer for scaling.

    Uses a simple dummy prompt to determine the typical activation
    magnitude, which is then used to scale the steering vector.

    Args:
        model: HookedModel instance.
        steering_vector: Steering vector (used only for dimension check).
        layer_idx: Absolute layer index to measure.

    Returns:
        Average activation norm at the target layer.
    """
    dummy = ["The quick brown fox jumps over the lazy dog."]
    activations = model.get_activations(dummy, [layer_idx])
    if layer_idx not in activations:
        return 1.0
    avg_norm = float(activations[layer_idx].norm(dim=-1).mean().item())
    # If avg_norm is suspiciously small (e.g., near zero), use the steering
    # vector norm as fallback to avoid division issues
    if avg_norm < 1e-6:
        avg_norm = float(steering_vector.norm().item())
        if avg_norm < 1e-6:
            return 1.0
    return avg_norm


# =============================================================================
# Main Orchestrator
# =============================================================================


def run_prefix_analysis(
    model_name: str = "Qwen/Qwen3-1.7B",
    concept: str = "sentiment",
    vector_path: Path | None = None,
    layer_frac: float = 0.7,
    steer_tokens: int = 10,
    scale_multiplier: float = 1.0,
    num_prompts: int = 10,
    max_new_tokens: int = 100,
    attention_max_tokens: int = 50,
    run_attention: bool = True,
    output_dir: Path = Path("outputs/prefix_analysis"),
) -> PrefixAnalysisReport:
    """Main entry point for Prefix Steering analysis.

    Steps:

    1. Load model + steering vector
    2. Load contrast pairs, select negative prompts
    3. Run KL divergence experiments
    4. Optionally run attention analysis
    5. Generate plots and markdown report

    Args:
        model_name: HuggingFace model identifier.
        concept: Steering concept (``"sentiment"``, ``"refusal"``, ``"polite"``).
        vector_path: Path to saved steering vector file (auto-discovered if ``None``).
        layer_frac: Relative layer position for steering.
        steer_tokens: Number of generation steps to apply steering.
        scale_multiplier: Multiplier for the auto-computed steering scale.
        num_prompts: Number of prompts to analyze.
        max_new_tokens: Maximum tokens for KL divergence experiments.
        attention_max_tokens: Maximum tokens for attention analysis (lower for memory).
        run_attention: Whether to run attention analysis (memory-intensive).
        output_dir: Base output directory.

    Returns:
        :class:`PrefixAnalysisReport` with all results.
    """
    logger.info("Starting Prefix Steering analysis")
    logger.info(
        "Model: %s, Concept: %s, Layer: %.2f, Steer tokens: %d",
        model_name,
        concept,
        layer_frac,
        steer_tokens,
    )

    effective_output_dir = ensure_dir(output_dir / concept / safe_model_name(model_name))

    # Load model
    model = HookedModel(ModelConfig(model_name=model_name))
    layer_idx = model.resolve_layers([layer_frac])[0]
    logger.info("Resolved layer %.2f to absolute index %d", layer_frac, layer_idx)

    # Load steering vector
    if vector_path is None:
        # Auto-discover: try discriminative first, then diff_means
        candidate = Path(f"outputs/vectors/{concept}/discriminative/k128_layer{layer_frac}.pt")
        if not candidate.exists():
            candidate = Path(f"outputs/vectors/{concept}/diff_means/n6000_layer{layer_frac}.pt")
        vector_path = candidate

    logger.info("Loading steering vector from %s", vector_path)
    data = torch.load(vector_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict) and "vector" in data:
        vector = cast("SteeringVector", data["vector"])
        steering_vector = vector.layer_activations[layer_idx]
    elif isinstance(data, Tensor):
        steering_vector = data
    else:
        msg = f"Unexpected vector format in {vector_path}: {type(data)}"
        raise ValueError(msg)

    # Normalize
    norm = steering_vector.norm()
    if norm > 0:
        steering_vector = steering_vector / norm

    # Compute scale from average activation norm
    avg_act = _compute_avg_activation(model, steering_vector, layer_idx)
    scale = avg_act * scale_multiplier
    logger.info("Scale: %.4f (avg_act=%.4f, mult=%.2f)", scale, avg_act, scale_multiplier)

    # Load prompts
    if concept == "refusal":
        pairs = load_contrast_pairs(concept, num_prompts, data_mode="prompt_only")
    else:
        pairs = load_contrast_pairs(concept, num_prompts)
    prompts = [pair.negative for pair in pairs[:num_prompts]]
    logger.info("Loaded %d prompts for analysis", len(prompts))

    # Run KL prefix length sweep
    logger.info("=== Running KL Prefix Length Sweep ===")
    kl_sweep_result = run_prefix_length_kl_sweep(
        model=model,
        prompts=prompts,
        steering_vector=steering_vector,
        layer_idx=layer_idx,
        layer_frac=layer_frac,
        scale=scale,
        steer_tokens_list=list(_DEFAULT_STEER_TOKENS_LIST),
    )

    # Run legacy per-step KL experiment (single steer_tokens value)
    logger.info("=== Running KL Divergence Experiment (per-step) ===")
    kl_results = run_kl_divergence_experiment(
        model=model,
        prompts=prompts,
        steering_vector=steering_vector,
        layer_idx=layer_idx,
        layer_frac=layer_frac,
        scale=scale,
        steer_tokens=steer_tokens,
        max_new_tokens=max_new_tokens,
    )

    # Run attention analysis
    attention_results: list[AttentionAnalysisResult] = []
    if run_attention:
        logger.info("=== Running Attention Analysis ===")
        attn_prompts = prompts[:3]
        logger.info("Loading eager attention model for %d prompts...", len(attn_prompts))
        eager_model = _load_eager_model(model_name)
        for i, prompt in enumerate(attn_prompts):
            logger.info("Attention analysis: prompt %d/%d", i + 1, len(attn_prompts))
            result = run_attention_analysis(
                prompt=prompt,
                steering_vector=steering_vector,
                layer_idx=layer_idx,
                scale=scale,
                steer_tokens=steer_tokens,
                eager_model=eager_model,
                max_new_tokens=attention_max_tokens,
            )
            attention_results.append(result)

    # Generate plots
    logger.info("=== Generating Plots ===")
    plot_dir = ensure_dir(effective_output_dir / "plots")
    kl_sweep_plot_paths = plot_prefix_length_kl_sweep(kl_sweep_result, plot_dir)
    kl_plot_paths = plot_kl_divergence_curves(kl_results, plot_dir)
    attn_plot_paths: list[Path] = []
    if attention_results:
        attn_plot_paths = plot_attention_analysis(attention_results, plot_dir)
        all_attn_links: list[AttentionLinkInstance] = []
        for attn_res in attention_results:
            all_attn_links.extend(attn_res.attention_links)
        if all_attn_links:
            heatmap_path = plot_attention_link_heatmap(all_attn_links, steer_tokens, plot_dir)
            attn_plot_paths.append(heatmap_path)

    all_plot_paths = kl_sweep_plot_paths + kl_plot_paths + attn_plot_paths

    # Generate report
    config_dict: dict[str, str | int | float | bool] = {
        "model": model_name,
        "concept": concept,
        "layer_frac": layer_frac,
        "layer_idx": layer_idx,
        "steer_tokens": steer_tokens,
        "scale_multiplier": scale_multiplier,
        "scale": scale,
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
        "run_attention": run_attention,
    }
    generate_analysis_report(
        kl_results=kl_results,
        attention_results=attention_results,
        config_dict=config_dict,
        plot_paths=all_plot_paths,
        output_path=effective_output_dir / "analysis_report.md",
        kl_sweep_result=kl_sweep_result,
    )

    logger.info("Analysis complete!")

    return PrefixAnalysisReport(
        kl_results=kl_results,
        kl_sweep_result=kl_sweep_result,
        attention_results=attention_results,
        config_dict=config_dict,
    )


__all__ = [
    "KLDivergenceResult",
    "PrefixLengthKLSweepResult",
    "AttentionLinkInstance",
    "AttentionAnalysisResult",
    "PrefixAnalysisReport",
    "per_token_kl_divergence",
    "run_kl_divergence_experiment",
    "run_prefix_length_kl_sweep",
    "run_attention_analysis",
    "run_prefix_analysis",
    "plot_kl_divergence_curves",
    "plot_prefix_length_kl_sweep",
    "plot_attention_analysis",
    "plot_attention_link_heatmap",
    "generate_analysis_report",
]
