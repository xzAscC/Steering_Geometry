#!/usr/bin/env python3
"""Validate JSON analysis results in outputs/unembed_analysis/json/."""

import json
import math
from pathlib import Path


def validate_json_file(filepath: Path) -> dict:
    """Validate a single JSON analysis file.

    Returns dict with validation results.
    """
    result = {
        "file": filepath.name,
        "valid_json": False,
        "has_required_fields": False,
        "valid_similarity_range": False,
        "no_nan_null": False,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    try:
        with open(filepath) as f:
            data = json.load(f)
        result["valid_json"] = True
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Read error: {e}")
        return result

    required_fields = ["concept", "model", "method", "results"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        result["errors"].append(f"Missing required fields: {missing}")
    else:
        result["has_required_fields"] = True

    if "results" not in data:
        return result

    results = data["results"]
    all_similarities = []
    all_tokens = []
    layer_count = 0

    for layer_key, layer_data in results.items():
        layer_count += 1

        if "tokens" not in layer_data:
            result["errors"].append(f"{layer_key}: missing 'tokens' field")
            continue
        if "similarities" not in layer_data:
            result["errors"].append(f"{layer_key}: missing 'similarities' field")
            continue

        tokens = layer_data["tokens"]
        similarities = layer_data["similarities"]

        all_tokens.extend(tokens)
        all_similarities.extend(similarities)

        for i, token in enumerate(tokens):
            if token is None:
                result["errors"].append(f"{layer_key}: token {i} is null")

        for i, sim in enumerate(similarities):
            if sim is None:
                result["errors"].append(f"{layer_key}: similarity {i} is null")
            elif isinstance(sim, float):
                if math.isnan(sim):
                    result["errors"].append(f"{layer_key}: similarity {i} is NaN")
                elif sim < -1.0 or sim > 1.0:
                    result["errors"].append(f"{layer_key}: similarity {i} out of range: {sim}")

    result["valid_similarity_range"] = all(
        -1.0 <= s <= 1.0 for s in all_similarities if isinstance(s, float)
    )
    result["no_nan_null"] = all(
        not (s is None or (isinstance(s, float) and math.isnan(s))) for s in all_similarities
    )

    result["stats"] = {
        "concept": data.get("concept", "unknown"),
        "model": data.get("model", "unknown"),
        "method": data.get("method", "unknown"),
        "layer_count": layer_count,
        "total_tokens": len(all_tokens),
        "total_similarities": len(all_similarities),
        "similarity_min": min(all_similarities) if all_similarities else None,
        "similarity_max": max(all_similarities) if all_similarities else None,
    }

    return result


def main():
    json_dir = Path("outputs/unembed_analysis/json")
    output_path = Path("outputs/unembed_analysis/VALIDATION.md")

    if not json_dir.exists():
        print(f"ERROR: Directory not found: {json_dir}")
        return 1

    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print(f"ERROR: No JSON files found in {json_dir}")
        return 1

    print(f"Validating {len(json_files)} JSON file(s)...\n")

    results = []
    for filepath in sorted(json_files):
        result = validate_json_file(filepath)
        results.append(result)

        status = (
            "✓ PASS"
            if all(
                [
                    result["valid_json"],
                    result["has_required_fields"],
                    result["valid_similarity_range"],
                    result["no_nan_null"],
                    not result["errors"],
                ]
            )
            else "✗ FAIL"
        )

        print(f"{status} {filepath.name}")

        if result["errors"]:
            for err in result["errors"]:
                print(f"    ERROR: {err}")

        if result["warnings"]:
            for warn in result["warnings"]:
                print(f"    WARNING: {warn}")

    lines = [
        "# Unembed Analysis Validation Report",
        "",
        "**Generated:** Validation run",
        f"**Files Checked:** {len(json_files)}",
        "",
        "## Summary",
        "",
    ]

    passed = sum(
        1
        for r in results
        if all(
            [
                r["valid_json"],
                r["has_required_fields"],
                r["valid_similarity_range"],
                r["no_nan_null"],
                not r["errors"],
            ]
        )
    )

    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| ✓ Pass | {passed} |")
    lines.append(f"| ✗ Fail | {len(results) - passed} |")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")

    for r in results:
        status = (
            "✓ PASS"
            if all(
                [
                    r["valid_json"],
                    r["has_required_fields"],
                    r["valid_similarity_range"],
                    r["no_nan_null"],
                    not r["errors"],
                ]
            )
            else "✗ FAIL"
        )

        lines.append(f"### {r['file']} ({status})")
        lines.append("")

        if r["stats"]:
            stats = r["stats"]
            lines.append(f"- **Concept:** {stats.get('concept', 'N/A')}")
            lines.append(f"- **Model:** {stats.get('model', 'N/A')}")
            lines.append(f"- **Method:** {stats.get('method', 'N/A')}")
            lines.append(f"- **Layers:** {stats.get('layer_count', 0)}")
            lines.append(f"- **Total Tokens:** {stats.get('total_tokens', 0)}")
            if stats.get("similarity_min") is not None:
                min_val = stats.get("similarity_min", 0)
                max_val = stats.get("similarity_max", 0)
                lines.append(f"- **Similarity Range:** [{min_val:.4f}, {max_val:.4f}]")
            else:
                lines.append("- **Similarity Range:** N/A")
            lines.append("")

        lines.append("| Check | Status |")
        lines.append("|-------|--------|")
        lines.append(f"| Valid JSON | {'✓' if r['valid_json'] else '✗'} |")
        lines.append(f"| Required Fields | {'✓' if r['has_required_fields'] else '✗'} |")
        lines.append(
            f"| Similarity Range [-1, 1] | {'✓' if r['valid_similarity_range'] else '✗'} |"
        )
        lines.append(f"| No NaN/Null Values | {'✓' if r['no_nan_null'] else '✗'} |")
        lines.append("")

        if r["errors"]:
            lines.append("**Errors:**")
            for err in r["errors"]:
                lines.append(f"- {err}")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    print(f"\nValidation report saved to: {output_path}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit(main())
