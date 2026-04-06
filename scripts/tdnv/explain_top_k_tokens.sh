#!/bin/bash
# Generate markdown explanation of top-k discriminative tokens
#
# Usage:
#   ./explain_top_k_tokens.sh --output outputs/tdnv/
#
# Creates: {output}/top_k_explanation.md

set -euo pipefail

# Default output directory
OUTPUT_DIR="outputs/tdnv/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Output file
OUTPUT_FILE="${OUTPUT_DIR}top_k_explanation.md"

# Generate markdown content
cat > "$OUTPUT_FILE" << 'EOF'
# Top-K Discriminative Tokens Explained

## What Does "Top-K Discriminative Tokens" Mean?

When extracting steering vectors from language models, we need to decide **which token positions** to use for computing the steering direction. Not all tokens are equally informative!

**Top-k discriminative tokens** are the k tokens that best distinguish between two classes (e.g., "honest" vs "dishonest" responses). Instead of using random tokens or just the last n tokens, we select tokens that carry the strongest signal for the concept we want to steer.

## The Scoring Formula

For each token position `i`, we compute a discriminativity score:

```
s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
```

Where:
- `h_i` = hidden state vector of token `i`
- `μ_same` = centroid (mean) of hidden states from the **same class**
- `μ_other` = centroid (mean) of hidden states from the **other class**
- `||·||²` = squared L2 norm (Euclidean distance squared)

## Why This Formula Works

The score `s_i` captures how well token `i` separates the two classes:

| Condition | Effect on Score | Interpretation |
|-----------|-----------------|----------------|
| Token close to its own class (`||h_i - μ_same||²` is **small**) | Score **increases** | Token is representative of its class |
| Token far from other class (`||h_i - μ_other||²` is **large**) | Score **increases** | Token clearly distinguishes from other class |

**High score = highly discriminative token**

### Visual Intuition

```
                    μ_other (other class centroid)
                         *
                        / \
                       /   \
    h_i ---------->  *     \
                     |\     \
                     | \     \
                     |  \     \
        discriminative  \     distance to μ_other (LARGE)
        token (high s_i) \    
                           * 
                        μ_same (own class centroid)
                        
    Token h_i is:
    ✓ CLOSE to μ_same (small distance)
    ✓ FAR from μ_other (large distance)
    → High discriminativity score!
```

## Step-by-Step Example

Consider extracting a steering vector for "honesty" from a model:

1. **Collect pairs**: "honest" responses vs "dishonest" responses
2. **Extract hidden states**: Get `h_i` for each token in each response
3. **Compute centroids**:
   - `μ_honest` = mean of honest token hidden states
   - `μ_dishonest` = mean of dishonest token hidden states
4. **Score each token** in honest responses:
   ```
   s_i = ||h_i - μ_dishonest||² - ||h_i - μ_honest||²
   ```
5. **Select top-k**: Keep only the k tokens with highest scores
6. **Compute steering vector**: Use only these discriminative tokens

## Comparison with Other Selection Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Top-k discriminative** | Select tokens with highest `s_i` | Maximizes signal-to-noise, concept-specific | Requires computing centroids |
| **Last-n tokens** | Use last n tokens of each sequence | Simple, fast | May miss important early tokens |
| **Random selection** | Randomly sample token positions | Unbiased baseline | No guarantee of signal quality |
| **All tokens** | Use every token position | Complete information | Includes noisy/irrelevant tokens |

## Mathematical Derivation

The discriminativity score can be rewritten as:

```
s_i = ||h_i - μ_other||² - ||h_i - μ_same||²
    = (h_i - μ_other)·(h_i - μ_other) - (h_i - μ_same)·(h_i - μ_same)
    = ||h_i||² - 2h_i·μ_other + ||μ_other||² - ||h_i||² + 2h_i·μ_same - ||μ_same||²
    = 2h_i·(μ_same - μ_other) + ||μ_other||² - ||μ_same||²
```

Since the last two terms are constant for all tokens, maximizing `s_i` is equivalent to maximizing:

```
h_i · (μ_same - μ_other)
```

This is the **projection** of `h_i` onto the direction pointing from the other class to the same class. Tokens with higher projections along this direction are more discriminative.

## When to Use Top-K Discriminative Selection

**Use this method when:**
- You want maximum steering effect with fewer tokens
- Your concept is well-defined with clear positive/negative examples
- You have enough data to reliably estimate centroids

**Consider alternatives when:**
- You have very few examples (centroids may be unreliable)
- The concept doesn't have clear class boundaries
- Computational budget is extremely limited

## Implementation Notes

In this codebase, top-k discriminative selection is implemented in:
- `src/steering_geometry/extract.py`: Main extraction pipeline
- `src/steering_geometry/tdnv.py`: TDNV metrics computation

The selection happens before computing the final steering vector via contrastive methods (difference of means or PCA).

---

*Generated by `scripts/tdnv/explain_top_k_tokens.sh`*
EOF

echo "Created: $OUTPUT_FILE"
echo ""
echo "Content summary:"
echo "  - Explains top-k discriminative token concept"
echo "  - Documents scoring formula: s_i = ||h_i - μ_other||² - ||h_i - μ_same||²"
echo "  - Compares with random/last-n selection methods"
echo "  - Includes mathematical derivation and visual intuition"
