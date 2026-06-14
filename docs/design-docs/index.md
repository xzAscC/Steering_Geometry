# Design Documents

This directory holds design decisions and architectural rationale.

## Index

| Doc | Status | Date | Summary |
|-----|--------|------|---------|
| [core-beliefs.md](core-beliefs.md) | Approved | 2024-01 | Core engineering beliefs for AI-assisted development |
| [sweep-evaluation.md](sweep-evaluation.md) | Approved | 2026-05 | Steering strength x prefix token count sweep evaluation with heatmaps |
| [robust-dim-extraction.md](robust-dim-extraction.md) | Approved | 2026-06 | Robust DiM extraction, K ablation, candidate pool ablation, and comparison experiments |
| [construction-diagnosis.md](construction-diagnosis.md) | Approved | 2026-06 | Construction diagnosis: token position, prompt vs response, example count experiments |
| [prefix-steering-analysis.md](prefix-steering-analysis.md) | Approved | 2026-06 | Prefix steering KL divergence analysis, attention patterns, and prefix vs all-token comparison |

## Adding New Design Docs

1. Create a new `.md` file with descriptive name
2. Include: Context, Decision, Consequences
3. Add entry to index table above
4. Update status as doc evolves (Draft → Approved → Superseded)

## Naming Convention

- Use kebab-case: `feature-name.md`
- Be descriptive: `auth-strategy.md` not `auth.md`
