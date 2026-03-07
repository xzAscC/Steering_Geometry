# AI Research Template

Welcome to the AI Research Template! This guide will help you set up your new research project.

## About This Template

This template is designed for **AI-assisted research projects** — experiments, prototypes, and research codebases where AI agents help with implementation.

**Key features:**
- Python 3.12+ with modern tooling (uv, ruff, mypy, pytest)
- AI-friendly documentation structure (AGENTS.md, ARCHITECTURE.md)
- Research workflow support (experiment tracking, quality metrics)
- GitHub CI/CD pre-configured

## Getting Started

### 1. Create Your Repository

Click **"Use this template"** → **"Create a new repository"**

### 2. Clone and Setup

```bash
# Clone your new repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
uv sync
```

### 3. Rename the Package

Replace `ai_research_template` with your project name:

```bash
# Update all references
find . -type f \( -name "*.py" -o -name "*.toml" \) -exec sed -i 's/ai_research_template/YOUR_PROJECT_NAME/g' {} +

# Rename the source directory
mv src/ai_research_template src/YOUR_PROJECT_NAME
```

### 4. Verify Everything Works

```bash
# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run mypy src/

# Run tests
uv run pytest
```

### 5. Start Your Research

1. Update `README.md` with your project description
2. Update `ARCHITECTURE.md` with your system design
3. Document experiments in `docs/design-docs/`
4. Track quality in `docs/QUALITY_SCORE.md`

## Files to Update After Cloning

| File | What to Update |
|------|---------------|
| `pyproject.toml` | Project name, description, dependencies |
| `README.md` | Project description, remove template-specific content |
| `ARCHITECTURE.md` | Your system architecture |
| `AGENTS.md` | Package references (line 35: `packages = ["YOUR_PROJECT_NAME"]`) |
| `docs/PLANS.md` | Your project roadmap |
| `.github/` | Update issue templates if needed |

## Need Help?

- Check `AGENTS.md` for AI agent instructions
- See `ARCHITECTURE.md` for system design template
- Review `docs/` for documentation structure

## Related Templates

- **ai-general-template** — For general-purpose projects (coming soon)

---

Happy researching! 🔬
