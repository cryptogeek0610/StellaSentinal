# Contributing to SOTI Stella Sentinel

Thank you for your interest in contributing to SOTI Stella Sentinel! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes and learn from them

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AnomalyDetection.git
   cd AnomalyDetection
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/AnomalyDetection.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Docker & Docker Compose (recommended)

### Using Docker (Recommended)

```bash
# Copy environment template
cp env.template .env

# Start all services
make up

# Run tests
make test-synthetic
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install frontend dependencies
cd frontend && npm install && cd ..
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-detector` - New features
- `fix/memory-leak-dashboard` - Bug fixes
- `docs/update-api-guide` - Documentation
- `refactor/optimize-queries` - Code refactoring

### Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits

3. **Keep commits focused** - one logical change per commit

4. **Write meaningful commit messages**:
   ```
   Short summary (50 chars or less)

   More detailed description if needed. Explain the what and why,
   not the how. Wrap at 72 characters.

   - Bullet points are okay
   - Use present tense: "Add feature" not "Added feature"

   Fixes #123
   ```

## Coding Standards

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Maximum line length: 88 characters (Black default)

```python
def detect_anomalies(
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Detect anomalies in device feature data.

    Args:
        features: DataFrame containing device features
        threshold: Anomaly score threshold (0-1)

    Returns:
        List of detected anomalies with metadata
    """
    ...
```

### TypeScript/React

- Use functional components with hooks
- Follow ESLint configuration
- Use TypeScript strict mode
- Prefer named exports

```typescript
interface AnomalyCardProps {
  anomaly: Anomaly;
  onSelect: (id: string) => void;
}

export function AnomalyCard({ anomaly, onSelect }: AnomalyCardProps) {
  // Component implementation
}
```

### SQL

- Use uppercase for keywords
- Use meaningful aliases
- Format for readability

```sql
SELECT
    d.device_id,
    d.device_name,
    a.anomaly_score
FROM devices d
INNER JOIN anomalies a ON d.device_id = a.device_id
WHERE a.detected_at >= @start_date
ORDER BY a.anomaly_score DESC;
```

## Testing

### Running Tests

```bash
# Python tests
pytest tests/

# With coverage
pytest tests/ --cov=src/device_anomaly --cov-report=html

# Specific test file
pytest tests/test_anomaly_detection.py

# Frontend tests
cd frontend && npm test
```

### Writing Tests

- Write tests for new features and bug fixes
- Aim for meaningful coverage, not 100%
- Use descriptive test names

```python
def test_isolation_forest_detects_obvious_anomaly():
    """Verify that extreme outliers are flagged as anomalies."""
    ...

def test_baseline_updates_with_new_data():
    """Ensure baselines incorporate recent observations."""
    ...
```

## Submitting Changes

### Pull Request Process

1. **Update documentation** if needed
2. **Add/update tests** for your changes
3. **Ensure all tests pass** locally
4. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request** on GitHub

### Pull Request Guidelines

- Fill out the PR template completely
- Reference related issues
- Keep PRs focused - one feature/fix per PR
- Respond to review feedback promptly
- Squash commits if requested

### PR Title Format

```
type(scope): description

Examples:
feat(api): add batch anomaly detection endpoint
fix(dashboard): resolve memory leak in chart component
docs(readme): update installation instructions
refactor(models): simplify baseline calculation
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or screenshots

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Proposed solution (if any)
- Alternatives considered

---

## Questions?

If you have questions, feel free to:
- Open a Discussion on GitHub
- Check existing issues and documentation
- Reach out to the maintainers

Thank you for contributing! 🎉

