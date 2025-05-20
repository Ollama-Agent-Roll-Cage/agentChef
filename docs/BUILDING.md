# Building AgentChef

This guide covers setting up a development environment and building AgentChef using modern Python tools.

## Development Environment Setup

### Create Virtual Environment with UV

```bash
# Create virtual environment with Python 3.11 or 3.12
uv venv -p 3.11 .venv
.venv\Scripts\activate
```

### Clean Environment

If you need to reset your environment:

```bash
# Uninstall all packages
uv pip freeze > requirements.txt
uv pip uninstall -r requirements.txt

# Or remove entire environment
Remove-Item -Recurse -Force .venv
```

### Install Core Dependencies

```bash
# Install essential build tools
uv pip install uv pip wheel setuptools build twine

# Install AgentChef in development mode with dev dependencies
uv pip install -e ".[dev]"
```

## Building the Package

### Clean Build Artifacts

```bash
# Windows PowerShell
Remove-Item -Path "dist","build","*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue
```

### Build Package

```bash
# Build source distribution and wheel
python -m build
```

## Publishing

### Test Release

Upload to TestPyPI first to verify everything works:

```bash
python -m twine upload --repository testpypi dist/*
```

### Production Release

Once tested, upload to PyPI:

```bash
python -m twine upload dist/*
```

## Development Tips

- Always use a fresh virtual environment for clean testing
- Run tests before building: `pytest`
- Check package contents: `python -m pip show agentchef`
- Verify installation: `python -c "import agentchef; print(agentchef.__version__)"`

## Troubleshooting

If you encounter permission issues:
- Ensure you're in an activated virtual environment
- Check file/directory permissions
- Remove old build artifacts before rebuilding

For build errors:
- Verify pyproject.toml configuration
- Check Python version compatibility
- Review package dependencies
