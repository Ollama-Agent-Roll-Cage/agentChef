# Make sure you have the latest tools
pip install --upgrade build twine

# Build the package
python -m build

# Upload to TestPyPI first to check everything works
python -m twine upload --repository testpypi dist/*

# Once verified on TestPyPI, upload to PyPI
python -m twine upload dist/*