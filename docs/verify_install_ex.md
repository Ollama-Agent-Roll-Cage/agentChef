# Create a new virtual environment for testing
python -m venv test_env
cd test_env
# On Windows:
Scripts\activate
# On Linux/Mac:
source bin/activate

# Install your package
pip install agentChef

# Test importing it
python -c "import agentChef; print(agentChef.__version__)"