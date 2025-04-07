# AgentChef Testing Suite

This directory contains comprehensive tests for the agentChef library. The tests cover individual modules as well as integrated functionality.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Common fixtures and configuration
├── test_conversation_generator.py # Tests for conversation generation
├── test_dataset_expander.py       # Tests for dataset expansion
├── test_dataset_cleaner.py        # Tests for dataset cleaning
├── test_ollama_interface.py       # Tests for Ollama interface
├── test_pandas_query.py           # Tests for pandas query integration
├── test_crawlers/
│   ├── __init__.py
│   ├── test_web_crawler.py        # Tests for web crawler
│   ├── test_arxiv_searcher.py     # Tests for ArXiv searcher
│   ├── test_duckduckgo_searcher.py # Tests for DuckDuckGo searcher
│   └── test_github_crawler.py     # Tests for GitHub crawler
├── test_udrags.py                 # Integration tests for UDRAGS system
└── test_end_to_end.py             # End-to-end workflow tests
```

## Prerequisites

To run the tests, you'll need:

1. Python 3.8 or higher
2. pytest and related packages
3. The agentChef library installed in development mode

## Installation

```bash
# Clone the repository
git clone https://github.com/Leoleojames1/agentChef.git
cd agentChef

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Running Tests

### Running all tests

```bash
pytest
```

### Running tests for a specific module

```bash
# Test conversation generator
pytest tests/test_conversation_generator.py

# Test dataset expander
pytest tests/test_dataset_expander.py

# Test UDRAGS integration
pytest tests/test_udrags.py
```

### Running tests with coverage

```bash
pytest --cov=agentChef
```

### Running tests by category

```bash
# Run only unit tests (excluding slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m "integration"

# Run only end-to-end tests
pytest -m "e2e"
```

## Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test complete workflows

## Mock Implementation

The test suite uses Python's `unittest.mock` library to mock external dependencies:

1. **Ollama API**: All interactions with Ollama models are mocked
2. **External APIs**: ArXiv API, DuckDuckGo search, GitHub interactions
3. **File operations**: File reading/writing operations

## Configuration

All tests use fixtures defined in `conftest.py`, which provides:

1. Common test data
2. Mock implementations of components
3. Temporary directory management
4. Test configuration

## Writing New Tests

When adding new tests:

1. Follow the existing test structure
2. Use fixtures from `conftest.py` where possible
3. Mock external dependencies appropriately
4. Include both positive and negative test cases
5. Add appropriate markers for test categorization

## Continuous Integration

The test suite is integrated with GitHub Actions and runs automatically on pull requests and pushes to the main branch.

## Test Data

Sample test data is provided in `conftest.py` for:

1. Paper content
2. Conversations
3. HTML content
4. ArXiv paper information

## Troubleshooting

### Missing Dependencies

If you encounter import errors, ensure you've installed the library with development dependencies:

```bash
pip install -e ".[dev]"
```

### Async Test Failures

For async test failures, try running with `pytest-asyncio`:

```bash
pip install pytest-asyncio
```

### Coverage Issues

For test coverage issues, install and run with `pytest-cov`:

```bash
pip install pytest-cov
pytest --cov=agentChef --cov-report=html
```
