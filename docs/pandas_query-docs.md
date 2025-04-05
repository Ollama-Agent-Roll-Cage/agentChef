# Pandas Query Integration Documentation

This module integrates LlamaIndex's PandasQueryEngine into the research and dataset generation system. It provides utilities for natural language querying of pandas DataFrames.

## Class: `PandasQueryIntegration`

Integrates LlamaIndex's PandasQueryEngine for natural language querying of pandas DataFrames.

### Constructor

```python
def __init__(self, openai_api_key=None, verbose=True, synthesize_response=True)
```

Initialize the PandasQueryIntegration.

**Parameters:**
- `openai_api_key` (str, optional): OpenAI API key. If None, uses environment variable.
- `verbose` (bool): Whether to print verbose output.
- `synthesize_response` (bool): Whether to synthesize a natural language response.

### Methods

#### `create_query_engine`

```python
def create_query_engine(self, df, custom_instructions=None)
```

Create a PandasQueryEngine for the given DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to query.
- `custom_instructions` (str, optional): Custom instructions for the query engine.

**Returns:**
- PandasQueryEngine: Query engine for natural language queries.

#### `query_dataframe`

```python
def query_dataframe(self, df, query, custom_instructions=None)
```

Query a DataFrame using natural language.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to query.
- `query` (str): Natural language query.
- `custom_instructions` (str, optional): Custom instructions for the query engine.

**Returns:**
- Dict[str, Any]: Query results including response and metadata.

#### `generate_dataset_insights`

```python
def generate_dataset_insights(self, df, num_insights=5)
```

Generate insights from a DataFrame using PandasQueryEngine.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to analyze.
- `num_insights` (int): Number of insights to generate.

**Returns:**
- List[Dict[str, Any]]: List of generated insights.

#### `compare_datasets`

```python
def compare_datasets(self, df1, df2, df1_name="Original", df2_name="Modified", aspects=None)
```

Compare two DataFrames and generate insights about the differences.

**Parameters:**
- `df1` (pd.DataFrame): First DataFrame.
- `df2` (pd.DataFrame): Second DataFrame.
- `df1_name` (str): Name of the first DataFrame.
- `df2_name` (str): Name of the second DataFrame.
- `aspects` (List[str], optional): Specific aspects to compare.

**Returns:**
- Dict[str, Any]: Comparison results.

## Class: `OllamaLlamaIndexIntegration`

Integration between Ollama and LlamaIndex for local LLM-powered DataFrame querying. This is a fallback when OpenAI API is not available.

### Constructor

```python
def __init__(self, ollama_model="llama3", verbose=True)
```

Initialize the OllamaLlamaIndexIntegration.

**Parameters:**
- `ollama_model` (str): Ollama model to use.
- `verbose` (bool): Whether to print verbose output.

### Methods

#### `query_dataframe_with_ollama`

```python
def query_dataframe_with_ollama(self, df, query)
```

Query a DataFrame using Ollama as the LLM backend.

This is a simplified version that doesn't use LlamaIndex directly but follows a similar approach. It sends the DataFrame info and query to Ollama and expects pandas code as a response.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to query.
- `query` (str): Natural language query.

**Returns:**
- Dict[str, Any]: Query results including response and pandas code.
