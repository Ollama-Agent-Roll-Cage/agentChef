# DatasetCleaner Documentation

## Class: `DatasetCleaner`

A class to clean and validate expanded datasets by comparing them to original conversations and identifying/fixing quality issues using NLP queries through an LLM interface. Works alongside DatasetExpander to ensure high-quality conversation data.

### Constructor

```python
def __init__(self, ollama_interface, output_dir="./cleaned_output", use_llama_index=True, openai_api_key=None)
```

Initialize the DatasetCleaner.

**Parameters:**
- `ollama_interface`: An interface to Ollama for generating text and analyzing datasets
- `output_dir` (str): Directory to save cleaned datasets
- `use_llama_index` (bool): Whether to use LlamaIndex for advanced DataFrame querying
- `openai_api_key` (str, optional): OpenAI API key for LlamaIndex integration

### Methods

#### `analyze_dataset`

```python
def analyze_dataset(self, original_conversations, expanded_conversations)
```

Analyze expanded conversations to identify potential quality issues compared to originals.

**Parameters:**
- `original_conversations`: List of original conversations
- `expanded_conversations`: List of expanded conversations

**Returns:**
- Dictionary with analysis results

#### `clean_dataset`

```python
def clean_dataset(self, original_conversations, expanded_conversations, cleaning_criteria=None)
```

Clean the expanded dataset by fixing identified issues.

**Parameters:**
- `original_conversations`: List of original conversations
- `expanded_conversations`: List of expanded conversations
- `cleaning_criteria`: Dictionary of criteria to use for cleaning:
  - `fix_hallucinations` (bool): Fix factual errors or hallucinations
  - `normalize_style` (bool): Ensure consistent style
  - `correct_grammar` (bool): Fix grammar issues
  - `ensure_coherence` (bool): Ensure conversation flow is coherent

**Returns:**
- List of cleaned conversations

### Private Methods

#### `_perform_advanced_analysis`

```python
def _perform_advanced_analysis(self, orig_df, expanded_df)
```

Perform advanced analysis using PandasQueryIntegration.

**Parameters:**
- `orig_df`: DataFrame with original conversations
- `expanded_df`: DataFrame with expanded conversations

**Returns:**
- Dictionary with advanced analysis results

#### `_clean_conversation`

```python
def _clean_conversation(self, original_conv, expanded_conv, criteria)
```

Clean a single conversation by fixing issues.

**Parameters:**
- `original_conv`: Original conversation
- `expanded_conv`: Expanded conversation with potential issues
- `criteria`: Cleaning criteria

**Returns:**
- Cleaned conversation

#### `_needs_cleaning`

```python
def _needs_cleaning(self, expanded_turn, original_turn, criteria)
```

Determine if a turn needs cleaning based on quick heuristics.

**Parameters:**
- `expanded_turn`: Expanded conversation turn
- `original_turn`: Original conversation turn
- `criteria`: Cleaning criteria

**Returns:**
- True if turn needs cleaning, False otherwise

#### `_clean_turn_content`

```python
def _clean_turn_content(self, original_content, expanded_content, source, turn_idx, criteria)
```

Clean the content of a conversation turn using LLM.

**Parameters:**
- `original_content`: Content from original turn
- `expanded_content`: Content from expanded turn
- `source`: Source of the turn ('human' or 'gpt')
- `turn_idx`: Index of the turn in the conversation
- `criteria`: Cleaning criteria

**Returns:**
- Cleaned turn content

#### `_convert_conversations_to_df`

```python
def _convert_conversations_to_df(self, conversations)
```

Convert conversations to a DataFrame for analysis.

**Parameters:**
- `conversations`: List of conversations to convert

**Returns:**
- DataFrame representation of conversations

#### `_analyze_length_differences`

```python
def _analyze_length_differences(self, orig_df, expanded_df)
```

Analyze differences in content length between original and expanded conversations.

**Parameters:**
- `orig_df`: DataFrame with original conversations
- `expanded_df`: DataFrame with expanded conversations

**Returns:**
- Dictionary with length difference analysis

#### `_analyze_semantic_quality`

```python
def _analyze_semantic_quality(self, original_conversations, expanded_conversations)
```

Analyze semantic quality issues in expanded conversations compared to originals.

**Parameters:**
- `original_conversations`: List of original conversations
- `expanded_conversations`: List of expanded conversations

**Returns:**
- Dictionary with semantic quality analysis
