# OllamaInterface Documentation

## Class: `OllamaInterface`

A unified interface for interacting with Ollama. This class provides a consistent way to access Ollama functionality throughout the agentChef package.

### Constructor

```python
def __init__(self, model_name="llama3")
```

Initialize the Ollama interface.

**Parameters:**
- `model_name` (str): Name of the Ollama model to use

### Methods

#### `chat`

```python
def chat(self, messages)
```

Send a chat request to Ollama.

**Parameters:**
- `messages` (List[Dict[str, str]]): List of message objects in the format:
  `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`

**Returns:**
- Dict[str, Any]: Response from Ollama or an error message

#### `embeddings`

```python
def embeddings(self, text)
```

Generate embeddings for text using Ollama.

**Parameters:**
- `text` (str): Text to create embeddings for

**Returns:**
- List[float]: Embedding vector or empty list on error

#### `is_available`

```python
def is_available(self)
```

Check if Ollama is available and working.

**Returns:**
- bool: True if Ollama is available and working, False otherwise
