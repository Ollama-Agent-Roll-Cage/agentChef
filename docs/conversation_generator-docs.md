# OllamaConversationGenerator Documentation

## Class: `OllamaConversationGenerator`

A lightweight class to generate formatted conversations using Ollama models. Produces conversations with alternating "from": "human" and "from": "gpt" entries. Enhanced with NLP hedging and analysis capabilities.

### Constructor

```python
def __init__(self, model_name="llama3", enable_hedging=True, ollama_interface=None)
```

Initialize the conversation generator.

**Parameters:**
- `model_name` (str): Name of the Ollama model to use for generating conversations.
- `enable_hedging` (bool): Whether to enable NLP hedging for more natural responses.
- `ollama_interface` (OllamaInterface, optional): Pre-configured Ollama interface. If None, a new interface will be created.

### Methods

#### `generate_conversation`

```python
def generate_conversation(self, content, num_turns=3, conversation_context="research", 
                         hedging_level="balanced", conversation_history=None)
```

Generate a conversation about the given content.

**Parameters:**
- `content` (str): The content to generate a conversation about.
- `num_turns` (int): Number of back-and-forth turns in the conversation.
- `conversation_context` (str): Context to guide the conversation topic.
- `hedging_level` (str): Level of hedging to use in responses ("confident", "balanced", "cautious").
- `conversation_history` (list, optional): Previous conversation history to build upon.

**Returns:**
- list: A list of conversation turns in the format `[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]` or None if generation fails.

#### `analyze_conversation_hedging`

```python
def analyze_conversation_hedging(self, conversations)
```

Analyze hedging patterns in a set of conversations.

**Parameters:**
- `conversations` (list): List of conversations to analyze

**Returns:**
- dict: Analysis of hedging patterns including counts, percentages, and examples

#### `generate_conversations_batch`

```python
def generate_conversations_batch(self, content_chunks, num_turns=3, context="research", 
                               hedging_level="balanced")
```

Generate multiple conversations from a list of content chunks.

**Parameters:**
- `content_chunks` (list): List of text chunks to generate conversations about.
- `num_turns` (int): Number of turns in each conversation.
- `context` (str): Context to guide the conversation topic.
- `hedging_level` (str): Level of hedging to use.

**Returns:**
- list: List of generated conversations.

#### `chunk_text`

```python
@staticmethod
def chunk_text(content, chunk_size=2000, overlap=200)
```

Split text content into overlapping chunks of specified size.

**Parameters:**
- `content` (str): Text to split into chunks.
- `chunk_size` (int): Maximum size of each chunk.
- `overlap` (int): Number of characters to overlap between chunks.

**Returns:**
- list: List of text chunks.

#### `generate_hedged_response`

```python
def generate_hedged_response(self, prompt, hedging_profile="balanced", 
                           knowledge_level="medium", subject_expertise="general")
```

Generate a response with appropriate hedging based on the specified profile.

**Parameters:**
- `prompt` (str): The prompt to respond to
- `hedging_profile` (str): Hedging profile to use ("confident", "balanced", "cautious")
- `knowledge_level` (str): Level of knowledge about the topic ("high", "medium", "low")
- `subject_expertise` (str): Level of expertise in the subject

**Returns:**
- str: Generated response with appropriate hedging

### Private Methods

#### `_get_hedging_instructions`

```python
def _get_hedging_instructions(self, hedging_level="balanced")
```

Get instructions for hedging in AI responses based on the specified level.

**Parameters:**
- `hedging_level` (str): Level of hedging ("confident", "balanced", "cautious")

**Returns:**
- str: Hedging instructions

#### `_validate_conversation_format`

```python
def _validate_conversation_format(self, conversation)
```

Validate that the conversation follows the expected format. Ensures each entry has "from" and "value" fields and corrects any issues.

**Parameters:**
- `conversation` (list): The conversation to validate.

**Raises:**
- ValueError: If the conversation format is invalid and can't be fixed.

#### `_basic_hedging_analysis`

```python
def _basic_hedging_analysis(self, conversations)
```

Perform basic analysis of hedging patterns without using the query engine.

**Parameters:**
- `conversations` (list): List of conversations to analyze

**Returns:**
- dict: Basic analysis of hedging patterns

#### `_conversations_to_df`

```python
def _conversations_to_df(self, conversations)
```

Convert conversations to a DataFrame for analysis.

**Parameters:**
- `conversations` (list): List of conversations to convert

**Returns:**
- pd.DataFrame: DataFrame representation of conversations
