from src.agentChef.augmentation.dataset_expander import DatasetExpander
from src.agentChef.ollama.ollama_interface import OllamaInterface

# Initialize components
ollama_interface = OllamaInterface(model_name="llama3")

# Use a custom output directory
expander = DatasetExpander(
    ollama_interface=ollama_interface,
    output_dir="./my_conversation_data"  # Files will be saved here
)

# Your conversations
conversations = [
    [
        {"from": "human", "value": "What are transformer models?"},
        {"from": "gpt", "value": "Transformer models are a type of neural network..."}
    ]
]

# Save to JSONL
output_path = expander.save_conversations_to_jsonl(conversations, "my_conversations")
print(f"Saved conversations to: {output_path}")
# This would print: "Saved conversations to: ./my_conversation_data/my_conversations.jsonl"

# Or save in multiple formats
outputs = expander.convert_to_multi_format(
    conversations, 
    "my_conversations",
    formats=['jsonl', 'parquet', 'csv']
)
print("JSONL path:", outputs['jsonl'])
print("Parquet path:", outputs['parquet']) 
print("CSV path:", outputs['csv'])