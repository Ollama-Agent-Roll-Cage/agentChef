# AgentChef Installation & Setup Guide

## Quick Installation

### 1. Install AgentChef
```bash
pip install agentChef
```

### 2. Install & Setup Ollama
```bash
# Download from https://ollama.ai/
# Then install a model:
ollama pull llama3.2:3b
ollama serve  # Keep this running
```

### 3. Verify Installation
```python
from agentChef import PandasRAG
import pandas as pd

# Test basic functionality
rag = PandasRAG()
agent_id = rag.register_agent("test", system_prompt="You are a test agent.")

df = pd.DataFrame({"test": [1, 2, 3]})
response = rag.query(df, "What is the sum of the test column?", agent_id)
print("✅ AgentChef is working:", response)
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB+ recommended for larger models)
- **Storage**: 2GB+ for models and data
- **OS**: Windows, macOS, or Linux

## Detailed Installation

### Option 1: Standard Installation
```bash
# Install AgentChef with core features
pip install agentChef

# Install optional UI dependencies
pip install agentChef[ui]

# Install development dependencies
pip install agentChef[dev]
```

### Option 2: Development Installation
```bash
# Clone repository
git clone https://github.com/yourusername/agentChef.git
cd agentChef

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## Ollama Setup

### Installation
1. Visit [https://ollama.ai/](https://ollama.ai/)
2. Download for your OS
3. Install following their instructions

### Model Installation
```bash
# Recommended models (choose based on your hardware)
ollama pull llama3.2:1b     # Fastest, 1.3GB
ollama pull llama3.2:3b     # Balanced, 2.0GB  
ollama pull llama3.1:8b     # Best quality, 4.7GB

# Specialized models
ollama pull codellama:7b    # For code analysis
ollama pull mistral:7b      # Alternative model
```

### Start Ollama Service
```bash
# Start the Ollama service (keep running)
ollama serve

# Verify it's working
ollama list
```

## Configuration

### Environment Variables (Optional)
```bash
# Data storage location
export AGENTCHEF_DATA_DIR="./my_agentchef_data"

# Ollama server location
export OLLAMA_HOST="http://localhost:11434"

# Logging level
export AGENTCHEF_LOG_LEVEL="INFO"
```

### First Run Configuration
```python
from agentChef import PandasRAG

# Initialize with custom settings
rag = PandasRAG(
    data_dir="./my_data",          # Where to store data
    model_name="llama3.2:3b",      # Which model to use
    log_level="INFO",              # Logging verbosity
    max_history_turns=10           # Conversation memory
)

print("✅ AgentChef configured successfully!")
```

## Verification

### Test Script
Create `test_agentchef.py`:

```python
import asyncio
from agentChef import PandasRAG, ResearchManager
import pandas as pd

async def test_agentchef():
    print("🧪 Testing AgentChef Installation...")
    
    # Test 1: PandasRAG
    print("1. Testing PandasRAG...")
    try:
        rag = PandasRAG(model_name="llama3.2:3b")
        agent_id = rag.register_agent("test", system_prompt="You are helpful.")
        
        df = pd.DataFrame({"numbers": [1, 2, 3, 4, 5]})
        response = rag.query(df, "What is the average of the numbers?", agent_id)
        print(f"   ✅ PandasRAG: {response[:50]}...")
    except Exception as e:
        print(f"   ❌ PandasRAG failed: {e}")
    
    # Test 2: ResearchManager
    print("2. Testing ResearchManager...")
    try:
        research = ResearchManager(model_name="llama3.2:3b")
        print("   ✅ ResearchManager initialized")
    except Exception as e:
        print(f"   ❌ ResearchManager failed: {e}")
    
    print("🎉 AgentChef testing complete!")

if __name__ == "__main__":
    asyncio.run(test_agentchef())
```

Run the test:
```bash
python test_agentchef.py
```

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'agentChef'"**
```bash
# Reinstall AgentChef
pip uninstall agentChef
pip install agentChef
```

**2. "Ollama is not available"**
```bash
# Check if Ollama is running
ps aux | grep ollama  # Linux/Mac
tasklist | findstr ollama  # Windows

# Start Ollama if not running
ollama serve
```

**3. "Model not found" errors**
```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.2:3b
```

**4. Performance issues**
```python
# Use a smaller model
rag = PandasRAG(model_name="llama3.2:1b")

# Reduce conversation history
rag = PandasRAG(max_history_turns=3)
```

**5. Permission errors**
```bash
# Fix data directory permissions
chmod 755 ./agentchef_data

# Or use a different location
rag = PandasRAG(data_dir="./my_custom_location")
```

### System-Specific Notes

**Windows:**
- Run Command Prompt as Administrator for installation
- Ensure Python is added to PATH
- Use PowerShell for better compatibility

**macOS:**
- Use Homebrew for Python installation if needed
- May need to install Xcode command line tools

**Linux:**
- Ensure Python development headers are installed
- May need to install additional dependencies:
  ```bash
  sudo apt-get install python3-dev python3-pip
  ```

## Next Steps

After successful installation:

1. **Try Examples**: Run the example scripts in `/examples`
2. **Read Guides**: Check out the PandasRAG and Custom Chef guides
3. **Build Something**: Create your first custom chef
4. **Join Community**: Connect with other AgentChef users

## Hardware Recommendations

### Minimum (for llama3.2:1b)
- 4GB RAM
- 2GB storage
- Basic CPU

### Recommended (for llama3.2:3b)
- 8GB RAM
- 5GB storage
- Multi-core CPU

### Optimal (for llama3.1:8b+)
- 16GB+ RAM
- 10GB+ storage
- High-performance CPU/GPU

## Getting Help

- **Documentation**: Check the `/docs` folder
- **Examples**: Review `/examples` for working code
- **Issues**: Report problems on GitHub
- **Community**: Join our Discord server

Ready to start cooking with AgentChef! 🍳🤖