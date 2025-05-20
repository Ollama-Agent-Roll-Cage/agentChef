# AgentChef Interfaces

AgentChef provides three main interfaces:

## 1. Command Line Interface

```bash
# Research Mode
agentchef research --topic "transformer models" --max-papers 5

# Generate Mode
agentchef generate --topic "transformer models" --turns 3 --expand 2

# Process Mode
agentchef process --input papers/ --format all
```

## 2. FastAPI Interface

Start the API server:
```bash
uvicorn agentChef.api.agent_chef_api:app --reload
```

Example endpoints:
```bash
# Research
POST /research/topic
{
    "topic": "transformer models",
    "max_papers": 5
}

# Generate
POST /generate/conversation
{
    "content": "research content",
    "num_turns": 3
}
```

## 3. Graphical Interface

```bash
# Start the UI
agentchef --mode ui

# Or programmatically
from agentChef.core.ui_components.RagchefUI import RagchefUI
ui = RagchefUI(research_manager)
ui.show()
```

## 4. MCP Protocol

The Model Context Protocol allows IDE integration:

```python
from agentChef.mcp.mcp_server import MCPServer

server = MCPServer()
await server.start()
```

Configure in VS Code:
```json
{
    "llm.server": "agentchef-mcp",
    "llm.port": 50505
}
```
