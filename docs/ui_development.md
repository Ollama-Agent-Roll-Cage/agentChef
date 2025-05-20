# AgentChef UI Development

## Current UI Components

- RagChef Research Interface
- Configuration Editor
- Progress Monitoring
- Data Visualization

## Upcoming Features

### Visual Node Editor
The node editor will allow visual workflow creation:
- Drag-and-drop components
- Visual data flow
- Real-time preview
- Custom node creation

### Parquet Data Viewer
Interactive data exploration:
- Column filtering
- Data preview
- Schema inspection
- Export options

### Component Library
Reusable UI elements:
- Progress bars
- Data tables
- Charts
- File pickers

## Creating Custom UIs

Inherit from base classes:
```python
from agentChef.core.ui_components import ChefUIBase

class CustomUI(ChefUIBase):
    def __init__(self, chef):
        super().__init__(chef)
        self.setup_ui()
```
