# UI Module Documentation

This module provides a simple graphical interface for the ragchef system. It offers a user-friendly interface for research, dataset generation, and analysis.

## Class: `WorkerThread`

Worker thread for running asynchronous operations.

### Constructor

```python
def __init__(self, func, *args, **kwargs)
```

Initialize the worker thread.

**Parameters:**
- `func`: Asynchronous function to run
- `*args`, `**kwargs`: Arguments for the function

### Methods

#### `run`

```python
def run(self)
```

Run the worker thread.

## Class: `RagchefUI`

Main window for the ragchef UI.

### Constructor

```python
def __init__(self, research_manager)
```

Initialize the UI.

**Parameters:**
- `research_manager`: ResearchManager instance

### Methods

#### `_setup_ui`

```python
def _setup_ui(self)
```

Set up the main UI components.

#### `_setup_research_tab`

```python
def _setup_research_tab(self)
```

Set up the Research tab.

#### `_setup_generate_tab`

```python
def _setup_generate_tab(self)
```

Set up the Generate tab.

#### `_setup_process_tab`

```python
def _setup_process_tab(self)
```

Set up the Process tab for existing papers.

#### `_setup_analyze_tab`

```python
def _setup_analyze_tab(self)
```

Set up the Analyze tab for dataset analysis.

#### Event Handlers

The UI includes various event handlers for button clicks and other user interactions:

- `_on_research_clicked()`: Handle research button click
- `_on_save_research_clicked()`: Handle save research results button click
- `_on_proceed_to_generate_clicked()`: Handle proceed to generate button click
- `_on_generate_clicked()`: Handle generate dataset button click
- `_on_browse_output_clicked()`: Handle browse output directory button click
- `_on_open_output_clicked()`: Handle open output directory button click
- `_on_proceed_to_analyze_clicked()`: Handle proceed to analyze button click
- `_on_browse_input_clicked()`: Handle browse input directory button click
- `_on_process_clicked()`: Handle process files button click
- `_on_proc_browse_output_clicked()`: Handle process output directory browse button click

## Usage

To use the UI, initialize a ResearchManager and pass it to the RagchefUI class:

```python
from agentChef.ragchef import ResearchManager
from agentChef.ui_module import RagchefUI
from PyQt6.QtWidgets import QApplication
import sys

# Initialize the research manager
manager = ResearchManager()

# Create the application
app = QApplication(sys.argv)
ui = RagchefUI(manager)
ui.show()

# Run the application
sys.exit(app.exec())
```

## Requirements

The UI module requires PyQt6 to be installed:

```bash
pip install PyQt6
```
