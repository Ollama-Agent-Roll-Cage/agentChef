import sys
import json
import asyncio
import tempfile
import os
import re
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QSplitter, QVBoxLayout, 
                           QWidget, QPushButton, QToolBar, QLabel, QStatusBar,
                           QComboBox, QFileDialog, QInputDialog, QMessageBox,
                           QCheckBox, QHBoxLayout, QButtonGroup, QRadioButton)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QProcess
from PyQt6.QtGui import QColor, QPalette, QIcon, QAction
import subprocess

# For Ollama integration
from ollama import AsyncClient

class OllamaWorker(QObject):
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model="llama3:latest"):
        super().__init__()
        self.model = model
        
    async def generate_content(self, prompt, notebook_context=None, cell_type="code", is_derive_mode=False):
        try:
            messages = []
            
            # Enhanced system prompt for better notebook structure
            if is_derive_mode:
                system_prompt = (
                    "You are a Jupyter notebook generator. Generate a well-structured notebook using this exact format:\n\n"
                    "<md>\n"
                    "# Title or section heading\n"
                    "Explanation text here\n"
                    "</md>\n\n"
                    "<code>\n"
                    "```python\n"
                    "# Python code here\n"
                    "print('hello world')\n"
                    "```\n"
                    "</code>\n\n"
                    "Important formatting rules:\n"
                    "1. ALWAYS wrap markdown content in <md></md> tags\n"
                    "2. ALWAYS wrap code content in <code></code> tags\n"
                    "3. ALWAYS wrap Python code blocks in ```python\n and ``` markers\n"
                    "4. NEVER mix markdown and code in the same section\n"
                    "5. ALWAYS include code examples in ```python blocks\n"
                    "6. Start each markdown section with a clear header\n"
                    "Add '<version_complete>Done</version_complete>' when finished"
                )
            else:
                system_prompt = (
                    "You are a Python notebook cell generator.\n"
                    f"Generate {'code' if cell_type == 'code' else 'markdown'} content using this exact format:\n\n"
                    "For code cells:\n"
                    "<code>\n"
                    "```python\n"
                    "# Python code here\n"
                    "print('hello world')\n"
                    "```\n"
                    "</code>\n\n"
                    "For markdown cells:\n"
                    "<md>\n"
                    "# Section heading\n"
                    "Explanation text here\n"
                    "</md>\n\n"
                    "Important: ALWAYS use these exact tags and formatting."
                )
            
            if notebook_context:
                system_prompt += f"\n\nCurrent notebook context:\n{notebook_context}"
            
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            result = ""
            async for part in await AsyncClient().chat(model=self.model, messages=messages, stream=True):
                content = part['message']['content']
                result += content
                self.result_ready.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def start_generation(self, prompt, notebook_context=None, cell_type="code", is_derive_mode=False):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.generate_content(prompt, notebook_context, cell_type, is_derive_mode))

class NotebookCell:
    def __init__(self, cell_type="code", source="", outputs=None):
        self.cell_type = cell_type
        self.source = source if isinstance(source, list) else [source]
        self.outputs = outputs or []
        self.metadata = {}
        
    def to_dict(self):
        cell_dict = {
            "cell_type": self.cell_type,
            "metadata": self.metadata,
            "source": self.source
        }
        
        if self.cell_type == "code":
            cell_dict["execution_count"] = None
            cell_dict["outputs"] = self.outputs
            
        return cell_dict
        
class NotebookDocument:
    def __init__(self):
        self.cells = []
        self.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py", "mimetype": "text/x-python", "name": "python",
                "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.10"
            }
        }
        
    def add_cell(self, cell, index=None):
        if index is not None and 0 <= index < len(self.cells):
            self.cells.insert(index, cell)
        else:
            self.cells.append(cell)
    
    def update_cell(self, index, content):
        if 0 <= index < len(self.cells):
            self.cells[index].source = content if isinstance(content, list) else [content]
            return True
        return False
    
    def delete_cell(self, index):
        if 0 <= index < len(self.cells):
            del self.cells[index]
            return True
        return False
        
    def to_dict(self):
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "metadata": self.metadata,
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
    
    def to_plain_text(self):
        text = "# Notebook Content:\n\n"
        for i, cell in enumerate(self.cells):
            content = "".join(cell.source)
            text += f"## Cell {i+1} ({cell.cell_type}):\n{content}\n\n"
        return text
        
    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        doc = cls()
        doc.metadata = data.get("metadata", doc.metadata)
        
        for cell_data in data.get("cells", []):
            cell = NotebookCell(
                cell_type=cell_data.get("cell_type", "code"),
                source=cell_data.get("source", []),
                outputs=cell_data.get("outputs", [])
            )
            cell.metadata = cell_data.get("metadata", {})
            doc.add_cell(cell)
            
        return doc

    def parse_tagged_content(self, content):
        """Parse content with markdown and code tags into separate cells"""
        cells = []
        
        # First try explicit tags
        if '<md>' in content or '<code>' in content:
            # Normalize tags and clean up code blocks
            content = re.sub(r'```python\n?', '', content)  # Remove ```python markers
            content = re.sub(r'```\n?', '', content)        # Remove closing ``` markers
            
            # Split content into sections based on tags
            parts = re.split(r'(<(?:md|code)>|</(?:md|code)>)', content)
            current_type = None
            current_content = []
            
            for part in parts:
                if part.strip():
                    if part == '<md>':
                        if current_type and current_content:
                            cells.append(NotebookCell(current_type, '\n'.join(current_content)))
                        current_type = 'markdown'
                        current_content = []
                    elif part == '<code>':
                        if current_type and current_content:
                            cells.append(NotebookCell(current_type, '\n'.join(current_content)))
                        current_type = 'code'
                        current_content = []
                    elif not part.startswith('</'):
                        if current_type:
                            current_content.append(part.strip())
            
            # Add final cell if there's content
            if current_type and current_content:
                cells.append(NotebookCell(current_type, '\n'.join(current_content)))
        
        # If no explicit tags, try to parse based on markdown/code patterns
        else:
            # Split by code blocks first
            parts = re.split(r'(```python[\s\S]*?```)', content)
            
            for part in parts:
                if part.strip():
                    if part.startswith('```python'):
                        # Extract code between markers
                        code = re.sub(r'```python\n?|```\n?', '', part).strip()
                        if code:
                            cells.append(NotebookCell('code', code))
                    else:
                        # This is markdown content
                        markdown = part.strip()
                        if markdown:
                            cells.append(NotebookCell('markdown', markdown))
        
        return cells

class NotebookApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Notebook")
        self.setGeometry(100, 100, 1200, 800)
        self.current_file = None
        self.notebook = NotebookDocument()
        self.ollama_model = "llama3:latest"
        self.current_editing_cell = None
        self.generating_cell_index = None
        self.is_dark_mode = False
        self.generation_mode = "single"  # "single" or "derive"
        self.derive_mode_in_progress = False
        self.derive_mode_prompt = ""
        self.iteration_count = 0
        
        # Theme colors (light mode default)
        self.update_colors(is_dark=False)
        
        # Setup UI
        self.setup_ui()
        
        # Initialize Ollama worker
        self.ollama_worker = OllamaWorker(self.ollama_model)
        self.ollama_worker.result_ready.connect(self.handle_ollama_result)
        self.ollama_worker.error_occurred.connect(self.handle_ollama_error)
        
        # Start with a blank notebook
        self.new_notebook()
        
        # Fetch available models
        self.fetch_ollama_models()
    
    def update_colors(self, is_dark=False):
        if is_dark:
            # Dark mode
            self.colors = {
                'bg_primary': '#1a1a2e',       # Dark blue background
                'bg_secondary': '#16213e',     # Slightly lighter blue for cells
                'accent': '#FFD700',           # Race car yellow (kept)
                'text': '#e1e1e1',             # Light text
                'cell_code': '#2c2c44',        # Dark blue-purple for code cells
                'cell_md': '#262640',          # Dark blue for markdown cells
                'cell_border': '#3a3a5c',      # Subtle border
                'button_bg': '#FFD700',        # Yellow button background
                'button_text': '#000000',      # Black button text
                'selected': '#FFA500',         # Orange for selected items
                'toolbar_bg': '#16213e',       # Toolbar background
                'status_bg': '#16213e'         # Status bar background
            }
        else:
            # Light mode
            self.colors = {
                'bg_primary': '#87CEEB',      # Sky blue background
                'bg_secondary': '#e1f5fe',    # Lighter blue
                'accent': '#FFD700',          # Race car yellow
                'text': '#343434',            # Dark text
                'cell_code': '#F0F8FF',       # Light blue for code cells
                'cell_md': '#FFFAF0',         # Warm white for markdown cells
                'cell_border': '#4682B4',     # Steel blue borders
                'button_bg': '#FFD700',       # Yellow button background
                'button_text': '#000000',     # Black button text
                'selected': '#FFA500',        # Orange for selected items
                'toolbar_bg': '#e1f5fe',      # Toolbar background  
                'status_bg': '#e1f5fe'        # Status bar background
            }
    
    def setup_theme(self):
        # Set application palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(self.colors['bg_primary']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(self.colors['text']))
        palette.setColor(QPalette.ColorRole.Base, QColor(self.colors['cell_code']))
        palette.setColor(QPalette.ColorRole.Button, QColor(self.colors['button_bg']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(self.colors['button_text']))
        self.setPalette(palette)
        
        # Set application stylesheet
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {self.colors['bg_primary']}; }}
            QPushButton {{ 
                background-color: {self.colors['button_bg']}; 
                color: {self.colors['button_text']}; 
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                margin: 2px;
            }}
            QPushButton:hover {{ background-color: {self.colors['selected']}; }}
            QToolBar {{ 
                background-color: {self.colors['toolbar_bg']}; 
                border: 1px solid {self.colors['cell_border']}; 
                border-radius: 4px;
                spacing: 5px;
                padding: 5px;
            }}
            QStatusBar {{ background-color: {self.colors['status_bg']}; }}
            QComboBox {{ 
                background-color: {self.colors['cell_code']}; 
                border: 1px solid {self.colors['cell_border']}; 
                border-radius: 4px;
                padding: 4px;
                color: {self.colors['text']};
            }}
            QRadioButton, QLabel, QCheckBox {{ 
                color: {self.colors['text']}; 
                font-weight: bold; 
                margin-right: 8px;
            }}
            QRadioButton::indicator, QCheckBox::indicator {{
                width: 15px;
                height: 15px;
            }}
            QSplitter::handle {{
                background-color: {self.colors['accent']};
            }}
            QMenuBar {{ 
                background-color: {self.colors['toolbar_bg']}; 
                color: {self.colors['text']}; 
            }}
            QMenuBar::item:selected {{ background-color: {self.colors['selected']}; }}
            QMenu {{ 
                background-color: {self.colors['bg_secondary']}; 
                color: {self.colors['text']}; 
            }}
            QMenu::item:selected {{ background-color: {self.colors['selected']}; }}
        """)
        
        # Also need to update editor content
        if hasattr(self, 'editor'):
            self.update_editor()
        
    def setup_ui(self):
        # Create central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("&New Notebook", self.new_notebook)
        file_menu.addAction("&Open Notebook", self.open_notebook)
        file_menu.addAction("&Save", self.save_notebook)
        file_menu.addAction("Save &As...", self.save_notebook_as)
        
        # Ollama menu
        ollama_menu = menu_bar.addMenu("&Ollama")
        ollama_menu.addAction("Select &Model", self.select_ollama_model)
        ollama_menu.addAction("&Refresh Models", self.fetch_ollama_models)
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        toggle_theme_action = QAction("Toggle Dark/Light Mode", self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(toggle_theme_action)
        
        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Cell operations
        cell_ops_layout = QHBoxLayout()
        cell_ops_widget = QWidget()
        cell_ops_widget.setLayout(cell_ops_layout)
        
        add_code_btn = QPushButton("+ Code")
        add_code_btn.setToolTip("Add a new code cell")
        add_code_btn.clicked.connect(lambda: self.add_cell("code"))
        cell_ops_layout.addWidget(add_code_btn)
        
        add_md_btn = QPushButton("+ Markdown")
        add_md_btn.setToolTip("Add a new markdown cell")
        add_md_btn.clicked.connect(lambda: self.add_cell("markdown"))
        cell_ops_layout.addWidget(add_md_btn)
        
        delete_cell_btn = QPushButton("Delete")
        delete_cell_btn.setToolTip("Delete selected cell")
        delete_cell_btn.clicked.connect(self.delete_current_cell)
        cell_ops_layout.addWidget(delete_cell_btn)
        
        toolbar.addWidget(cell_ops_widget)
        toolbar.addSeparator()
        
        # Generation mode selection
        gen_mode_widget = QWidget()
        gen_mode_layout = QHBoxLayout(gen_mode_widget)
        gen_mode_layout.setContentsMargins(0, 0, 0, 0)
        
        gen_mode_layout.addWidget(QLabel("Mode:"))
        
        self.mode_group = QButtonGroup(self)
        
        self.single_mode_radio = QRadioButton("Single Cell")
        self.single_mode_radio.setChecked(True)
        self.single_mode_radio.toggled.connect(lambda: self.set_generation_mode("single"))
        gen_mode_layout.addWidget(self.single_mode_radio)
        self.mode_group.addButton(self.single_mode_radio)
        
        self.derive_mode_radio = QRadioButton("Derive Notebook")
        self.derive_mode_radio.toggled.connect(lambda: self.set_generation_mode("derive"))
        gen_mode_layout.addWidget(self.derive_mode_radio)
        self.mode_group.addButton(self.derive_mode_radio)
        
        toolbar.addWidget(gen_mode_widget)
        toolbar.addSeparator()
        
        # Ollama controls widget
        ollama_widget = QWidget()
        ollama_layout = QHBoxLayout(ollama_widget)
        ollama_layout.setContentsMargins(0, 0, 0, 0)
        
        ollama_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        ollama_layout.addWidget(self.model_combo)
        
        self.use_context_checkbox = QCheckBox("Use context")
        self.use_context_checkbox.setChecked(True)
        ollama_layout.addWidget(self.use_context_checkbox)
        
        toolbar.addWidget(ollama_widget)
        
        # Generate buttons
        gen_buttons_widget = QWidget()
        gen_buttons_layout = QHBoxLayout(gen_buttons_widget)
        gen_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create distinctive, more prominent buttons
        generate_code_btn = QPushButton("💻 Generate Code")
        generate_code_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent']};
                font-size: 14px;
                padding: 10px 20px;
            }}
        """)
        generate_code_btn.clicked.connect(lambda: self.generate_with_ollama("code"))
        gen_buttons_layout.addWidget(generate_code_btn)
        
        generate_md_btn = QPushButton("📝 Generate Markdown")
        generate_md_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['accent']};
                font-size: 14px;
                padding: 10px 20px;
            }}
        """)
        generate_md_btn.clicked.connect(lambda: self.generate_with_ollama("markdown"))
        gen_buttons_layout.addWidget(generate_md_btn)
        
        toolbar.addWidget(gen_buttons_widget)
        
        toolbar.addSeparator()
        
        # Run button
        run_btn = QPushButton("▶ Run Notebook")
        run_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #28a745;
                color: white;
                font-size: 14px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
        """)
        run_btn.clicked.connect(self.run_notebook)
        toolbar.addWidget(run_btn)
        
        # Create editor and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.editor = QWebEngineView()
        self.editor.page().loadFinished.connect(self.on_editor_loaded)
        
        self.preview = QWebEngineView()
        
        splitter.addWidget(self.editor)
        splitter.addWidget(self.preview)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Apply theme
        self.setup_theme()
        
        # Setup JavaScript communication channel
        self.editor.page().javaScriptConsoleMessage = self.handle_js_console
        
    def set_generation_mode(self, mode):
        if mode in ["single", "derive"]:
            self.generation_mode = mode
            self.status_label.setText(f"Generation mode set to: {mode}")
    
    def toggle_theme(self):
        # Toggle the dark mode flag
        self.is_dark_mode = not self.is_dark_mode
        
        # Update colors based on new setting
        self.update_colors(is_dark=self.is_dark_mode)
        
        # Apply the theme
        self.setup_theme()
        
    def handle_js_console(self, level, message, line, source):
        # Handle cell content updates from JavaScript
        if message.startswith("CELL_UPDATE:"):
            try:
                _, cell_index, content = message.split(":", 2)
                self.update_cell_content(int(cell_index), content)
            except Exception as e:
                print(f"Error handling cell update: {str(e)}")
                
        # Handle cell selection
        elif message.startswith("CELL_SELECTED:"):
            try:
                _, cell_index = message.split(":", 1)
                self.current_editing_cell = int(cell_index)
                print(f"Selected cell: {self.current_editing_cell}")  # Debug output
                self.update_status_bar()
            except Exception as e:
                print(f"Error handling cell selection: {str(e)}")

    def update_status_bar(self):
        """Update status bar with current cell selection and other info"""
        if self.current_editing_cell is not None and 0 <= self.current_editing_cell < len(self.notebook.cells):
            cell = self.notebook.cells[self.current_editing_cell]
            status = f"Selected: Cell {self.current_editing_cell + 1} ({cell.cell_type})"
            if self.current_file:
                status += f" | File: {os.path.basename(self.current_file)}"
            self.status_label.setText(status)
        else:
            self.current_editing_cell = None
            self.status_label.setText("No cell selected")
        
    def on_editor_loaded(self, success):
        if success:
            # Add event handlers for cell editing and selection
            self.editor.page().runJavaScript("""
                document.querySelectorAll('.cell-content').forEach(cell => {
                    cell.addEventListener('input', function() {
                        const cellIndex = this.closest('.cell').dataset.index;
                        console.log('CELL_UPDATE:' + cellIndex + ':' + this.innerText);
                    });
                    
                    cell.addEventListener('focus', function() {
                        const cellIndex = this.closest('.cell').dataset.index;
                        console.log('CELL_SELECTED:' + cellIndex);
                    });
                });
            """)
            
    def update_cell_content(self, cell_index, content):
        # Update the notebook document with edited cell content
        self.notebook.update_cell(cell_index, content)
        
    def new_notebook(self):
        self.notebook = NotebookDocument()
        self.notebook.add_cell(NotebookCell("markdown", "# New Neural Notebook\n\nCreate powerful notebooks with AI assistance."))
        self.notebook.add_cell(NotebookCell("code", "# Your code here\n"))
        self.current_file = None
        self.current_editing_cell = 0  # Set initial selection
        self.update_editor()
        self.status_label.setText("New notebook created")
        
    def open_notebook(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Notebook", "", "Jupyter Notebooks (*.ipynb)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    notebook_json = f.read()
                    
                self.notebook = NotebookDocument.from_json(notebook_json)
                self.current_file = file_path
                self.update_editor()
                self.status_label.setText(f"Opened: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open notebook: {str(e)}")
                
    def save_notebook(self):
        if not self.current_file:
            return self.save_notebook_as()
            
        try:
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(self.notebook.to_json())
            self.status_label.setText(f"Saved: {os.path.basename(self.current_file)}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save notebook: {str(e)}")
            return False
            
    def save_notebook_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Notebook As", "", "Jupyter Notebooks (*.ipynb)"
        )
        
        if file_path:
            if not file_path.endswith('.ipynb'):
                file_path += '.ipynb'
                
            self.current_file = file_path
            return self.save_notebook()
        return False
        
    def add_cell(self, cell_type):
        default_content = "# Your code here\n" if cell_type == "code" else "## New Section\n"
            
        # Add after the current cell if one is selected
        if self.current_editing_cell is not None:
            self.notebook.add_cell(NotebookCell(cell_type, default_content), self.current_editing_cell + 1)
            self.current_editing_cell += 1
        else:
            self.notebook.add_cell(NotebookCell(cell_type, default_content))
            
        self.update_editor()
        
    def delete_current_cell(self):
        print(f"Current editing cell: {self.current_editing_cell}")  # Debug output
        if self.current_editing_cell is not None and 0 <= self.current_editing_cell < len(self.notebook.cells):
            if self.notebook.delete_cell(self.current_editing_cell):
                # Reset cell selection after deletion
                if len(self.notebook.cells) > 0:
                    # Keep the same index unless it's now out of bounds
                    self.current_editing_cell = min(self.current_editing_cell, len(self.notebook.cells) - 1)
                else:
                    self.current_editing_cell = None
                self.update_editor()
                self.status_label.setText("Cell deleted")
                return
    
        QMessageBox.warning(self, "Warning", "Please select a cell to delete")
        self.status_label.setText("No cell selected")
        
    def update_editor(self):
        # Create interactive HTML representation of the notebook
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ 
                    font-family: Arial; 
                    margin: 0; 
                    padding: 10px;
                    background-color: {self.colors['bg_primary']};
                    color: {self.colors['text']};
                }}
                .cell {{ 
                    border: 1px solid {self.colors['cell_border']}; 
                    margin: 10px 0; 
                    padding: 10px 10px 10px 45px; 
                    position: relative;
                    border-radius: 6px;
                }}
                .cell:focus-within {{
                    border: 2px solid {self.colors['accent']};
                    box-shadow: 0 0 8px 1px rgba(255, 215, 0, 0.5);
                }}
                .cell-code {{ background-color: {self.colors['cell_code']}; }}
                .cell-markdown {{ background-color: {self.colors['cell_md']}; }}
                .cell-type {{ 
                    position: absolute; 
                    top: 5px; 
                    right: 10px; 
                    color: {self.colors['text']}; 
                    font-size: 10px;
                    user-select: none;
                    opacity: 0.7;
                }}
                .cell-run-button {{
                    position: absolute;
                    left: 8px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 25px;
                    height: 25px;
                    background-color: {self.colors['button_bg']};
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                .cell-run-button:hover {{ background-color: {self.colors['selected']}; }}
                .cell-run-button::after {{
                    content: '▶';
                    font-size: 12px;
                    color: {self.colors['button_text']};
                }}
                .cell-content {{ 
                    margin: 0; 
                    white-space: pre-wrap; 
                    min-height: 1.5em;
                    outline: none;
                    font-family: monospace;
                    line-height: 1.5;
                }}
                [contenteditable] {{ outline: none; }}
            </style>
        </head>
        <body>
            <div id="notebook-container">
        """
        
        for i, cell in enumerate(self.notebook.cells):
            cell_type_class = f"cell-{cell.cell_type}"
            cell_content = "".join(cell.source)
            
            html += f"""
                <div class="cell {cell_type_class}" data-index="{i}">
                    <div class="cell-run-button" title="Run cell"></div>
                    <div class="cell-type">{cell.cell_type}</div>
                    <div class="cell-content" contenteditable="true" spellcheck="false">{cell_content}</div>
                </div>
            """
            
        html += """
            </div>
            <script>
                // Initialize selection tracking
                let currentSelectedCell = null;
                
                function selectCell(cell) {
                    // Clear previous selection visual indicator if any
                    if (currentSelectedCell) {
                        currentSelectedCell.style.boxShadow = '';
                    }
                    
                    // Set new selection
                    currentSelectedCell = cell;
                    cell.style.boxShadow = '0 0 10px rgba(255, 215, 0, 0.8)';
                    
                    // Notify Python of selection
                    const cellIndex = cell.dataset.index;
                    console.log('CELL_SELECTED:' + cellIndex);
                }

                // Initial cell selection
                document.addEventListener('DOMContentLoaded', function() {
                    const firstCell = document.querySelector('.cell');
                    if (firstCell) {
                        selectCell(firstCell);
                    }
                });

                // Make entire cell div clickable
                document.querySelectorAll('.cell').forEach(cell => {
                    cell.addEventListener('click', function(e) {
                        selectCell(this);
                        
                        // Focus content area unless clicking run button
                        if (!e.target.classList.contains('cell-run-button')) {
                            const content = this.querySelector('.cell-content');
                            if (content) {
                                content.focus();
                            }
                        }
                    });
                    
                    // Handle content focus
                    const content = cell.querySelector('.cell-content');
                    if (content) {
                        content.addEventListener('focus', function() {
                            selectCell(cell);
                        });
                    }
                });
            </script>
        </body>
        </html>
        """
        
        self.editor.setHtml(html)
        
    def run_notebook(self):
        if not self.current_file or not self.save_notebook():
            QMessageBox.warning(self, "Warning", "Save the notebook first")
            return
            
        # Open the notebook in the default Jupyter app
        try:
            if sys.platform == 'win32':
                os.startfile(self.current_file)
            elif sys.platform == 'darwin':
                subprocess.call(['open', self.current_file])
            else:
                subprocess.call(['jupyter', 'notebook', self.current_file])
            self.status_label.setText("Opened notebook in Jupyter")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Jupyter: {str(e)}")
            
    def fetch_ollama_models(self):
        """Fetch available Ollama models using subprocess directly"""
        self.status_label.setText("Fetching Ollama models...")
        self.model_combo.clear()
        
        try:
            # Use subprocess to run ollama list command
            if sys.platform == 'win32':
                proc = subprocess.Popen(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            else:
                proc = subprocess.Popen(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            stdout, stderr = proc.communicate()
            
            if proc.returncode != 0:
                raise Exception(f"Ollama command failed: {stderr.decode('utf-8')}")
                
            # Parse the output
            output = stdout.decode('utf-8')
            lines = output.strip().split('\n')
            
            # Skip header line
            if len(lines) > 1:
                models = []
                for line in lines[1:]:  # Skip header line
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        models.append(model_name)
                
                if models:
                    self.model_combo.addItems(models)
                    # Set current model if it exists in the list
                    current_index = self.model_combo.findText(self.ollama_model)
                    if current_index >= 0:
                        self.model_combo.setCurrentIndex(current_index)
                    self.status_label.setText(f"Found {len(models)} Ollama models")
                else:
                    self.model_combo.addItem("No models found")
                    self.status_label.setText("No Ollama models found")
            else:
                self.model_combo.addItem("No models found")
                self.status_label.setText("No Ollama models found")
                
        except Exception as e:
            self.status_label.setText(f"Error fetching models: {str(e)}")
            self.model_combo.addItem("Error loading models")
            
    def select_ollama_model(self):
        # Get current models from the combo box
        models = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        if not models or models[0] == "No models found" or models[0] == "Error loading models":
            QMessageBox.warning(self, "Warning", "No Ollama models available. Please install models with 'ollama pull <model>'")
            return
            
        model, ok = QInputDialog.getItem(
            self, "Select Ollama Model", "Choose a model:", models, 0, False
        )
        
        if ok and model:
            self.ollama_model = model
            self.ollama_worker = OllamaWorker(self.ollama_model)
            self.ollama_worker.result_ready.connect(self.handle_ollama_result)
            self.ollama_worker.error_occurred.connect(self.handle_ollama_error)
            
            # Update combo box selection
            index = self.model_combo.findText(model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                
            self.status_label.setText(f"Ollama model set to {model}")
            
    def generate_with_ollama(self, cell_type):
        selected_model = self.model_combo.currentText()
        if selected_model in ["No models found", "Error loading models"]:
            QMessageBox.warning(self, "Warning", "Please select a valid Ollama model")
            return
            
        # Update the current model from the combo box
        if selected_model != self.ollama_model:
            self.ollama_model = selected_model
            self.ollama_worker = OllamaWorker(self.ollama_model)
            self.ollama_worker.result_ready.connect(self.handle_ollama_result)
            self.ollama_worker.error_occurred.connect(self.handle_ollama_error)
        
        # Handle derive mode differently
        if self.generation_mode == "derive":
            self.generate_in_derive_mode(cell_type)
            return
        
        # Single cell mode
        prompt, ok = QInputDialog.getText(
            self, 
            f"Generate {cell_type.capitalize()}",
            "Enter your prompt:",
            text=f"Generate {'code to' if cell_type == 'code' else 'documentation for'} "
        )
        
        if ok and prompt:
            # Prepare notebook context if enabled
            notebook_context = None
            if self.use_context_checkbox.isChecked():
                notebook_context = self.notebook.to_plain_text()
            
            # Create a placeholder cell for the generated content
            placeholder_text = f"Generating {cell_type} content with {self.ollama_model}..."
            
            # Add the placeholder cell at the current position or at the end
            if self.current_editing_cell is not None:
                self.notebook.add_cell(NotebookCell(cell_type, placeholder_text), self.current_editing_cell + 1)
                self.current_editing_cell += 1
                self.generating_cell_index = self.current_editing_cell
            else:
                self.notebook.add_cell(NotebookCell(cell_type, placeholder_text))
                self.generating_cell_index = len(self.notebook.cells) - 1
                
            self.update_editor()
            self.status_label.setText(f"Generating {cell_type} with {self.ollama_model}...")
            
            # Run Ollama in a separate thread to keep UI responsive
            import threading
            thread = threading.Thread(
                target=self.ollama_worker.start_generation,
                args=(prompt, notebook_context, cell_type, False)  # False for not derive mode
            )
            thread.daemon = True
            thread.start()
            
    def generate_in_derive_mode(self, cell_type):
        """Handle notebook generation in derive mode"""
        # If already in derive mode and in progress, just continue
        if self.derive_mode_in_progress:
            self.continue_derive_mode()
            return
            
        # Starting a new derive mode session
        prompt, ok = QInputDialog.getText(
            self, 
            "Derive Complete Notebook",
            "Describe the notebook you want to create:",
            text="Create a notebook that "
        )
        
        if not ok or not prompt:
            return
            
        # Save the prompt for later iterations
        self.derive_mode_prompt = prompt
        self.derive_mode_in_progress = True
        self.iteration_count = 0
        
        # Create initial notebook structure or clear existing
        if len(self.notebook.cells) <= 1:
            # Start with a title if notebook is empty
            self.notebook = NotebookDocument()
            self.notebook.add_cell(NotebookCell("markdown", f"# {prompt}\n\nGenerating notebook..."))
        else:
            # Ask if user wants to start fresh or build on existing notebook
            result = QMessageBox.question(
                self, 
                "Derive Mode", 
                "Do you want to build on the existing notebook or create a new one?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if result == QMessageBox.StandardButton.No:
                self.notebook = NotebookDocument()
                self.notebook.add_cell(NotebookCell("markdown", f"# {prompt}\n\nGenerating notebook..."))
        
        # Add a cell where generation will begin
        self.notebook.add_cell(NotebookCell("code", "# Generating initial code..."))
        self.generating_cell_index = len(self.notebook.cells) - 1
        self.update_editor()
        
        # Prepare context
        notebook_context = self.notebook.to_plain_text() if self.use_context_checkbox.isChecked() else None
        
        # Start generation
        self.status_label.setText(f"Starting notebook derivation with {self.ollama_model}...")
        
        import threading
        thread = threading.Thread(
            target=self.ollama_worker.start_generation,
            args=(self.derive_mode_prompt, notebook_context, cell_type, True)  # True for derive mode
        )
        thread.daemon = True
        thread.start()
        
    def continue_derive_mode(self):
        """Continue existing derive mode iteration"""
        # Increment iteration counter
        self.iteration_count += 1
        
        # Add a placeholder cell for continuation
        self.notebook.add_cell(NotebookCell("code", f"# Continuing notebook generation (iteration {self.iteration_count})..."))
        self.generating_cell_index = len(self.notebook.cells) - 1
        self.update_editor()
        
        # Create continuation prompt
        continuation_prompt = f"Continue developing the notebook for: {self.derive_mode_prompt}\nThis is iteration {self.iteration_count}. Please continue where you left off."
        
        # Get context
        notebook_context = self.notebook.to_plain_text() if self.use_context_checkbox.isChecked() else None
        
        # Start generation
        self.status_label.setText(f"Continuing notebook derivation (iteration {self.iteration_count})...")
        
        import threading
        thread = threading.Thread(
            target=self.ollama_worker.start_generation,
            args=(continuation_prompt, notebook_context, "code", True)  # True for derive mode
        )
        thread.daemon = True
        thread.start()
            
    def handle_ollama_result(self, result):
        if self.generating_cell_index is None:
            return

        # Check for derive mode completion
        if self.generation_mode == "derive" and self.derive_mode_in_progress:
            if '<version_complete>' in result:
                # Extract content before completion tag
                content = result.split('<version_complete>')[0]
                self.parse_and_update_cells(content)
                self.derive_mode_in_progress = False
                self.status_label.setText("Notebook generation complete!")
            else:
                # Update current cell with intermediate content
                if not result.strip().endswith('...'):
                    self.parse_and_update_cells(result)
                else:
                    self.notebook.update_cell(self.generating_cell_index, result)
        else:
            # Single cell mode
            self.parse_and_update_cells(result)
        
        self.update_editor()

    def parse_and_update_cells(self, content):
        """Single unified method to parse and update cells"""
        new_cells = self.notebook.parse_tagged_content(content)
        if new_cells:
            # Delete the placeholder cell
            self.notebook.delete_cell(self.generating_cell_index)
            
            # Add all new cells after the current position
            for i, cell in enumerate(new_cells):
                insert_position = self.generating_cell_index + i
                self.notebook.add_cell(cell, insert_position)
            
            # Update the generating cell index to point to the last added cell
            self.generating_cell_index += len(new_cells) - 1

    def parse_multi_cell_response(self):
        """Parse a response that may contain multiple cells"""
        if self.generating_cell_index is None:
            return
            
        # Get content of the current cell
        current_content = "".join(self.notebook.cells[self.generating_cell_index].source)
        
        # Check if we have markdown section headers (##) that could indicate multiple sections
        # or multiple code cells that should be separated
        markdown_sections = re.split(r'\n#{2,3} ', current_content)
        
        if len(markdown_sections) > 1:
            # We found multiple markdown headers, split into multiple cells
            
            # Keep the first part in the current cell (might not start with ##)
            first_part = markdown_sections[0]
            self.notebook.update_cell(self.generating_cell_index, first_part)
            
            # Create new cells for the other sections
            for i, section in enumerate(markdown_sections[1:], 1):
                # Add the ## back to the beginning of the section
                section_content = f"## {section}"
                
                # Insert after the current cell
                insert_position = self.generating_cell_index + i
                self.notebook.add_cell(NotebookCell("markdown", section_content), insert_position)
        
        # Now look for code blocks that should be separate cells
        elif "```python" in current_content or "```" in current_content:
            # Try to split by code blocks
            parts = re.split(r'```(?:python)?\n', current_content)
            
            if len(parts) > 1:
                # First part is likely markdown
                self.notebook.update_cell(self.generating_cell_index, parts[0])
                
                # Process the rest of the parts
                for i, part in enumerate(parts[1:], 1):
                    # Remove the closing ``` if present
                    code_content = part.split("```")[0] if "```" in part else part
                    
                    # Insert after the current cell
                    insert_position = self.generating_cell_index + i
                    cell_type = "code"  # Assume all code blocks are Python
                    
                    self.notebook.add_cell(NotebookCell(cell_type, code_content), insert_position)
        
    def handle_ollama_error(self, error_msg):
        QMessageBox.critical(self, "Ollama Error", f"Error generating content: {error_msg}")
        self.status_label.setText("Ollama generation failed")
        
        # Remove the placeholder cell if there was an error
        if self.generating_cell_index is not None:
            self.notebook.delete_cell(self.generating_cell_index)
            self.generating_cell_index = None
            self.update_editor()
            
        # Reset derive mode if applicable
        if self.derive_mode_in_progress:
            self.derive_mode_in_progress = False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NotebookApp()
    window.show()
    sys.exit(app.exec())
