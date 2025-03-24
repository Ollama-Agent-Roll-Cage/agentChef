
"""
TO run either use 
python research_ui.py
or
python research_ui.py --topic "Your research topic" --model "model_name" --iterations 10 --time-limit 30
"""

import sys
import os
import re
import argparse
import time
import markdown
from pathlib import Path
from datetime import datetime
from research_thread import IterativeResearchThread
from research_utils import setup_logging

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
                            QComboBox, QSpinBox, QCheckBox, QFileDialog, QSplitter,
                            QMessageBox, QDialog, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
from PyQt6.QtGui import QIcon, QFont, QDesktopServices, QTextCursor, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Ollama
import ollama

# Default paths
DEFAULT_DATA_DIR = os.path.join(Path.home(), '.research_assistant')
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

# Setup logging
logger = setup_logging(DEFAULT_DATA_DIR)

class FloatingResearchWindow(QMainWindow):
    """Main window for the research assistant."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Research Assistant")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set initial size
        self.resize(400, 600)
        
        # For window dragging
        self.dragging = False
        self.offset = None
        
        # Initialize UI
        self.setup_ui()
        
        # Research thread
        self.research_thread = None
        
        # Available models from Ollama
        self.available_models = []
        self.fetch_available_models()
        
        # Settings
        self.max_iterations = 10
        self.time_limit_minutes = 30
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main container widget with background
        self.container = QWidget()
        self.container.setObjectName("container")
        self.container.setStyleSheet("""
            #container {
                background-color: rgba(30, 30, 40, 0.95);
                border-radius: 15px;
                border: 1px solid #4444ff;
            }
        """)
        
        self.setCentralWidget(self.container)
        
        # Main layout
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header with title and close button
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🧠 Research Assistant")
        title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
        """)
        
        close_button = QPushButton("×")
        close_button.setFixedSize(30, 30)
        close_button.setStyleSheet("""
            QPushButton {
                color: #ffffff;
                background-color: transparent;
                border: none;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff4444;
            }
        """)
        close_button.clicked.connect(self.close)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(close_button)
        
        layout.addLayout(header_layout)
        
        # Research prompt input
        prompt_label = QLabel("Research Topic:")
        prompt_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Describe the research topic or question...")
        self.prompt_input.setMinimumHeight(80)
        self.prompt_input.setMaximumHeight(120)
        self.prompt_input.setStyleSheet("""
            QTextEdit {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.prompt_input)
        
        # Model selection
        model_layout = QHBoxLayout()
        
        model_label = QLabel("Ollama Model:")
        model_label.setStyleSheet("color: #ffffff;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 4px 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
            }
        """)
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)
        
        # Research parameters
        params_layout = QHBoxLayout()
        
        iteration_label = QLabel("Max Iterations:")
        iteration_label.setStyleSheet("color: #ffffff;")
        params_layout.addWidget(iteration_label)
        
        self.iteration_spin = QSpinBox()
        self.iteration_spin.setRange(1, 50)
        self.iteration_spin.setValue(10)
        self.iteration_spin.setStyleSheet("""
            QSpinBox {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 4px 8px;
            }
        """)
        self.iteration_spin.valueChanged.connect(self.update_settings)
        params_layout.addWidget(self.iteration_spin)
        
        time_label = QLabel("Time Limit (mins):")
        time_label.setStyleSheet("color: #ffffff;")
        params_layout.addWidget(time_label)
        
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 120)
        self.time_spin.setValue(30)
        self.time_spin.setStyleSheet("""
            QSpinBox {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 4px 8px;
            }
        """)
        self.time_spin.valueChanged.connect(self.update_settings)
        params_layout.addWidget(self.time_spin)
        
        layout.addLayout(params_layout)
        
        # Start research button
        self.research_button = QPushButton("🔍 Start Research")
        self.research_button.setStyleSheet("""
            QPushButton {
                background-color: #4444ff;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5555ff;
            }
            QPushButton:pressed {
                background-color: #3333cc;
            }
        """)
        self.research_button.clicked.connect(self.start_research)
        layout.addWidget(self.research_button)
        
        # Progress bar
        self.progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4444ff;
                border-radius: 7px;
            }
        """)
        
        self.cancel_button = QPushButton("✕")
        self.cancel_button.setFixedSize(25, 25)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: #ffffff;
                border: none;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_research)
        self.cancel_button.setVisible(False)
        
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.cancel_button)
        
        layout.addLayout(self.progress_layout)
        self.progress_bar.setVisible(False)
        
        # Console output
        console_label = QLabel("Research Progress:")
        console_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(console_label)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMinimumHeight(200)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: rgba(20, 20, 30, 0.9);
                color: #33ff33;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas, Monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.console_output)
        
        # View Results button
        self.view_results_button = QPushButton("📋 View Results")
        self.view_results_button.setStyleSheet("""
            QPushButton {
                background-color: #44aa44;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #55bb55;
            }
            QPushButton:pressed {
                background-color: #338833;
            }
        """)
        self.view_results_button.clicked.connect(self.view_results)
        self.view_results_button.setVisible(False)
        layout.addWidget(self.view_results_button)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.offset = event.position().toPoint()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.offset:
            new_pos = event.globalPosition().toPoint() - self.offset
            self.move(new_pos)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            
    def fetch_available_models(self):
        """Fetch available models from Ollama using the Python API."""
        self.log_message("Fetching available Ollama models...")
        self.model_combo.clear()
        
        try:
            # Use the Python API instead of subprocess
            models_list = ollama.list()
            
            if hasattr(models_list, 'models') and models_list.models:
                model_names = [model.model for model in models_list.models]
                
                if model_names:
                    self.model_combo.addItems(model_names)
                    self.available_models = model_names
                    self.log_message(f"Found {len(model_names)} Ollama models")
                else:
                    self.model_combo.addItem("No models found")
                    self.log_message("No Ollama models found")
            else:
                self.model_combo.addItem("No models found")
                self.log_message("No models found in response")
                    
        except Exception as e:
            self.log_message(f"Error fetching models: {str(e)}")
            self.model_combo.addItem("Error fetching models")
    
    def update_settings(self):
        """Update research settings from UI."""
        self.max_iterations = self.iteration_spin.value()
        self.time_limit_minutes = self.time_spin.value()
    
    def log_message(self, message):
        """Add a message to the console output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Set cursor to end of document
        cursor = self.console_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_output.setTextCursor(cursor)
        
        # Add the new message
        self.console_output.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_research(self):
        """Start the research process."""
        # Get research prompt
        research_prompt = self.prompt_input.toPlainText().strip()
        if not research_prompt:
            QMessageBox.warning(self, "Input Required", "Please enter a research topic.")
            return
        
        # Get selected model
        selected_model = self.model_combo.currentText()
        if selected_model in ["No models found", "Error fetching models"]:
            QMessageBox.warning(self, "Model Required", "Please select a valid Ollama model.")
            return
        
        # Update UI
        self.research_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        self.view_results_button.setVisible(False)
        self.console_output.clear()
        
        self.log_message(f"Starting research on: {research_prompt}")
        self.log_message(f"Using model: {selected_model}")
        self.log_message(f"Max iterations: {self.max_iterations}")
        self.log_message(f"Time limit: {self.time_limit_minutes} minutes")
        
        # Create and start research thread
        self.research_thread = IterativeResearchThread(
            research_topic=research_prompt,
            model_name=selected_model,
            max_iterations=self.max_iterations,
            time_limit_minutes=self.time_limit_minutes
        )
        
        # Connect signals
        self.research_thread.update_signal.connect(self.log_message)
        self.research_thread.progress_signal.connect(self.update_progress)
        self.research_thread.result_signal.connect(self.handle_research_result)
        self.research_thread.error_signal.connect(self.handle_research_error)
        self.research_thread.search_result_signal.connect(self.handle_search_results)
        self.research_thread.paper_signal.connect(self.handle_paper_info)
        
        # Start the thread
        self.research_thread.start()
    
    def cancel_research(self):
        """Cancel the current research process."""
        if self.research_thread and self.research_thread.isRunning():
            self.log_message("Cancelling research...")
            self.research_thread.stop()
            
            # Wait for thread to finish
            if not self.research_thread.wait(3000):  # 3 second timeout
                self.research_thread.terminate()
                self.log_message("Research thread terminated.")
            
            self.reset_ui_after_research()
    
    def update_progress(self, value, message):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")
    
    def handle_research_result(self, result):
        """Handle the final research result."""
        self.log_message("Research complete!")
        self.final_report = result
        self.reset_ui_after_research()
        self.view_results_button.setVisible(True)
    
    def handle_research_error(self, error_message):
        """Handle research errors."""
        self.log_message(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Research Error", error_message)
        self.reset_ui_after_research()
    
    def handle_search_results(self, results):
        """Handle search results update."""
        self.log_message(f"Found {len(results)} search results")
    
    def handle_paper_info(self, paper):
        """Handle paper information update."""
        self.log_message(f"Processed paper: {paper['title']}")
    
    def reset_ui_after_research(self):
        """Reset the UI after research completes or is cancelled."""
        self.research_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
    
    def view_results(self):
        """Show the research results in a separate window."""
        if hasattr(self, 'final_report'):
            result_dialog = ResearchResultDialog(self.final_report, parent=self)
            result_dialog.exec()
        else:
            QMessageBox.warning(self, "No Results", "No research results available.")

class ResearchResultDialog(QDialog):
    """Dialog for displaying research results."""
    def __init__(self, markdown_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Research Report")
        self.resize(800, 600)
        
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)
        
        # Add CSS for styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    padding: 0;
                    color: #333;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #2c3e50;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                h1 {{ font-size: 28px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ font-size: 24px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                p {{ margin: 15px 0; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                code {{ 
                    background: #f8f8f8; 
                    padding: 3px 5px; 
                    border-radius: 3px; 
                    font-family: Consolas, monospace;
                }}
                pre {{ 
                    background: #f8f8f8; 
                    padding: 15px; 
                    border-radius: 5px; 
                    overflow-x: auto;
                    border: 1px solid #ddd;
                }}
                blockquote {{
                    margin: 15px 0;
                    padding: 10px 15px;
                    border-left: 4px solid #3498db;
                    background: #f8f8f8;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                ul, ol {{
                    margin: 15px 0;
                    padding-left: 30px;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create splitter for preview and raw markdown
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create tabs for different views
        tab_widget = QTabWidget()
        
        # HTML Preview tab
        html_view = QWebEngineView()
        html_view.setHtml(styled_html)
        tab_widget.addTab(html_view, "Rendered Preview")
        
        # Raw markdown tab
        markdown_edit = QTextEdit()
        markdown_edit.setPlainText(markdown_content)
        markdown_edit.setReadOnly(False)  # Allow editing
        tab_widget.addTab(markdown_edit, "Markdown Source")
        
        splitter.addWidget(tab_widget)
        layout.addWidget(splitter)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Report")
        save_button.clicked.connect(lambda: self.save_report(markdown_content))
        button_layout.addWidget(save_button)
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: self.copy_to_clipboard(markdown_content))
        button_layout.addWidget(copy_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def save_report(self, content):
        """Save the report to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Research Report", "", "Markdown Files (*.md);;HTML Files (*.html);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.html'):
                # Convert to HTML and save
                html_content = markdown.markdown(content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            else:
                # If not .md extension, add it
                if not file_path.endswith('.md'):
                    file_path += '.md'
                
                # Save as markdown
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            QMessageBox.information(self, "Success", f"Report saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def copy_to_clipboard(self, content):
        """Copy the report to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(content)
        QMessageBox.information(self, "Success", "Report copied to clipboard")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Research Assistant")
    parser.add_argument("--topic", type=str, help="Research topic to analyze")
    parser.add_argument("--model", type=str, default="llama3:latest", help="Ollama model to use")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--time-limit", type=int, default=30, help="Time limit in minutes")
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Research Assistant")
    
    # Set global stylesheet
    app.setStyleSheet("""
        QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit {
            font-family: Arial;
        }
    """)
    
    # Create main window
    window = FloatingResearchWindow()
    window.show()
    
    # If topic provided via command line, start research automatically
    if args.topic:
        window.prompt_input.setPlainText(args.topic)
        
        # Set model if valid
        if args.model:
            index = window.model_combo.findText(args.model)
            if index >= 0:
                window.model_combo.setCurrentIndex(index)
        
        # Set iterations and time limit
        window.iteration_spin.setValue(args.iterations)
        window.time_spin.setValue(args.time_limit)
        
        # Wait a bit for UI to initialize, then start research
        QTimer.singleShot(1000, window.start_research)
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
