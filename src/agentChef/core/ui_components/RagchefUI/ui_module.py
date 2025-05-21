"""
ragchef UI Module - Simple graphical interface for the ragchef system.
Provides a user-friendly interface for research, dataset generation, and analysis.

Written By: @AgentChef
Date: 4/4/2025
"""

import os
import sys
import json
import asyncio
import threading
import random  # Add this import
from pathlib import Path
import ollama

from datetime import datetime, timezone
UTC = timezone.utc

import logging
logger = logging.getLogger(__name__)  # Add this line
import webbrowser
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QFileDialog,
        QProgressBar, QSpinBox, QCheckBox, QGroupBox, QFormLayout, QSplitter,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDialog
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer, QPointF  # Add QPointF here
    from PyQt6.QtGui import QFont, QIcon, QTextCursor, QPainter, QColor, QBrush
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt6 not available. Install with 'pip install PyQt6'")

from agentChef.logs.agentchef_logging import log, setup_file_logging

class WorkerThread(QThread):
    """Worker thread for running asynchronous operations."""
    
    update_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        """
        Initialize the worker thread.
        
        Args:
            func: Asynchronous function to run
            *args, **kwargs: Arguments for the function
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        """Run the worker thread."""
        try:
            # Create a callback for progress updates
            def update_callback(message):
                self.update_signal.emit(message)
                
            # Add the callback to kwargs
            self.kwargs['callback'] = update_callback
            
            # Create and run event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.func(*self.args, **self.kwargs))
            self.result_signal.emit(result)
            
            loop.close()
            
        except Exception as e:
            logger.exception("Error in worker thread")
            self.error_signal.emit(f"Error: {str(e)}")

class ParticleSystem(QWidget):
    """Simple particle system for background effects."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make widget transparent and stay on top
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.raise_()  # Keep on top
        self.particles = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # 60 FPS
        
        # Create initial particles with higher opacity
        for _ in range(50):
            self.particles.append({
                'x': random.randint(0, self.width()),
                'y': random.randint(0, self.height()),
                'speed': random.uniform(0.5, 2.0),
                'size': random.uniform(1.5, 4),  # Slightly larger particles
                'opacity': random.uniform(0.3, 0.7)  # Increased opacity range
            })
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        for p in self.particles:
            painter.setOpacity(p['opacity'])
            color = QColor("#FF1744")  # Neon red
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(p['x'], p['y']), p['size'], p['size'])
            
            # Update particle position
            p['y'] += p['speed']
            if p['y'] > self.height():
                p['y'] = -10
                p['x'] = random.randint(0, self.width())

class ConversationViewer(QWidget):
    """Widget for displaying generated conversations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Conversation display with transparency
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(31, 41, 55, 0.85);  /* Dark background with 85% opacity */
                color: #FFFFFF;
                border: 1px solid rgba(55, 65, 81, 0.8);  /* Semi-transparent border */
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        self.layout.addWidget(self.display)

    def add_message(self, role: str, content: str):
        """Add a new message to the conversation."""
        color = "#FF1744" if role == "human" else "#00E5FF"
        self.display.append(f'<p style="margin: 4px 0;"><span style="color: {color}">{role}:</span> {content}</p>')

class RagchefUI(QMainWindow):
    """Main window for the ragchef UI."""
    
    def __init__(self, research_manager):
        """Initialize the UI."""
        super().__init__()
        self.research_manager = research_manager
        self.worker_thread = None
        
        # Set up logging for UI
        setup_file_logging("./logs/ui")
        logger.info("Initializing RagchefUI...")
        
        # Set window attributes and style
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet("""
            QMainWindow {
                background-color: palette(window);
            }
            QWidget {
                background-color: palette(base);
            }
            QToolTip {
                background-color: palette(toolTipBase);
                color: palette(toolTipText);
                border: 1px solid palette(dark);
            }
        """)
        
        self.setWindowTitle("ragchef - Unified Dataset Research, Augmentation, & Generation System")
        self.setMinimumSize(1000, 700)
        
        # Set up the UI
        self._setup_ui()
        
        # Add particle system background
        self.particles = ParticleSystem(self)
        self.particles.setGeometry(self.rect())
        
        # Create particle toggle button in the corner
        self.particle_toggle = QPushButton("🌟", self)
        self.particle_toggle.setToolTip("Toggle Particle Effects")
        self.particle_toggle.setFixedSize(32, 32)
        self.particle_toggle.setStyleSheet("""
            QPushButton {
                background-color: rgba(31, 41, 55, 0.7);
                border: 1px solid rgba(255, 23, 68, 0.8);
                border-radius: 16px;
                color: #FF1744;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: rgba(255, 23, 68, 0.3);
            }
        """)
        self.particle_toggle.clicked.connect(self._toggle_particles)
        
        # Set dark theme
        self.setup_dark_theme()
    
    def setup_dark_theme(self):
        """Set up dark theme and modern styling with transparency."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgba(17, 24, 39, 0.95);  /* Very slight transparency */
            }
            QWidget {
                color: #FFFFFF;
            }
            QPushButton {
                background-color: rgba(55, 65, 81, 0.85);  /* Semi-transparent background */
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: rgba(75, 85, 99, 0.9);  /* More opaque on hover */
            }
            QPushButton:pressed {
                background-color: rgba(31, 41, 55, 0.95);  /* Most opaque when pressed */
            }
            QGroupBox {
                background-color: rgba(31, 41, 55, 0.8);  /* Transparent group boxes */
                border: 1px solid rgba(55, 65, 81, 0.8);
                border-radius: 8px;
                padding-top: 16px;
                margin-top: 8px;
            }
            QGroupBox::title {
                color: #FF1744;
                subcontrol-position: top center;
                padding: 0 4px;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: rgba(55, 65, 81, 0.85);  /* Semi-transparent inputs */
                border: 1px solid rgba(75, 85, 99, 0.8);
                border-radius: 4px;
                padding: 4px;
                color: #FFFFFF;
            }
            QProgressBar {
                border: 1px solid rgba(75, 85, 99, 0.8);
                border-radius: 4px;
                text-align: center;
                background-color: rgba(31, 41, 55, 0.7);  /* Transparent background */
            }
            QProgressBar::chunk {
                background-color: rgba(255, 23, 68, 0.9);  /* Semi-transparent neon red */
            }
            QTabWidget::pane {
                background-color: rgba(31, 41, 55, 0.8);  /* Transparent tab panes */
                border: 1px solid rgba(55, 65, 81, 0.8);
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: rgba(55, 65, 81, 0.85);  /* Semi-transparent tabs */
                color: #FFFFFF;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: rgba(255, 23, 68, 0.8);  /* Semi-transparent neon red for selected tab */
            }
            QScrollBar {
                background-color: rgba(31, 41, 55, 0.6);  /* Transparent scrollbars */
                width: 12px;
                height: 12px;
            }
            QScrollBar::handle {
                background-color: rgba(75, 85, 99, 0.8);  /* Semi-transparent scroll handle */
                border-radius: 6px;
                min-height: 24px;
            }
            QScrollBar::handle:hover {
                background-color: rgba(255, 23, 68, 0.8);  /* Neon red on hover */
            }
        """)
    
    def _setup_ui(self):
        """Set up the main UI components."""
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.research_tab = QWidget()
        self.generate_tab = QWidget()
        self.process_tab = QWidget()
        self.analyze_tab = QWidget()
        
        # Add tabs to widget
        self.tabs.addTab(self.research_tab, "Research")
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.addTab(self.process_tab, "Process")
        self.tabs.addTab(self.analyze_tab, "Analyze")
        
        # Set up each tab
        self._setup_research_tab()
        self._setup_generate_tab()
        self._setup_process_tab()
        self._setup_analyze_tab()
    
    def _setup_research_tab(self):
        """Set up the Research tab."""
        layout = QVBoxLayout()
        
        # Research form
        form_group = QGroupBox("Research Settings")
        form_layout = QFormLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItem("Loading models...")
        form_layout.addRow("Ollama Model:", self.model_combo)
        
        # Populate models asynchronously
        self._populate_model_list()
        
        # Topic input
        self.topic_input = QLineEdit()
        form_layout.addRow("Research Topic:", self.topic_input)
        
        # Max papers
        self.max_papers_spin = QSpinBox()
        self.max_papers_spin.setRange(1, 20)
        self.max_papers_spin.setValue(5)
        form_layout.addRow("Max Papers:", self.max_papers_spin)
        
        # Max search results
        self.max_search_spin = QSpinBox()
        self.max_search_spin.setRange(1, 50)
        self.max_search_spin.setValue(10)
        form_layout.addRow("Max Search Results:", self.max_search_spin)
        
        # GitHub options
        self.include_github_check = QCheckBox("Include GitHub Repositories")
        form_layout.addRow("", self.include_github_check)
        
        self.github_repos_input = QLineEdit()
        self.github_repos_input.setPlaceholderText("Repository URLs (comma-separated)")
        self.github_repos_input.setEnabled(False)
        form_layout.addRow("GitHub Repos:", self.github_repos_input)
        
        # Connect checkbox to enable/disable repo input
        self.include_github_check.stateChanged.connect(
            lambda state: self.github_repos_input.setEnabled(state == Qt.CheckState.Checked)
        )
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Research button
        self.research_button = QPushButton("Start Research")
        self.research_button.clicked.connect(self._on_research_clicked)
        layout.addWidget(self.research_button)
        
        # Progress area
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.research_progress = QProgressBar()
        self.research_progress.setRange(0, 0)  # Indeterminate
        self.research_progress.setVisible(False)
        progress_layout.addWidget(self.research_progress)
        
        self.research_log = QTextEdit()
        self.research_log.setReadOnly(True)
        progress_layout.addWidget(self.research_log)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group, stretch=1)
        
        # Results area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.research_results = QTextEdit()
        self.research_results.setReadOnly(True)
        results_layout.addWidget(self.research_results)
        
        save_layout = QHBoxLayout()
        self.save_research_button = QPushButton("Save Results")
        self.save_research_button.clicked.connect(self._on_save_research_clicked)
        self.save_research_button.setEnabled(False)
        save_layout.addWidget(self.save_research_button)
        
        self.proceed_to_generate_button = QPushButton("Proceed to Generate")
        self.proceed_to_generate_button.clicked.connect(self._on_proceed_to_generate_clicked)
        self.proceed_to_generate_button.setEnabled(False)
        save_layout.addWidget(self.proceed_to_generate_button)
        
        results_layout.addLayout(save_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        # Add conversation viewer
        self.conversation_viewer = ConversationViewer()
        layout.addWidget(self.conversation_viewer, stretch=1)
        
        self.research_tab.setLayout(layout)
    
    def _populate_model_list(self):
        """Populate the model selection combo box with available Ollama models."""
        try:
            # Get list of models from Ollama
            models = ollama.list()
            
            # Clear and populate combo box
            self.model_combo.clear()
            
            # Handle different response formats
            if hasattr(models, 'models'):
                # New ListResponse format
                model_names = [model.model for model in models.models if hasattr(model, 'model')]
            elif isinstance(models, list):
                # List of Model objects format
                model_names = [model.model for model in models if hasattr(model, 'model')]
            elif isinstance(models, dict) and 'models' in models:
                # Old format
                model_names = [model['name'] for model in models['models'] if 'name' in model]
            else:
                model_names = ['llama2']  # Default fallback
                
            self.model_combo.addItems(model_names)
            
            # Set default model
            default_model = "llama2"
            index = self.model_combo.findText(default_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                
        except Exception as e:
            logger.error(f"Error populating model list: {str(e)}")
            self.model_combo.clear()
            self.model_combo.addItem("llama2")  # Fallback to default

    def _setup_generate_tab(self):
        """Set up the Generate tab."""
        layout = QVBoxLayout()
        
        # Generation settings
        settings_group = QGroupBox("Generation Settings")
        settings_layout = QFormLayout()
        
        # Number of turns
        self.turns_spin = QSpinBox()
        self.turns_spin.setRange(1, 10)
        self.turns_spin.setValue(3)
        settings_layout.addRow("Conversation Turns:", self.turns_spin)
        
        # Expansion factor
        self.expansion_spin = QSpinBox()
        self.expansion_spin.setRange(1, 10)
        self.expansion_spin.setValue(3)
        settings_layout.addRow("Expansion Factor:", self.expansion_spin)
        
        # Hedging level
        self.hedging_combo = QComboBox()
        self.hedging_combo.addItems(["confident", "balanced", "cautious"])
        self.hedging_combo.setCurrentText("balanced")
        settings_layout.addRow("Hedging Level:", self.hedging_combo)
        
        # Clean checkbox
        self.clean_check = QCheckBox("Clean Expanded Dataset")
        self.clean_check.setChecked(True)
        settings_layout.addRow("", self.clean_check)
        
        # Static fields settings
        static_group = QGroupBox("Static Fields")
        static_layout = QVBoxLayout()
        
        self.human_static_check = QCheckBox("Keep Human Messages Static")
        self.human_static_check.setChecked(True)
        static_layout.addWidget(self.human_static_check)
        
        self.gpt_static_check = QCheckBox("Keep GPT Messages Static")
        self.gpt_static_check.setChecked(False)
        static_layout.addWidget(self.gpt_static_check)
        
        static_group.setLayout(static_layout)
        settings_layout.addRow("", static_group)
        
        # Output format
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jsonl", "parquet", "csv", "all"])
        settings_layout.addRow("Output Format:", self.format_combo)
        
        # Output directory
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(str(self.research_manager.datasets_dir))
        self.output_dir_layout.addWidget(self.output_dir_input)
        
        self.browse_output_button = QPushButton("Browse...")
        self.browse_output_button.clicked.connect(self._on_browse_output_clicked)
        self.output_dir_layout.addWidget(self.browse_output_button)
        
        settings_layout.addRow("Output Directory:", self.output_dir_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Generate button
        self.generate_button = QPushButton("Generate Dataset")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self.generate_button)
        
        # Progress area
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.generate_progress = QProgressBar()
        self.generate_progress.setRange(0, 0)  # Indeterminate
        self.generate_progress.setVisible(False)
        progress_layout.addWidget(self.generate_progress)
        
        self.generate_log = QTextEdit()
        self.generate_log.setReadOnly(True)
        progress_layout.addWidget(self.generate_log)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group, stretch=1)
        
        # Results area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.generate_results = QTextEdit()
        self.generate_results.setReadOnly(True)
        results_layout.addWidget(self.generate_results)
        
        open_layout = QHBoxLayout()
        self.open_output_button = QPushButton("Open Output Directory")
        self.open_output_button.clicked.connect(self._on_open_output_clicked)
        open_layout.addWidget(self.open_output_button)
        
        self.proceed_to_analyze_button = QPushButton("Analyze Dataset")
        self.proceed_to_analyze_button.clicked.connect(self._on_proceed_to_analyze_clicked)
        self.proceed_to_analyze_button.setEnabled(False)
        open_layout.addWidget(self.proceed_to_analyze_button)
        
        results_layout.addLayout(open_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        self.generate_tab.setLayout(layout)
    
    def _setup_process_tab(self):
        """Set up the Process tab for existing papers."""
        layout = QVBoxLayout()
        
        # Input selection
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout()
        
        self.input_dir_layout = QHBoxLayout()
        self.input_dir_label = QLabel("Input Directory:")
        self.input_dir_layout.addWidget(self.input_dir_label)
        
        self.input_dir_edit = QLineEdit()
        self.input_dir_layout.addWidget(self.input_dir_edit)
        
        self.browse_input_button = QPushButton("Browse...")
        self.browse_input_button.clicked.connect(self._on_browse_input_clicked)
        self.input_dir_layout.addWidget(self.browse_input_button)
        
        input_layout.addLayout(self.input_dir_layout)
        
        # Files list
        self.files_label = QLabel("Selected Files:")
        input_layout.addWidget(self.files_label)
        
        self.files_list = QTextEdit()
        self.files_list.setReadOnly(True)
        self.files_list.setMaximumHeight(100)
        input_layout.addWidget(self.files_list)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Processing settings
        settings_group = QGroupBox("Processing Settings")
        settings_layout = QFormLayout()
        
        # Number of turns
        self.proc_turns_spin = QSpinBox()
        self.proc_turns_spin.setRange(1, 10)
        self.proc_turns_spin.setValue(3)
        settings_layout.addRow("Conversation Turns:", self.proc_turns_spin)
        
        # Expansion factor
        self.proc_expansion_spin = QSpinBox()
        self.proc_expansion_spin.setRange(1, 10)
        self.proc_expansion_spin.setValue(3)
        settings_layout.addRow("Expansion Factor:", self.proc_expansion_spin)
        
        # Clean checkbox
        self.proc_clean_check = QCheckBox("Clean Expanded Dataset")
        self.proc_clean_check.setChecked(True)
        settings_layout.addRow("", self.proc_clean_check)
        
        # Output format
        self.proc_format_combo = QComboBox()
        self.proc_format_combo.addItems(["jsonl", "parquet", "csv", "all"])
        settings_layout.addRow("Output Format:", self.proc_format_combo)
        
        # Output directory
        self.proc_output_dir_layout = QHBoxLayout()
        self.proc_output_dir_input = QLineEdit()
        self.proc_output_dir_input.setText(str(self.research_manager.datasets_dir))
        self.proc_output_dir_layout.addWidget(self.proc_output_dir_input)
        
        self.proc_browse_output_button = QPushButton("Browse...")
        self.proc_browse_output_button.clicked.connect(self._on_proc_browse_output_clicked)
        self.proc_output_dir_layout.addWidget(self.proc_browse_output_button)
        
        settings_layout.addRow("Output Directory:", self.proc_output_dir_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Process button
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self._on_process_clicked)
        layout.addWidget(self.process_button)
        
        # Progress area
        proc_progress_group = QGroupBox("Progress")
        proc_progress_layout = QVBoxLayout()
        
        self.process_progress = QProgressBar()
        self.process_progress.setRange(0, 0)  # Indeterminate
        self.process_progress.setVisible(False)
        proc_progress_layout.addWidget(self.process_progress)
        
        self.process_log = QTextEdit()
        self.process_log.setReadOnly(True)
        proc_progress_layout.addWidget(self.process_log)
        
        proc_progress_group.setLayout(proc_progress_layout)
        layout.addWidget(proc_progress_group, stretch=1)
        
        # Results area
        proc_results_group = QGroupBox("Results")
        proc_results_layout = QVBoxLayout()
        
        self.process_results = QTextEdit()
        self.process_results.setReadOnly(True)
        proc_results_layout.addWidget(self.process_results)
        
        proc_results_group.setLayout(proc_results_layout)
        layout.addWidget(proc_results_group, stretch=1)
        
        self.process_tab.setLayout(layout)

    def _setup_analyze_tab(self):
        """Set up the Analyze tab for dataset analysis."""
        layout = QVBoxLayout()
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QFormLayout()
        
        # Original dataset
        self.orig_dataset_layout = QHBoxLayout()
        self.orig_dataset_input = QLineEdit()
        self.orig_dataset_layout.addWidget(self.orig_dataset_input)
        
        self.browse_orig_button = QPushButton("Browse...")
        self.browse_orig_button.clicked.connect(self._on_browse_orig_clicked)
        self.orig_dataset_layout.addWidget(self.browse_orig_button)
        
        dataset_layout.addRow("Original Dataset:", self.orig_dataset_layout)
        
        # Expanded dataset
        self.exp_dataset_layout = QHBoxLayout()
        self.exp_dataset_input = QLineEdit()
        self.exp_dataset_layout.addWidget(self.exp_dataset_input)
        
        self.browse_exp_button = QPushButton("Browse...")
        self.browse_exp_button.clicked.connect(self._on_browse_exp_clicked)
        self.exp_dataset_layout.addWidget(self.browse_exp_button)
        
        dataset_layout.addRow("Expanded Dataset:", self.exp_dataset_layout)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout()
        
        self.basic_stats_check = QCheckBox("Basic Statistics")
        self.basic_stats_check.setChecked(True)
        analysis_layout.addWidget(self.basic_stats_check)
        
        self.quality_check = QCheckBox("Quality Analysis")
        self.quality_check.setChecked(True)
        analysis_layout.addWidget(self.quality_check)
        
        self.comparison_check = QCheckBox("Dataset Comparison")
        self.comparison_check.setChecked(True)
        analysis_layout.addWidget(self.comparison_check)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Analyze button
        self.analyze_button = QPushButton("Analyze Datasets")
        self.analyze_button.clicked.connect(self._on_analyze_clicked)
        layout.addWidget(self.analyze_button)
        
        # Results area
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        results_layout.addWidget(self.analysis_results)
        
        save_analysis_button = QPushButton("Save Analysis Results")
        save_analysis_button.clicked.connect(self._on_save_analysis_clicked)
        results_layout.addWidget(save_analysis_button)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        self.analyze_tab.setLayout(layout)

    # Event handlers
    def _on_research_clicked(self):
        """Handle research button click."""
        logger.info("Starting research operation...")
        topic = self.topic_input.text().strip()
        if not topic:
            QMessageBox.warning(self, "Input Required", "Please enter a research topic.")
            return
            
        self.research_button.setEnabled(False)
        self.research_progress.setVisible(True)
        self.research_log.clear()
        
        # Get GitHub repos if enabled
        github_repos = None
        if self.include_github_check.isChecked():
            repos = self.github_repos_input.text().strip()
            if repos:
                github_repos = [r.strip() for r in repos.split(',')]
        
        # Get selected model
        model = self.model_combo.currentText()
        
        # Start research in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.research_topic,
            topic=topic,
            max_papers=self.max_papers_spin.value(),
            max_search_results=self.max_search_spin.value(),
            include_github=self.include_github_check.isChecked(),
            github_repos=github_repos,
            model_name=model  # Pass model name to research_topic
        )
        
        self.worker_thread.update_signal.connect(self._on_research_update)
        self.worker_thread.result_signal.connect(self._on_research_complete)
        self.worker_thread.error_signal.connect(self._on_research_error)
        
        self.worker_thread.start()
    
    def _on_research_update(self, message):
        """Handle research progress updates."""
        self.research_log.append(message)
        self.research_log.moveCursor(QTextCursor.MoveOperation.End)
        
        # Check if message contains conversation data
        if "human:" in message.lower() or "gpt:" in message.lower():
            # Extract role and content
            parts = message.split(":", 1)
            if len(parts) == 2:
                role = parts[0].strip().lower()
                content = parts[1].strip()
                self.conversation_viewer.add_message(role, content)
    
    def _on_research_complete(self, result):
        """Handle research completion."""
        self.research_button.setEnabled(True)
        self.research_progress.setVisible(False)
        self.save_research_button.setEnabled(True)
        self.proceed_to_generate_button.setEnabled(True)
        
        # Display summary
        summary = result.get('summary', 'No summary available')
        self.research_results.setPlainText(summary)
        
        # Store results for later use
        self.current_research_results = result
    
    def _on_research_error(self, error_msg):
        """Handle research errors."""
        logger.error(f"Research error: {error_msg}")
        self.research_button.setEnabled(True)
        self.research_progress.setVisible(False)
        
        # Create message box with error details but continue UI operation
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Research Error")
        msg.setText("The research operation encountered some issues.")
        msg.setDetailedText(error_msg)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        
        # Still show partial results if available
        if hasattr(self, 'current_research_results'):
            self.research_results.setPlainText(self.current_research_results.get('summary', 'Partial results available'))
            self.save_research_button.setEnabled(True)

    def _on_save_research_clicked(self):
        """Handle save research button click."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Research Results",
            "",
            "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    if hasattr(self, 'current_research_results'):
                        json.dump(self.current_research_results, f, indent=2)
                    else:
                        f.write(self.research_results.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving file: {str(e)}")

    def _on_proceed_to_generate_clicked(self):
        """Handle proceed to generate button click."""
        self.tabs.setCurrentIndex(1)  # Switch to Generate tab

    def _on_browse_output_clicked(self):
        """Handle browse output directory button click."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self.research_manager.datasets_dir)
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def _on_browse_input_clicked(self):
        """Handle browse input directory button click."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            str(self.research_manager.papers_dir)
        )
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            # Update files list
            try:
                files = list(Path(dir_path).glob('*.txt')) + list(Path(dir_path).glob('*.md'))
                self.files_list.setPlainText('\n'.join(str(f) for f in files))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error reading directory: {str(e)}")

    def _on_proc_browse_output_clicked(self):
        """Handle browse output directory button click in process tab."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self.research_manager.datasets_dir)
        )
        if dir_path:
            self.proc_output_dir_input.setText(dir_path)

    def _on_browse_orig_clicked(self):
        """Handle browse original dataset button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Original Dataset",
            str(self.research_manager.datasets_dir),
            "Dataset Files (*.jsonl *.parquet *.csv);;All Files (*.*)"
        )
        if file_path:
            self.orig_dataset_input.setText(file_path)

    def _on_browse_exp_clicked(self):
        """Handle browse expanded dataset button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Expanded Dataset",
            str(self.research_manager.datasets_dir),
            "Dataset Files (*.jsonl *.parquet *.csv);;All Files (*.*)"
        )
        if file_path:
            self.exp_dataset_input.setText(file_path)

    def _on_generate_clicked(self):
        """Handle generate dataset button click."""
        if not hasattr(self, 'current_research_results'):
            QMessageBox.warning(self, "Input Required", "Please complete research first or load existing papers.")
            return

        self.generate_button.setEnabled(False)
        self.generate_progress.setVisible(True)
        self.generate_log.clear()

        # Start generation in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.generate_conversation_dataset,
            papers=self.current_research_results.get('processed_papers', []),
            num_turns=self.turns_spin.value(),
            expansion_factor=self.expansion_spin.value(),
            clean=self.clean_check.isChecked()
        )

        self.worker_thread.update_signal.connect(self._on_generate_update)
        self.worker_thread.result_signal.connect(self._on_generate_complete)
        self.worker_thread.error_signal.connect(self._on_generate_error)

        self.worker_thread.start()

    def _on_generate_update(self, message):
        """Handle generation progress updates."""
        self.generate_log.append(message)
        self.generate_log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_generate_complete(self, result):
        """Handle generation completion."""
        self.generate_button.setEnabled(True)
        self.generate_progress.setVisible(False)
        self.proceed_to_analyze_button.setEnabled(True)

        # Show summary in results
        summary = f"Generated {len(result.get('conversations', []))} conversations\n"
        summary += f"Expanded to {len(result.get('expanded_conversations', []))} variations\n"
        summary += f"Output saved to: {result.get('output_path', 'unknown')}"
        self.generate_results.setPlainText(summary)

    def _on_generate_error(self, error_msg):
        """Handle generation errors."""
        logger.error(f"Generation error: {error_msg}")
        self.generate_button.setEnabled(True)
        self.generate_progress.setVisible(False)
        QMessageBox.critical(self, "Generation Error", error_msg)

    def _on_open_output_clicked(self):
        """Handle open output directory button click."""
        output_dir = self.output_dir_input.text()
        if not output_dir:
            output_dir = str(self.research_manager.datasets_dir)
        
        try:
            if sys.platform == 'win32':
                os.startfile(output_dir)
            else:
                webbrowser.open(f'file://{output_dir}')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open directory: {str(e)}")

    def _on_proceed_to_analyze_clicked(self):
        """Handle proceed to analyze button click."""
        self.tabs.setCurrentIndex(3)  # Switch to Analyze tab

    def _on_process_clicked(self):
        """Handle process files button click."""
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "Input Required", "Please select an input directory.")
            return

        self.process_button.setEnabled(False)
        self.process_progress.setVisible(True)
        self.process_log.clear()

        # Get list of paper files
        paper_files = list(Path(input_dir).glob('*.txt')) + list(Path(input_dir).glob('*.md'))
        if not paper_files:
            QMessageBox.warning(self, "No Files Found", "No text or markdown files found in the input directory.")
            self.process_button.setEnabled(True)
            self.process_progress.setVisible(False)
            return

        # Start processing in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.process_paper_files,
            paper_files=paper_files,
            output_format=self.proc_format_combo.currentText(),
            num_turns=self.proc_turns_spin.value(),
            expansion_factor=self.proc_expansion_spin.value(),
            clean=self.proc_clean_check.isChecked()
        )

        self.worker_thread.update_signal.connect(self._on_process_update)
        self.worker_thread.result_signal.connect(self._on_process_complete)
        self.worker_thread.error_signal.connect(self._on_process_error)

        self.worker_thread.start()

    def _on_process_update(self, message):
        """Handle process progress updates."""
        self.process_log.append(message)
        self.process_log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_process_complete(self, result):
        """Handle process completion."""
        self.process_button.setEnabled(True)
        self.process_progress.setVisible(False)

        # Show summary in results
        summary = f"Processed {result.get('conversations_count', 0)} conversations\n\nOutput files:"
        for fmt, path in result.get('output_paths', {}).items():
            if isinstance(path, str):
                summary += f"\n{fmt}: {path}"
        self.process_results.setPlainText(summary)

    def _on_process_error(self, error_msg):
        """Handle process errors."""
        self.process_button.setEnabled(True)
        self.process_progress.setVisible(False)
        QMessageBox.critical(self, "Process Error", error_msg)

    def _on_analyze_clicked(self):
        """Handle analyze button click."""
        orig_path = self.orig_dataset_input.text().strip()
        exp_path = self.exp_dataset_input.text().strip()
        
        if not orig_path or not exp_path:
            QMessageBox.warning(self, "Input Required", "Please select both original and expanded datasets.")
            return
        
        self.analyze_button.setEnabled(False)
        self.analysis_results.clear()
        
        # Start analysis in worker thread
        self.worker_thread = WorkerThread(
            self.research_manager.analyze_expanded_dataset,
            orig_path=orig_path,
            exp_path=exp_path,
            analysis_options={
                "basic_stats": self.basic_stats_check.isChecked(),
                "quality": self.quality_check.isChecked(),
                "comparison": self.comparison_check.isChecked()
            }
        )
        
        self.worker_thread.update_signal.connect(self._append_to_analysis_results)
        self.worker_thread.result_signal.connect(self._on_analysis_complete)
        self.worker_thread.error_signal.connect(self._on_analysis_error)
        
        self.worker_thread.start()
    
    def _append_to_analysis_results(self, text):
        """Append text to analysis results."""
        self.analysis_results.append(text)
        self.analysis_results.moveCursor(QTextCursor.MoveOperation.End)
    
    def _on_analysis_complete(self, results):
        """Handle analysis completion."""
        self.analyze_button.setEnabled(True)
        
        # Format and display results
        formatted_results = json.dumps(results, indent=2)
        self.analysis_results.setPlainText(formatted_results)
    
    def _on_analysis_error(self, error_msg):
        """Handle analysis errors."""
        logger.error(f"Analysis error: {error_msg}")
        self.analyze_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error_msg)
    
    def _on_save_analysis_clicked(self):
        """Save analysis results to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Results",
            "",
            "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.analysis_results.toPlainText())
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving file: {str(e)}")

    def _toggle_particles(self):
        """Toggle particle effects on/off."""
        if self.particles.isVisible():
            self.particles.hide()
            self.particle_toggle.setText("☆")
        else:
            self.particles.show()
            self.particle_toggle.setText("🌟")
    
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Update particle system size
        self.particles.setGeometry(self.rect())
        # Keep toggle button in top-right corner with padding
        self.particle_toggle.move(self.width() - 40, 8)

    def paintEvent(self, event):
        """Override paint event to ensure proper styling."""
        # No need for custom painting, let Qt handle it
        super().paintEvent(event)