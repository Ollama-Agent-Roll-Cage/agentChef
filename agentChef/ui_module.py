"""
UDRAGS UI Module - Simple graphical interface for the UDRAGS system.
Provides a user-friendly interface for research, dataset generation, and analysis.

Written By: @AgentChef
Date: 4/4/2025
"""

import os
import sys
import json
import asyncio
import threading
from pathlib import Path

from datetime import datetime, timezone
UTC = timezone.utc

import logging
import webbrowser
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QFileDialog,
        QProgressBar, QSpinBox, QCheckBox, QGroupBox, QFormLayout, QSplitter,
        QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDialog
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
    from PyQt6.QtGui import QFont, QIcon, QTextCursor
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt6 not available. Install with 'pip install PyQt6'")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('udrags_ui.log'),
        logging.StreamHandler()
    ]
)

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

class UdragsUI(QMainWindow):
    """Main window for the UDRAGS UI."""
    
    def __init__(self, research_manager):
        """
        Initialize the UI.
        
        Args:
            research_manager: ResearchManager instance
        """
        super().__init__()
        self.research_manager = research_manager
        self.worker_thread = None
        
        self.setWindowTitle("UDRAGS - Unified Dataset Research, Augmentation, & Generation System")
        self.setMinimumSize(1000, 700)
        
        # Set up the UI
        self._setup_ui()
    
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
        self.github_repos_input.setPlaceholder("Repository URLs (comma-separated)")
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
        
        self.research_tab.setLayout(layout)
    
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