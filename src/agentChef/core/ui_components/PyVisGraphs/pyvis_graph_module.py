"""
PyVis Graph Visualization Module for AgentChef

This module provides reusable PyVis network visualization components for AgentChef UIs.
Supports knowledge graphs, citation networks, ontologies, and general graph visualizations.

Features:
- Interactive network visualizations with customizable physics
- Dark/light theme support
- Save/export functionality
- Integration with AgentChef data formats
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import os
import re
import tempfile
import logging
import math
import random
import json

# PyVis imports
from pyvis.network import Network

# Import the fixed options helper
from .fixed_options import create_options_json

# PyQt imports
try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
        QFrame, QFileDialog, QCheckBox, QSlider, QGroupBox, QFormLayout,
        QSpinBox, QSplitter
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtCore import QUrl, Qt, pyqtSignal, QSize, QTimer
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt6 not available. UI components will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


class PyVisGraphWidget(QWidget):
    """
    Interactive PyVis graph visualization widget for AgentChef UIs.
    
    This widget provides a self-contained PyVis visualization area with controls
    for physics settings, node filtering, and graph manipulation.
    """
    
    graphModified = pyqtSignal()  # Signal emitted when graph is modified
    
    def __init__(self, parent=None, dark_theme=True):
        """
        Initialize the PyVis graph widget.
        
        Args:
            parent: Parent widget
            dark_theme: Whether to use dark theme for the graph
        """
        super().__init__(parent)
        self.dark_theme = dark_theme
        self.graph_data = {"nodes": {}, "edges": []}
        self.html_path = None
        self.temp_dir = tempfile.mkdtemp(prefix="agentchef_pyvis_")
        
        # Set up the UI
        self._setup_ui()
        
        logger.info("PyVisGraphWidget initialized")
    
    def _setup_ui(self):
        """Set up the widget UI components."""
        if not HAS_QT:
            logger.error("PyQt6 not available, cannot set up PyVis UI")
            return
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Physics controls
        physics_group = QGroupBox("Physics Settings")
        physics_layout = QFormLayout(physics_group)
        
        self.gravity_slider = QSlider(Qt.Orientation.Horizontal)
        self.gravity_slider.setRange(-3000, 0)
        self.gravity_slider.setValue(-1200)
        self.gravity_slider.setToolTip("Gravitational constant")
        self.gravity_slider.valueChanged.connect(self._update_physics)
        physics_layout.addRow("Gravity:", self.gravity_slider)
        
        self.spring_slider = QSlider(Qt.Orientation.Horizontal)
        self.spring_slider.setRange(0, 100)
        self.spring_slider.setValue(50)
        self.spring_slider.setToolTip("Spring strength")
        self.spring_slider.valueChanged.connect(self._update_physics)
        physics_layout.addRow("Spring:", self.spring_slider)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)
        
        self.node_size_spin = QSpinBox()
        self.node_size_spin.setRange(10, 80)
        self.node_size_spin.setValue(25)
        self.node_size_spin.valueChanged.connect(self._update_node_options)
        viz_layout.addRow("Node Size:", self.node_size_spin)
        
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Force-directed", "Hierarchical", "Circular"])
        self.layout_combo.currentIndexChanged.connect(self._update_layout)
        viz_layout.addRow("Layout:", self.layout_combo)
        
        # Add control groups to controls layout
        controls_layout.addWidget(physics_group)
        controls_layout.addWidget(viz_group)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Graph")
        self.save_button.clicked.connect(self.save_graph)
        button_layout.addWidget(self.save_button)
        
        self.refresh_button = QPushButton("Refresh View")
        self.refresh_button.clicked.connect(self.refresh_graph)
        button_layout.addWidget(self.refresh_button)
        
        # Add controls and buttons to main layout
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(button_layout)
        
        # Add web view for PyVis
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(400)
        main_layout.addWidget(self.web_view, stretch=1)
        
        # Create empty graph to start
        self._create_empty_graph()
    
    def _create_empty_graph(self):
        """Create an empty graph to initialize the view."""
        self._generate_pyvis_graph()
        self.load_graph_html(self.html_path)
    
    def _update_physics(self):
        """Update the physics settings of the graph."""
        self._generate_pyvis_graph()
        self.load_graph_html(self.html_path)
    
    def _update_node_options(self):
        """Update the node appearance options."""
        self._generate_pyvis_graph()
        self.load_graph_html(self.html_path)
    
    def _update_layout(self):
        """Update the graph layout algorithm."""
        self._generate_pyvis_graph()
        self.load_graph_html(self.html_path)
    
    def _generate_pyvis_graph(self):
        """Generate a PyVis graph from the current data and settings."""
        # Create network with theme-appropriate settings
        bgcolor = "#1a1b26" if self.dark_theme else "#ffffff"
        font_color = "#a9b1d6" if self.dark_theme else "#000000"
        
        net = Network(
            height="100%",
            width="100%",
            bgcolor=bgcolor,
            font_color=font_color,
            notebook=False,
            cdn_resources="in_line"
        )
        
        # Configure physics based on slider values
        gravity = self.gravity_slider.value()
        spring_strength = self.spring_slider.value() / 1000
        
        net.barnes_hut(
            gravity=gravity,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=spring_strength,
            damping=0.09,
            overlap=0
        )
        
        # Add nodes and edges
        node_size = self.node_size_spin.value()
        
        # Add nodes from graph data
        for node_id, node_data in self.graph_data["nodes"].items():
            try:
                color = node_data.get("color", "#7aa2f7")
                label = node_data.get("label", str(node_id))
                title = node_data.get("title", label)
                
                net.add_node(
                    node_id, 
                    label=label, 
                    title=title,
                    color=color, 
                    size=node_size
                )
            except Exception as e:
                logger.error(f"Error adding node {node_id}: {e}")
        
        # Add edges from graph data
        for edge in self.graph_data["edges"]:
            try:
                from_id = edge.get("from")
                to_id = edge.get("to")
                title = edge.get("title", "")
                width = edge.get("width", 1)
                
                if from_id in self.graph_data["nodes"] and to_id in self.graph_data["nodes"]:
                    net.add_edge(from_id, to_id, title=title, width=width)
            except Exception as e:
                logger.error(f"Error adding edge: {e}")
          # Set options for better appearance and interaction
        layout_type = self.layout_combo.currentText().lower()
        options_json = create_options_json(
            node_size=node_size,
            gravity=gravity,
            spring_strength=spring_strength,
            layout_type=layout_type
        )
        
        # Apply options to network - using the json string
        net.set_options(options_json)
        
        # Save to HTML file
        self.html_path = os.path.join(self.temp_dir, "network.html")
        try:
            net.save_graph(self.html_path)
        except Exception as e:
            logger.error(f"Error saving PyVis graph: {e}")
    
    def load_graph_html(self, html_path):
        """Load the generated HTML graph into the web view."""
        if not os.path.exists(html_path):
            logger.error(f"HTML file not found: {html_path}")
            return
            
        try:
            url = QUrl.fromLocalFile(html_path)
            self.web_view.load(url)
        except Exception as e:
            logger.error(f"Error loading graph HTML: {e}")
    
    def set_graph_data(self, nodes: Dict[str, Dict], edges: List[Dict]):
        """
        Set the graph data to display.
        
        Args:
            nodes: Dictionary mapping node IDs to node data
            edges: List of edge dictionaries with 'from' and 'to' fields
        """
        self.graph_data["nodes"] = nodes
        self.graph_data["edges"] = edges
        self._generate_pyvis_graph()
        self.load_graph_html(self.html_path)
        
        logger.info(f"Graph updated with {len(nodes)} nodes and {len(edges)} edges")
    
    def refresh_graph(self):
        """Refresh the graph visualization."""
        self._generate_pyvis_graph()
        self.load_graph_html(self.html_path)
    
    def save_graph(self):
        """Save the current graph to an HTML file."""
        if not self.html_path:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "", "HTML Files (*.html)"
        )
        
        if save_path:
            if not save_path.endswith(".html"):
                save_path += ".html"
                
            try:
                # Copy the current graph HTML to the save location
                import shutil
                shutil.copy(self.html_path, save_path)
                logger.info(f"Graph saved to: {save_path}")
            except Exception as e:
                logger.error(f"Error saving graph: {e}")
    
    def set_dark_theme(self, dark: bool):
        """Set the graph theme to dark or light."""
        if self.dark_theme != dark:
            self.dark_theme = dark
            self._generate_pyvis_graph()
            self.load_graph_html(self.html_path)


class KnowledgeGraphBuilder:
    """
    Helper class for building knowledge graphs from various data sources.
    
    This class converts different data formats (papers, datasets, ontologies)
    into PyVis-compatible graph structures.
    """
    
    @staticmethod
    def from_paper_citations(papers: List[Dict], max_depth: int = 1) -> Dict:
        """
        Build a citation network from paper data.
        
        Args:
            papers: List of paper dictionaries with citation information
            max_depth: Maximum depth of citation graph
            
        Returns:
            Graph data dictionary with nodes and edges
        """
        nodes = {}
        edges = []
        processed_ids = set()
        
        # Process papers to build citation network
        for paper in papers:
            paper_id = paper.get("arxiv_id", paper.get("id", "unknown"))
            if paper_id in processed_ids:
                continue
                
            processed_ids.add(paper_id)
            
            # Add paper node
            nodes[paper_id] = {
                "label": paper.get("title", "Untitled"),
                "title": f"{paper.get('title', 'Untitled')}<br>Authors: {', '.join(paper.get('authors', []))}",
                "color": "#7aa2f7",  # Default color
                "type": "paper"
            }
            
            # Add citation edges
            citations = paper.get("citations", [])
            for citation in citations[:max_depth * 5]:  # Limit citations by depth
                cited_id = citation.get("arxiv_id", citation.get("id", "unknown"))
                
                # Add cited paper node if not already present
                if cited_id not in nodes:
                    nodes[cited_id] = {
                        "label": citation.get("title", "Untitled"),
                        "title": f"{citation.get('title', 'Untitled')}<br>Authors: {', '.join(citation.get('authors', []))}",
                        "color": "#bb9af7",  # Different color for cited papers
                        "type": "cited_paper"
                    }
                
                # Add citation edge
                edges.append({
                    "from": paper_id,
                    "to": cited_id,
                    "title": "Cites",
                    "arrows": "to"
                })
        
        return {"nodes": nodes, "edges": edges}
    
    @staticmethod
    def from_ontology(concepts: Dict[str, List[str]]) -> Dict:
        """
        Build a graph from an ontology structure.
        
        Args:
            concepts: Dictionary mapping concepts to related concepts
            
        Returns:
            Graph data dictionary with nodes and edges
        """
        nodes = {}
        edges = []
        
        # Process concepts
        for concept, related in concepts.items():
            # Add main concept node if not present
            if concept not in nodes:
                nodes[concept] = {
                    "label": concept,
                    "title": concept,
                    "color": "#7aa2f7",  # Default color
                    "type": "concept"
                }
            
            # Add related concept nodes and edges
            for rel in related:
                # Add related concept node if not present
                if rel not in nodes:
                    nodes[rel] = {
                        "label": rel,
                        "title": rel,
                        "color": "#bb9af7",  # Different color for related concepts
                        "type": "related_concept"
                    }
                
                # Add relationship edge
                edges.append({
                    "from": concept,
                    "to": rel,
                    "title": "Related to",
                    "arrows": "to,from"
                })
        
        return {"nodes": nodes, "edges": edges}
    
    @staticmethod
    def from_dataset_relationships(dataset: List[Dict], entity_field: str, 
                                  relation_field: str, target_field: str) -> Dict:
        """
        Build a graph from dataset relationships.
        
        Args:
            dataset: List of dictionaries containing relationship data
            entity_field: Field name for source entity
            relation_field: Field name for relation type
            target_field: Field name for target entity
            
        Returns:
            Graph data dictionary with nodes and edges
        """
        nodes = {}
        edges = []
        
        for item in dataset:
            entity = item.get(entity_field)
            relation = item.get(relation_field)
            target = item.get(target_field)
            
            if not all([entity, relation, target]):
                continue
            
            # Add source entity node
            if entity not in nodes:
                nodes[entity] = {
                    "label": entity,
                    "title": entity,
                    "color": "#7aa2f7",  # Blue color for source
                    "type": "entity"
                }
            
            # Add target entity node
            if target not in nodes:
                nodes[target] = {
                    "label": target,
                    "title": target,
                    "color": "#bb9af7",  # Purple color for target
                    "type": "entity"
                }
            
            # Add relationship edge
            edges.append({
                "from": entity,
                "to": target,
                "title": relation,
                "arrows": "to"
            })
        
        return {"nodes": nodes, "edges": edges}


# Example usage if module is run directly
if __name__ == "__main__" and HAS_QT:
    import sys
    
    app = QApplication(sys.argv)
    
    # Create sample graph data
    sample_nodes = {
        "A": {"label": "Concept A", "title": "Concept A: Main idea", "color": "#7aa2f7"},
        "B": {"label": "Concept B", "title": "Concept B: Related idea", "color": "#bb9af7"},
        "C": {"label": "Concept C", "title": "Concept C: Another idea", "color": "#2ac3de"}
    }
    
    sample_edges = [
        {"from": "A", "to": "B", "title": "Relates to"},
        {"from": "A", "to": "C", "title": "Influences"},
        {"from": "B", "to": "C", "title": "Depends on"}
    ]
    
    # Create and show widget
    widget = PyVisGraphWidget(dark_theme=True)
    widget.setWindowTitle("PyVis Graph Example")
    widget.resize(1000, 800)
    widget.set_graph_data(sample_nodes, sample_edges)
    widget.show()
    
    sys.exit(app.exec())