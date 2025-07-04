"""
Node Wire Diagram UI Module for AgentChef

This module provides reusable UI components for creating node-based visual programming
interfaces with draggable nodes, input/output connections, and wire management.

Features:
- Draggable nodes with customizable input/output ports
- Visual wire connections between nodes
- Automatic wire routing and validation
- Node selection and manipulation
- Zoom and pan canvas support
- Programmatic node and connection management
- Export/import of node graph configurations

Usage:
    diagram = NodeWireDiagram()
    diagram.add_node("process", "Data Processor", inputs=["data"], outputs=["result"])
    diagram.show()
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Set
import json
import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
        QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem,
        QGraphicsRectItem, QGraphicsTextItem, QGraphicsPathItem, QMenu,
        QFileDialog, QMessageBox, QDialog, QFormLayout, QLineEdit, QSpinBox,
        QComboBox, QTextEdit, QGroupBox, QCheckBox, QSlider, QFrame,
        QGraphicsProxyWidget, QToolBar, QStatusBar, QMainWindow, QSplitter
    )
    from PyQt6.QtCore import (
        Qt, QRectF, QPointF, QSizeF, pyqtSignal, QObject, QTimer,
        QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
    )
    from PyQt6.QtGui import (
        QPen, QBrush, QColor, QPainter, QPainterPath, QFont, QAction,
        QPixmap, QTransform, QPolygonF, QLinearGradient, QRadialGradient
    )
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt6 not available. Node diagram UI will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


class PortType(Enum):
    """Types of node ports."""
    INPUT = "input"
    OUTPUT = "output"


class NodeType(Enum):
    """Types of nodes."""
    STANDARD = "standard"
    INPUT = "input"
    OUTPUT = "output"
    PROCESS = "process"
    DECISION = "decision"
    CUSTOM = "custom"


@dataclass
class PortConfig:
    """Configuration for a node port."""
    name: str
    port_type: PortType
    data_type: str = "any"
    required: bool = True
    multiple_connections: bool = False
    description: str = ""
    

@dataclass
class NodeConfig:
    """Configuration for a node."""
    node_id: str
    title: str
    node_type: NodeType = NodeType.STANDARD
    inputs: List[PortConfig] = field(default_factory=list)
    outputs: List[PortConfig] = field(default_factory=list)
    position: Tuple[float, float] = (0, 0)
    size: Tuple[float, float] = (120, 80)
    color: str = "#4a90e2"
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ConnectionConfig:
    """Configuration for a wire connection."""
    connection_id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    color: str = "#666666"
    style: str = "solid"  # solid, dashed, dotted
    

if HAS_QT:
    class NodePort(QGraphicsEllipseItem):
        """A connection port on a node."""
        
        def __init__(self, config: PortConfig, parent_node, radius=6):
            super().__init__(-radius, -radius, radius*2, radius*2)
            self.config = config
            self.parent_node = parent_node
            self.radius = radius
            self.connections = []
            self.is_hovered = False
            
            # Set appearance
            self.setAcceptHoverEvents(True)
            self.setZValue(10)  # Above nodes
            self._update_appearance()
            
        def _update_appearance(self):
            """Update the visual appearance of the port."""
            if self.config.port_type == PortType.INPUT:
                color = QColor("#ff6b6b") if self.is_hovered else QColor("#e74c3c")
            else:  # OUTPUT
                color = QColor("#4ecdc4") if self.is_hovered else QColor("#2ecc71")
                
            self.setBrush(QBrush(color))
            self.setPen(QPen(QColor("#2c3e50"), 2))
            
        def hoverEnterEvent(self, event):
            """Handle mouse hover enter."""
            self.is_hovered = True
            self._update_appearance()
            self.setToolTip(f"{self.config.name}\nType: {self.config.data_type}\n{self.config.description}")
            super().hoverEnterEvent(event)
            
        def hoverLeaveEvent(self, event):
            """Handle mouse hover leave."""
            self.is_hovered = False
            self._update_appearance()
            super().hoverLeaveEvent(event)
            
        def add_connection(self, wire):
            """Add a wire connection to this port."""
            if not self.config.multiple_connections and self.connections:
                return False
            self.connections.append(wire)
            return True
            
        def remove_connection(self, wire):
            """Remove a wire connection from this port."""
            if wire in self.connections:
                self.connections.remove(wire)
                
        def get_scene_position(self):
            """Get the port's position in scene coordinates."""
            return self.mapToScene(self.boundingRect().center())


    class NodeWire(QGraphicsPathItem):
        """A wire connection between two ports."""
        
        def __init__(self, source_port: NodePort, target_port: NodePort = None, config: ConnectionConfig = None):
            super().__init__()
            self.source_port = source_port
            self.target_port = target_port
            self.config = config or ConnectionConfig("", "", "", "", "")
            self.is_temporary = target_port is None
            self.end_position = QPointF(0, 0)
            
            # Set appearance
            self.setZValue(5)  # Between background and nodes
            self.setAcceptHoverEvents(True)
            self._update_appearance()
            self._update_path()
            
        def _update_appearance(self):
            """Update the visual appearance of the wire."""
            color = QColor(self.config.color) if self.config.color else QColor("#666666")
            pen = QPen(color, 3)
            
            if self.config.style == "dashed":
                pen.setStyle(Qt.PenStyle.DashLine)
            elif self.config.style == "dotted":
                pen.setStyle(Qt.PenStyle.DotLine)
                
            self.setPen(pen)
            
        def _update_path(self):
            """Update the wire path between source and target."""
            if not self.source_port:
                return
                
            start_pos = self.source_port.get_scene_position()
            
            if self.target_port:
                end_pos = self.target_port.get_scene_position()
            else:
                end_pos = self.end_position
                
            path = QPainterPath()
            path.moveTo(start_pos)
            
            # Create a smooth curve
            control_offset = abs(end_pos.x() - start_pos.x()) * 0.5
            control_offset = max(control_offset, 50)  # Minimum curve
            
            control1 = QPointF(start_pos.x() + control_offset, start_pos.y())
            control2 = QPointF(end_pos.x() - control_offset, end_pos.y())
            
            path.cubicTo(control1, control2, end_pos)
            self.setPath(path)
            
        def set_end_position(self, position: QPointF):
            """Set the end position for temporary wires."""
            self.end_position = position
            self._update_path()
            
        def connect_to_port(self, target_port: NodePort):
            """Connect this wire to a target port."""
            if target_port.add_connection(self):
                self.target_port = target_port
                self.is_temporary = False
                self._update_path()
                return True
            return False
            
        def disconnect(self):
            """Disconnect this wire from its ports."""
            if self.source_port:
                self.source_port.remove_connection(self)
            if self.target_port:
                self.target_port.remove_connection(self)
                
        def update_position(self):
            """Update the wire path when nodes move."""
            self._update_path()


    class DiagramNode(QGraphicsRectItem):
        """A node in the diagram with input/output ports."""
        
        def __init__(self, config: NodeConfig):
            super().__init__()
            self.config = config
            self.input_ports = {}
            self.output_ports = {}
            self.is_selected = False
            self.is_hovered = False
            
            # Set basic properties
            self.setRect(0, 0, config.size[0], config.size[1])
            self.setPos(config.position[0], config.position[1])
            self.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
            self.setAcceptHoverEvents(True)
            self.setZValue(7)  # Above wires, below ports
            
            # Create visual elements
            self._create_title()
            self._create_ports()
            self._update_appearance()
            
        def _create_title(self):
            """Create the title text for the node."""
            self.title_text = QGraphicsTextItem(self.config.title, self)
            self.title_text.setPos(10, 5)
            font = QFont("Arial", 10, QFont.Weight.Bold)
            self.title_text.setFont(font)
            self.title_text.setDefaultTextColor(QColor("#ffffff"))
            
        def _create_ports(self):
            """Create input and output ports for the node."""
            rect = self.rect()
            port_radius = 6
            
            # Create input ports (left side)
            if self.config.inputs:
                port_spacing = rect.height() / (len(self.config.inputs) + 1)
                for i, port_config in enumerate(self.config.inputs):
                    port = NodePort(port_config, self, port_radius)
                    port.setParentItem(self)
                    y_pos = port_spacing * (i + 1)
                    port.setPos(-port_radius, y_pos)
                    self.input_ports[port_config.name] = port
                    
            # Create output ports (right side)
            if self.config.outputs:
                port_spacing = rect.height() / (len(self.config.outputs) + 1)
                for i, port_config in enumerate(self.config.outputs):
                    port = NodePort(port_config, self, port_radius)
                    port.setParentItem(self)
                    y_pos = port_spacing * (i + 1)
                    port.setPos(rect.width() + port_radius, y_pos)
                    self.output_ports[port_config.name] = port
                    
        def _update_appearance(self):
            """Update the visual appearance of the node."""
            base_color = QColor(self.config.color)
            
            if self.is_selected:
                # Brighter color when selected
                brush_color = base_color.lighter(120)
                pen_color = QColor("#ffd700")  # Gold border
                pen_width = 3
            elif self.is_hovered:
                # Slightly brighter when hovered
                brush_color = base_color.lighter(110)
                pen_color = base_color.darker(150)
                pen_width = 2
            else:
                # Normal appearance
                brush_color = base_color
                pen_color = base_color.darker(150)
                pen_width = 2
                
            # Create gradient brush
            gradient = QLinearGradient(0, 0, 0, self.rect().height())
            gradient.setColorAt(0, brush_color.lighter(120))
            gradient.setColorAt(1, brush_color.darker(120))
            
            self.setBrush(QBrush(gradient))
            self.setPen(QPen(pen_color, pen_width))
            
        def hoverEnterEvent(self, event):
            """Handle mouse hover enter."""
            self.is_hovered = True
            self._update_appearance()
            self.setToolTip(f"{self.config.title}\n{self.config.description}")
            super().hoverEnterEvent(event)
            
        def hoverLeaveEvent(self, event):
            """Handle mouse hover leave."""
            self.is_hovered = False
            self._update_appearance()
            super().hoverLeaveEvent(event)
            
        def itemChange(self, change, value):
            """Handle item changes (selection, position, etc.)."""
            if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
                self.is_selected = bool(value)
                self._update_appearance()
            elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
                # Update wire positions when node moves
                self._update_connected_wires()
                # Update config position
                pos = self.pos()
                self.config.position = (pos.x(), pos.y())
                
            return super().itemChange(change, value)
            
        def _update_connected_wires(self):
            """Update all wires connected to this node."""
            for port in list(self.input_ports.values()) + list(self.output_ports.values()):
                for wire in port.connections:
                    wire.update_position()
                    
        def get_port(self, port_name: str, port_type: PortType) -> Optional[NodePort]:
            """Get a specific port by name and type."""
            if port_type == PortType.INPUT:
                return self.input_ports.get(port_name)
            else:
                return self.output_ports.get(port_name)
                
        def get_all_ports(self) -> List[NodePort]:
            """Get all ports on this node."""
            return list(self.input_ports.values()) + list(self.output_ports.values())


    class NodeDiagramView(QGraphicsView):
        """Custom graphics view for the node diagram."""
        
        # Signals
        nodeSelected = pyqtSignal(str)  # node_id
        nodeDeselected = pyqtSignal(str)  # node_id
        connectionCreated = pyqtSignal(str, str, str, str)  # source_node, source_port, target_node, target_port
        connectionDeleted = pyqtSignal(str)  # connection_id
        
        def __init__(self, scene):
            super().__init__(scene)
            self.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.setMouseTracking(True)
            
            # Connection state
            self.is_connecting = False
            self.temp_wire = None
            self.connection_start_port = None
            
            # View settings
            self.setMinimumSize(400, 300)
            self.setBackgroundBrush(QBrush(QColor("#2c3e50")))
            
        def wheelEvent(self, event):
            """Handle mouse wheel for zooming."""
            # Zoom in/out
            scale_factor = 1.2
            if event.angleDelta().y() < 0:
                scale_factor = 1 / scale_factor
                
            self.scale(scale_factor, scale_factor)
            
        def mousePressEvent(self, event):
            """Handle mouse press events."""
            if event.button() == Qt.MouseButton.LeftButton:
                item = self.itemAt(event.position().toPoint())
                
                if isinstance(item, NodePort):
                    self._start_connection(item)
                    return
                elif isinstance(item, NodeWire):
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        # Delete wire on Ctrl+Click
                        self._delete_wire(item)
                        return
                        
            elif event.button() == Qt.MouseButton.RightButton:
                item = self.itemAt(event.position().toPoint())
                if isinstance(item, DiagramNode):
                    self._show_node_context_menu(item, event.position().toPoint())
                    return
                elif isinstance(item, NodeWire):
                    self._show_wire_context_menu(item, event.position().toPoint())
                    return
                else:
                    self._show_canvas_context_menu(event.position().toPoint())
                    return
                    
            super().mousePressEvent(event)
            
        def mouseMoveEvent(self, event):
            """Handle mouse move events."""
            if self.is_connecting and self.temp_wire:
                # Update temporary wire position
                scene_pos = self.mapToScene(event.position().toPoint())
                self.temp_wire.set_end_position(scene_pos)
                
            super().mouseMoveEvent(event)
            
        def mouseReleaseEvent(self, event):
            """Handle mouse release events."""
            if self.is_connecting and event.button() == Qt.MouseButton.LeftButton:
                item = self.itemAt(event.position().toPoint())
                
                if isinstance(item, NodePort):
                    self._complete_connection(item)
                else:
                    self._cancel_connection()
                    
            super().mouseReleaseEvent(event)
            
        def _start_connection(self, port: NodePort):
            """Start creating a connection from a port."""
            if port.config.port_type == PortType.INPUT and port.connections:
                # Input ports with existing connections can't start new ones
                return
                
            self.is_connecting = True
            self.connection_start_port = port
            
            # Create temporary wire
            self.temp_wire = NodeWire(port)
            self.scene().addItem(self.temp_wire)
            
        def _complete_connection(self, target_port: NodePort):
            """Complete a connection to a target port."""
            if not self.is_connecting or not self.connection_start_port:
                return
                
            source_port = self.connection_start_port
            
            # Validate connection
            if self._can_connect(source_port, target_port):
                # Create permanent connection
                connection_id = f"{source_port.parent_node.config.node_id}:{source_port.config.name}->{target_port.parent_node.config.node_id}:{target_port.config.name}"
                
                config = ConnectionConfig(
                    connection_id=connection_id,
                    source_node=source_port.parent_node.config.node_id,
                    source_port=source_port.config.name,
                    target_node=target_port.parent_node.config.node_id,
                    target_port=target_port.config.name
                )
                
                wire = NodeWire(source_port, target_port, config)
                if wire.connect_to_port(target_port):
                    source_port.add_connection(wire)
                    self.scene().addItem(wire)
                    
                    # Emit signal
                    self.connectionCreated.emit(
                        config.source_node, config.source_port,
                        config.target_node, config.target_port
                    )
                    
            self._cancel_connection()
            
        def _cancel_connection(self):
            """Cancel the current connection attempt."""
            if self.temp_wire:
                self.scene().removeItem(self.temp_wire)
                self.temp_wire = None
                
            self.is_connecting = False
            self.connection_start_port = None
            
        def _can_connect(self, source_port: NodePort, target_port: NodePort) -> bool:
            """Check if two ports can be connected."""
            # Can't connect to same node
            if source_port.parent_node == target_port.parent_node:
                return False
                
            # Must be different port types
            if source_port.config.port_type == target_port.config.port_type:
                return False
                
            # Check if target already has connection (if not allowing multiple)
            if not target_port.config.multiple_connections and target_port.connections:
                return False
                
            # Check data type compatibility (simplified)
            if (source_port.config.data_type != "any" and 
                target_port.config.data_type != "any" and
                source_port.config.data_type != target_port.config.data_type):
                return False
                
            return True
            
        def _delete_wire(self, wire: NodeWire):
            """Delete a wire connection."""
            wire.disconnect()
            self.scene().removeItem(wire)
            
            if wire.config.connection_id:
                self.connectionDeleted.emit(wire.config.connection_id)
                
        def _show_node_context_menu(self, node: DiagramNode, position):
            """Show context menu for a node."""
            menu = QMenu(self)
            
            delete_action = QAction("Delete Node", self)
            delete_action.triggered.connect(lambda: self._delete_node(node))
            menu.addAction(delete_action)
            
            duplicate_action = QAction("Duplicate Node", self)
            duplicate_action.triggered.connect(lambda: self._duplicate_node(node))
            menu.addAction(duplicate_action)
            
            menu.addSeparator()
            
            properties_action = QAction("Properties", self)
            properties_action.triggered.connect(lambda: self._show_node_properties(node))
            menu.addAction(properties_action)
            
            menu.exec(self.mapToGlobal(position))
            
        def _show_wire_context_menu(self, wire: NodeWire, position):
            """Show context menu for a wire."""
            menu = QMenu(self)
            
            delete_action = QAction("Delete Connection", self)
            delete_action.triggered.connect(lambda: self._delete_wire(wire))
            menu.addAction(delete_action)
            
            menu.exec(self.mapToGlobal(position))
            
        def _show_canvas_context_menu(self, position):
            """Show context menu for empty canvas."""
            menu = QMenu(self)
            
            # Add node submenu
            add_menu = menu.addMenu("Add Node")
            
            standard_action = QAction("Standard Node", self)
            standard_action.triggered.connect(lambda: self._add_node_at_position(position, NodeType.STANDARD))
            add_menu.addAction(standard_action)
            
            process_action = QAction("Process Node", self)
            process_action.triggered.connect(lambda: self._add_node_at_position(position, NodeType.PROCESS))
            add_menu.addAction(process_action)
            
            decision_action = QAction("Decision Node", self)
            decision_action.triggered.connect(lambda: self._add_node_at_position(position, NodeType.DECISION))
            add_menu.addAction(decision_action)
            
            menu.exec(self.mapToGlobal(position))
            
        def _delete_node(self, node: DiagramNode):
            """Delete a node and its connections."""
            # Remove all connected wires
            for port in node.get_all_ports():
                for wire in port.connections[:]:  # Copy list to avoid modification during iteration
                    self._delete_wire(wire)
                    
            # Remove node from scene
            self.scene().removeItem(node)
            
        def _duplicate_node(self, node: DiagramNode):
            """Duplicate a node."""
            # Create new config with offset position
            new_config = NodeConfig(
                node_id=f"{node.config.node_id}_copy",
                title=f"{node.config.title} Copy",
                node_type=node.config.node_type,
                inputs=node.config.inputs[:],  # Copy lists
                outputs=node.config.outputs[:],
                position=(node.config.position[0] + 50, node.config.position[1] + 50),
                size=node.config.size,
                color=node.config.color,
                description=node.config.description,
                properties=node.config.properties.copy()
            )
            
            new_node = DiagramNode(new_config)
            self.scene().addItem(new_node)
            
        def _show_node_properties(self, node: DiagramNode):
            """Show properties dialog for a node."""
            dialog = NodePropertiesDialog(node.config, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Update node with new config
                node.config = dialog.get_config()
                node.title_text.setPlainText(node.config.title)
                node._update_appearance()
                
        def _add_node_at_position(self, position, node_type: NodeType):
            """Add a new node at the specified position."""
            scene_pos = self.mapToScene(position)
            
            # Create basic config based on type
            if node_type == NodeType.PROCESS:
                config = NodeConfig(
                    node_id=f"process_{len(self.scene().items())}",
                    title="Process Node",
                    node_type=node_type,
                    inputs=[PortConfig("input", PortType.INPUT)],
                    outputs=[PortConfig("output", PortType.OUTPUT)],
                    position=(scene_pos.x(), scene_pos.y()),
                    color="#4a90e2"
                )
            elif node_type == NodeType.DECISION:
                config = NodeConfig(
                    node_id=f"decision_{len(self.scene().items())}",
                    title="Decision Node",
                    node_type=node_type,
                    inputs=[PortConfig("input", PortType.INPUT)],
                    outputs=[
                        PortConfig("true", PortType.OUTPUT),
                        PortConfig("false", PortType.OUTPUT)
                    ],
                    position=(scene_pos.x(), scene_pos.y()),
                    color="#e67e22"
                )
            else:  # STANDARD
                config = NodeConfig(
                    node_id=f"node_{len(self.scene().items())}",
                    title="Standard Node",
                    node_type=node_type,
                    inputs=[PortConfig("input", PortType.INPUT)],
                    outputs=[PortConfig("output", PortType.OUTPUT)],
                    position=(scene_pos.x(), scene_pos.y()),
                    color="#9b59b6"
                )
                
            node = DiagramNode(config)
            self.scene().addItem(node)


    class NodePropertiesDialog(QDialog):
        """Dialog for editing node properties."""
        
        def __init__(self, config: NodeConfig, parent=None):
            super().__init__(parent)
            self.config = config.copy() if hasattr(config, 'copy') else config
            self.setWindowTitle("Node Properties")
            self.setMinimumWidth(400)
            self._setup_ui()
            
        def _setup_ui(self):
            """Set up the dialog UI."""
            layout = QVBoxLayout(self)
            
            # Basic properties
            basic_group = QGroupBox("Basic Properties")
            basic_layout = QFormLayout(basic_group)
            
            self.title_edit = QLineEdit(self.config.title)
            basic_layout.addRow("Title:", self.title_edit)
            
            self.id_edit = QLineEdit(self.config.node_id)
            basic_layout.addRow("ID:", self.id_edit)
            
            self.description_edit = QTextEdit(self.config.description)
            self.description_edit.setMaximumHeight(60)
            basic_layout.addRow("Description:", self.description_edit)
            
            self.color_edit = QLineEdit(self.config.color)
            basic_layout.addRow("Color:", self.color_edit)
            
            layout.addWidget(basic_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            button_layout.addWidget(cancel_button)
            
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(self.accept)
            ok_button.setDefault(True)
            button_layout.addWidget(ok_button)
            
            layout.addLayout(button_layout)
            
        def get_config(self) -> NodeConfig:
            """Get the updated configuration."""
            self.config.title = self.title_edit.text()
            self.config.node_id = self.id_edit.text()
            self.config.description = self.description_edit.toPlainText()
            self.config.color = self.color_edit.text()
            return self.config


class NodeWireDiagram(QMainWindow):
    """
    Main widget for the node-wire diagram interface.
    
    This widget provides a complete interface for creating, editing, and managing
    node-based diagrams with visual programming capabilities.
    """
    
    # Signals
    nodeAdded = pyqtSignal(str)  # node_id
    nodeRemoved = pyqtSignal(str)  # node_id
    nodeSelected = pyqtSignal(str)  # node_id
    connectionCreated = pyqtSignal(str, str, str, str)  # source_node, source_port, target_node, target_port
    connectionRemoved = pyqtSignal(str)  # connection_id
    diagramChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the node wire diagram."""
        super().__init__(parent)
        
        self.nodes = {}  # node_id -> DiagramNode
        self.connections = {}  # connection_id -> NodeWire
        self.current_file = None
        
        if HAS_QT:
            self._setup_ui()
            self.setWindowTitle("Node Wire Diagram")
            self.resize(1200, 800)
        else:
            logger.error("PyQt6 not available, cannot set up NodeWireDiagram UI")
    
    def _setup_ui(self):
        """Set up the main UI components."""
        if not HAS_QT:
            return
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for diagram and properties
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Diagram area
        diagram_widget = QWidget()
        diagram_layout = QVBoxLayout(diagram_widget)
        
        # Create scene and view
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-2000, -2000, 4000, 4000)
        
        self.view = NodeDiagramView(self.scene)
        self.view.nodeSelected.connect(self.nodeSelected.emit)
        self.view.nodeDeselected.connect(self.nodeDeselected.emit)
        self.view.connectionCreated.connect(self._on_connection_created)
        self.view.connectionDeleted.connect(self._on_connection_deleted)
        
        diagram_layout.addWidget(self.view)
        splitter.addWidget(diagram_widget)
        
        # Properties panel
        self.properties_panel = self._create_properties_panel()
        splitter.addWidget(self.properties_panel)
        
        # Set splitter sizes
        splitter.setSizes([800, 300])
        
        # Create toolbar
        self._create_toolbar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create menu bar
        self._create_menu_bar()
        
    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # File operations
        new_action = QAction("New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_diagram)
        toolbar.addAction(new_action)
        
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_diagram)
        toolbar.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_diagram)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Edit operations
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setEnabled(False)  # TODO: Implement undo/redo
        toolbar.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.setEnabled(False)  # TODO: Implement undo/redo
        toolbar.addAction(redo_action)
        
        toolbar.addSeparator()
        
        # View operations
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(lambda: self.view.scale(1.2, 1.2))
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(lambda: self.view.scale(1/1.2, 1/1.2))
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("Fit to View", self)
        fit_action.triggered.connect(self.fit_to_view)
        toolbar.addAction(fit_action)
        
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New", self.new_diagram, "Ctrl+N")
        file_menu.addAction("Open", self.open_diagram, "Ctrl+O")
        file_menu.addAction("Save", self.save_diagram, "Ctrl+S")
        file_menu.addAction("Save As", self.save_diagram_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        file_menu.addAction("Export as Image", self.export_image)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Select All", self.select_all_nodes, "Ctrl+A")
        edit_menu.addAction("Delete Selected", self.delete_selected, "Delete")
        edit_menu.addSeparator()
        edit_menu.addAction("Clear Diagram", self.clear_diagram)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Zoom In", lambda: self.view.scale(1.2, 1.2), "Ctrl++")
        view_menu.addAction("Zoom Out", lambda: self.view.scale(1/1.2, 1/1.2), "Ctrl+-")
        view_menu.addAction("Reset Zoom", lambda: self.view.resetTransform(), "Ctrl+0")
        view_menu.addAction("Fit to View", self.fit_to_view)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Validate Connections", self.validate_connections)
        tools_menu.addAction("Auto Layout", self.auto_layout)
        tools_menu.addAction("Statistics", self.show_statistics)
        
    def _create_properties_panel(self):
        """Create the properties panel."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(250)
        
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Properties")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Node list
        node_group = QGroupBox("Nodes")
        node_layout = QVBoxLayout(node_group)
        
        self.node_list = QComboBox()
        self.node_list.addItem("No nodes")
        self.node_list.currentTextChanged.connect(self._on_node_selection_changed)
        node_layout.addWidget(self.node_list)
        
        # Node details
        self.node_details = QTextEdit()
        self.node_details.setMaximumHeight(100)
        self.node_details.setReadOnly(True)
        node_layout.addWidget(self.node_details)
        
        layout.addWidget(node_group)
        
        # Connection list
        connection_group = QGroupBox("Connections")
        connection_layout = QVBoxLayout(connection_group)
        
        self.connection_list = QComboBox()
        self.connection_list.addItem("No connections")
        connection_layout.addWidget(self.connection_list)
        
        layout.addWidget(connection_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout(stats_group)
        
        self.node_count_label = QLabel("0")
        stats_layout.addRow("Nodes:", self.node_count_label)
        
        self.connection_count_label = QLabel("0")
        stats_layout.addRow("Connections:", self.connection_count_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
        
    def _on_connection_created(self, source_node, source_port, target_node, target_port):
        """Handle when a new connection is created."""
        connection_id = f"{source_node}:{source_port}->{target_node}:{target_port}"
        
        # Find the wire in the scene
        for item in self.scene.items():
            if isinstance(item, NodeWire) and item.config.connection_id == connection_id:
                self.connections[connection_id] = item
                break
                
        self._update_properties_panel()
        self.connectionCreated.emit(source_node, source_port, target_node, target_port)
        self.diagramChanged.emit()
        
    def _on_connection_deleted(self, connection_id):
        """Handle when a connection is deleted."""
        if connection_id in self.connections:
            del self.connections[connection_id]
            
        self._update_properties_panel()
        self.connectionRemoved.emit(connection_id)
        self.diagramChanged.emit()
        
    def _on_node_selection_changed(self, node_text):
        """Handle node selection change in properties panel."""
        if node_text == "No nodes":
            self.node_details.clear()
            return
            
        # Find node by title
        for node in self.nodes.values():
            if node.config.title == node_text:
                details = f"ID: {node.config.node_id}\n"
                details += f"Type: {node.config.node_type.value}\n"
                details += f"Position: ({node.config.position[0]:.1f}, {node.config.position[1]:.1f})\n"
                details += f"Inputs: {len(node.config.inputs)}\n"
                details += f"Outputs: {len(node.config.outputs)}\n"
                if node.config.description:
                    details += f"Description: {node.config.description}\n"
                self.node_details.setText(details)
                break
                
    def _update_properties_panel(self):
        """Update the properties panel with current data."""
        # Update node list
        self.node_list.clear()
        if self.nodes:
            for node in self.nodes.values():
                self.node_list.addItem(node.config.title)
        else:
            self.node_list.addItem("No nodes")
            
        # Update connection list
        self.connection_list.clear()
        if self.connections:
            for conn_id in self.connections.keys():
                self.connection_list.addItem(conn_id)
        else:
            self.connection_list.addItem("No connections")
            
        # Update statistics
        self.node_count_label.setText(str(len(self.nodes)))
        self.connection_count_label.setText(str(len(self.connections)))
        
    def add_node(self, node_config: NodeConfig) -> bool:
        """
        Add a new node to the diagram.
        
        Args:
            node_config: Configuration for the new node
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_QT:
            logger.error("PyQt6 not available, cannot add node")
            return False
            
        if node_config.node_id in self.nodes:
            logger.warning(f"Node with ID {node_config.node_id} already exists")
            return False
            
        try:
            node = DiagramNode(node_config)
            self.scene.addItem(node)
            self.nodes[node_config.node_id] = node
            
            self._update_properties_panel()
            self.nodeAdded.emit(node_config.node_id)
            self.diagramChanged.emit()
            
            logger.info(f"Added node: {node_config.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding node: {e}")
            return False
            
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the diagram.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_QT:
            return False
            
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found")
            return False
            
        try:
            node = self.nodes[node_id]
            
            # Remove all connections to/from this node
            connections_to_remove = []
            for port in node.get_all_ports():
                for wire in port.connections[:]:
                    connections_to_remove.append(wire.config.connection_id)
                    wire.disconnect()
                    self.scene.removeItem(wire)
                    
            # Remove connections from tracking
            for conn_id in connections_to_remove:
                if conn_id in self.connections:
                    del self.connections[conn_id]
                    
            # Remove node
            self.scene.removeItem(node)
            del self.nodes[node_id]
            
            self._update_properties_panel()
            self.nodeRemoved.emit(node_id)
            self.diagramChanged.emit()
            
            logger.info(f"Removed node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing node: {e}")
            return False
            
    def connect_nodes(self, source_node_id: str, source_port: str, 
                     target_node_id: str, target_port: str) -> bool:
        """
        Create a connection between two nodes.
        
        Args:
            source_node_id: ID of the source node
            source_port: Name of the source port
            target_node_id: ID of the target node
            target_port: Name of the target port
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_QT:
            return False
            
        try:
            source_node = self.nodes.get(source_node_id)
            target_node = self.nodes.get(target_node_id)
            
            if not source_node or not target_node:
                logger.error("Source or target node not found")
                return False
                
            source_port_obj = source_node.get_port(source_port, PortType.OUTPUT)
            target_port_obj = target_node.get_port(target_port, PortType.INPUT)
            
            if not source_port_obj or not target_port_obj:
                logger.error("Source or target port not found")
                return False
                
            # Create connection config
            connection_id = f"{source_node_id}:{source_port}->{target_node_id}:{target_port}"
            config = ConnectionConfig(
                connection_id=connection_id,
                source_node=source_node_id,
                source_port=source_port,
                target_node=target_node_id,
                target_port=target_port
            )
            
            # Create wire
            wire = NodeWire(source_port_obj, target_port_obj, config)
            
            if wire.connect_to_port(target_port_obj):
                source_port_obj.add_connection(wire)
                self.scene.addItem(wire)
                self.connections[connection_id] = wire
                
                self._update_properties_panel()
                self.connectionCreated.emit(source_node_id, source_port, target_node_id, target_port)
                self.diagramChanged.emit()
                
                logger.info(f"Connected: {connection_id}")
                return True
            else:
                logger.error("Failed to connect ports")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting nodes: {e}")
            return False
            
    def disconnect_nodes(self, connection_id: str) -> bool:
        """
        Remove a connection between nodes.
        
        Args:
            connection_id: ID of the connection to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_QT:
            return False
            
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
            
        try:
            wire = self.connections[connection_id]
            wire.disconnect()
            self.scene.removeItem(wire)
            del self.connections[connection_id]
            
            self._update_properties_panel()
            self.connectionRemoved.emit(connection_id)
            self.diagramChanged.emit()
            
            logger.info(f"Disconnected: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting nodes: {e}")
            return False
            
    def get_diagram_data(self) -> Dict[str, Any]:
        """
        Get the current diagram as a serializable dictionary.
        
        Returns:
            Dictionary containing all diagram data
        """
        data = {
            "nodes": {},
            "connections": {}
        }
        
        # Export nodes
        for node_id, node in self.nodes.items():
            config = node.config
            data["nodes"][node_id] = {
                "node_id": config.node_id,
                "title": config.title,
                "node_type": config.node_type.value,
                "position": config.position,
                "size": config.size,
                "color": config.color,
                "description": config.description,
                "properties": config.properties,
                "inputs": [
                    {
                        "name": port.name,
                        "data_type": port.data_type,
                        "required": port.required,
                        "multiple_connections": port.multiple_connections,
                        "description": port.description
                    }
                    for port in config.inputs
                ],
                "outputs": [
                    {
                        "name": port.name,
                        "data_type": port.data_type,
                        "required": port.required,
                        "multiple_connections": port.multiple_connections,
                        "description": port.description
                    }
                    for port in config.outputs
                ]
            }
            
        # Export connections
        for conn_id, wire in self.connections.items():
            config = wire.config
            data["connections"][conn_id] = {
                "connection_id": config.connection_id,
                "source_node": config.source_node,
                "source_port": config.source_port,
                "target_node": config.target_node,
                "target_port": config.target_port,
                "color": config.color,
                "style": config.style
            }
            
        return data
        
    def load_diagram_data(self, data: Dict[str, Any]) -> bool:
        """
        Load diagram from a dictionary.
        
        Args:
            data: Dictionary containing diagram data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.clear_diagram()
            
            # Load nodes
            for node_data in data.get("nodes", {}).values():
                # Convert port data back to PortConfig objects
                inputs = [
                    PortConfig(
                        name=port["name"],
                        port_type=PortType.INPUT,
                        data_type=port.get("data_type", "any"),
                        required=port.get("required", True),
                        multiple_connections=port.get("multiple_connections", False),
                        description=port.get("description", "")
                    )
                    for port in node_data.get("inputs", [])
                ]
                
                outputs = [
                    PortConfig(
                        name=port["name"],
                        port_type=PortType.OUTPUT,
                        data_type=port.get("data_type", "any"),
                        required=port.get("required", True),
                        multiple_connections=port.get("multiple_connections", True),
                        description=port.get("description", "")
                    )
                    for port in node_data.get("outputs", [])
                ]
                
                config = NodeConfig(
                    node_id=node_data["node_id"],
                    title=node_data["title"],
                    node_type=NodeType(node_data.get("node_type", "standard")),
                    inputs=inputs,
                    outputs=outputs,
                    position=tuple(node_data.get("position", (0, 0))),
                    size=tuple(node_data.get("size", (120, 80))),
                    color=node_data.get("color", "#4a90e2"),
                    description=node_data.get("description", ""),
                    properties=node_data.get("properties", {})
                )
                
                self.add_node(config)
                
            # Load connections
            for conn_data in data.get("connections", {}).values():
                self.connect_nodes(
                    conn_data["source_node"],
                    conn_data["source_port"],
                    conn_data["target_node"],
                    conn_data["target_port"]
                )
                
            self.status_bar.showMessage(f"Loaded {len(self.nodes)} nodes and {len(self.connections)} connections")
            return True
            
        except Exception as e:
            logger.error(f"Error loading diagram data: {e}")
            self.status_bar.showMessage(f"Error loading diagram: {str(e)}")
            return False
            
    def save_diagram(self):
        """Save the current diagram to file."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_diagram_as()
            
    def save_diagram_as(self):
        """Save the diagram to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Diagram", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self._save_to_file(file_path):
                self.current_file = file_path
                self.setWindowTitle(f"Node Wire Diagram - {Path(file_path).name}")
                
    def _save_to_file(self, file_path: str) -> bool:
        """Save diagram data to a file."""
        try:
            data = self.get_diagram_data()
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.status_bar.showMessage(f"Saved to {file_path}")
            logger.info(f"Saved diagram to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving diagram: {e}")
            self.status_bar.showMessage(f"Error saving: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Error saving diagram: {str(e)}")
            return False
            
    def open_diagram(self):
        """Open a diagram from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Diagram", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self._load_from_file(file_path)
            
    def _load_from_file(self, file_path: str) -> bool:
        """Load diagram data from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if self.load_diagram_data(data):
                self.current_file = file_path
                self.setWindowTitle(f"Node Wire Diagram - {Path(file_path).name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading diagram: {e}")
            self.status_bar.showMessage(f"Error loading: {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Error loading diagram: {str(e)}")
            return False
            
    def new_diagram(self):
        """Create a new empty diagram."""
        self.clear_diagram()
        self.current_file = None
        self.setWindowTitle("Node Wire Diagram - New")
        self.status_bar.showMessage("New diagram created")
        
    def clear_diagram(self):
        """Clear all nodes and connections from the diagram."""
        self.scene.clear()
        self.nodes.clear()
        self.connections.clear()
        self._update_properties_panel()
        self.diagramChanged.emit()
        
    def select_all_nodes(self):
        """Select all nodes in the diagram."""
        for node in self.nodes.values():
            node.setSelected(True)
            
    def delete_selected(self):
        """Delete all selected items."""
        selected_items = self.scene.selectedItems()
        
        for item in selected_items:
            if isinstance(item, DiagramNode):
                self.remove_node(item.config.node_id)
            elif isinstance(item, NodeWire):
                if item.config.connection_id:
                    self.disconnect_nodes(item.config.connection_id)
                    
    def fit_to_view(self):
        """Fit all items to the view."""
        if self.nodes:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.view.resetTransform()
            
    def export_image(self):
        """Export the diagram as an image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                # Get scene bounds
                scene_rect = self.scene.itemsBoundingRect()
                if scene_rect.isEmpty():
                    scene_rect = QRectF(0, 0, 800, 600)
                    
                # Create pixmap
                pixmap = QPixmap(scene_rect.size().toSize())
                pixmap.fill(Qt.GlobalColor.white)
                
                # Render scene to pixmap
                painter = QPainter(pixmap)
                self.scene.render(painter, QRectF(), scene_rect)
                painter.end()
                
                # Save pixmap
                if pixmap.save(file_path):
                    self.status_bar.showMessage(f"Exported to {file_path}")
                else:
                    self.status_bar.showMessage("Export failed")
                    
            except Exception as e:
                logger.error(f"Error exporting image: {e}")
                QMessageBox.critical(self, "Export Error", f"Error exporting image: {str(e)}")
                
    def validate_connections(self):
        """Validate all connections in the diagram."""
        issues = []
        
        for node_id, node in self.nodes.items():
            # Check required inputs
            for input_config in node.config.inputs:
                if input_config.required:
                    port = node.get_port(input_config.name, PortType.INPUT)
                    if not port or not port.connections:
                        issues.append(f"Node '{node.config.title}' has unconnected required input '{input_config.name}'")
                        
        if issues:
            msg = "Validation Issues Found:\n\n" + "\n".join(issues)
            QMessageBox.warning(self, "Validation Results", msg)
        else:
            QMessageBox.information(self, "Validation Results", "No issues found!")
            
    def auto_layout(self):
        """Automatically arrange nodes in the diagram."""
        if not self.nodes:
            return
            
        # Simple grid layout
        nodes_per_row = math.ceil(math.sqrt(len(self.nodes)))
        node_spacing_x = 200
        node_spacing_y = 150
        
        for i, node in enumerate(self.nodes.values()):
            row = i // nodes_per_row
            col = i % nodes_per_row
            
            x = col * node_spacing_x
            y = row * node_spacing_y
            
            node.setPos(x, y)
            node.config.position = (x, y)
            
        self.diagramChanged.emit()
        self.status_bar.showMessage("Auto layout applied")
        
    def show_statistics(self):
        """Show diagram statistics."""
        total_connections = len(self.connections)
        node_types = {}
        
        for node in self.nodes.values():
            node_type = node.config.node_type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
        stats = f"Diagram Statistics:\n\n"
        stats += f"Total Nodes: {len(self.nodes)}\n"
        stats += f"Total Connections: {total_connections}\n\n"
        stats += "Node Types:\n"
        for node_type, count in node_types.items():
            stats += f"  {node_type.title()}: {count}\n"
            
        QMessageBox.information(self, "Diagram Statistics", stats)


# Example usage and helper functions
def create_sample_diagram() -> NodeWireDiagram:
    """Create a sample diagram with some nodes and connections."""
    if not HAS_QT:
        logger.error("PyQt6 not available, cannot create sample diagram")
        return None
        
    diagram = NodeWireDiagram()
    
    # Add some sample nodes
    input_node = NodeConfig(
        node_id="input_1",
        title="Data Input",
        node_type=NodeType.INPUT,
        outputs=[PortConfig("data", PortType.OUTPUT, "data")],
        position=(50, 100),
        color="#2ecc71"
    )
    
    process_node = NodeConfig(
        node_id="process_1", 
        title="Data Processor",
        node_type=NodeType.PROCESS,
        inputs=[PortConfig("input", PortType.INPUT, "data")],
        outputs=[
            PortConfig("output", PortType.OUTPUT, "data"),
            PortConfig("debug", PortType.OUTPUT, "string")
        ],
        position=(300, 100),
        color="#4a90e2"
    )
    
    output_node = NodeConfig(
        node_id="output_1",
        title="Data Output", 
        node_type=NodeType.OUTPUT,
        inputs=[PortConfig("result", PortType.INPUT, "data")],
        position=(550, 100),
        color="#e74c3c"
    )
    
    # Add nodes to diagram
    diagram.add_node(input_node)
    diagram.add_node(process_node)
    diagram.add_node(output_node)
    
    # Connect nodes
    diagram.connect_nodes("input_1", "data", "process_1", "input")
    diagram.connect_nodes("process_1", "output", "output_1", "result")
    
    return diagram


# Example usage if module is run directly
if __name__ == "__main__" and HAS_QT:
    import sys
    
    app = QApplication(sys.argv)
    
    # Create and show diagram
    diagram = create_sample_diagram()
    if diagram:
        diagram.show()
        sys.exit(app.exec())
    else:
        print("Failed to create diagram - PyQt6 not available")
        sys.exit(1)