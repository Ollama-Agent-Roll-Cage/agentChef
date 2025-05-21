"""Menu module for AgentChef UI."""

import os
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSignal, QUrl, QObject, pyqtSlot
from PyQt6.QtWebChannel import QWebChannel

class WebBridge(QObject):
    """Bridge between web UI and Python."""
    launchWizard = pyqtSignal()  # Signal to emit when wizard should be launched

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot()
    def triggerWizard(self):
        """Called from JavaScript to launch wizard."""
        print("Wizard launch triggered from menu")
        self.launchWizard.emit()

class AgentChefMenu(QMainWindow):
    """Main menu window for AgentChef."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Chef")
        self.resize(800, 600)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the HTML-based menu interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Set up web view
        self.web_view = QWebEngineView(central_widget)
        self.web_view.setGeometry(0, 0, 800, 600)
        
        # Set up web channel for JS-Python communication
        self.channel = QWebChannel()
        self.bridge = WebBridge()
        self.channel.registerObject("backend", self.bridge)
        self.web_view.page().setWebChannel(self.channel)
        
        # Find menu HTML file
        package_dir = Path(__file__).parent.parent.parent
        menu_file = package_dir / "core" / "ui_components" / "menu" / "agentChefMenu.html"
        
        # Create menu directory if needed
        menu_dir = package_dir / "core" / "ui_components" / "menu"
        menu_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default menu HTML if it doesn't exist
        if not menu_file.exists():
            self._create_default_menu(menu_file)
        
        # Load the HTML file
        self.web_view.setUrl(QUrl.fromLocalFile(str(menu_file)))
    
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self.web_view.setGeometry(0, 0, self.width(), self.height())
    
    def _create_default_menu(self, file_path: Path):
        """Create a basic menu HTML file if it doesn't exist."""
        html_content = None
        
        # Try to find the template in the package
        template_path = Path(__file__).parent / "menu" / "agentChefMenu.html"
        if template_path.exists():
            html_content = template_path.read_text(encoding='utf-8')
        
        if not html_content:
            # Fallback minimal menu
            html_content = """<!DOCTYPE html>
            <html><body>
                <h1>Agent Chef Menu</h1>
                <button onclick="window.backend.triggerWizard()">Launch RAGChef</button>
            </body></html>"""
        
        file_path.write_text(html_content, encoding='utf-8')
