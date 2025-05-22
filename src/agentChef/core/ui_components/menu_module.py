"""Menu module for AgentChef UI."""

import os
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QMainWindow, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import pyqtSignal, QUrl, QObject, pyqtSlot, QTimer
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineSettings

class WebBridge(QObject):
    """Bridge between web UI and Python."""
    launchWizard = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot()
    def triggerWizard(self):
        """Called from JavaScript to launch wizard."""
        print("DEBUG: Wizard launch triggered from menu")
        self.launchWizard.emit()

class AgentChefMenu(QMainWindow):
    """Main menu window for AgentChef."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Chef")
        self.resize(800, 600)
        
        # Create QWebEngineView after setting up window
        self.web_view = None
        
        # Initialize UI asynchronously after Qt event loop starts
        QTimer.singleShot(0, self.setup_ui)

    def setup_ui(self):
        """Set up the HTML-based menu interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create web view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        
        # Configure WebEngine settings
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        
        # Set up page and channel
        page = self.web_view.page()
        
        # Create channel and bridge
        self.channel = QWebChannel()
        self.bridge = WebBridge()
        self.channel.registerObject("backend", self.bridge)
        page.setWebChannel(self.channel)
        
        # Load menu file
        menu_file = Path(__file__).parent / "menu" / "agentChefMenu.html"
        menu_dir = Path(__file__).parent / "menu"
        menu_dir.mkdir(parents=True, exist_ok=True)
        
        if not menu_file.exists():
            self._create_default_menu(menu_file)
        
        # Load the HTML file
        url = QUrl.fromLocalFile(str(menu_file.absolute()))
        self.web_view.setUrl(url)
        
        # Make sure the web view is properly sized
        self.web_view.setGeometry(0, 0, self.width(), self.height())
    
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        if self.web_view is not None:  # Add this check
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
