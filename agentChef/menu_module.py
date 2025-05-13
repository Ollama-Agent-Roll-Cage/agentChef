from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QUrl, pyqtSignal, pyqtSlot, QObject, QFileInfo
from PyQt6.QtGui import QColor, QPalette
from pathlib import Path
import os
import logging

# Import the path from __init__
from . import MENU_HTML_PATH, PACKAGE_DIR

class WebInterface(QObject):
    """Interface for JavaScript to Python communication"""
    launchRequested = pyqtSignal()
    
    @pyqtSlot()
    def launchWizard(self):
        self.launchRequested.emit()

class AgentChefMenu(QMainWindow):
    """Main menu window for Agent Chef using HTML/CSS interface."""
    
    # Signal emitted when user clicks "Launch Wizard"
    launch_wizard = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent Chef")
        self.setMinimumSize(800, 600)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Debug log the package directory and HTML path
        self.logger.debug(f"Package directory: {PACKAGE_DIR}")
        self.logger.debug(f"Menu HTML path: {MENU_HTML_PATH}")
        
        # Set up the dark theme
        self.setup_theme()
        
        # Create web view
        self.web_view = QWebEngineView()
        self.setCentralWidget(self.web_view)
        
        # Set up web channel for JS communication
        self.channel = QWebChannel()
        self.web_interface = WebInterface()
        self.channel.registerObject("backend", self.web_interface)
        self.web_view.page().setWebChannel(self.channel)
        
        # Connect signals
        self.web_interface.launchRequested.connect(self.launch_wizard)
        
        # Inject the required JavaScript before loading the page
        self.web_view.page().runJavaScript("""
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.backend = channel.objects.backend;
            });
            
            function launchWizard() {
                if (window.backend) {
                    backend.launchWizard();
                }
            }
        """)
        
        # Get absolute path to HTML file
        if not MENU_HTML_PATH.exists():
            self.logger.error(f"Menu HTML file not found at: {MENU_HTML_PATH}")
            # Create a basic error page
            error_html = """
            <html><body style="background-color: #111827; color: #FFFFFF; font-family: Arial;">
                <h1>Error: Menu Template Not Found</h1>
                <p>The menu interface template could not be loaded.</p>
                <button onclick="launchWizard()">Launch Research Wizard Anyway</button>
            </body></html>
            """
            self.web_view.setHtml(error_html, QUrl.fromLocalFile(str(PACKAGE_DIR)))
        else:
            # Load the HTML file with proper base URL
            self.logger.debug(f"Loading menu HTML from: {MENU_HTML_PATH}")
            url = QUrl.fromLocalFile(str(MENU_HTML_PATH.absolute()))
            self.web_view.setUrl(url)
            self.logger.debug(f"Set URL to: {url.toString()}")
    
    def setup_theme(self):
        """Set up dark theme palette."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#1F2937"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#374151"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#FFFFFF"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#374151"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#FFFFFF"))
        self.setPalette(palette)
    
    def _handle_js_console(self, level, message, line_number, source_id):
        """Handle messages from JavaScript."""
        if "launchWizard" in message:
            self.launch_wizard.emit()
