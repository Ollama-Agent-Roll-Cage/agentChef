import sys
import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QTimer
try:
    from PyQt6.QtWebEngineCore import QWebEngineProfile
    from PyQt6.QtWebChannel import QWebChannel
    HAS_QT = True
except ImportError:
    HAS_QT = False
    logging.warning("PyQt6/WebEngine not installed. Install with: pip install PyQt6 PyQt6-WebEngine")
from agentChef.core.ui_components.menu_module import AgentChefMenu
from agentChef.core.ui_components.RagchefUI.ui_module import RagchefUI
from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.utils.const import DEFAULT_DATA_DIR

def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Ensure required directories exist
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up basic logging first
    setup_logging()
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Enable high DPI support
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    
    try:
        # Initialize WebEngine
        if not QWebEngineProfile.defaultProfile():
            QWebEngineProfile()
        
        # Create research manager
        manager = ResearchManager()
        
        # Create menu
        menu = AgentChefMenu()
        
        # Create wizard but don't show yet
        wizard = RagchefUI(manager)
        
        # Connect menu signal to show wizard and hide menu
        def on_launch_wizard():
            print("DEBUG: Launching RAGChef wizard")
            wizard.show()
            menu.hide()
        
        # Connect the signal after menu is fully initialized
        QTimer.singleShot(100, lambda: menu.bridge.launchWizard.connect(on_launch_wizard))
        
        # Show menu
        menu.show()
        
        return app.exec()
        
    except Exception as e:
        logging.exception("Error in main")
        raise

if __name__ == "__main__":
    sys.exit(main())
