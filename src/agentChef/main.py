import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from agentChef.core.ui_components.menu_module import AgentChefMenu
from agentChef.core.ui_components.RagchefUI.ui_module import RagchefUI
from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.utils.const import DEFAULT_DATA_DIR

def setup_logging():
    """Set up basic logging configuration."""
    import logging
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
    
    try:
        # Create research manager
        manager = ResearchManager()
        
        # Create menu first
        menu = AgentChefMenu()
        
        # Create wizard but don't show yet
        wizard = RagchefUI(manager)
        
        # Connect menu signal to show wizard and hide menu
        def on_launch_wizard():
            print("Launching RAGChef wizard")
            wizard.show()
            menu.hide()
        
        # Connect the signal
        menu.bridge.launchWizard.connect(on_launch_wizard)
        
        # Show menu first
        menu.show()
        
        return app.exec()
        
    except Exception as e:
        import logging
        logging.exception("Error in main")
        raise

if __name__ == "__main__":
    sys.exit(main())
