import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView  # Required for HTML menu
from .ui_components.menu_module import AgentChefMenu
from .ui_components.ui_module import RagchefUI
from .research_manager import ResearchManager

def main():
    app = QApplication(sys.argv)
    
    # Create research manager
    manager = ResearchManager()
    
    # Create menu first
    menu = AgentChefMenu()
    
    # Create wizard but don't show yet
    wizard = RagchefUI(manager)
    
    # Connect menu signal to show wizard and hide menu
    def on_launch_wizard():
        wizard.show()
        menu.hide()
    
    menu.launch_wizard.connect(on_launch_wizard)
    
    # Show menu first
    menu.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
