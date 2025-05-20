import pytest
from unittest.mock import patch, MagicMock
from PyQt6.QtWidgets import QApplication

from agentChef.core.ui_components.RagchefUI.ui_module import RagchefUI
from agentChef.core.ui_components.menu.agentChefMenu import AgentChefMenu

@pytest.fixture
def app():
    """Create QApplication instance for testing."""
    return QApplication([])

def test_ragchef_ui_initialization(app):
    """Test RagchefUI initialization."""
    with patch('agentChef.core.ui_components.RagchefUI.ui_module.ResearchManager') as mock_manager:
        ui = RagchefUI()
        assert ui.research_manager == mock_manager.return_value

def test_menu_initialization(app):
    """Test AgentChefMenu initialization."""
    with patch('agentChef.core.ui_components.menu.agentChefMenu.ConfigManager') as mock_config:
        menu = AgentChefMenu()
        assert menu.config_manager == mock_config.return_value
