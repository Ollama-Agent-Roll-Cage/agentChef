import pytest
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

from agentChef.config.config import Config
from agentChef.config.config_manager import ConfigManager
from agentChef.utils.const import (
    CONFIG_KEY_DATA_DIR,
    CONFIG_KEY_LOG_LEVEL,
    CONFIG_KEY_MAX_RETRIES
)

@pytest.fixture
def temp_config_file(tmp_path):
    """Fixture to provide a temporary config file."""
    config_file = tmp_path / "test_config.ini"
    return config_file

def test_config_initialization():
    """Test basic config initialization."""
    config = Config()
    assert config.data_dir is not None
    assert config.log_level is not None

def test_load_config_file(temp_config_file):
    """Test loading configuration from a file."""
    temp_config_file.write_text("""
[agentchef]
log_level = DEBUG
max_retries = 5
    """)
    
    Config.load_from_file(str(temp_config_file))
    config = Config()
    assert config.log_level == "DEBUG"
    assert config.max_retries == 5

def test_environment_variables():
    """Test environment variable overrides."""
    with patch.dict(os.environ, {
        'AGENTCHEF_LOG_LEVEL': 'DEBUG',
        'AGENTCHEF_MAX_RETRIES': '10'
    }):
        config = Config()
        assert config.log_level == "DEBUG"
        assert config.max_retries == 10

def test_config_manager():
    """Test ConfigManager operations."""
    manager = ConfigManager()
    
    # Test getting current config
    current_config = manager.get_current_config()
    assert isinstance(current_config, dict)
    assert CONFIG_KEY_DATA_DIR in current_config
    
    # Test getting config sources
    sources = manager.get_config_source()
    assert isinstance(sources, dict)
    assert CONFIG_KEY_LOG_LEVEL in sources
