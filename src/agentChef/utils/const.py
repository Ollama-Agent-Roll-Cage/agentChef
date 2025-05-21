"""Constants used throughout AgentChef."""
import os
from pathlib import Path

# MCP Protocol
MCP_PORT = 50505

# Configuration
CONFIG_SECTION = "agentchef"
DEFAULT_CONFIG_FILENAME = "agentchef.ini"

# Config keys
CONFIG_KEY_DATA_DIR = "data_dir"
CONFIG_KEY_LOG_LEVEL = "log_level"
CONFIG_KEY_MAX_RETRIES = "max_retries"
CONFIG_KEY_TIMEOUT = "timeout"
CONFIG_KEY_USER_AGENT = "user_agent"
CONFIG_KEY_GITHUB_TOKEN = "github_token"

# Default values
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
DEFAULT_USER_AGENT = "AgentChef/1.0"

# Environment variables 
ENV_DATA_DIR = "AGENTCHEF_DATA_DIR"
ENV_LOG_LEVEL = "AGENTCHEF_LOG_LEVEL"
ENV_MAX_RETRIES = "AGENTCHEF_MAX_RETRIES"
ENV_TIMEOUT = "AGENTCHEF_TIMEOUT"
ENV_USER_AGENT = "AGENTCHEF_USER_AGENT"
ENV_GITHUB_TOKEN = "AGENTCHEF_GITHUB_TOKEN"

# Data directories
DEFAULT_DATA_DIR = Path(os.getenv("AGENTCHEF_DATA_DIR", Path.home() / ".agentchef"))
PAPERS_DIR = DEFAULT_DATA_DIR / "papers"
DATASETS_DIR = DEFAULT_DATA_DIR / "datasets"
ASSETS_DIR = DEFAULT_DATA_DIR / "assets"
LOGS_DIR = DEFAULT_DATA_DIR / "logs"

# Create required directories
for dir_path in [DEFAULT_DATA_DIR, PAPERS_DIR, DATASETS_DIR, ASSETS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Return codes
SUCCESS = 0
ERROR = 1

# Base directories
PACKAGE_DIR = Path(__file__).parent.parent
UI_DIR = PACKAGE_DIR / 'core' / 'ui_components'
MENU_DIR = UI_DIR / 'menu'
MENU_HTML_PATH = MENU_DIR / 'agentChefMenu.html'

# Create required directories
for directory in [DEFAULT_DATA_DIR, UI_DIR, MENU_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'model_name': 'llama2',
    'enable_ui': True,
    'data_dir': str(DEFAULT_DATA_DIR),
    'log_level': 'INFO'
}