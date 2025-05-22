from pathlib import Path
from oarc_log import setup_logging, get_logger

def init_logging():
    """Initialize centralized logging."""
    log_dir = Path.home() / ".agentchef" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(
        log_file=log_dir / "agentchef.log",
        level="INFO"
    )
    
    return get_logger("agentchef")