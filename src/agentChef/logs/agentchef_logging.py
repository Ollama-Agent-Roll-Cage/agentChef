"""
Centralized logging configuration for agentChef using OARC-Log.
Other modules should import from this module instead of directly from oarc_log.
"""

from oarc_log import log, enable_debug_logging
import logging
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO") -> None:
    """Set up basic logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    try:
        # Convert string to logging level
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure basic logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handle oarc_log's ContextAwareLogger differently
        # Get the underlying logger instead of trying to set level on ContextAwareLogger
        if hasattr(log, '_logger'):
            # If oarc_log wraps a standard logger
            log._logger.setLevel(level)
        elif hasattr(log, 'logger'):
            # Alternative attribute name
            log.logger.setLevel(level)
        else:
            # Fallback: set level on root logger
            logging.getLogger().setLevel(level)
        
        if log_level.upper() == "DEBUG":
            enable_debug_logging()
            
    except Exception as e:
        print(f"Warning: Could not set up logging: {e}")

def setup_file_logging(log_dir: str, filename: str = "agentchef.log") -> None:
    """Configure logging to write to a file in the specified directory.
    
    Args:
        log_dir (str): Directory to store log files
        filename (str): Name of the log file
    """
    try:
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure logging to file
        file_handler = logging.FileHandler(
            Path(log_dir) / filename,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
        
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")

def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"agentchef.{module_name}")

def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug logging."""
    if enabled:
        enable_debug_logging()
        # Set standard logging to DEBUG as well
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Set standard logging to INFO
        logging.getLogger().setLevel(logging.INFO)

# Re-export commonly used functions
__all__ = ['log', 'enable_debug_logging', 'setup_logging', 'setup_file_logging', 'get_module_logger', 'set_debug']
