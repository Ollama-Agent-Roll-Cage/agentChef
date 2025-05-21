"""
Centralized logging configuration for agentChef using OARC-Log.
Other modules should import from this module instead of directly from oarc_log.

Example usage:
    from agentChef.logs.agentchef_logging import log, setup_file_logging

    # Set up file logging if needed
    setup_file_logging("./logs")
    
    # Use different log levels throughout your code:
    
    # Debug - Detailed information for diagnosing problems
    log.debug("Processing item with parameters: x={}, y={}", x, y)
    log.debug("SQL query executed: %s", query)  # For detailed debugging info
    
    # Info - Confirmation that things are working as expected
    log.info("Started processing task")  # General progress info
    log.info("Successfully processed %d items", num_items)
    
    # Warning - An indication that something unexpected happened
    # or indicative of some problem in the near future
    log.warning("Config file missing, using default values")
    log.warning("API rate limit reached, waiting %d seconds", delay)
    
    # Error - The software has not been able to perform some function
    log.error("Failed to connect to database: %s", err)
    log.error("Unable to process file", exc_info=True)  # Include traceback
    
    # Critical - Program/application wide failures
    log.critical("System shutdown initiated due to hardware failure")
    log.critical("Unable to access required resources", exc_info=True)

Notes on Log Levels:
-------------------
debug:    Use for detailed diagnostic information
info:     Use for general operational information
warning:  Use for unexpected but recoverable issues
error:    Use for failures in specific operations
critical: Use for application-wide failures

Best Practices:
--------------
1. Always use appropriate log levels for the context
2. Include relevant variables and error details in messages
3. Use exc_info=True with error/critical for exceptions
4. Format messages using parameters, not f-strings
5. Keep messages clear and actionable
"""

from oarc_log import log, enable_debug_logging
import logging
from pathlib import Path
from typing import Optional

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
        import logging
        file_handler = logging.FileHandler(
            Path(log_dir) / filename,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Get the underlying logger from oarc_log
        logger = logging.getLogger('agentchef')
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}")

def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"agentchef.{module_name}")

def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug logging."""
    if enabled:
        enable_debug_logging()
    else:
        log.setLevel(logging.INFO)

# Re-export commonly used functions
__all__ = ['log', 'enable_debug_logging', 'setup_file_logging', 'get_module_logger', 'set_debug']
