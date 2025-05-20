import pytest
from unittest.mock import patch, MagicMock
import logging

from agentChef.logs.agentchef_logging import setup_file_logging, log

def test_setup_file_logging():
    """Test setting up file logging."""
    with patch('logging.FileHandler') as mock_handler:
        setup_file_logging("test.log")
        mock_handler.assert_called_once_with("test.log")

def test_log_levels():
    """Test different logging levels."""
    with patch('logging.Logger.debug') as mock_debug, \
         patch('logging.Logger.info') as mock_info, \
         patch('logging.Logger.warning') as mock_warning, \
         patch('logging.Logger.error') as mock_error:
        
        log.debug("test debug")
        log.info("test info")
        log.warning("test warning")
        log.error("test error")
        
        mock_debug.assert_called_once_with("test debug")
        mock_info.assert_called_once_with("test info")
        mock_warning.assert_called_once_with("test warning")
        mock_error.assert_called_once_with("test error")