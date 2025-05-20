"""
Build utilities for AgentChef package.

This module provides static utility methods for cleaning build artifacts,
building the package, and publishing to PyPI or TestPyPI. It supports both
synchronous and asynchronous operations, and handles cross-platform cleanup
of build directories.
"""

import os
import sys
import subprocess
import asyncio
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BuildUtils:
    """
    Static utility methods for building, cleaning, and publishing AgentChef package.

    This class provides methods to:
      - Clean build artifacts (dist, build, egg-info)
      - Build the package using PEP 517 build backend
      - Publish the package to PyPI or TestPyPI using Twine (supports async)
    All methods are cross-platform and handle errors gracefully.
    """
    

    @staticmethod
    def clean_build_directories():
        """Clean build, dist and egg directories."""
        try:
            dirs_to_clean = ['build', 'dist', '*.egg-info']
            for d in dirs_to_clean:
                paths = Path('.').glob(d)
                for path in paths:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error cleaning directories: {e}")
            return False
    

    @staticmethod
    def build_package():
        """Build the package using pyproject.toml."""
        try:
            subprocess.run(['python', '-m', 'build'], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error building package: {e}")
            return False


    @staticmethod
    async def publish_package(test=False, username=None, password=None, config_file=None):
        """Publish the package to PyPI.
        
        Args:
            test (bool): If True, upload to TestPyPI instead of PyPI
            username (str): PyPI username (if not using keyring or config file)
            password (str): PyPI password (if not using keyring or config file)
            config_file (str): Path to PyPI config file (.pypirc)
            
        Returns:
            dict: Result of the operation with keys 'success' and 'message'
            
        Raises:
            RuntimeError: If the upload fails
        """
        try:
            cmd = ["twine", "upload"]
            if test:
                cmd.extend(["--repository", "testpypi"])
            
            # Add authentication if provided
            if username:
                cmd.extend(["--username", username])
            if password:
                cmd.extend(["--password", password])
            if config_file:
                cmd.extend(["--config-file", config_file])
                
            cmd.extend(["dist/*"])
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                return {
                    "success": True,
                    "message": "Package published successfully!",
                    "output": stdout.decode()
                }
            else:
                error_msg = stderr.decode()
                return {
                    "success": False,
                    "message": f"Error publishing package: {error_msg}",
                    "error": error_msg
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error publishing package: {str(e)}",
                "error": str(e)
            }
