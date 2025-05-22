"""
OARC-Crawlers MCP Tools

This module provides a unified Model Context Protocol (MCP) API wrapper for the OARC-Crawlers package.
It exposes YouTube, GitHub, DuckDuckGo, web crawling, and ArXiv tools as MCP-compatible endpoints,
enabling seamless integration with FastMCP servers and VS Code extensions.

Author: @Borcherdingl, RawsonK
Date: 2025-04-18
"""

import sys
import asyncio
from typing import Dict, List, Optional

from fastmcp import FastMCP
from aiohttp.client_exceptions import ClientError

from oarc_log import log
from oarc_utils.decorators import singleton
from oarc_utils.errors import (
    MCPError,
    TransportError
)

from oarc_crawlers.core import (
    YTCrawler,
    GHCrawler,
    DDGCrawler,
    WebCrawler,
    ArxivCrawler,
)
from oarc_crawlers.utils.const import FAILURE, VERSION

import logging
import json
from pathlib import Path
from typing import Any

from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.core.ollama.ollama_interface import OllamaInterface
from agentChef.utils.mcp_utils import MCPUtils
from agentChef.utils.const import MCP_PORT

logger = logging.getLogger(__name__)

@singleton
class MCPServer:
    """
    MCPTools provides a unified Model Context Protocol (MCP) API for OARC-Crawlers.

    This class exposes YouTube, GitHub, DuckDuckGo, web crawling, and ArXiv tools as
    MCP-compatible endpoints, enabling seamless integration with FastMCP servers,
    VS Code extensions, and other MCP clients.

    It handles initialization, tool registration, server management, and installation
    for streamlined deployment and developer experience.
    """
    
    def __init__(self, data_dir: Optional[str] = None, name: str = "AgentChef", port: int = MCP_PORT):
        self.data_dir = data_dir
        self.port = port
        self.name = name
        
        # Initialize components using AgentChef's classes
        self.ollama = OllamaInterface()
        self.research_manager = ResearchManager(data_dir=data_dir)
        
        # Initialize MCP server
        self.mcp = FastMCP(
            name=name,
            dependencies=self._get_dependencies(),
            description="AgentChef MCP server providing research, dataset generation, and analysis capabilities"
        )
        
        self._register_tools()

    def _get_dependencies(self) -> List[str]:
        return [
            "agentchef",
            "ollama",
            "pandas",
            "pyarrow",
            "PyQt6",
            "aiohttp"
        ]

    def _register_tools(self):
        # Register research tools using AgentChef's ResearchManager
        @self.mcp.tool()
        async def research_topic(topic: str, max_papers: int = 5) -> Dict:
            """Research a topic using multiple sources."""
            return await self.research_manager.research_topic(topic, max_papers)

        @self.mcp.tool()
        async def generate_dataset(content: str, num_turns: int = 3) -> Dict:
            """Generate a conversation dataset."""
            return await self.research_manager.generate_conversation_dataset(
                papers=[{"content": content}],
                num_turns=num_turns
            )

        # Register analysis tools
        @self.mcp.tool()
        async def analyze_dataset(dataset_path: str) -> Dict:
            """Analyze a dataset using natural language queries."""
            return await self.research_manager.analyze_dataset(dataset_path)

    async def start_server(self):
        """Start the MCP server"""
        try:
            # Create or update .vscode/mcp.json configuration
            try:
                self._update_vscode_config()
            except Exception as e:
                log.warning(f"Failed to update VS Code configuration: {e}")
                
            # Start server using FastMCP's run method
            log.info(f"Starting server on port {self.port}")
            self.mcp.run(
                port=self.port,
                transport="ws"  # Use WebSocket transport for VS Code
            )
            log.info(f"MCP server started on port {self.port}")
            
            while True:
                await asyncio.sleep(1)
                
        except ClientError as e:
            log.error(f"Client error: {e}")
            raise TransportError(f"Connection error: {e}")
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            raise MCPError(f"MCP server error: {e}")

    def _update_vscode_config(self):
        """Create or update .vscode/mcp.json for VS Code integration"""
        import os
        import json
        
        try:
            # Find the project root directory (where .vscode would typically be)
            current_dir = os.path.abspath(os.path.curdir)
            vscode_dir = os.path.join(current_dir, ".vscode")
            
            # Create .vscode directory if it doesn't exist
            if not os.path.exists(vscode_dir):
                os.makedirs(vscode_dir)
                log.debug(f"Created .vscode directory at {vscode_dir}")
                
            # Path to mcp.json config file
            config_file = os.path.join(vscode_dir, "mcp.json")
            
            # Create or update the configuration
            config = {}
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                except json.JSONDecodeError:
                    log.warning(f"Existing mcp.json at {config_file} is invalid, creating new one")
            
            # Update the servers configuration
            if "servers" not in config:
                config["servers"] = {}
                
            config["servers"][self.name] = {
                "type": "ws",
                "url": f"ws://localhost:{self.port}"
            }
            
            # Write the updated configuration
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
                
            log.info(f"Updated VS Code configuration at {config_file}")
        except Exception as e:
            log.warning(f"Failed to update VS Code configuration: {e}")

    def run(self, transport: str = "ws", **kwargs):
        """Run the MCP server."""
        try:
            return asyncio.run(self.start_server())
        except KeyboardInterrupt:
            log.error("Server stopped by user")
            sys.exit(FAILURE)
        except (TransportError, MCPError) as e:
            log.error(f"MCP server error: {e}")
            raise MCPError(f"MCP server error: {e}")
        except Exception as e:
            log.error(f"Unexpected error in MCP server: {str(e)}")
            raise MCPError(f"Unexpected error in MCP server: {str(e)}")
    
    def install(self, name: str = None):
        """Install the MCP server for VS Code integration."""
        from oarc_crawlers.utils.mcp_utils import MCPUtils
        
        # Create a wrapper script with a global 'server' variable that FastMCP can find
        script_content = f"""
from oarc_crawlers.core.mcp.mcp_server import MCPServer

# Create server as a global variable - this is what FastMCP looks for
server = MCPServer(name="{self.name}")

if __name__ == "__main__":
    server.run()
"""
        
        return MCPUtils.install_mcp_with_content(
            script_content=script_content,
            name=name, 
            mcp_name=self.name,
            dependencies=self.mcp.dependencies
        )

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle an MCP client connection."""
        while True:
            try:
                data = await reader.read(4096)
                if not data:
                    break
                    
                request = json.loads(data.decode())
                response = await self.handle_request(request)
                
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
                
            except Exception as e:
                logger.error(f"Error handling MCP request: {e}")
                error_response = {"error": str(e)}
                writer.write(json.dumps(error_response).encode() + b"\n")
                await writer.drain()
                break
                
        writer.close()
        await writer.wait_closed()
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        command = request.get("command")
        params = request.get("params", {})
        
        handlers = {
            "research": self._handle_research,
            "generate": self._handle_generate,
            "expand": self._handle_expand,
            "clean": self._handle_clean,
            "chat": self._handle_chat
        }
        
        handler = handlers.get(command)
        if handler:
            return await handler(params)
        else:
            return {"error": f"Unknown command: {command}"}
            
    async def _handle_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle research requests."""
        try:
            result = await self.research_manager.research_topic(**params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    async def _handle_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generation requests."""
        try:
            result = await self.research_manager.generate_conversation_dataset(**params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    async def _handle_expand(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dataset expansion requests."""
        try:
            result = await self.research_manager.expand_dataset(**params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    async def _handle_clean(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dataset cleaning requests."""
        try:
            result = await self.research_manager.clean_dataset(**params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    async def start(self):
        """Start the MCP server."""
        self.server = await asyncio.start_server(
            self.handle_client, 
            self.host, 
            self.port
        )
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f'MCP server started on {addr}')
        
        async with self.server:
            await self.server.serve_forever()
            
    def stop(self):
        """Stop the MCP server."""
        if self.server:
            self.server.close()
            logger.info("MCP server stopped")
