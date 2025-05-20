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
from agentChef.core.generation.conversation_generator import OllamaConversationGenerator
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
    
    def __init__(self, data_dir: Optional[str] = None, name: str = "OARC-Crawlers", port: int = 3000):
        """Initialize the OARC-Crawlers MCP wrapper."""
        self.data_dir = data_dir
        self.port = port
        self.name = name  # Store the name as a property on the server
        
        # Initialize MCP server with required configuration
        self.mcp = FastMCP(
            name=name,
            dependencies=[
            "pytube",         # YouTube downloading
            "beautifulsoup4", # Web crawling/parsing
            "pandas",         # Data handling
            "pyarrow",        # Parquet storage
            "aiohttp",        # Async HTTP requests
            "gitpython",      # GitHub repo cloning
            "pytchat"         # YouTube live chat
            ],
            description="OARC's dynamic webcrawler module collection providing YouTube, GitHub, DuckDuckGo, web crawling, and ArXiv paper extraction capabilities.",
            version=VERSION
        )
        
        # Initialize our creepy crawlers
        self.youtube = YTCrawler(data_dir=data_dir)
        self.github = GHCrawler(data_dir=data_dir)
        self.ddg = DDGCrawler(data_dir=data_dir)
        self.bs = WebCrawler(data_dir=data_dir)
        self.arxiv = ArxivCrawler(data_dir=data_dir)
        
        # Register OARC-Crawler MCP tools
        self._register_tools()

    def _register_tools(self):
        """Register all crawler tools with MCP."""
        # --- YouTube tools ---
        @self.mcp.tool(
            name="download_youtube_video",
            description="Download a YouTube video with specified format and resolution."
        )
        async def download_youtube_video(url: str, format: str = "mp4", 
                                      resolution: str = "highest") -> Dict:
            """Download a YouTube video."""
            try:
                return await self.youtube.download_video(url, format, resolution)
            except Exception as e:
                log.error(f"Error downloading video: {e}")
                return {"error": str(e)}
            
        @self.mcp.tool()
        async def download_youtube_playlist(playlist_url: str, 
                                         max_videos: int = 10) -> Dict:
            """Download videos from a YouTube playlist."""
            return await self.youtube.download_playlist(playlist_url, 
                                                      max_videos=max_videos)
            
        @self.mcp.tool()
        async def extract_youtube_captions(url: str, 
                                        languages: List[str] = ["en"]) -> Dict:
            """Extract captions from a YouTube video."""
            return await self.youtube.extract_captions(url, languages)
        
        # --- GitHub tools ---
        @self.mcp.tool()
        async def clone_github_repo(repo_url: str) -> str:
            """Clone and analyze a GitHub repository."""
            return await self.github.clone_and_store_repo(repo_url)
            
        @self.mcp.tool()
        async def analyze_github_repo(repo_url: str) -> str:
            """Get a summary analysis of a GitHub repository."""
            return await self.github.get_repo_summary(repo_url)
            
        @self.mcp.tool()
        async def find_similar_code(repo_url: str, code_snippet: str) -> str:
            """Find similar code in a GitHub repository."""
            return await self.github.find_similar_code(repo_url, code_snippet)
        
        # --- DuckDuckGo tools ---
        @self.mcp.tool()
        async def ddg_text_search(query: str, max_results: int = 5) -> str:
            """Perform a DuckDuckGo text search."""
            return await self.ddg.text_search(query, max_results)
            
        @self.mcp.tool()
        async def ddg_image_search(query: str, max_results: int = 10) -> str:
            """Perform a DuckDuckGo image search."""
            return await self.ddg.image_search(query, max_results)
            
        @self.mcp.tool()
        async def ddg_news_search(query: str, max_results: int = 20) -> str:
            """Perform a DuckDuckGo news search."""
            return await self.ddg.news_search(query, max_results)
        
        # ---  Web tools ---
        @self.mcp.tool()
        async def crawl_webpage(url: str) -> str:
            """Crawl and extract content from a webpage."""
            return await self.bs.fetch_url_content(url)
            
        @self.mcp.tool()
        async def crawl_documentation(url: str) -> str:
            """Crawl and extract content from a documentation site."""
            return await self.bs.crawl_documentation_site(url)
        
        # --- ArXiv tools ---
        @self.mcp.tool()
        async def fetch_arxiv_paper(arxiv_id: str) -> Dict:
            """Fetch paper information from ArXiv."""
            return await self.arxiv.fetch_paper_info(arxiv_id)
            
        @self.mcp.tool()
        async def download_arxiv_source(arxiv_id: str) -> Dict:
            """Download LaTeX source files for an ArXiv paper."""
            return await self.arxiv.download_source(arxiv_id)

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
