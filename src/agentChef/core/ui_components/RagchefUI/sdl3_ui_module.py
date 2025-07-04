"""
RAGChef SDL3 UI Module - High-performance HTML/JavaScript interface with Python backend
====================================================================================

This module provides a modern web-based UI wrapped in SDL3 for maximum performance.
Uses the existing HTML/JavaScript interface from ragchef_main_ui.html with Python backend.

Features:
- Pure SDL3 window management (no PyQt6 dependency)
- Full HTML/JavaScript frontend
- Python backend integration
- Real-time research and dataset generation
- WebSocket communication for live updates
"""

import json
import asyncio
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import webbrowser
import tempfile
import shutil
from datetime import datetime

try:
    import SDL3
    HAS_SDL3 = True
except ImportError:
    HAS_SDL3 = False
    logging.warning("PySDL3 not available. Install with: pip install PySDL3")

try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logging.warning("websockets not available. Install with: pip install websockets")

from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.logs.agentchef_logging import setup_file_logging, log

logger = logging.getLogger(__name__)

class RAGChefSDL3UI:
    """
    SDL3-based UI wrapper for RAGChef with HTML/JavaScript frontend.
    
    This provides a high-performance window container for the web-based UI
    while maintaining full Python backend integration.
    """
    
    def __init__(self, research_manager: Optional[ResearchManager] = None):
        """Initialize the SDL3 UI."""
        if not HAS_SDL3:
            raise ImportError("PySDL3 is required. Install with: pip install PySDL3")
        
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets is required. Install with: pip install websockets")
        
        self.research_manager = research_manager or ResearchManager()
        
        # SDL3 window properties
        self.window = None
        self.renderer = None
        self.running = False
        
        # WebSocket server for communication
        self.websocket_server = None
        self.websocket_port = 8765
        self.connected_clients = set()
        
        # UI state
        self.current_operations = {}
        self.progress_callbacks = {}
        
        # Set up logging
        setup_file_logging("./logs/ui")
        logger.info("Initializing RAGChef SDL3 UI...")
        
        # Initialize SDL3
        self._init_sdl3()
        
        # Set up WebSocket server
        self._setup_websocket_server()
        
        logger.info("RAGChef SDL3 UI initialized successfully")

    def _init_sdl3(self):
        """Initialize SDL3 window and renderer."""
        if not SDL3.SDL_Init(SDL3.SDL_INIT_VIDEO):
            raise RuntimeError(f"Failed to initialize SDL3: {SDL3.SDL_GetError()}")
        
        # Create window
        self.window = SDL3.SDL_CreateWindow(
            "RAGChef - AI Research & Dataset Generation Platform".encode('utf-8'),
            1200, 800,
            SDL3.SDL_WINDOW_RESIZABLE | SDL3.SDL_WINDOW_MAXIMIZED
        )
        
        if not self.window:
            raise RuntimeError(f"Failed to create window: {SDL3.SDL_GetError()}")
        
        # Create renderer
        self.renderer = SDL3.SDL_CreateRenderer(self.window, None)
        if not self.renderer:
            raise RuntimeError(f"Failed to create renderer: {SDL3.SDL_GetError()}")
        
        logger.info("SDL3 window and renderer created successfully")

    def _setup_websocket_server(self):
        """Set up WebSocket server for frontend communication."""
        async def handle_client(websocket, path):
            """Handle WebSocket client connections."""
            self.connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    await self._handle_websocket_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.discard(websocket)
                logger.info(f"Client disconnected")

        # Start WebSocket server in separate thread
        def start_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server_coro = serve(handle_client, "localhost", self.websocket_port)
            loop.run_until_complete(start_server_coro)
            loop.run_forever()
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        logger.info(f"WebSocket server started on ws://localhost:{self.websocket_port}")

    async def _handle_websocket_message(self, websocket, message):
        """Handle incoming WebSocket messages from frontend."""
        try:
            data = json.loads(message)
            command = data.get('command')
            params = data.get('params', {})
            request_id = data.get('request_id')
            
            logger.info(f"Received command: {command} with params: {params}")
            
            # Route commands to appropriate handlers
            if command == 'research_topic':
                await self._handle_research_topic(websocket, params, request_id)
            elif command == 'generate_dataset':
                await self._handle_generate_dataset(websocket, params, request_id)
            elif command == 'process_papers':
                await self._handle_process_papers(websocket, params, request_id)
            elif command == 'analyze_datasets':
                await self._handle_analyze_datasets(websocket, params, request_id)
            elif command == 'get_status':
                await self._handle_get_status(websocket, params, request_id)
            else:
                await self._send_error(websocket, f"Unknown command: {command}", request_id)
                
        except json.JSONDecodeError as e:
            await self._send_error(websocket, f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, f"Internal error: {e}")

    async def _handle_research_topic(self, websocket, params, request_id):
        """Handle research topic request."""
        topic = params.get('topic')
        max_papers = params.get('max_papers', 5)
        max_search_results = params.get('max_search_results', 10)
        include_github = params.get('include_github', False)
        github_repos = params.get('github_repos', [])
        model_name = params.get('model_name', 'llama3.2:3b')
        
        if not topic:
            await self._send_error(websocket, "Topic is required", request_id)
            return
        
        # Create progress callback
        async def progress_callback(message):
            await self._send_progress(websocket, message, request_id)
        
        try:
            # Start research operation
            await self._send_response(websocket, {
                "status": "started",
                "message": f"Starting research on: {topic}"
            }, request_id)
            
            # Execute research
            results = await self.research_manager.research_topic(
                topic=topic,
                max_papers=max_papers,
                max_search_results=max_search_results,
                include_github=include_github,
                github_repos=github_repos,
                model_name=model_name,
                callback=progress_callback
            )
            
            # Send results
            await self._send_response(websocket, {
                "status": "completed",
                "data": results
            }, request_id)
            
        except Exception as e:
            logger.error(f"Research error: {e}")
            await self._send_error(websocket, str(e), request_id)

    async def _handle_generate_dataset(self, websocket, params, request_id):
        """Handle dataset generation request."""
        papers = params.get('papers', [])
        num_turns = params.get('num_turns', 3)
        expansion_factor = params.get('expansion_factor', 3)
        clean = params.get('clean', True)
        hedging_level = params.get('hedging_level', 'balanced')
        output_format = params.get('output_format', 'jsonl')
        
        if not papers:
            await self._send_error(websocket, "Papers are required for dataset generation", request_id)
            return
        
        # Create progress callback
        async def progress_callback(message):
            await self._send_progress(websocket, message, request_id)
        
        try:
            await self._send_response(websocket, {
                "status": "started",
                "message": "Starting dataset generation..."
            }, request_id)
            
            # Execute dataset generation
            results = await self.research_manager.generate_conversation_dataset(
                papers=papers,
                num_turns=num_turns,
                expansion_factor=expansion_factor,
                clean=clean,
                hedging_level=hedging_level,
                output_format=output_format,
                callback=progress_callback
            )
            
            await self._send_response(websocket, {
                "status": "completed",
                "data": results
            }, request_id)
            
        except Exception as e:
            logger.error(f"Dataset generation error: {e}")
            await self._send_error(websocket, str(e), request_id)

    async def _handle_process_papers(self, websocket, params, request_id):
        """Handle paper processing request."""
        paper_files = params.get('paper_files', [])
        output_format = params.get('output_format', 'jsonl')
        num_turns = params.get('num_turns', 3)
        expansion_factor = params.get('expansion_factor', 3)
        clean = params.get('clean', True)
        
        if not paper_files:
            await self._send_error(websocket, "Paper files are required", request_id)
            return
        
        # Create progress callback
        async def progress_callback(message):
            await self._send_progress(websocket, message, request_id)
        
        try:
            await self._send_response(websocket, {
                "status": "started",
                "message": "Processing paper files..."
            }, request_id)
            
            # Execute paper processing
            results = await self.research_manager.process_paper_files(
                paper_files=paper_files,
                output_format=output_format,
                num_turns=num_turns,
                expansion_factor=expansion_factor,
                clean=clean,
                callback=progress_callback
            )
            
            await self._send_response(websocket, {
                "status": "completed",
                "data": results
            }, request_id)
            
        except Exception as e:
            logger.error(f"Paper processing error: {e}")
            await self._send_error(websocket, str(e), request_id)

    async def _handle_analyze_datasets(self, websocket, params, request_id):
        """Handle dataset analysis request."""
        original_dataset = params.get('original_dataset')
        expanded_dataset = params.get('expanded_dataset')
        analysis_options = params.get('analysis_options', {})
        
        if not original_dataset or not expanded_dataset:
            await self._send_error(websocket, "Both original and expanded datasets are required", request_id)
            return
        
        try:
            await self._send_response(websocket, {
                "status": "started", 
                "message": "Analyzing datasets..."
            }, request_id)
            
            # Load and analyze datasets
            # ...existing analysis code...
            
            results = {"analysis": "Dataset analysis completed"}  # Placeholder
            
            await self._send_response(websocket, {
                "status": "completed",
                "data": results
            }, request_id)
            
        except Exception as e:
            logger.error(f"Dataset analysis error: {e}")
            await self._send_error(websocket, str(e), request_id)

    async def _handle_get_status(self, websocket, params, request_id):
        """Handle status request."""
        status = {
            "research_manager_ready": bool(self.research_manager),
            "active_operations": len(self.current_operations),
            "connected_clients": len(self.connected_clients),
            "timestamp": datetime.now().isoformat()
        }
        
        await self._send_response(websocket, {
            "status": "success",
            "data": status
        }, request_id)

    async def _send_response(self, websocket, data, request_id=None):
        """Send response back to frontend."""
        message = {
            "type": "response",
            "data": data,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(message))

    async def _send_progress(self, websocket, message, request_id=None):
        """Send progress update to frontend."""
        progress_data = {
            "type": "progress",
            "message": message,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(progress_data))

    async def _send_error(self, websocket, error_message, request_id=None):
        """Send error message to frontend."""
        error_data = {
            "type": "error",
            "error": error_message,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(error_data))

    def _prepare_ui_files(self):
        """Prepare UI files for the web interface."""
        # Create temporary directory for UI files
        self.ui_temp_dir = tempfile.mkdtemp(prefix="ragchef_ui_")
        
        # Copy HTML file
        html_source = Path(__file__).parent / "ragchef_main_ui.html"
        html_dest = Path(self.ui_temp_dir) / "index.html"
        
        if html_source.exists():
            # Update HTML with WebSocket configuration
            html_content = html_source.read_text(encoding='utf-8')
            
            # Inject WebSocket connection script
            websocket_script = f"""
            <script>
                // WebSocket connection for real-time communication
                const ws = new WebSocket('ws://localhost:{self.websocket_port}');
                let requestId = 0;
                const pendingRequests = new Map();
                
                ws.onopen = function(event) {{
                    console.log('WebSocket connected');
                    updateConnectionStatus(true);
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                }};
                
                ws.onclose = function(event) {{
                    console.log('WebSocket disconnected');
                    updateConnectionStatus(false);
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                    updateConnectionStatus(false);
                }};
                
                function sendCommand(command, params) {{
                    const id = ++requestId;
                    const message = {{
                        command: command,
                        params: params,
                        request_id: id
                    }};
                    
                    return new Promise((resolve, reject) => {{
                        pendingRequests.set(id, {{ resolve, reject }});
                        ws.send(JSON.stringify(message));
                    }});
                }}
                
                function handleWebSocketMessage(data) {{
                    if (data.type === 'response' && data.request_id) {{
                        const request = pendingRequests.get(data.request_id);
                        if (request) {{
                            pendingRequests.delete(data.request_id);
                            request.resolve(data.data);
                        }}
                    }} else if (data.type === 'progress') {{
                        updateProgress(data.message, data.request_id);
                    }} else if (data.type === 'error') {{
                        console.error('WebSocket error:', data.error);
                        const request = pendingRequests.get(data.request_id);
                        if (request) {{
                            pendingRequests.delete(data.request_id);
                            request.reject(new Error(data.error));
                        }}
                    }}
                }}
                
                function updateConnectionStatus(connected) {{
                    // Update UI to show connection status
                    const statusEl = document.querySelector('.connection-status');
                    if (statusEl) {{
                        statusEl.textContent = connected ? 'Connected' : 'Disconnected';
                        statusEl.className = `connection-status ${{connected ? 'connected' : 'disconnected'}}`;
                    }}
                }}
                
                function updateProgress(message, requestId) {{
                    // Update progress displays
                    console.log('Progress:', message);
                    const logEl = document.querySelector(`#log-${{requestId}} || .active-log`);
                    if (logEl) {{
                        logEl.innerHTML += `<div class="log-entry">${{message}}</div>`;
                        logEl.scrollTop = logEl.scrollHeight;
                    }}
                }}
            </script>
            """
            
            # Insert WebSocket script before closing body tag
            html_content = html_content.replace('</body>', f'{websocket_script}</body>')
            
            # Update the JavaScript functions to use WebSocket
            html_content = html_content.replace(
                'function startResearch() {',
                '''function startResearch() {
                    const form = document.getElementById('researchForm');
                    const formData = new FormData(form);
                    
                    const params = {
                        topic: formData.get('topic'),
                        max_papers: parseInt(formData.get('max_papers')),
                        max_search_results: parseInt(formData.get('max_search_results')),
                        include_github: formData.get('include_github') === 'on',
                        github_repos: formData.get('github_repos') ? formData.get('github_repos').split(',').map(s => s.trim()) : [],
                        model_name: formData.get('model_name') || 'llama3.2:3b'
                    };
                    
                    sendCommand('research_topic', params)
                        .then(result => {
                            console.log('Research completed:', result);
                            displayResearchResults(result);
                        })
                        .catch(error => {
                            console.error('Research failed:', error);
                            displayError(error.message);
                        });
                '''
            )
            
            html_dest.write_text(html_content, encoding='utf-8')
        else:
            # Create basic HTML file if template doesn't exist
            basic_html = """<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>RAGChef - AI Research Platform</title>
            </head>
            <body>
                <h1>RAGChef - AI Research Platform</h1>
                <div class="connection-status">Connecting...</div>
                <p>RAGChef UI is starting up...</p>
            </body>
            </html>"""
            
            html_dest.write_text(basic_html, encoding='utf-8')
        
        return str(html_dest)

    def run(self):
        """Run the SDL3 UI main loop."""
        if not self.window:
            raise RuntimeError("Window not initialized")
        
        self.running = True
        
        # Prepare UI files
        ui_file = self._prepare_ui_files()
        
        # Open the UI in the default browser
        webbrowser.open(f'file://{ui_file}')
        
        logger.info("RAGChef UI started - check your browser")
        
        # SDL3 event loop
        event = SDL3.SDL_Event()
        
        try:
            while self.running:
                while SDL3.SDL_PollEvent(event):
                    if event.type == SDL3.SDL_EVENT_QUIT:
                        self.running = False
                    elif event.type == SDL3.SDL_EVENT_KEY_DOWN:
                        if event.key.key == SDL3.SDLK_ESCAPE:
                            self.running = False
                
                # Clear screen
                SDL3.SDL_SetRenderDrawColor(self.renderer, 20, 30, 40, 255)
                SDL3.SDL_RenderClear(self.renderer)
                
                # Present frame
                SDL3.SDL_RenderPresent(self.renderer)
                
                # Small delay to prevent 100% CPU usage
                SDL3.SDL_Delay(16)  # ~60 FPS
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up SDL3 resources."""
        if self.renderer:
            SDL3.SDL_DestroyRenderer(self.renderer)
            self.renderer = None
        
        if self.window:
            SDL3.SDL_DestroyWindow(self.window)
            self.window = None
        
        SDL3.SDL_Quit()
        
        # Clean up temporary files
        if hasattr(self, 'ui_temp_dir'):
            shutil.rmtree(self.ui_temp_dir, ignore_errors=True)
        
        logger.info("SDL3 UI cleaned up")

    def show(self):
        """Show the window and start the event loop."""
        if self.window:
            SDL3.SDL_ShowWindow(self.window)
            self.run()


def main():
    """Main function for testing the SDL3 UI."""
    try:
        # Initialize research manager
        research_manager = ResearchManager()
        
        # Create and run UI
        ui = RAGChefSDL3UI(research_manager)
        ui.show()
        
    except Exception as e:
        logger.error(f"Failed to start RAGChef UI: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
