import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from websockets.exceptions import ConnectionClosed

from agentChef.mcp.mcp_server import MCPServer

@pytest.fixture
def mcp_server():
    """Create an MCP server instance for testing."""
    server = MCPServer()
    return server

@pytest.fixture
def test_client(mcp_server):
    """Create a test client for the FastAPI app."""
    return TestClient(mcp_server.app)

async def test_websocket_connection(test_client):
    """Test WebSocket connection and basic message handling."""
    with test_client.websocket_connect("/mcp") as websocket:
        # Test research command
        await websocket.send_json({
            "command": "research",
            "params": {"topic": "test"}
        })
        response = await websocket.receive_json()
        assert "result" in response

async def test_mcp_research_handling(mcp_server):
    """Test research command handling."""
    with patch('agentChef.api.agent_chef_api.research_manager') as mock_manager:
        mock_manager.research_topic.return_value = {"status": "success"}
        
        response = await mcp_server.handle_request({
            "command": "research",
            "params": {"topic": "test"}
        })
        
        assert "result" in response
        assert response["result"]["status"] == "success"

async def test_mcp_error_handling(mcp_server):
    """Test error handling in MCP server."""
    response = await mcp_server.handle_request({
        "command": "invalid",
        "params": {}
    })
    
    assert "error" in response
    assert "Unknown command" in response["error"]
