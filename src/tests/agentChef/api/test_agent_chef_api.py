import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from agentChef.api.agent_chef_api import app, research_manager

client = TestClient(app)

@pytest.fixture
def mock_research_manager():
    """Fixture to provide a mocked ResearchManager."""
    with patch('agentChef.api.agent_chef_api.research_manager') as mock:
        yield mock

def test_research_topic(mock_research_manager):
    """Test the research topic endpoint."""
    mock_research_manager.research_topic.return_value = {
        "topic": "test topic",
        "papers": []
    }
    
    response = client.post(
        "/research/topic",
        json={
            "topic": "test topic",
            "max_papers": 5
        }
    )
    
    assert response.status_code == 200
    assert response.json()["topic"] == "test topic"

def test_generate_conversation(mock_research_manager):
    """Test the generate conversation endpoint."""
    mock_research_manager.generate_conversation_dataset.return_value = {
        "conversations": []
    }
    
    response = client.post(
        "/generate/conversation",
        json={
            "content": "test content",
            "num_turns": 3
        }
    )
    
    assert response.status_code == 200

def test_generate_dataset(mock_research_manager):
    """Test dataset generation endpoint."""
    mock_response = {
        "conversations": [
            [{"from": "human", "value": "test"}]
        ]
    }
    mock_research_manager.generate_conversation_dataset.return_value = mock_response
    
    response = client.post(
        "/generate/dataset",
        json={
            "content": "test content",
            "num_turns": 3
        }
    )
    
    assert response.status_code == 200
    assert "conversations" in response.json()

def test_expand_dataset():
    """Test dataset expansion endpoint."""
    with patch('agentChef.api.agent_chef_api.dataset_expander') as mock_expander:
        mock_expander.expand_conversation_dataset.return_value = [
            [{"from": "human", "value": "expanded"}]
        ]
        
        response = client.post(
            "/expand/dataset",
            json={
                "conversations": [[{"from": "human", "value": "test"}]],
                "expansion_factor": 2
            }
        )
        
        assert response.status_code == 200
        assert len(response.json()) > 0

def test_clean_dataset():
    """Test dataset cleaning endpoint."""
    with patch('agentChef.api.agent_chef_api.dataset_cleaner') as mock_cleaner:
        mock_cleaner.clean_dataset.return_value = [
            [{"from": "human", "value": "cleaned"}]
        ]
        
        response = client.post(
            "/clean/dataset",
            json={
                "conversations": [[{"from": "human", "value": "test"}]],
                "criteria": {"fix_hallucinations": True}
            }
        )
        
        assert response.status_code == 200

def test_error_handling():
    """Test API error handling."""
    response = client.post(
        "/research/topic",
        json={"invalid": "data"}
    )
    assert response.status_code == 422  # Validation error
