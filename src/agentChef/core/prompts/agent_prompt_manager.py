"""agent_prompt_manager.py
Manages agent-specific prompts and templates for AgentChef system.
Provides abstraction for loading, storing, and retrieving prompts for different agent types.

Written By: @BorcherdingL
Date: 6/29/2025
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class AgentPromptManager:
    """Manages agent-specific prompts and profiles."""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize the agent prompt manager."""
        self.prompts_dir = Path(prompts_dir) if prompts_dir else Path("./agent_prompts")
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        self.agents_dir = self.prompts_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates_dir = self.prompts_dir / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded profiles
        self._profile_cache = {}
        
    def create_agent_profile(self, agent_id: str, system_prompt: str, 
                           description: str = None, **kwargs) -> bool:
        """
        Create or update an agent profile.
        
        Args:
            agent_id: Unique identifier for the agent
            system_prompt: System prompt for the agent
            description: Description of the agent's purpose
            **kwargs: Additional agent metadata
            
        Returns:
            bool: True if successful
        """
        try:
            profile = {
                "agent_id": agent_id,
                "system_prompt": system_prompt,
                "description": description or f"Agent: {agent_id}",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                **kwargs
            }
            
            # Save to file
            profile_file = self.agents_dir / f"{agent_id}.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._profile_cache[agent_id] = profile
            
            return True
            
        except Exception as e:
            print(f"Error creating agent profile: {e}")
            return False
    
    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        """
        Get an agent profile.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dict containing agent profile or empty dict
        """
        try:
            # Check cache first
            if agent_id in self._profile_cache:
                return self._profile_cache[agent_id]
            
            # Load from file
            profile_file = self.agents_dir / f"{agent_id}.json"
            if profile_file.exists():
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    self._profile_cache[agent_id] = profile
                    return profile
            
            return {}
            
        except Exception as e:
            print(f"Error getting agent profile: {e}")
            return {}
    
    def list_agent_profiles(self) -> List[str]:
        """List all available agent profiles."""
        try:
            profiles = []
            for file_path in self.agents_dir.glob("*.json"):
                agent_id = file_path.stem
                profiles.append(agent_id)
            return profiles
        except Exception as e:
            print(f"Error listing agent profiles: {e}")
            return []

# Example usage and default agent configurations
DEFAULT_AGENTS = {
    "ragchef": {
        "domain": "research",
        "personality": "analytical",
        "expertise": ["research", "paper analysis", "knowledge extraction"],
        "prompts": {
            "paper_analysis": """
            You are RAGChef, a research analysis agent specializing in academic papers.
            
            Query: {query}
            
            Instructions for paper analysis:
            - Extract key findings, methodologies, and contributions
            - Identify relevant citations and related work
            - Assess the quality and significance of the research
            - Generate conversation topics for educational purposes
            
            DataFrame info: {df_info}
            
            Generate pandas code for research analysis:
            """
        }
    },
    "conversationchef": {
        "domain": "conversation",
        "personality": "engaging",
        "expertise": ["dialogue generation", "conversation patterns", "engagement"],
        "prompts": {
            "dialogue_analysis": """
            You are ConversationChef, specializing in dialogue and conversation analysis.
            
            Query: {query}
            
            Instructions for conversation analysis:
            - Analyze conversation flow and engagement patterns
            - Identify successful dialogue strategies
            - Measure response quality and relevance
            - Generate insights for conversation improvement
            
            DataFrame info: {df_info}
            
            Generate pandas code for conversation analysis:
            """
        }
    }
}

def initialize_default_agents(prompt_manager: AgentPromptManager) -> None:
    """Initialize the prompt manager with default agent configurations."""
    for agent_name, config in DEFAULT_AGENTS.items():
        prompt_manager.register_agent(agent_name, config)
    logger.info("Initialized default agents")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create prompt manager
    pm = AgentPromptManager()
    
    # Initialize default agents
    initialize_default_agents(pm)
    
    # Test getting a prompt
    prompt = pm.get_prompt(
        "ragchef", 
        "paper_analysis",
        query="What are the main findings in this dataset?",
        df_info="Dataset contains research papers with titles, abstracts, and citations"
    )
    
    print("Generated prompt:")
    print(prompt)
