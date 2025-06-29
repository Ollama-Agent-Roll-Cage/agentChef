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
    """
    Manages prompts and templates for different agent types.
    Supports loading from files, dictionaries, and dynamic prompt generation.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the AgentPromptManager.
        
        Args:
            base_dir: Base directory for storing agent prompts and templates
        """
        self.base_dir = Path(base_dir) if base_dir else Path("data/agent_prompts")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for loaded prompts
        self.agent_prompts: Dict[str, Dict[str, Any]] = {}
        self.prompt_templates: Dict[str, str] = {}
        self.dynamic_prompts: Dict[str, Callable] = {}
        
        # Default prompt templates
        self._load_default_prompts()
        
    def _load_default_prompts(self):
        """Load default prompt templates for common agent types."""
        self.prompt_templates.update({
            "conversation_analysis": """
            You are analyzing conversation data for agent: {agent_name}.
            Dataset context: {context}
            
            Query: {query}
            
            Instructions:
            - Focus on conversation patterns and agent behavior
            - Identify key themes and topics discussed
            - Note conversation quality and engagement metrics
            - Provide insights relevant to agent training
            
            DataFrame info:
            {df_info}
            
            Generate pandas code to answer the query:
            """,
            
            "knowledge_extraction": """
            You are extracting knowledge for agent: {agent_name}.
            Knowledge domain: {domain}
            
            Query: {query}
            
            Instructions:
            - Extract factual information and key concepts
            - Identify relationships between different pieces of knowledge
            - Organize information in a structured format
            - Focus on information relevant to the agent's domain expertise
            
            DataFrame info:
            {df_info}
            
            Generate pandas code to extract and organize the knowledge:
            """,
            
            "template_generation": """
            You are generating conversation templates for agent: {agent_name}.
            Template type: {template_type}
            
            Query: {query}
            
            Instructions:
            - Create reusable conversation patterns
            - Include placeholders for dynamic content
            - Ensure templates match the agent's personality and expertise
            - Generate varied templates to avoid repetition
            
            DataFrame info:
            {df_info}
            
            Generate pandas code to create conversation templates:
            """,
            
            "performance_analysis": """
            You are analyzing performance metrics for agent: {agent_name}.
            Analysis focus: {focus}
            
            Query: {query}
            
            Instructions:
            - Measure conversation quality and effectiveness
            - Identify areas for improvement
            - Compare performance across different scenarios
            - Provide actionable insights for agent optimization
            
            DataFrame info:
            {df_info}
            
            Generate pandas code to analyze performance:
            """
        })
    
    def register_agent(self, agent_name: str, agent_config: Dict[str, Any]) -> bool:
        """
        Register a new agent with its specific prompts and configuration.
        
        Args:
            agent_name: Unique identifier for the agent
            agent_config: Configuration including prompts, templates, and metadata
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate agent config
            if not isinstance(agent_config, dict):
                logger.error(f"Agent config must be a dictionary for {agent_name}")
                return False
            
            # Set default values
            config = {
                "name": agent_name,
                "domain": agent_config.get("domain", "general"),
                "personality": agent_config.get("personality", "helpful"),
                "expertise": agent_config.get("expertise", []),
                "prompts": agent_config.get("prompts", {}),
                "templates": agent_config.get("templates", {}),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Store in memory
            self.agent_prompts[agent_name] = config
            
            # Save to file
            agent_file = self.base_dir / f"{agent_name}.json"
            with open(agent_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Registered agent: {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_name}: {e}")
            return False
    
    def load_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Load agent configuration from file or memory.
        
        Args:
            agent_name: Name of the agent to load
            
        Returns:
            Dict containing agent configuration or None if not found
        """
        # Check memory first
        if agent_name in self.agent_prompts:
            return self.agent_prompts[agent_name]
        
        # Try loading from file
        agent_file = self.base_dir / f"{agent_name}.json"
        if agent_file.exists():
            try:
                with open(agent_file, 'r') as f:
                    config = json.load(f)
                self.agent_prompts[agent_name] = config
                return config
            except Exception as e:
                logger.error(f"Failed to load agent {agent_name}: {e}")
        
        return None
    
    def get_prompt(self, agent_name: str, prompt_type: str, **kwargs) -> Optional[str]:
        """
        Get a formatted prompt for a specific agent and prompt type.
        
        Args:
            agent_name: Name of the agent
            prompt_type: Type of prompt to retrieve
            **kwargs: Variables to substitute in the prompt template
            
        Returns:
            Formatted prompt string or None if not found
        """
        try:
            # Load agent config
            agent_config = self.load_agent(agent_name)
            if not agent_config:
                logger.warning(f"Agent {agent_name} not found, using default prompts")
                agent_config = {"name": agent_name, "domain": "general"}
            
            # Get prompt template
            prompt_template = None
            
            # Check agent-specific prompts first
            if "prompts" in agent_config and prompt_type in agent_config["prompts"]:
                prompt_template = agent_config["prompts"][prompt_type]
            
            # Fall back to default templates
            elif prompt_type in self.prompt_templates:
                prompt_template = self.prompt_templates[prompt_type]
            
            # Check dynamic prompts
            elif prompt_type in self.dynamic_prompts:
                return self.dynamic_prompts[prompt_type](agent_config, **kwargs)
            
            if not prompt_template:
                logger.error(f"Prompt type {prompt_type} not found for agent {agent_name}")
                return None
            
            # Prepare template variables
            template_vars = {
                "agent_name": agent_name,
                "domain": agent_config.get("domain", "general"),
                "personality": agent_config.get("personality", "helpful"),
                **kwargs
            }
            
            # Format the template
            formatted_prompt = prompt_template.format(**template_vars)
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Error getting prompt for {agent_name}: {e}")
            return None
    
    def add_prompt_template(self, template_name: str, template: str) -> bool:
        """
        Add a new prompt template to the manager.
        
        Args:
            template_name: Name of the template
            template: Template string with placeholders
            
        Returns:
            bool: True if added successfully
        """
        try:
            self.prompt_templates[template_name] = template
            logger.info(f"Added prompt template: {template_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add prompt template {template_name}: {e}")
            return False
    
    def add_dynamic_prompt(self, prompt_name: str, prompt_func: Callable) -> bool:
        """
        Add a dynamic prompt function that generates prompts based on context.
        
        Args:
            prompt_name: Name of the dynamic prompt
            prompt_func: Function that takes (agent_config, **kwargs) and returns a prompt
            
        Returns:
            bool: True if added successfully
        """
        try:
            if not callable(prompt_func):
                logger.error(f"Prompt function for {prompt_name} must be callable")
                return False
            
            self.dynamic_prompts[prompt_name] = prompt_func
            logger.info(f"Added dynamic prompt: {prompt_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add dynamic prompt {prompt_name}: {e}")
            return False
    
    def update_agent_prompt(self, agent_name: str, prompt_type: str, prompt: str) -> bool:
        """
        Update a specific prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            prompt_type: Type of prompt to update
            prompt: New prompt string
            
        Returns:
            bool: True if updated successfully
        """
        try:
            agent_config = self.load_agent(agent_name)
            if not agent_config:
                logger.error(f"Agent {agent_name} not found")
                return False
            
            if "prompts" not in agent_config:
                agent_config["prompts"] = {}
            
            agent_config["prompts"][prompt_type] = prompt
            agent_config["updated_at"] = datetime.now().isoformat()
            
            # Update memory and file
            self.agent_prompts[agent_name] = agent_config
            agent_file = self.base_dir / f"{agent_name}.json"
            with open(agent_file, 'w') as f:
                json.dump(agent_config, f, indent=2)
            
            logger.info(f"Updated prompt {prompt_type} for agent {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update prompt for {agent_name}: {e}")
            return False
    
    def list_agents(self) -> List[str]:
        """
        List all registered agents.
        
        Returns:
            List of agent names
        """
        # Get agents from memory
        memory_agents = set(self.agent_prompts.keys())
        
        # Get agents from files
        file_agents = set()
        if self.base_dir.exists():
            for file_path in self.base_dir.glob("*.json"):
                agent_name = file_path.stem
                file_agents.add(agent_name)
        
        return list(memory_agents.union(file_agents))
    
    def list_prompt_types(self, agent_name: Optional[str] = None) -> List[str]:
        """
        List available prompt types for an agent or globally.
        
        Args:
            agent_name: Optional agent name to get specific prompts
            
        Returns:
            List of prompt type names
        """
        prompt_types = set(self.prompt_templates.keys())
        prompt_types.update(self.dynamic_prompts.keys())
        
        if agent_name:
            agent_config = self.load_agent(agent_name)
            if agent_config and "prompts" in agent_config:
                prompt_types.update(agent_config["prompts"].keys())
        
        return list(prompt_types)
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict with agent summary or None if not found
        """
        agent_config = self.load_agent(agent_name)
        if not agent_config:
            return None
        
        return {
            "name": agent_config.get("name"),
            "domain": agent_config.get("domain"),
            "personality": agent_config.get("personality"),
            "expertise": agent_config.get("expertise", []),
            "num_prompts": len(agent_config.get("prompts", {})),
            "num_templates": len(agent_config.get("templates", {})),
            "created_at": agent_config.get("created_at"),
            "updated_at": agent_config.get("updated_at")
        }

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
