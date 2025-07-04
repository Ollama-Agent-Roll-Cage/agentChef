"""conversation_storage.py
Agent-focused storage system for conversations, knowledge, and templates.
Provides abstraction over storage backends and supports agent-specific data organization.

Written By: @BorcherdingL
Date: 6/29/2025
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Import the ParquetStorage - try multiple possible locations
try:
    from oarc_crawlers import ParquetStorage
except ImportError:
    try:
        # Try the specific import path from the attachment
        import sys
        import importlib.util
        # Load ParquetStorage from the attachment path if available
        spec = importlib.util.find_spec("parquet_storage")
        if spec:
            parquet_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parquet_module)
            ParquetStorage = parquet_module.ParquetStorage
        else:
            raise ImportError("ParquetStorage not found")
    except ImportError:
        # Fallback to local implementation if not available
        class ParquetStorage:
            @staticmethod
            def save_to_parquet(data, file_path):
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = data
                df.to_parquet(file_path, index=False)
                return True
            
            @staticmethod
            def load_from_parquet(file_path):
                if Path(file_path).exists():
                    return pd.read_parquet(file_path)
                return None
            
            @staticmethod
            def append_to_parquet(data, file_path):
                existing = ParquetStorage.load_from_parquet(file_path)
                if existing is not None:
                    if isinstance(data, dict):
                        new_df = pd.DataFrame([data])
                    elif isinstance(data, list):
                        new_df = pd.DataFrame(data)
                    else:
                        new_df = data
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    return ParquetStorage.save_to_parquet(combined, file_path)
                else:
                    return ParquetStorage.save_to_parquet(data, file_path)

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ConversationMetadata:
    """Metadata for stored conversations."""
    agent_name: str
    conversation_id: str
    created_at: str
    updated_at: str
    context: str = "general"
    template_format: str = "standard"
    num_turns: int = 0
    quality_score: Optional[float] = None
    tags: List[str] = None
    source: str = "generated"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class KnowledgeEntry:
    """Entry in an agent's knowledge base."""
    agent_name: str
    entry_id: str
    topic: str
    content: str
    knowledge_type: str = "fact"  # fact, process, template, example
    confidence: float = 1.0
    source: str = "unknown"
    created_at: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class ConversationStorage:
    """
    Agent-focused storage system for conversations, knowledge, and templates.
    Organizes data by agent and provides efficient querying capabilities.
    """
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the ConversationStorage system.
        
        Args:
            base_dir: Base directory for storing agent data
        """
        self.base_dir = Path(base_dir) if base_dir else Path("data/agent_storage")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        self.conversations_dir = self.base_dir / "conversations"
        self.knowledge_dir = self.base_dir / "knowledge"
        self.templates_dir = self.base_dir / "templates"
        self.metadata_dir = self.base_dir / "metadata"
        
        for dir_path in [self.conversations_dir, self.knowledge_dir, 
                        self.templates_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Storage backend
        self.storage = ParquetStorage()
        
    def _get_agent_dir(self, agent_name: str, data_type: str) -> Path:
        """Get the directory path for a specific agent and data type."""
        type_dir = getattr(self, f"{data_type}_dir")
        agent_dir = type_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir
    
    def _generate_conversation_id(self, agent_name: str) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{agent_name}_{timestamp}"
    
    def save_conversation(self, 
                         agent_name: str, 
                         conversation: List[Dict[str, Any]], 
                         metadata: Optional[ConversationMetadata] = None,
                         conversation_id: Optional[str] = None) -> str:
        """
        Save a conversation for a specific agent.
        
        Args:
            agent_name: Name of the agent
            conversation: List of conversation turns
            metadata: Optional metadata for the conversation
            conversation_id: Optional specific conversation ID
            
        Returns:
            str: The conversation ID
        """
        try:
            # Generate ID if not provided
            if not conversation_id:
                conversation_id = self._generate_conversation_id(agent_name)
            
            # Create metadata if not provided
            if not metadata:
                metadata = ConversationMetadata(
                    agent_name=agent_name,
                    conversation_id=conversation_id,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    num_turns=len(conversation)
                )
            
            # Prepare conversation data with metadata
            conversation_data = []
            for i, turn in enumerate(conversation):
                turn_data = {
                    "conversation_id": conversation_id,
                    "agent_name": agent_name,
                    "turn_index": i,
                    "from": turn.get("from", "unknown"),
                    "value": turn.get("value", ""),
                    "timestamp": turn.get("timestamp", datetime.now().isoformat()),
                    **turn  # Include any additional fields
                }
                conversation_data.append(turn_data)
            
            # Save conversation data
            agent_dir = self._get_agent_dir(agent_name, "conversations")
            conv_file = agent_dir / f"{conversation_id}.parquet"
            success = self.storage.save_to_parquet(conversation_data, conv_file)
            
            if success:
                # Save metadata separately
                metadata_file = self.metadata_dir / agent_name / f"{conversation_id}_meta.json"
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                
                logger.info(f"Saved conversation {conversation_id} for agent {agent_name}")
                return conversation_id
            else:
                logger.error(f"Failed to save conversation {conversation_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving conversation for {agent_name}: {e}")
            return None
    
    def load_conversation(self, agent_name: str, conversation_id: str) -> Optional[Tuple[List[Dict[str, Any]], ConversationMetadata]]:
        """
        Load a specific conversation for an agent.
        
        Args:
            agent_name: Name of the agent
            conversation_id: ID of the conversation to load
            
        Returns:
            Tuple of (conversation_turns, metadata) or None if not found
        """
        try:
            # Load conversation data
            agent_dir = self._get_agent_dir(agent_name, "conversations")
            conv_file = agent_dir / f"{conversation_id}.parquet"
            
            df = self.storage.load_from_parquet(conv_file)
            if df is None:
                return None
            
            # Convert back to conversation format
            conversation = []
            for _, row in df.sort_values('turn_index').iterrows():
                turn = {
                    "from": row["from"],
                    "value": row["value"]
                }
                # Include any additional fields (excluding our internal ones)
                exclude_fields = {"conversation_id", "agent_name", "turn_index", "timestamp"}
                for col in df.columns:
                    if col not in exclude_fields and col not in turn:
                        turn[col] = row[col]
                conversation.append(turn)
            
            # Load metadata
            metadata = None
            metadata_file = self.metadata_dir / agent_name / f"{conversation_id}_meta.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = ConversationMetadata(**metadata_dict)
            
            return conversation, metadata
            
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id} for {agent_name}: {e}")
            return None
    
    def save_knowledge(self, agent_name: str, knowledge_entries: List[KnowledgeEntry]) -> bool:
        """
        Save knowledge entries for an agent.
        
        Args:
            agent_name: Name of the agent
            knowledge_entries: List of knowledge entries to save
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if not knowledge_entries:
                return True
            
            # Convert to DataFrame format
            knowledge_data = [asdict(entry) for entry in knowledge_entries]
            
            # Save to agent's knowledge file
            agent_dir = self._get_agent_dir(agent_name, "knowledge")
            knowledge_file = agent_dir / "knowledge_base.parquet"
            
            # Append to existing knowledge if file exists
            if knowledge_file.exists():
                success = self.storage.append_to_parquet(knowledge_data, knowledge_file)
            else:
                success = self.storage.save_to_parquet(knowledge_data, knowledge_file)
            
            if success:
                logger.info(f"Saved {len(knowledge_entries)} knowledge entries for {agent_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving knowledge for {agent_name}: {e}")
            return False
    
    def load_knowledge(self, agent_name: str) -> List['KnowledgeEntry']:
        """
        Load knowledge entries for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of KnowledgeEntry objects
        """
        try:
            agent_dir = self._get_agent_dir(agent_name, "knowledge")
            knowledge_file = agent_dir / "knowledge_base.parquet"
            
            if not knowledge_file.exists():
                return []
            
            df = self.storage.load_from_parquet(knowledge_file)
            if df is None:
                return []
            
            # Convert DataFrame rows to KnowledgeEntry objects
            entries = []
            for _, row in df.iterrows():
                entry = KnowledgeEntry(
                    agent_name=row.get('agent_name', agent_name),
                    entry_id=row.get('entry_id', ''),
                    topic=row.get('topic', ''),
                    content=row.get('content', ''),
                    knowledge_type=row.get('knowledge_type', 'fact'),
                    confidence=row.get('confidence', 1.0),
                    source=row.get('source', 'unknown'),
                    created_at=row.get('created_at', ''),
                    tags=row.get('tags', [])
                )
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Error loading knowledge for {agent_name}: {e}")
            return []
    
    def save_template(self, agent_name: str, template_name: str, template_data: Dict[str, Any]) -> bool:
        """
        Save a conversation template for an agent.
        
        Args:
            agent_name: Name of the agent
            template_name: Name of the template
            template_data: Template data including structure and metadata
            
        Returns:
            bool: True if saved successfully
        """
        try:
            agent_dir = self._get_agent_dir(agent_name, "templates")
            template_file = agent_dir / f"{template_name}.json"
            
            # Add metadata
            template_data.update({
                "agent_name": agent_name,
                "template_name": template_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            with open(template_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            logger.info(f"Saved template {template_name} for agent {agent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template for {agent_name}: {e}")
            return False
    
    def load_template(self, agent_name: str, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a conversation template for an agent.
        
        Args:
            agent_name: Name of the agent
            template_name: Name of the template
            
        Returns:
            Template data or None if not found
        """
        try:
            agent_dir = self._get_agent_dir(agent_name, "templates")
            template_file = agent_dir / f"{template_name}.json"
            
            if not template_file.exists():
                return None
            
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            
            return template_data
            
        except Exception as e:
            logger.error(f"Error loading template {template_name} for {agent_name}: {e}")
            return None
    
    def list_conversations(self, agent_name: str) -> List[str]:
        """
        List all conversation IDs for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of conversation IDs
        """
        try:
            agent_dir = self._get_agent_dir(agent_name, "conversations")
            conversation_files = agent_dir.glob("*.parquet")
            return [f.stem for f in conversation_files]
        except Exception as e:
            logger.error(f"Error listing conversations for {agent_name}: {e}")
            return []
    
    def list_agents(self) -> List[str]:
        """
        List all agents with stored data.
        
        Returns:
            List of agent names
        """
        agents = set()
        
        # Check all data directories
        for data_dir in [self.conversations_dir, self.knowledge_dir, self.templates_dir]:
            if data_dir.exists():
                agents.update(d.name for d in data_dir.iterdir() if d.is_dir())
        
        return list(agents)
    
    def get_conversations(self, agent_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Get conversations for an agent as a DataFrame.
        
        Args:
            agent_name: Name of the agent
            limit: Maximum number of conversations to return
            
        Returns:
            DataFrame with conversation data
        """
        try:
            agent_dir = self._get_agent_dir(agent_name, "conversations")
            if not agent_dir.exists():
                return pd.DataFrame()
            
            conversations = []
            
            # Get all conversation files
            conv_files = list(agent_dir.glob("*.parquet"))
            conv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
            
            for conv_file in conv_files[:limit]:
                try:
                    df = self.storage.load_from_parquet(conv_file)
                    if df is not None:
                        # Convert the parquet data to conversation format
                        for _, row in df.iterrows():
                            conversations.append({
                                'role': row.get('from', 'unknown'),
                                'content': row.get('value', ''),
                                'timestamp': row.get('timestamp', ''),
                                'conversation_id': row.get('conversation_id', ''),
                                'agent_name': row.get('agent_name', agent_name)
                            })
                        
                except Exception as e:
                    logger.warning(f"Error reading conversation file {conv_file}: {e}")
                    continue
            
            return pd.DataFrame(conversations)
            
        except Exception as e:
            logger.error(f"Error getting conversations for {agent_name}: {e}")
            return pd.DataFrame()

    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        Get statistics for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict with agent statistics
        """
        try:
            conversations_df = self.get_conversations(agent_name, limit=1000)  # Get more for stats
            knowledge_entries = self.load_knowledge(agent_name)
            
            stats = {
                'total_conversations': len(conversations_df),
                'total_knowledge_entries': len(knowledge_entries),
                'conversation_roles': conversations_df['role'].value_counts().to_dict() if not conversations_df.empty else {},
                'knowledge_types': {},
                'last_activity': '',
                'agent_name': agent_name
            }
            
            # Get knowledge type distribution
            if knowledge_entries:
                knowledge_types = {}
                for entry in knowledge_entries:
                    k_type = entry.knowledge_type
                    knowledge_types[k_type] = knowledge_types.get(k_type, 0) + 1
                stats['knowledge_types'] = knowledge_types
                
                # Get most recent activity
                recent_entries = sorted(knowledge_entries, key=lambda x: x.created_at, reverse=True)
                if recent_entries:
                    stats['last_activity'] = recent_entries[0].created_at
            
            # Get most recent conversation timestamp
            if not conversations_df.empty and 'timestamp' in conversations_df.columns:
                recent_convs = conversations_df['timestamp'].dropna()
                if not recent_convs.empty:
                    most_recent = recent_convs.max()
                    if most_recent and (not stats['last_activity'] or most_recent > stats['last_activity']):
                        stats['last_activity'] = most_recent
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting agent stats for {agent_name}: {e}")
            return {
                'total_conversations': 0,
                'total_knowledge_entries': 0,
                'agent_name': agent_name,
                'error': str(e)
            }

# ...existing code...

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create storage system
    storage = ConversationStorage()
    
    # Example conversation
    conversation = [
        {"from": "human", "value": "What is machine learning?"},
        {"from": "gpt", "value": "Machine learning is a subset of artificial intelligence..."}
    ]
    
    # Save conversation
    conv_id = storage.save_conversation("ragchef", conversation)
    print(f"Saved conversation: {conv_id}")
    
    # Load conversation back
    loaded = storage.load_conversation("ragchef", conv_id)
    if loaded:
        loaded_conv, loaded_meta = loaded
        print(f"Loaded conversation with {len(loaded_conv)} turns")
    
    # Example knowledge entry
    knowledge = [
        KnowledgeEntry(
            agent_name="ragchef",
            entry_id="ml_def_001",
            topic="machine learning",
            content="Machine learning is a method of data analysis that automates analytical model building.",
            knowledge_type="fact",
            source="training_data"
        )
    ]
    
    # Save knowledge
    storage.save_knowledge("ragchef", knowledge)
    
    # Get agent stats
    stats = storage.get_agent_stats("ragchef")
    print(f"Agent stats: {stats}")
