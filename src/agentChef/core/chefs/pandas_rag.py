"""
PandasRAG - Script-Friendly AgentChef Interface
===========================================================

A simplified interface for working with AgentChef's pandas querying and 
agent-centric storage system in scripts, without needing the web UI.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime

# Set up basic logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PandasRAG:
    """
    Simple, script-friendly interface for AgentChef that includes conversation history
    in prompts for better context-aware responses.
    """
    
    def __init__(self, data_dir: str = "./agentchef_data", model_name: str = "llama3.2", 
                 log_level: str = "INFO", max_history_turns: int = 10):
        """
        Initialize PandasRAG with conversation history support.
        
        Args:
            data_dir: Directory to store agent data
            model_name: Ollama model to use
            log_level: Logging level
            max_history_turns: Maximum number of conversation turns to include in context
        """
        # Set up logging with graceful fallback
        try:
            from ...logs.agentchef_logging import setup_logging
            setup_logging(log_level)
        except ImportError:
            # Fallback to basic logging
            level = getattr(logging, log_level.upper(), logging.INFO)
            logging.basicConfig(level=level)
        
        self.logger = logging.getLogger(__name__)
        
        # Set up data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_name = model_name
        self.max_history_turns = max_history_turns
        self.conversation_history = {}  # Per-agent conversation history cache
        
        # Initialize Ollama interface with graceful fallback
        self.ollama = None
        try:
            from ..ollama.ollama_interface import OllamaInterface
            self.ollama = OllamaInterface(model_name=model_name)
            self.logger.info("Ollama interface initialized successfully")
        except ImportError:
            self.logger.warning("Ollama interface not available")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama interface: {e}")

        # Initialize components with graceful fallback
        self.storage = None
        self.prompt_manager = None
        try:
            from ..storage.conversation_storage import ConversationStorage
            from ..prompts.agent_prompt_manager import AgentPromptManager
            
            self.storage = ConversationStorage(self.data_dir)
            self.prompt_manager = AgentPromptManager(self.data_dir / "prompts")
            self.logger.info("Agent components initialized successfully")
        except ImportError:
            self.logger.warning("Agent components not available. Some features will be limited.")
        except Exception as e:
            self.logger.warning(f"Failed to initialize agent components: {e}")

        self.query_engine = None  # Initialized on first use
        self._agents = {}  # Local agent storage if prompt manager not available
        
        self.logger.info(f"PandasRAG initialized with data directory: {self.data_dir}")
    
    def register_agent(self, 
                       agent_id: str, 
                       system_prompt: str = None,
                       description: str = None,
                       **kwargs) -> str:
        """
        Register a new agent with optional system prompt and description.
        
        Args:
            agent_id: Unique identifier for the agent
            system_prompt: System prompt for the agent
            description: Description of the agent's purpose
            **kwargs: Additional agent metadata
            
        Returns:
            The registered agent_id
        """
        if system_prompt is None:
            system_prompt = f"You are {agent_id}, a helpful AI assistant specialized in data analysis."
        
        if self.prompt_manager:
            try:
                self.prompt_manager.create_agent_profile(
                    agent_id=agent_id,
                    system_prompt=system_prompt,
                    description=description or f"Agent: {agent_id}",
                    **kwargs
                )
            except Exception as e:
                self.logger.warning(f"Failed to create agent profile with prompt manager: {e}")
                # Fall back to local storage
                self._agents[agent_id] = {
                    'system_prompt': system_prompt,
                    'description': description,
                    **kwargs
                }
        else:
            # Store agent info locally if prompt manager not available
            self._agents[agent_id] = {
                'system_prompt': system_prompt,
                'description': description,
                **kwargs
            }
        
        self.logger.info(f"Registered agent: {agent_id}")
        return agent_id
    
    def list_agents(self) -> List[str]:
        """List all registered agents."""
        if self.prompt_manager:
            try:
                return self.prompt_manager.list_agent_profiles()
            except Exception:
                pass
        return list(self._agents.keys())
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a specific agent."""
        if self.prompt_manager:
            try:
                return self.prompt_manager.get_agent_profile(agent_id)
            except Exception:
                pass
        return self._agents.get(agent_id, {})

    def create_empty_conversation(self, agent_id: str) -> bool:
        """Create an empty conversation for an agent."""
        if agent_id not in self.conversation_history:
            self.conversation_history[agent_id] = []
            self.logger.info(f"Created empty conversation history for {agent_id}")
            return True
        return False

    def save_conversation(self, agent_id: str, role: str, content: str) -> bool:
        """Save a conversation turn."""
        try:
            if agent_id not in self.conversation_history:
                self.conversation_history[agent_id] = []
            
            self.conversation_history[agent_id].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Also save to persistent storage if available
            if self.storage:
                try:
                    conversation = [{"from": role, "value": content}]
                    from ..storage.conversation_storage import ConversationMetadata
                    metadata = ConversationMetadata(
                        agent_name=agent_id,
                        conversation_id="",
                        created_at="",
                        updated_at="",
                        context="manual_entry",
                        source="pandas_rag"
                    )
                    self.storage.save_conversation(agent_id, conversation, metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to save to persistent storage: {e}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
            return False

    def get_conversations(self, agent_id: str, limit: int = 10) -> pd.DataFrame:
        """Get conversations for an agent."""
        try:
            if self.storage:
                try:
                    return self.storage.get_conversations(agent_id, limit=limit)
                except Exception as e:
                    self.logger.warning(f"Failed to get from persistent storage: {e}")
            
            # Fall back to in-memory storage
            if agent_id in self.conversation_history:
                history = self.conversation_history[agent_id][-limit:]
                return pd.DataFrame(history)
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting conversations: {e}")
            return pd.DataFrame()

    def add_knowledge(self, agent_id: str, content: str, source: str = "manual", 
                     metadata: Dict[str, Any] = None) -> bool:
        """Add knowledge to an agent."""
        try:
            if self.storage:
                from ..storage.conversation_storage import KnowledgeEntry
                entry = KnowledgeEntry(
                    agent_name=agent_id,
                    entry_id=f"manual_{datetime.now().timestamp()}",
                    topic=source,
                    content=content,
                    knowledge_type="manual",
                    source=source,
                    created_at=datetime.now().isoformat(),
                    tags=metadata.get("tags", []) if metadata else []
                )
                return self.storage.save_knowledge(agent_id, [entry])
            return False
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {e}")
            return False

    def get_knowledge(self, agent_id: str) -> pd.DataFrame:
        """Get knowledge for an agent."""
        try:
            if self.storage:
                knowledge_entries = self.storage.load_knowledge(agent_id)
                return pd.DataFrame([{
                    'content': entry.content,
                    'source': entry.source,
                    'created_at': entry.created_at,
                    'tags': entry.tags
                } for entry in knowledge_entries])
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error getting knowledge: {e}")
            return pd.DataFrame()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all agents and their data."""
        summary = {
            "total_agents": len(self.list_agents()),
            "data_dir": str(self.data_dir),
            "agents": {}
        }
        
        for agent_id in self.list_agents():
            conversations = self.get_conversations(agent_id)
            knowledge = self.get_knowledge(agent_id)
            agent_info = self.get_agent_info(agent_id)
            
            summary["agents"][agent_id] = {
                "conversation_turns": len(conversations),
                "knowledge_items": len(knowledge),
                "profile": agent_info
            }
        
        return summary

    def export_data(self, agent_id: str, export_dir: str) -> Dict[str, str]:
        """Export agent data to files."""
        export_path = Path(export_dir)
        export_path.mkdir(exist_ok=True)
        
        exported = {}
        
        try:
            # Export conversations
            conversations = self.get_conversations(agent_id)
            if not conversations.empty:
                conv_file = export_path / f"{agent_id}_conversations.csv"
                conversations.to_csv(conv_file, index=False)
                exported["conversations"] = str(conv_file)
            
            # Export knowledge
            knowledge = self.get_knowledge(agent_id)
            if not knowledge.empty:
                kb_file = export_path / f"{agent_id}_knowledge.csv"
                knowledge.to_csv(kb_file, index=False)
                exported["knowledge"] = str(kb_file)
            
            # Export agent profile
            profile = self.get_agent_info(agent_id)
            if profile:
                import json
                profile_file = export_path / f"{agent_id}_profile.json"
                with open(profile_file, 'w') as f:
                    json.dump(profile, f, indent=2)
                exported["profile"] = str(profile_file)
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
        
        return exported

    def query(self, dataframe: pd.DataFrame, question: str, agent_id: str = "default", 
              save_conversation: bool = True, include_history: bool = True,
              max_history: Optional[int] = None) -> str:
        """
        Query a pandas DataFrame with natural language, including conversation history.
        
        Args:
            dataframe: DataFrame to query
            question: Natural language question
            agent_id: Agent identifier
            save_conversation: Whether to save this query as a conversation
            include_history: Whether to include conversation history in the prompt
            max_history: Override default max history turns
            
        Returns:
            Natural language response
        """
        try:
            if not self.ollama or not self.ollama.is_available():
                return "Ollama is not available. Please check your installation and ensure Ollama is running."
            
            # Get conversation history for context
            history_context = ""
            if include_history:
                history_context = self._get_conversation_history_context(
                    agent_id, 
                    max_turns=max_history or self.max_history_turns
                )
            
            # Enhanced prompt with history context
            enhanced_prompt = self._create_history_aware_prompt(
                dataframe, question, history_context, agent_id
            )
            
            # Get response from Ollama
            response = self.ollama.chat(messages=[
                {"role": "system", "content": enhanced_prompt},
                {"role": "user", "content": question}
            ])
            
            result = response.get('message', {}).get('content', 'No response generated')
            
            # Save conversation if requested
            if save_conversation:
                self.save_conversation(agent_id, "user", question)
                self.save_conversation(agent_id, "assistant", result)
            
            # Update in-memory conversation cache
            self._update_conversation_cache(agent_id, question, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            return f"Error processing query: {str(e)}"

    def _get_conversation_history_context(self, agent_id: str, max_turns: int = 10) -> str:
        """Get recent conversation history for context."""
        try:
            # First try to get from in-memory cache
            if agent_id in self.conversation_history:
                cached_history = self.conversation_history[agent_id]
                recent_history = cached_history[-max_turns:] if len(cached_history) > max_turns else cached_history
                
                if recent_history:
                    context_lines = []
                    for turn in recent_history:
                        if 'question' in turn and 'response' in turn:
                            context_lines.append(f"Human: {turn['question']}")
                            context_lines.append(f"Assistant: {turn['response']}")
                        elif turn.get('role') == 'user':
                            context_lines.append(f"Human: {turn['content']}")
                        elif turn.get('role') == 'assistant':
                            context_lines.append(f"Assistant: {turn['content']}")
                    return "\n".join(context_lines)
            
            # If not in cache, try to load from storage
            if self.storage:
                try:
                    conversations = self.storage.get_conversations(agent_id, limit=max_turns)
                    if not conversations.empty:
                        context_lines = []
                        for _, conv in conversations.tail(max_turns).iterrows():
                            if conv.get('role') == 'human' or conv.get('role') == 'user':
                                context_lines.append(f"Human: {conv['content']}")
                            elif conv.get('role') == 'agent' or conv.get('role') == 'assistant':
                                context_lines.append(f"Assistant: {conv['content']}")
                        return "\n".join(context_lines)
                except Exception as e:
                    self.logger.warning(f"Could not load conversation history from storage: {e}")
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve conversation history: {e}")
            return ""

    def _create_history_aware_prompt(self, dataframe: pd.DataFrame, question: str, 
                                   history_context: str, agent_id: str) -> str:
        """Create a prompt that includes conversation history for better context."""
        # Get agent-specific prompt if available
        agent_prompt = ""
        agent_info = self.get_agent_info(agent_id)
        if agent_info and "system_prompt" in agent_info:
            agent_prompt = agent_info["system_prompt"]
        
        # Default system prompt if none found
        if not agent_prompt:
            agent_prompt = f"You are {agent_id}, a data analysis assistant specialized in analyzing pandas DataFrames and providing clear insights."
        
        # Enhanced DataFrame information with actual sample calculations
        df_info = f"""
DataFrame Information:
- Shape: {dataframe.shape}
- Columns: {list(dataframe.columns)}
- Data types: {dict(dataframe.dtypes)}
- Sample data (first 3 rows):
{dataframe.head(3).to_string()}

Key Statistics:
- Total records: {len(dataframe)}
- Unique products: {dataframe['product'].nunique() if 'product' in dataframe.columns else 'N/A'}
- Date range: {dataframe['quarter'].unique().tolist() if 'quarter' in dataframe.columns else 'N/A'}
- Regions covered: {dataframe['region'].unique().tolist() if 'region' in dataframe.columns else 'N/A'}
"""
    
        # Conversation history section
        history_section = ""
        if history_context:
            history_section = f"""
Previous Conversation History:
{history_context}

Based on this conversation history, you should:
1. Maintain context from previous exchanges
2. Reference earlier discussions when relevant
3. Build upon previous insights
4. Avoid repeating information already covered unless specifically requested
5. Use the ACTUAL data from the DataFrame above, not sample data
"""
    
        # Combined prompt with emphasis on using real data
        system_prompt = f"""{agent_prompt}

{df_info}

{history_section}

IMPORTANT: Always use the ACTUAL data provided above, not sample data. When showing calculations:
1. Reference the real DataFrame structure and values
2. Provide specific numbers from the actual dataset
3. Use the exact column names and data types shown
4. Show Python code that works with this specific DataFrame
5. Give precise insights based on the real data patterns

Your task is to analyze the DataFrame and answer the user's question with:
1. Clear, natural language explanations using REAL data
2. Relevant insights from the ACTUAL dataset provided
3. Context from previous conversations when applicable
4. Specific details and examples from the REAL data
5. Python code that works with the actual DataFrame structure

Current Question: {question}

Provide a comprehensive, context-aware response that uses the ACTUAL data while building on the conversation history."""

        return system_prompt

    def _update_conversation_cache(self, agent_id: str, question: str, response: str):
        """Update the in-memory conversation cache."""
        if agent_id not in self.conversation_history:
            self.conversation_history[agent_id] = []
        
        self.conversation_history[agent_id].append({
            "question": question,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the most recent conversations in memory
        if len(self.conversation_history[agent_id]) > self.max_history_turns * 2:
            self.conversation_history[agent_id] = self.conversation_history[agent_id][-self.max_history_turns * 2:]

    def chat_with_data(self, dataframe: pd.DataFrame, agent_id: str = "default") -> 'DataChatSession':
        """Start an interactive chat session with a DataFrame."""
        return DataChatSession(self, dataframe, agent_id)

    def get_conversation_summary(self, agent_id: str, num_exchanges: int = 5) -> str:
        """Get a summary of recent conversation exchanges."""
        try:
            history_context = self._get_conversation_history_context(agent_id, num_exchanges * 2)
            
            if not history_context:
                return f"No conversation history found for agent {agent_id}."
            
            if not self.ollama or not self.ollama.is_available():
                return f"Cannot generate summary - Ollama not available. Recent history: {history_context[:200]}..."
            
            # Use the agent to summarize the conversation
            summary_prompt = f"""Please provide a concise summary of this conversation history:

{history_context}

Focus on:
1. Main topics discussed
2. Key insights discovered
3. Data analysis patterns
4. Important findings or conclusions

Provide a brief, structured summary."""

            response = self.ollama.chat(messages=[
                {"role": "system", "content": "You are a conversation summarizer."},
                {"role": "user", "content": summary_prompt}
            ])
            
            return response.get('message', {}).get('content', 'Could not generate summary')
            
        except Exception as e:
            self.logger.error(f"Error generating conversation summary: {e}")
            return f"Error generating summary: {str(e)}"


class DataChatSession:
    """Interactive chat session with a DataFrame that maintains conversation context."""
    
    def __init__(self, pandas_rag: PandasRAG, dataframe: pd.DataFrame, agent_id: str):
        self.pandas_rag = pandas_rag
        self.dataframe = dataframe
        self.agent_id = agent_id
        self.session_started = datetime.now()
        
    def ask(self, question: str) -> str:
        """Ask a question about the DataFrame with full conversation context."""
        return self.pandas_rag.query(
            self.dataframe, 
            question, 
            self.agent_id, 
            save_conversation=True,
            include_history=True
        )
    
    def clear_history(self):
        """Clear conversation history for this session."""
        if self.agent_id in self.pandas_rag.conversation_history:
            del self.pandas_rag.conversation_history[self.agent_id]
        
        # Also clear from persistent storage
        if self.pandas_rag.storage:
            try:
                self.pandas_rag.storage.clear_conversations(self.agent_id)
            except Exception as e:
                self.pandas_rag.logger.warning(f"Could not clear persistent storage: {e}")
    
    def get_summary(self) -> str:
        """Get a summary of this chat session."""
        return self.pandas_rag.get_conversation_summary(self.agent_id)
    
    def save_session(self, session_name: str) -> bool:
        """Save this chat session for later reference."""
        try:
            if self.pandas_rag.storage:
                # Add session metadata
                metadata = {
                    "session_name": session_name,
                    "dataframe_shape": self.dataframe.shape,
                    "dataframe_columns": list(self.dataframe.columns),
                    "session_started": self.session_started.isoformat(),
                    "session_ended": datetime.now().isoformat()
                }
                
                return self.pandas_rag.storage.save_session_metadata(self.agent_id, metadata)
            return False
        except Exception as e:
            self.pandas_rag.logger.error(f"Error saving session: {e}")
            return False
