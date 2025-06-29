"""
PandasRAG - Script-Friendly AgentChef Interface
===========================================================

A simplified interface for working with AgentChef's pandas querying and 
agent-centric storage system in scripts, without needing the web UI.

Usage:
------
```python
from agentChef import PandasRAG
import pandas as pd

# Initialize with default settings
rag = PandasRAG()

# Or with custom data directory
rag = PandasRAG(data_dir="./my_data")

# Register an agent
agent_id = rag.register_agent("research_assistant", 
                             system_prompt="You are a helpful research assistant.")

# Load data and query
df = pd.read_csv("data.csv")
response = rag.query(df, "What are the main trends in this data?", agent_id=agent_id)

# Store and retrieve conversations
rag.save_conversation(agent_id, "user", "What trends do you see?")
rag.save_conversation(agent_id, "assistant", response)
conversations = rag.get_conversations(agent_id)
```
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime

from .core.prompts.agent_prompt_manager import AgentPromptManager
from .core.storage.conversation_storage import ConversationStorage
from .core.llamaindex.pandas_query import PandasQueryIntegration
from .logs.agentchef_logging import setup_logging


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
        # Set up logging
        setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Set up data directory
        if data_dir is None:
            data_dir = os.getcwd() / Path("data") if isinstance(os.getcwd(), Path) else Path(os.getcwd()) / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_name = model_name
        self.prompt_manager = AgentPromptManager(self.data_dir)
        self.storage = ConversationStorage(self.data_dir)
        self.query_engine = None  # Initialized on first use
        
        # Add conversation history management
        self.max_history_turns = max_history_turns
        self.conversation_history = {}  # Per-agent conversation history cache
        
        # Initialize components with history support
        if HAS_AGENT_COMPONENTS:
            self.storage = ConversationStorage(self.data_dir)
            self.prompt_manager = AgentPromptManager(self.data_dir / "prompts")
            
            # Enhanced query engine with history support
            self.query_engine = HistoryAwarePandasQueryIntegration(
                agent_name="default",
                max_history_turns=max_history_turns,
                storage=self.storage
            )
        else:
            self.logger.warning("Agent components not available. Some features will be limited.")
            self.storage = None
            self.prompt_manager = None
            self.query_engine = None

        self.logger.info(f"PandasRAG initialized with data directory: {self.data_dir}")
    
    def _get_query_engine(self) -> PandasQueryIntegration:
        """Lazy initialization of query engine."""
        if self.query_engine is None:
            self.query_engine = PandasQueryIntegration(
                prompt_manager=self.prompt_manager,
                storage=self.storage,
                model_name=self.model_name
            )
        return self.query_engine
    
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
            system_prompt = f"You are {agent_id}, a helpful AI assistant."
        
        self.prompt_manager.create_agent_profile(
            agent_id=agent_id,
            system_prompt=system_prompt,
            description=description or f"Agent: {agent_id}",
            **kwargs
        )
        
        self.logger.info(f"Registered agent: {agent_id}")
        return agent_id
    
    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return self.prompt_manager.list_agent_profiles()
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a specific agent."""
        return self.prompt_manager.get_agent_profile(agent_id)
    
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
            if not self.ollama.is_available():
                return "Ollama is not available. Please check your installation."
            
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
            if save_conversation and self.storage:
                self._save_query_conversation(agent_id, question, result, {
                    "dataframe_shape": dataframe.shape,
                    "dataframe_columns": list(dataframe.columns),
                    "included_history": include_history,
                    "history_turns": len(history_context.split('\n')) if history_context else 0
                })
            
            # Update in-memory conversation cache
            self._update_conversation_cache(agent_id, question, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            return f"Error processing query: {str(e)}"

    def _get_conversation_history_context(self, agent_id: str, max_turns: int = 10) -> str:
        """
        Get recent conversation history for context.
        
        Args:
            agent_id: Agent identifier
            max_turns: Maximum number of conversation turns to include
            
        Returns:
            Formatted conversation history string
        """
        try:
            # First try to get from in-memory cache
            if agent_id in self.conversation_history:
                cached_history = self.conversation_history[agent_id]
                recent_history = cached_history[-max_turns:] if len(cached_history) > max_turns else cached_history
                
                if recent_history:
                    context_lines = []
                    for turn in recent_history:
                        context_lines.append(f"Human: {turn['question']}")
                        context_lines.append(f"Assistant: {turn['response']}")
                    return "\n".join(context_lines)
            
            # If not in cache, try to load from storage
            if self.storage:
                conversations = self.storage.get_conversations(agent_id, limit=max_turns)
                if not conversations.empty:
                    context_lines = []
                    for _, conv in conversations.tail(max_turns).iterrows():
                        if conv['role'] == 'human':
                            context_lines.append(f"Human: {conv['content']}")
                        elif conv['role'] == 'agent':
                            context_lines.append(f"Assistant: {conv['content']}")
                    return "\n".join(context_lines)
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve conversation history: {e}")
            return ""

    def _create_history_aware_prompt(self, dataframe: pd.DataFrame, question: str, 
                                   history_context: str, agent_id: str) -> str:
        """
        Create a prompt that includes conversation history for better context.
        
        Args:
            dataframe: DataFrame being queried
            question: Current question
            history_context: Previous conversation history
            agent_id: Agent identifier
            
        Returns:
            Enhanced system prompt with history context
        """
        # Get agent-specific prompt if available
        agent_prompt = ""
        if self.prompt_manager:
            agent_config = self.prompt_manager.get_agent_config(agent_id)
            if agent_config and "system_prompt" in agent_config:
                agent_prompt = agent_config["system_prompt"]
        
        # Default system prompt if none found
        if not agent_prompt:
            agent_prompt = f"You are {agent_id}, a data analysis assistant."
        
        # DataFrame information
        df_info = f"""
DataFrame Information:
- Shape: {dataframe.shape}
- Columns: {list(dataframe.columns)}
- Sample data:
{dataframe.head(3).to_string()}
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
4. Avoid repeating information already covered
"""
        
        # Combined prompt
        system_prompt = f"""{agent_prompt}

{df_info}

{history_section}

Your task is to analyze the DataFrame and answer the user's question with:
1. Clear, natural language explanations
2. Relevant insights from the data
3. Context from previous conversations when applicable
4. Specific details and examples from the data

Current Question: {question}

Provide a comprehensive, context-aware response that builds on the conversation history."""

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
        """
        Start an interactive chat session with a DataFrame that maintains full conversation history.
        
        Args:
            dataframe: DataFrame to analyze
            agent_id: Agent identifier
            
        Returns:
            DataChatSession object for interactive querying
        """
        return DataChatSession(self, dataframe, agent_id)

    def get_conversation_summary(self, agent_id: str, num_exchanges: int = 5) -> str:
        """
        Get a summary of recent conversation exchanges.
        
        Args:
            agent_id: Agent identifier
            num_exchanges: Number of recent exchanges to summarize
            
        Returns:
            Conversation summary
        """
        try:
            history_context = self._get_conversation_history_context(agent_id, num_exchanges * 2)
            
            if not history_context:
                return f"No conversation history found for agent {agent_id}."
            
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
            self.pandas_rag.storage.clear_conversations(self.agent_id)
    
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
            logging.error(f"Error saving session: {e}")
            return False


class HistoryAwarePandasQueryIntegration:
    """Enhanced pandas query integration that includes conversation history in prompts."""
    
    def __init__(self, agent_name: str = "default", max_history_turns: int = 10, 
                 storage: Optional[Any] = None):
        self.agent_name = agent_name
        self.max_history_turns = max_history_turns
        self.storage = storage
        
        # Initialize Ollama interface
        try:
            from agentChef.core.ollama.ollama_interface import OllamaInterface
            self.ollama = OllamaInterface()
        except ImportError:
            self.ollama = None
    
    def query_with_history(self, df: pd.DataFrame, query: str, 
                          conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Query DataFrame with conversation history context.
        
        Args:
            df: DataFrame to query
            query: Natural language query
            conversation_history: Previous conversation context
            
        Returns:
            Dict with query results and metadata
        """
        try:
            # Format conversation history
            history_text = ""
            if conversation_history:
                for msg in conversation_history[-self.max_history_turns:]:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if role == 'human':
                        history_text += f"Previous Question: {content}\n"
                    elif role == 'agent':
                        history_text += f"Previous Answer: {content}\n"
            
            # Enhanced prompt with history
            system_prompt = f"""You are a data analyst working with a pandas DataFrame.

DataFrame Info:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Sample:
{df.head(2).to_string()}

{history_text}

Current Question: {query}

Based on the DataFrame and conversation history, provide a comprehensive answer that:
1. References relevant previous discussions
2. Provides specific insights from the data
3. Builds upon earlier analysis
4. Gives concrete examples from the DataFrame

Answer:"""
            
            # Get response
            if self.ollama:
                response = self.ollama.chat(messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ])
                
                result = response.get('message', {}).get('content', 'No response')
                
                return {
                    "response": result,
                    "agent_name": self.agent_name,
                    "included_history_turns": len(conversation_history) if conversation_history else 0,
                    "dataframe_shape": df.shape
                }
            else:
                return {"error": "Ollama interface not available"}
                
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}
