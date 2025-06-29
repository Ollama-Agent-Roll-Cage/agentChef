"""pandas_query_integration.py
Integrates LlamaIndex's PandasQueryEngine into the research and dataset generation system.
This module provides utilities for natural language querying of pandas DataFrames with
agent-specific prompt support and conversation storage integration.

Written By: @BorcherdingL
Date: 6/29/2025
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# LlamaIndex imports - handle gracefully
try:
    from llama_index.experimental.query_engine import PandasQueryEngine
    from llama_index.core import PromptTemplate
    HAS_QUERY_ENGINE = True
except ImportError:
    HAS_QUERY_ENGINE = False

# Import our new modular components
try:
    from agentChef.core.prompts.agent_prompt_manager import AgentPromptManager
    from agentChef.core.storage.conversation_storage import ConversationStorage
    HAS_AGENT_COMPONENTS = True
except ImportError:
    HAS_AGENT_COMPONENTS = False
    logging.warning("Agent components not available. Some features will be limited.")

# Setup logging
logger = logging.getLogger(__name__)

class AbstractQueryEngine:
    """
    Abstract base class for query engines that can be specialized for different agents.
    Provides a common interface while allowing agent-specific customization.
    """
    
    def __init__(self, agent_name: str = "default", **kwargs):
        """
        Initialize the abstract query engine.
        
        Args:
            agent_name: Name of the agent using this query engine
            **kwargs: Additional configuration parameters
        """
        self.agent_name = agent_name
        self.config = kwargs
        
        # Initialize prompt manager if available
        if HAS_AGENT_COMPONENTS:
            self.prompt_manager = AgentPromptManager()
            self.storage = ConversationStorage()
        else:
            self.prompt_manager = None
            self.storage = None
    
    def get_agent_prompt(self, prompt_type: str, **kwargs) -> Optional[str]:
        """
        Get an agent-specific prompt for the query.
        
        Args:
            prompt_type: Type of prompt to retrieve
            **kwargs: Variables for prompt formatting
            
        Returns:
            Formatted prompt string or None
        """
        if self.prompt_manager:
            return self.prompt_manager.get_prompt(self.agent_name, prompt_type, **kwargs)
        return None
    
    def save_query_result(self, query: str, result: Any, metadata: Dict[str, Any] = None) -> bool:
        """
        Save query result as a conversation for the agent.
        
        Args:
            query: The original query
            result: Query result
            metadata: Additional metadata
            
        Returns:
            bool: True if saved successfully
        """
        if not self.storage:
            return False
        
        try:
            # Create a conversation from the query and result
            conversation = [
                {"from": "human", "value": query},
                {"from": "gpt", "value": str(result)}
            ]
            
            # Create metadata
            from agentChef.core.storage.conversation_storage import ConversationMetadata
            conv_metadata = ConversationMetadata(
                agent_name=self.agent_name,
                conversation_id="",  # Will be generated
                created_at="",  # Will be generated
                updated_at="",  # Will be generated
                context="data_query",
                template_format="query_response",
                num_turns=2,
                source="pandas_query",
                tags=["query", "data_analysis"]
            )
            
            if metadata:
                conv_metadata.tags.extend(metadata.get("tags", []))
            
            conv_id = self.storage.save_conversation(
                self.agent_name, 
                conversation, 
                conv_metadata
            )
            
            return conv_id is not None
            
        except Exception as e:
            logger.error(f"Failed to save query result: {e}")
            return False
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a query. To be implemented by subclasses.
        
        Args:
            query: Natural language query
            **kwargs: Additional query parameters
            
        Returns:
            Dict containing query results
        """
        raise NotImplementedError("Subclasses must implement query method")

class PandasQueryIntegration(AbstractQueryEngine):
    """
    Enhanced PandasQueryEngine integration with agent-specific prompts and storage.
    """
    
    def __init__(self, agent_name: str = "default", verbose=True, synthesize_response=True, **kwargs):
        """
        Initialize the PandasQueryIntegration.
        
        Args:
            agent_name: Name of the agent using this query engine
            verbose: Whether to print verbose output
            synthesize_response: Whether to synthesize a natural language response
            **kwargs: Additional configuration
        """
        super().__init__(agent_name, **kwargs)
        
        if not HAS_QUERY_ENGINE:
            logger.warning("PandasQueryEngine not available. Some features will be limited.")
            
        self.verbose = verbose
        self.synthesize_response = synthesize_response
        
        # Check if we can use local models
        try:
            from agentChef.core.ollama.ollama_interface import OllamaInterface
            self.ollama = OllamaInterface()
            self.use_local_models = True
        except ImportError:
            self.use_local_models = False
            logger.warning("Local models not available. Some features will be limited.")
        
    def create_query_engine(self, df: pd.DataFrame, 
                          prompt_type: str = "conversation_analysis",
                          custom_instructions: Optional[str] = None) -> Any:
        """
        Create a PandasQueryEngine with agent-specific prompts.
        
        Args:
            df: DataFrame to query
            prompt_type: Type of agent prompt to use
            custom_instructions: Override instructions
            
        Returns:
            PandasQueryEngine: Query engine for natural language queries
        """
        if not HAS_QUERY_ENGINE:
            logger.error("PandasQueryEngine not available")
            return None
            
        try:
            # Create the query engine
            query_engine = PandasQueryEngine(
                df=df,
                verbose=self.verbose,
                synthesize_response=self.synthesize_response
            )
            
            # Get agent-specific prompt or use custom instructions
            instructions = custom_instructions
            if not instructions and self.prompt_manager:
                # Prepare DataFrame info for the prompt
                df_info = f"Shape: {df.shape}, Columns: {list(df.columns)}"
                agent_prompt = self.get_agent_prompt(
                    prompt_type,
                    df_info=df_info,
                    context=f"DataFrame analysis for {self.agent_name}"
                )
                if agent_prompt:
                    instructions = agent_prompt
            
            # Update prompts if we have instructions
            if instructions:
                try:
                    # Get the current prompts
                    prompts = query_engine.get_prompts()
                    
                    # Create a new prompt with agent-specific instructions
                    new_pandas_prompt = PromptTemplate(instructions)
                    
                    # Update the prompts
                    query_engine.update_prompts({"pandas_prompt": new_pandas_prompt})
                    logger.info(f"Updated query engine with {prompt_type} prompt for agent {self.agent_name}")
                except Exception as e:
                    logger.warning(f"Could not update prompts: {e}")
                
            return query_engine
            
        except Exception as e:
            logger.error(f"Error creating PandasQueryEngine: {str(e)}")
            raise
    
    def query_dataframe(self, df: pd.DataFrame, query: str, 
                       prompt_type: str = "conversation_analysis",
                       custom_instructions: Optional[str] = None,
                       save_result: bool = True) -> Dict[str, Any]:
        """
        Query a DataFrame using natural language with agent-specific prompts.
        
        Args:
            df: DataFrame to query
            query: Natural language query
            prompt_type: Type of agent prompt to use
            custom_instructions: Override instructions
            save_result: Whether to save the result as a conversation
            
        Returns:
            Dict containing query results and metadata
        """
        try:
            # Create the query engine with agent-specific prompts
            query_engine = self.create_query_engine(df, prompt_type, custom_instructions)
            
            if not query_engine:
                return {
                    "error": "Could not create query engine",
                    "response": "Query engine not available",
                    "pandas_instructions": ""
                }
            
            # Execute the query
            response = query_engine.query(query)
            
            # Prepare result
            result = {
                "response": str(response),
                "pandas_instructions": response.metadata.get("pandas_instruction_str", ""),
                "raw_response": response,
                "agent_name": self.agent_name,
                "prompt_type": prompt_type
            }
            
            # Save result as conversation if requested
            if save_result:
                self.save_query_result(
                    query, 
                    result["response"], 
                    {"prompt_type": prompt_type, "tags": ["dataframe_query"]}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying DataFrame: {str(e)}")
            return {
                "error": str(e),
                "response": f"Error querying DataFrame: {str(e)}",
                "pandas_instructions": "",
                "agent_name": self.agent_name
            }
    
    def generate_agent_insights(self, df: pd.DataFrame, num_insights: int = 5) -> List[Dict[str, Any]]:
        """
        Generate insights from a DataFrame using agent-specific analysis prompts.
        
        Args:
            df: DataFrame to analyze
            num_insights: Number of insights to generate
            
        Returns:
            List of generated insights
        """
        # Get agent-specific insight queries if available
        insight_queries = self._get_agent_insight_queries(num_insights)
        
        # Generate insights
        insights = []
        for i, (query, prompt_type) in enumerate(insight_queries[:num_insights]):
            result = self.query_dataframe(
                df, 
                query, 
                prompt_type=prompt_type,
                save_result=False  # Don't save individual insights
            )
            insights.append({
                "query": query,
                "insight": result["response"],
                "pandas_code": result["pandas_instructions"],
                "prompt_type": prompt_type,
                "index": i + 1
            })
            
        # Save insights summary as conversation
        if insights:
            summary_query = f"Generated {len(insights)} insights for agent {self.agent_name}"
            summary_response = f"Analysis complete. Generated insights: {[insight['query'] for insight in insights]}"
            self.save_query_result(
                summary_query, 
                summary_response, 
                {"tags": ["insights", "analysis_summary"]}
            )
            
        return insights
    
    def _get_agent_insight_queries(self, num_insights: int) -> List[Tuple[str, str]]:
        """
        Get agent-specific insight queries.
        
        Args:
            num_insights: Number of insights requested
            
        Returns:
            List of (query, prompt_type) tuples
        """
        # Default queries with corresponding prompt types
        default_queries = [
            ("What is the overall structure and key characteristics of this dataset?", "conversation_analysis"),
            ("Identify any patterns or trends in the data that are relevant for agent training.", "knowledge_extraction"),
            ("What are the most important features or variables for conversation generation?", "template_generation"),
            ("Assess the quality and completeness of the data for agent development.", "performance_analysis"),
            ("Extract key knowledge points that the agent should learn from this data.", "knowledge_extraction"),
            ("Identify successful conversation patterns that can be used as templates.", "template_generation"),
            ("What insights can help improve the agent's performance?", "performance_analysis"),
            ("Summarize the main topics and themes present in the conversations.", "conversation_analysis")
        ]
        
        # Try to get agent-specific queries from prompt manager
        if self.prompt_manager and hasattr(self.prompt_manager, 'get_insight_queries'):
            agent_queries = self.prompt_manager.get_insight_queries(self.agent_name, num_insights)
            if agent_queries:
                return agent_queries
        
        return default_queries[:num_insights]

    def compare_agent_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        df1_name: str = "Original", df2_name: str = "Modified",
                        aspects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two DataFrames with agent-specific comparison prompts.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            df1_name: Name of the first DataFrame
            df2_name: Name of the second DataFrame
            aspects: Specific aspects to compare
            
        Returns:
            Dict containing comparison results
        """
        if aspects is None:
            aspects = ["shape", "schema", "missing_values", "statistics", "distributions"]
        
        # Create a combined DataFrame with a 'dataset' column
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        
        df1_copy['_dataset'] = df1_name
        df2_copy['_dataset'] = df2_name
        
        # Ensure column names match for concatenation
        common_columns = list(set(df1.columns).intersection(set(df2.columns)))
        
        if not common_columns:
            return {
                "error": "No common columns between datasets",
                "comparison": f"The datasets have no common columns. {df1_name} columns: {list(df1.columns)}, {df2_name} columns: {list(df2.columns)}"
            }
        
        # Use only common columns and add the dataset identifier
        df1_subset = df1_copy[common_columns + ['_dataset']]
        df2_subset = df2_copy[common_columns + ['_dataset']]
        
        # Concatenate for comparison
        combined_df = pd.concat([df1_subset, df2_subset], axis=0, ignore_index=True)
        
        # Generate comparison queries for each aspect
        comparison_results = {}
        for aspect in aspects:
            comparison_query = f"Compare the {aspect} between {df1_name} and {df2_name} datasets."
            result = self.query_dataframe(
                combined_df, 
                comparison_query, 
                prompt_type="performance_analysis",
                save_result=False
            )
            comparison_results[aspect] = result
        
        # Generate overall summary
        summary_query = f"Provide a comprehensive summary of the key differences between {df1_name} and {df2_name} datasets for agent {self.agent_name}."
        summary_result = self.query_dataframe(
            combined_df, 
            summary_query, 
            prompt_type="performance_analysis",
            save_result=True
        )
        
        return {
            "comparison_details": comparison_results,
            "overall_summary": summary_result["response"],
            "common_columns": common_columns,
            "unique_columns_df1": list(set(df1.columns) - set(common_columns)),
            "unique_columns_df2": list(set(df2.columns) - set(common_columns)),
            "agent_name": self.agent_name
        }


class OllamaLlamaIndexIntegration:
    """
    Integration between Ollama and LlamaIndex for local LLM-powered DataFrame querying.
    This is a fallback when OpenAI API is not available.
    """
    
    def __init__(self, ollama_model="llama3", verbose=True):
        """
        Initialize the OllamaLlamaIndexIntegration.
        
        Args:
            ollama_model (str): Ollama model to use.
            verbose (bool): Whether to print verbose output.
        """
        self.ollama_model = ollama_model
        self.verbose = verbose
        
        # Check if LlamaIndex is available
        if not HAS_QUERY_ENGINE:
            raise ImportError(
                "LlamaIndex not installed. Install with: pip install llama-index llama-index-experimental"
            )
            
        # Check if Ollama is available
        try:
            import ollama
        except ImportError:
            ollama = None
    
    def query_dataframe_with_ollama(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Query a DataFrame using Ollama as the LLM backend.
        
        This is a simplified version that doesn't use LlamaIndex directly but follows a similar approach.
        It sends the DataFrame info and query to Ollama and expects pandas code as a response.
        
        Args:
            df (pd.DataFrame): DataFrame to query.
            query (str): Natural language query.
            
        Returns:
            Dict[str, Any]: Query results including response and pandas code.
        """
        try:
            # Get DataFrame info
            df_info = f"DataFrame Info:\n{df.info()}\n\nSample (first 5 rows):\n{df.head().to_string()}"
            
            # Create system prompt
            system_prompt = """You are a data analysis assistant working with pandas DataFrames.
            You will be given a DataFrame description and a natural language query.
            
            Your task is to:
            1. Convert the natural language query into executable pandas Python code
            2. The code should be a solution to the query
            3. Return ONLY the pandas code expression that answers the query
            4. Do not include explanatory text, just the code
            
            The user will execute your code to get the answer."""
            
            # Create user prompt
            user_prompt = f"""DataFrame Information:
            {df_info}
            
            Query: {query}
            
            Please provide just the pandas code to answer this query."""
            
            # Get response from Ollama
            response = self.ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract the pandas code
            pandas_code = response['message']['content'].strip()
            
            # Clean up the code (remove markdown code blocks if present)
            if '```python' in pandas_code or '```' in pandas_code:
                # Extract code between backticks
                import re
                code_match = re.search(r'```(?:python)?\n(.*?)\n```', pandas_code, re.DOTALL)
                if code_match:
                    pandas_code = code_match.group(1).strip()
            
            # Execute the pandas code
            try:
                result = eval(pandas_code, {"df": df, "pd": pd})
                
                # Convert the result to string if it's not already
                if not isinstance(result, str):
                    if isinstance(result, pd.DataFrame):
                        result_str = result.to_string()
                    else:
                        result_str = str(result)
                else:
                    result_str = result
                
                return {
                    "response": result_str,
                    "pandas_code": pandas_code,
                    "raw_result": result
                }
                
            except Exception as e:
                return {
                    "error": f"Error executing pandas code: {str(e)}",
                    "response": f"Error executing pandas code: {str(e)}",
                    "pandas_code": pandas_code
                }
                
        except Exception as e:
            logger.error(f"Error querying DataFrame with Ollama: {str(e)}")
            return {
                "error": str(e),
                "response": f"Error querying DataFrame with Ollama: {str(e)}",
                "pandas_code": ""
            }


# Example usage
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Sample DataFrame
    df = pd.DataFrame({
        "city": ["Toronto", "Tokyo", "Berlin", "Sydney", "New York"],
        "population": [2930000, 13960000, 3645000, 5312000, 8419000],
        "country": ["Canada", "Japan", "Germany", "Australia", "USA"],
        "continent": ["North America", "Asia", "Europe", "Oceania", "North America"]
    })
    
    # Test with PandasQueryIntegration
    try:
        pandas_query = PandasQueryIntegration(verbose=True)
        
        # Test simple query
        result = pandas_query.query_dataframe(df, "What is the city with the highest population?")
        print(f"Response: {result['response']}")
        print(f"Pandas Code: {result['pandas_instructions']}")
        
        # Generate insights
        insights = pandas_query.generate_dataset_insights(df, num_insights=2)
        for insight in insights:
            print(f"\nQuery: {insight['query']}")
            print(f"Insight: {insight['insight']}")
            
    except ImportError:
        print("LlamaIndex not installed, skipping PandasQueryIntegration test")
        
        # Test with OllamaLlamaIndexIntegration
        try:
            ollama_query = OllamaLlamaIndexIntegration(ollama_model="llama3")
            
            # Test simple query
            result = ollama_query.query_dataframe_with_ollama(df, "What is the city with the highest population?")
            print(f"Response: {result['response']}")
            print(f"Pandas Code: {result['pandas_code']}")
            
        except ImportError:
            print("Ollama not installed, skipping OllamaLlamaIndexIntegration test")
