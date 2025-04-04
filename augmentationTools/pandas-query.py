"""pandas_query_integration.py
Integrates LlamaIndex's PandasQueryEngine into the research and dataset generation system.
This module provides utilities for natural language querying of pandas DataFrames.

Written By: @BorcherdingL
Date: 4/4/2025
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any

# LlamaIndex imports
try:
    from llama_index.experimental.query_engine import PandasQueryEngine
    from llama_index.core import PromptTemplate
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    logging.warning("LlamaIndex not installed. PandasQueryEngine will not be available.")

# Setup logging
logger = logging.getLogger(__name__)

class PandasQueryIntegration:
    """
    Integrates LlamaIndex's PandasQueryEngine for natural language querying of pandas DataFrames.
    """
    
    def __init__(self, openai_api_key=None, verbose=True, synthesize_response=True):
        """
        Initialize the PandasQueryIntegration.
        
        Args:
            openai_api_key (str, optional): OpenAI API key. If None, uses environment variable.
            verbose (bool): Whether to print verbose output.
            synthesize_response (bool): Whether to synthesize a natural language response.
        """
        self.verbose = verbose
        self.synthesize_response = synthesize_response
        
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Check if LlamaIndex is available
        if not HAS_LLAMA_INDEX:
            raise ImportError(
                "LlamaIndex not installed. Install with: pip install llama-index llama-index-experimental"
            )
    
    def create_query_engine(self, df: pd.DataFrame, custom_instructions: Optional[str] = None) -> Any:
        """
        Create a PandasQueryEngine for the given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to query.
            custom_instructions (str, optional): Custom instructions for the query engine.
            
        Returns:
            PandasQueryEngine: Query engine for natural language queries.
        """
        try:
            # Create the query engine
            query_engine = PandasQueryEngine(
                df=df,
                verbose=self.verbose,
                synthesize_response=self.synthesize_response
            )
            
            # Update prompts if custom instructions are provided
            if custom_instructions:
                # Get the current prompts
                prompts = query_engine.get_prompts()
                
                # Create a new prompt with custom instructions
                new_pandas_prompt = PromptTemplate(
                    f"""You are working with a pandas dataframe in Python.
                    The name of the dataframe is `df`.
                    This is the result of `print(df.head())`:
                    {{df_str}}

                    Follow these instructions:
                    {custom_instructions}
                    Query: {{query_str}}

                    Expression:"""
                )
                
                # Update the prompts
                query_engine.update_prompts({"pandas_prompt": new_pandas_prompt})
                
            return query_engine
            
        except Exception as e:
            logger.error(f"Error creating PandasQueryEngine: {str(e)}")
            raise
    
    def query_dataframe(self, df: pd.DataFrame, query: str, custom_instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Query a DataFrame using natural language.
        
        Args:
            df (pd.DataFrame): DataFrame to query.
            query (str): Natural language query.
            custom_instructions (str, optional): Custom instructions for the query engine.
            
        Returns:
            Dict[str, Any]: Query results including response and metadata.
        """
        try:
            # Create the query engine
            query_engine = self.create_query_engine(df, custom_instructions)
            
            # Execute the query
            response = query_engine.query(query)
            
            # Return the response and metadata
            return {
                "response": str(response),
                "pandas_instructions": response.metadata.get("pandas_instruction_str", ""),
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Error querying DataFrame: {str(e)}")
            return {
                "error": str(e),
                "response": f"Error querying DataFrame: {str(e)}",
                "pandas_instructions": ""
            }
    
    def generate_dataset_insights(self, df: pd.DataFrame, num_insights: int = 5) -> List[Dict[str, Any]]:
        """
        Generate insights from a DataFrame using PandasQueryEngine.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze.
            num_insights (int): Number of insights to generate.
            
        Returns:
            List[Dict[str, Any]]: List of generated insights.
        """
        # Define common analysis queries
        analysis_queries = [
            "What is the overall shape and structure of this dataset?",
            "Identify any missing values or data quality issues in the dataset.",
            "What are the key statistical properties of the numerical columns?",
            "Are there any significant correlations between variables?",
            "What insights can you provide about the distribution of categorical variables?",
            "Identify any potential outliers in the numerical columns.",
            "What time-based patterns or trends exist in the data?",
            "Which factors seem most predictive of the target variable?",
            "Summarize the key findings from this dataset in 3-5 bullet points."
        ]
        
        # Select queries to run
        selected_queries = analysis_queries[:min(num_insights, len(analysis_queries))]
        
        # Generate insights
        insights = []
        for query in selected_queries:
            result = self.query_dataframe(df, query)
            insights.append({
                "query": query,
                "insight": result["response"],
                "pandas_code": result["pandas_instructions"]
            })
            
        return insights

    def compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        df1_name: str = "Original", df2_name: str = "Modified",
                        aspects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare two DataFrames and generate insights about the differences.
        
        Args:
            df1 (pd.DataFrame): First DataFrame.
            df2 (pd.DataFrame): Second DataFrame.
            df1_name (str): Name of the first DataFrame.
            df2_name (str): Name of the second DataFrame.
            aspects (List[str], optional): Specific aspects to compare.
            
        Returns:
            Dict[str, Any]: Comparison results.
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
        
        # Create the query engine for the combined dataset
        query_engine = self.create_query_engine(combined_df)
        
        # Define comparison queries based on aspects
        comparison_queries = []
        
        if "shape" in aspects:
            comparison_queries.append(f"Compare the number of rows and columns between {df1_name} and {df2_name}.")
        
        if "schema" in aspects:
            comparison_queries.append(f"Compare the data types and structure between {df1_name} and {df2_name}.")
        
        if "missing_values" in aspects:
            comparison_queries.append(f"Compare the missing values between {df1_name} and {df2_name}.")
        
        if "statistics" in aspects:
            comparison_queries.append(f"Compare the statistical properties of numerical columns between {df1_name} and {df2_name}.")
        
        if "distributions" in aspects:
            comparison_queries.append(f"Compare the distributions of key variables between {df1_name} and {df2_name}.")
        
        # Execute comparison queries
        comparison_results = {}
        for query in comparison_queries:
            try:
                response = query_engine.query(query)
                comparison_results[query] = {
                    "response": str(response),
                    "pandas_code": response.metadata.get("pandas_instruction_str", "")
                }
            except Exception as e:
                comparison_results[query] = {
                    "error": str(e),
                    "response": f"Error comparing datasets: {str(e)}"
                }
        
        # Generate an overall summary
        try:
            summary_query = f"Provide a comprehensive summary of the key differences between {df1_name} and {df2_name} datasets."
            summary_response = query_engine.query(summary_query)
            overall_summary = str(summary_response)
        except Exception as e:
            overall_summary = f"Error generating summary: {str(e)}"
        
        return {
            "comparison_details": comparison_results,
            "overall_summary": overall_summary,
            "common_columns": common_columns,
            "unique_columns_df1": list(set(df1.columns) - set(common_columns)),
            "unique_columns_df2": list(set(df2.columns) - set(common_columns))
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
        if not HAS_LLAMA_INDEX:
            raise ImportError(
                "LlamaIndex not installed. Install with: pip install llama-index llama-index-experimental"
            )
            
        # Check if Ollama is available
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError("Ollama not installed. Install with: pip install ollama")
    
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
