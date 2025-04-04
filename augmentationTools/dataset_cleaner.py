import os
import pandas as pd
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import re
import numpy as np

class DatasetCleaner:
    """
    A class to clean and validate expanded datasets by comparing them to original conversations
    and identifying/fixing quality issues using NLP queries through an LLM interface.
    
    Works alongside DatasetExpander to ensure high-quality conversation data.
    """
    
    def __init__(self, ollama_interface, output_dir="./cleaned_output"):
        """
        Initialize the DatasetCleaner.
        
        Args:
            ollama_interface: An interface to Ollama for generating text and analyzing datasets
            output_dir (str): Directory to save cleaned datasets
        """
        self.ollama_interface = ollama_interface
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
    def analyze_dataset(self, 
                       original_conversations: List[List[Dict[str, str]]], 
                       expanded_conversations: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        Analyze expanded conversations to identify potential quality issues compared to originals.
        
        Args:
            original_conversations: List of original conversations
            expanded_conversations: List of expanded conversations
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing expanded dataset quality...")
        
        analysis_results = {
            "total_original": len(original_conversations),
            "total_expanded": len(expanded_conversations),
            "issues_by_type": {},
            "detailed_issues": []
        }
        
        # Convert to dataframes for easier analysis
        orig_df = self._convert_conversations_to_df(original_conversations)
        expanded_df = self._convert_conversations_to_df(expanded_conversations)
        
        # Analyze content length differences
        orig_df['content_length'] = orig_df['value'].apply(len)
        expanded_df['content_length'] = expanded_df['value'].apply(len)
        
        length_diff = self._analyze_length_differences(orig_df, expanded_df)
        analysis_results["length_analysis"] = length_diff
        
        # Run NLP analysis on sample pairs
        semantic_issues = self._analyze_semantic_quality(original_conversations, expanded_conversations)
        analysis_results["issues_by_type"] = semantic_issues["issues_by_type"]
        analysis_results["detailed_issues"] = semantic_issues["detailed_issues"]
        
        return analysis_results
    
    def clean_dataset(self, 
                     original_conversations: List[List[Dict[str, str]]], 
                     expanded_conversations: List[List[Dict[str, str]]],
                     cleaning_criteria: Dict[str, bool] = None) -> List[List[Dict[str, str]]]:
        """
        Clean the expanded dataset by fixing identified issues.
        
        Args:
            original_conversations: List of original conversations
            expanded_conversations: List of expanded conversations
            cleaning_criteria: Dictionary of criteria to use for cleaning:
                - fix_hallucinations (bool): Fix factual errors or hallucinations
                - normalize_style (bool): Ensure consistent style
                - correct_grammar (bool): Fix grammar issues
                - ensure_coherence (bool): Ensure conversation flow is coherent
            
        Returns:
            List of cleaned conversations
        """
        if cleaning_criteria is None:
            cleaning_criteria = {
                "fix_hallucinations": True,
                "normalize_style": True,
                "correct_grammar": True,
                "ensure_coherence": True
            }
        
        self.logger.info("Cleaning expanded dataset...")
        
        # First analyze the dataset
        analysis = self.analyze_dataset(original_conversations, expanded_conversations)
        
        # Clean conversations based on analysis
        cleaned_conversations = []
        
        for idx, expanded_conv in enumerate(tqdm(expanded_conversations, desc="Cleaning conversations")):
            # Find corresponding original conversation (if it exists)
            original_idx = idx % len(original_conversations)
            original_conv = original_conversations[original_idx]
            
            # Check if this conversation has issues that need fixing
            needs_cleaning = False
            for issue in analysis["detailed_issues"]:
                if issue["conversation_idx"] == idx:
                    needs_cleaning = True
                    break
            
            if needs_cleaning:
                # Clean this conversation
                cleaned_conv = self._clean_conversation(
                    original_conv, 
                    expanded_conv,
                    cleaning_criteria
                )
                cleaned_conversations.append(cleaned_conv)
            else:
                # Keep as is if no issues detected
                cleaned_conversations.append(expanded_conv)
                
        return cleaned_conversations
    
    def _clean_conversation(self, 
                           original_conv: List[Dict[str, str]], 
                           expanded_conv: List[Dict[str, str]],
                           criteria: Dict[str, bool]) -> List[Dict[str, str]]:
        """
        Clean a single conversation by fixing issues.
        
        Args:
            original_conv: Original conversation
            expanded_conv: Expanded conversation with potential issues
            criteria: Cleaning criteria
            
        Returns:
            Cleaned conversation
        """
        cleaned_conv = []
        
        # Compare each turn in the conversation
        for i, (expanded_turn, original_turn) in enumerate(zip(expanded_conv, original_conv)):
            source = expanded_turn['from']
            
            # Skip if this turn doesn't need cleaning (based on source type)
            if source not in ["human", "gpt"]:
                cleaned_conv.append(expanded_turn.copy())
                continue
                
            # Check and clean this turn
            if self._needs_cleaning(expanded_turn, original_turn, criteria):
                cleaned_value = self._clean_turn_content(
                    original_turn['value'],
                    expanded_turn['value'],
                    source,
                    i,
                    criteria
                )
                
                cleaned_conv.append({
                    'from': source,
                    'value': cleaned_value
                })
            else:
                cleaned_conv.append(expanded_turn.copy())
                
        return cleaned_conv
    
    def _needs_cleaning(self, 
                       expanded_turn: Dict[str, str], 
                       original_turn: Dict[str, str],
                       criteria: Dict[str, bool]) -> bool:
        """
        Determine if a turn needs cleaning based on quick heuristics.
        
        Args:
            expanded_turn: Expanded conversation turn
            original_turn: Original conversation turn
            criteria: Cleaning criteria
            
        Returns:
            True if turn needs cleaning, False otherwise
        """
        expanded_value = expanded_turn['value']
        original_value = original_turn['value']
        
        # Check for obvious issues
        if criteria.get("correct_grammar", False):
            # Look for common grammar issues
            grammar_issues = re.search(r'\b(i\s+is|they\s+is|we\s+is|you\s+is)\b', 
                                      expanded_value, 
                                      re.IGNORECASE)
            if grammar_issues:
                return True
        
        if criteria.get("ensure_coherence", False):
            # Check if expanded content is too short or too long compared to original
            if len(expanded_value) < len(original_value) * 0.5 or len(expanded_value) > len(original_value) * 2:
                return True
        
        # More sophisticated checks would need LLM analysis
        return False
    
    def _clean_turn_content(self, 
                           original_content: str, 
                           expanded_content: str, 
                           source: str,
                           turn_idx: int,
                           criteria: Dict[str, bool]) -> str:
        """
        Clean the content of a conversation turn using LLM.
        
        Args:
            original_content: Content from original turn
            expanded_content: Content from expanded turn
            source: Source of the turn ('human' or 'gpt')
            turn_idx: Index of the turn in the conversation
            criteria: Cleaning criteria
            
        Returns:
            Cleaned turn content
        """
        system_prompt = """You are a conversation data cleaning assistant. Your task is to fix issues in AI-generated 
        conversation data while maintaining the original intent and meaning. Follow these guidelines:
        
        1. Fix any factual errors or hallucinations by referring to the original content
        2. Ensure the text maintains a consistent style appropriate for the source (human or AI assistant)
        3. Correct grammar and spelling issues
        4. Ensure the content is coherent in the context of a conversation
        5. The cleaned content should be a refined version of the expanded content, not just a copy of the original
        
        Return only the cleaned content without additional commentary or explanations."""
        
        # Build criteria-specific instructions
        criteria_instructions = []
        if criteria.get("fix_hallucinations", False):
            criteria_instructions.append("- Fix any factual inconsistencies or hallucinations by referring to the original content")
        if criteria.get("normalize_style", False):
            criteria_instructions.append("- Ensure consistent style and tone appropriate for the source type")
        if criteria.get("correct_grammar", False):
            criteria_instructions.append("- Fix any grammar, spelling, or punctuation errors")
        if criteria.get("ensure_coherence", False):
            criteria_instructions.append("- Ensure the content flows naturally in a conversation context")
        
        criteria_str = "\n".join(criteria_instructions)
        
        user_prompt = f"""Original content: {original_content}
        
        Expanded content (needs cleaning): {expanded_content}
        
        Source type: {source}
        Turn index: {turn_idx}
        
        Please clean the expanded content according to these criteria:
        {criteria_str}
        
        Provide only the cleaned content without any additional text."""

        try:
            response = self.ollama_interface.chat(messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            
            cleaned_content = response['message']['content'].strip()
            return cleaned_content
            
        except Exception as e:
            self.logger.error(f"Error cleaning turn content: {str(e)}")
            return expanded_content  # Return the original expanded content if cleaning fails
    
    def _convert_conversations_to_df(self, conversations: List[List[Dict[str, str]]]) -> pd.DataFrame:
        """
        Convert conversations to a DataFrame for analysis.
        
        Args:
            conversations: List of conversations
            
        Returns:
            DataFrame with conversation data
        """
        data = []
        
        for i, conv in enumerate(conversations):
            for j, turn in enumerate(conv):
                data.append({
                    'conversation_idx': i,
                    'turn_idx': j,
                    'from': turn['from'],
                    'value': turn['value']
                })
                
        return pd.DataFrame(data)
    
    def _analyze_length_differences(self, orig_df: pd.DataFrame, expanded_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze differences in content length between original and expanded datasets.
        
        Args:
            orig_df: DataFrame with original conversations
            expanded_df: DataFrame with expanded conversations
            
        Returns:
            Dictionary with length analysis results
        """
        orig_stats = orig_df.groupby('from')['content_length'].agg(['mean', 'std', 'min', 'max']).to_dict()
        expanded_stats = expanded_df.groupby('from')['content_length'].agg(['mean', 'std', 'min', 'max']).to_dict()
        
        # Calculate percent differences
        diff = {}
        for source in orig_stats['mean'].keys():
            if source in expanded_stats['mean']:
                diff[source] = {
                    'mean_diff_pct': (expanded_stats['mean'][source] - orig_stats['mean'][source]) / orig_stats['mean'][source] * 100,
                    'std_diff_pct': (expanded_stats['std'][source] - orig_stats['std'][source]) / orig_stats['std'][source] * 100,
                    'min_diff': expanded_stats['min'][source] - orig_stats['min'][source],
                    'max_diff': expanded_stats['max'][source] - orig_stats['max'][source]
                }
                
        return {
            'original_stats': orig_stats,
            'expanded_stats': expanded_stats,
            'differences': diff
        }
    
    def _analyze_semantic_quality(self, 
                                 original_conversations: List[List[Dict[str, str]]], 
                                 expanded_conversations: List[List[Dict[str, str]]],
                                 sample_size: int = 10) -> Dict[str, Any]:
        """
        Analyze semantic quality of expanded conversations using LLM.
        
        Args:
            original_conversations: List of original conversations
            expanded_conversations: List of expanded conversations
            sample_size: Number of conversations to sample for analysis
            
        Returns:
            Dictionary with semantic quality analysis results
        """
        self.logger.info(f"Analyzing semantic quality on {sample_size} samples...")
        
        # Sample conversation indices
        if len(expanded_conversations) <= sample_size:
            sample_indices = list(range(len(expanded_conversations)))
        else:
            sample_indices = sorted(np.random.choice(
                len(expanded_conversations), 
                size=sample_size, 
                replace=False
            ))
        
        issues_by_type = {
            "hallucination": 0,
            "style_inconsistency": 0,
            "grammar_error": 0,
            "incoherence": 0,
            "other": 0
        }
        
        detailed_issues = []
        
        for idx in tqdm(sample_indices, desc="Analyzing samples"):
            expanded_conv = expanded_conversations[idx]
            
            # Find corresponding original conversation
            original_idx = idx % len(original_conversations)
            original_conv = original_conversations[original_idx]
            
            # Analyze this conversation pair
            issues = self._analyze_conversation_pair(original_conv, expanded_conv, idx)
            
            # Update counts
            for issue in issues:
                issue_type = issue["issue_type"]
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
                detailed_issues.append(issue)
                
        return {
            "issues_by_type": issues_by_type,
            "detailed_issues": detailed_issues
        }
    
    def _analyze_conversation_pair(self, 
                                  original_conv: List[Dict[str, str]], 
                                  expanded_conv: List[Dict[str, str]],
                                  conv_idx: int) -> List[Dict[str, Any]]:
        """
        Analyze a pair of original and expanded conversations using LLM.
        
        Args:
            original_conv: Original conversation
            expanded_conv: Expanded conversation
            conv_idx: Index of the expanded conversation
            
        Returns:
            List of issue dictionaries
        """
        system_prompt = """You are a conversation data quality analyst. Your task is to identify quality issues in 
        expanded (paraphrased) conversations compared to the original conversations. For each issue you find, 
        categorize it as one of the following:
        
        - hallucination: When the expanded content introduces facts not implied by the original
        - style_inconsistency: When the expanded content has an inconsistent or inappropriate style
        - grammar_error: When the expanded content has grammar, spelling, or punctuation errors
        - incoherence: When the expanded content lacks logical flow or coherence
        - other: Any other quality issues
        
        Format your response as a JSON list of objects with these fields:
        - issue_type: One of the categories above
        - turn_idx: The index of the turn in the conversation where the issue occurs
        - description: A brief description of the issue
        - severity: A number from 1-5 where 5 is most severe
        """
        
        user_prompt = f"""Original conversation:
        {json.dumps(original_conv, indent=2)}
        
        Expanded conversation:
        {json.dumps(expanded_conv, indent=2)}
        
        Identify quality issues in the expanded conversation compared to the original.
        Limit your analysis to significant issues that would affect the quality of the dataset.
        
        Return your analysis as a JSON list as described in your instructions."""

        try:
            response = self.ollama_interface.chat(messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            
            # Extract JSON from response
            response_text = response['message']['content'].strip()
            json_text = re.search(r'(\[.*\])', response_text, re.DOTALL)
            
            if json_text:
                issues = json.loads(json_text.group(1))
                
                # Add conversation index to each issue
                for issue in issues:
                    issue["conversation_idx"] = conv_idx
                
                return issues
            else:
                self.logger.warning(f"Failed to parse issues JSON from response: {response_text}")
                return []
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation pair: {str(e)}")
            return []
    
    def query_dataset(self, 
                     expanded_conversations: List[List[Dict[str, str]]], 
                     query: str) -> Dict[str, Any]:
        """
        Run a natural language query on the expanded dataset using PandasQueryEngine-like functionality.
        
        Args:
            expanded_conversations: List of expanded conversations
            query: Natural language query about the dataset
            
        Returns:
            Dictionary with query results
        """
        self.logger.info(f"Running query on dataset: {query}")
        
        # Convert to DataFrame for analysis
        df = self._convert_conversations_to_df(expanded_conversations)
        
        # Additional derived features to help with analysis
        df['content_length'] = df['value'].apply(len)
        df['word_count'] = df['value'].apply(lambda x: len(x.split()))
        df['sentence_count'] = df['value'].apply(lambda x: len(re.findall(r'[.!?]+', x)) + 1)
        df['avg_word_length'] = df['value'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
        df['question'] = df['value'].apply(lambda x: x.strip().endswith('?'))
        
        # Create a sample of the dataframe for the LLM to understand its structure
        sample_df = df.head(5)
        df_info = f"DataFrame Info:\n{df.info()}\n\nSample:\n{sample_df.to_string()}"
        
        system_prompt = """You are a data analysis assistant working with conversation datasets. 
        You will be given a pandas DataFrame containing conversation data and a natural language query about this data.
        
        Your task is to:
        1. Convert the natural language query into executable pandas Python code
        2. Explain your approach
        3. Summarize the results in a user-friendly way
        
        The DataFrame has these columns:
        - conversation_idx: Index of the conversation
        - turn_idx: Index of the turn within a conversation
        - from: Source of the message ('human' or 'gpt')
        - value: Text content of the message
        - content_length: Character count of the value
        - word_count: Number of words in the value
        - sentence_count: Number of sentences in the value
        - avg_word_length: Average word length in the value
        - question: Boolean indicating if the message is a question
        
        Return your response as a JSON object with these fields:
        - pandas_code: The pandas code that would answer the query
        - explanation: Explanation of your approach
        - summary: User-friendly summary of the results
        """
        
        user_prompt = f"""DataFrame Information:
        {df_info}
        
        Query: {query}
        
        Please analyze this conversation dataset according to the query."""

        try:
            response = self.ollama_interface.chat(messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            
            # Extract JSON from response
            response_text = response['message']['content'].strip()
            
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Extract JSON if it's embedded in text
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Create structured result if not in JSON format
                    result = {
                        "pandas_code": "Could not extract pandas code",
                        "explanation": "Failed to parse JSON response",
                        "summary": response_text
                    }
            
            # Try to execute the pandas code if present
            if 'pandas_code' in result and result['pandas_code']:
                try:
                    # Create a safe environment with access to the DataFrame
                    exec_globals = {'df': df, 'pd': pd, 'np': np, 're': re}
                    exec_result = eval(result['pandas_code'], exec_globals)
                    
                    # Add the execution result
                    if isinstance(exec_result, pd.DataFrame):
                        result['execution_result'] = exec_result.to_dict(orient='records')
                    else:
                        result['execution_result'] = str(exec_result)
                except Exception as e:
                    result['execution_error'] = str(e)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running query on dataset: {str(e)}")
            return {
                "error": str(e),
                "query": query
            }
    
    def save_cleaning_report(self, 
                           analysis_results: Dict[str, Any], 
                           filename: str = "cleaning_report.json") -> str:
        """
        Save dataset analysis results to a file.
        
        Args:
            analysis_results: Dictionary with analysis results
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2)
                
        self.logger.info(f"Saved cleaning report to {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    import ollama
    
    # Define a simple ollama_interface (same as in DatasetExpander)
    class OllamaInterface:
        def __init__(self, model_name="llama3"):
            self.model = model_name
            
        def chat(self, messages):
            return ollama.chat(model=self.model, messages=messages)
    
    # Initialize the cleaner
    ollama_interface = OllamaInterface(model_name="llama3")
    cleaner = DatasetCleaner(ollama_interface, output_dir="./cleaned_data")
    
    # Import the dataset expander
    from dataset_expander import DatasetExpander
    
    # Load expanded dataset (sample code)
    expander = DatasetExpander(ollama_interface, output_dir="./expanded_data")
    original_conversations = expander.load_conversations_from_jsonl("./expanded_data/original_conversations.jsonl")
    expanded_conversations = expander.load_conversations_from_jsonl("./expanded_data/expanded_conversations.jsonl")
    
    # Analyze dataset
    analysis = cleaner.analyze_dataset(original_conversations, expanded_conversations)
    cleaner.save_cleaning_report(analysis, "dataset_analysis.json")
    
    # Example NLP query
    query_result = cleaner.query_dataset(
        expanded_conversations,
        "What is the average response length difference between human and gpt messages?"
    )
    print(f"Query result: {json.dumps(query_result, indent=2)}")
    
    # Clean dataset
    cleaned_conversations = cleaner.clean_dataset(
        original_conversations, 
        expanded_conversations,
        cleaning_criteria={
            "fix_hallucinations": True,
            "normalize_style": True,
            "correct_grammar": True,
            "ensure_coherence": True
        }
    )
    
    # Save cleaned dataset
    clean_output_path = expander.save_conversations_to_jsonl(
        cleaned_conversations, 
        "cleaned_conversations"
    )
    
    print(f"Cleaned dataset saved to: {clean_output_path}")