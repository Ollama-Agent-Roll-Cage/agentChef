import pandas as pd
import json
import re
import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import numpy as np
import random

# Import the PandasQueryIntegration
try:
    from agentChef.pandas_query import PandasQueryIntegration, OllamaLlamaIndexIntegration
    HAS_QUERY_INTEGRATION = True
except ImportError:
    HAS_QUERY_INTEGRATION = False
    logging.warning("LlamaIndex not installed. PandasQueryEngine will not be available.")

class DatasetExpander:
    """
    A class to expand datasets by generating paraphrases and variations of conversation data,
    with control over which fields remain static and which are dynamically generated.
    Works with conversation data in the format produced by OllamaConversationGenerator.
    """
    
    def __init__(self, ollama_interface, output_dir="./output", use_llama_index=True):
        """
        Initialize the DatasetExpander.
        
        Args:
            ollama_interface: An interface to Ollama for generating text
            output_dir (str): Directory to save expanded datasets
            use_llama_index (bool): Whether to use LlamaIndex for advanced DataFrame analysis
        """
        self.ollama_interface = ollama_interface
        self.output_dir = output_dir
        self.use_llama_index = use_llama_index
        
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Initialize pandas query integration if available
        self.pandas_query = None
        self.ollama_query = None
        
        if use_llama_index and HAS_QUERY_INTEGRATION:
            try:
                self.pandas_query = PandasQueryIntegration(
                    verbose=True,
                    synthesize_response=True
                )
                self.logger.info("Initialized PandasQueryIntegration for advanced DataFrame analysis")
            except ImportError:
                self.logger.warning("LlamaIndex not available. Falling back to OllamaLlamaIndexIntegration.")
                try:
                    self.ollama_query = OllamaLlamaIndexIntegration(
                        ollama_model=getattr(ollama_interface, 'model', "llama3"),
                        verbose=True
                    )
                    self.logger.info("Initialized OllamaLlamaIndexIntegration for DataFrame analysis")
                except ImportError:
                    self.logger.warning("Neither LlamaIndex nor Ollama are available for advanced DataFrame querying")
    
    def set_model(self, model_name: str):
        """Update the model used for dataset expansion."""
        if self.ollama_interface:
            self.ollama_interface.set_model(model_name)
        self.logger.info(f"Set dataset expander model to: {model_name}")
        
        # Update query engine if available
        if self.ollama_query:
            self.ollama_query.set_model(model_name)

    def expand_conversation_dataset(self, 
                                  conversations: List[List[Dict[str, str]]], 
                                  expansion_factor: int = 3,
                                  static_fields: Dict[str, bool] = None,
                                  reference_fields: List[str] = None) -> List[List[Dict[str, str]]]:
        """
        Expand a dataset of conversations by generating paraphrases.
        
        Args:
            conversations: List of conversations, where each conversation is a list of turns
                         Each turn is a dict with 'from' and 'value' keys
            expansion_factor: Number of variations to generate for each conversation
            static_fields: Dict mapping field names ("human", "gpt") to boolean indicating if they should remain static
                         If None, defaults to {'human': False, 'gpt': False} (all fields are dynamic)
            reference_fields: List of fields to use as reference values when generating paraphrases
        
        Returns:
            List of expanded conversations
        """
        if static_fields is None:
            static_fields = {'human': False, 'gpt': False}
        
        if reference_fields is None:
            reference_fields = []
            
        expanded_conversations = []
        
        for i, conversation in enumerate(tqdm(conversations, desc="Expanding conversations")):
            self.logger.info(f"Expanding conversation {i+1}/{len(conversations)}")
            
            # Create variations of this conversation
            for j in range(expansion_factor):
                expanded_conversation = []
                
                # Extract reference values from the original conversation if needed
                reference_values = {}
                for turn in conversation:
                    if turn['from'] in reference_fields:
                        reference_values[turn['from']] = turn['value']
                
                # Process each turn in the conversation
                for k, turn in enumerate(conversation):
                    source = turn['from']  # 'human' or 'gpt'
                    
                    if static_fields.get(source, False):
                        # Keep this field static (unchanged)
                        expanded_conversation.append(turn.copy())
                    else:
                        # Generate a paraphrase for this turn
                        original_value = turn['value']
                        paraphrased_value = self.paraphrase_text(
                            original_value, 
                            reference_values,
                            is_question=self._is_question(original_value)
                        )
                        
                        expanded_conversation.append({
                            'from': source,
                            'value': paraphrased_value
                        })
                
                expanded_conversations.append(expanded_conversation)
                
        return expanded_conversations
    
    def paraphrase_text(self, text: str, reference_values: Dict[str, str] = None, is_question: bool = None) -> str:
        """
        Generate a paraphrase of the given text.
        
        Args:
            text: Text to paraphrase
            reference_values: Dictionary of reference values to incorporate
            is_question: Whether the text is a question (if None, will be detected automatically)
        
        Returns:
            Paraphrased text
        """
        if not text.strip():
            return text
            
        if is_question is None:
            is_question = self._is_question(text)
            
        if reference_values is None:
            reference_values = {}
            
        system_prompt = """You are a paraphrasing assistant. Your task is to rephrase the given text while 
        maintaining its original meaning and incorporating any provided reference values. 
        Do not add any explanatory text or meta-information."""
        
        user_prompt = f"""Original text: {text}
        Reference values: {reference_values}
        Is question: {is_question}
        
        Please rephrase the text, maintaining its core meaning and incorporating the reference values where appropriate. 
        If it's a question, keep it as a question. If it's a statement, keep it as a statement.
        Ensure the paraphrased text is coherent and contextually relevant.
        Provide only the paraphrased text without any additional explanations or formatting."""

        try:
            response = self.ollama_interface.chat(messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])
            
            paraphrased_text = response['message']['content'].strip()
            
            # Ensure question mark is present if this is a question
            if is_question and not paraphrased_text.endswith('?'):
                paraphrased_text += '?'
            elif not is_question and paraphrased_text.endswith('?'):
                paraphrased_text = paraphrased_text[:-1] + '.'
            
            # Clean the generated content
            cleaned_text = self.clean_generated_content(paraphrased_text, is_question)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error paraphrasing text: {str(e)}")
            return text  # Return original text on error
    
    def verify_paraphrase(self, original: str, paraphrased: str, reference: Dict[str, str], is_question: bool) -> str:
        """
        Verify that the paraphrased text maintains the meaning of the original.
        
        Args:
            original: Original text
            paraphrased: Paraphrased text
            reference: Reference values
            is_question: Whether the text is a question
            
        Returns:
            Verified or corrected paraphrased text
        """
        system_prompt = """You are a verification assistant. Your task is to ensure that the paraphrased content 
        maintains the original meaning, format (question or statement), and incorporates the reference values correctly.
        If the paraphrase is accurate, return it as-is. If not, provide a corrected version."""
        
        user_prompt = f"""Original: {original}
        Paraphrased: {paraphrased}
        Reference values: {reference}
        Is question: {is_question}
        
        Verify that the paraphrased content maintains the original meaning, format (question or statement), 
        and correctly incorporates the reference values. If it does, return the paraphrased content. 
        If not, provide a corrected version that accurately reflects the original meaning, format, 
        and includes the reference values.
        Do not include any explanatory text or meta-information in your response."""

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        
        verified_text = response['message']['content'].strip()
        
        # Only add question mark if:
        # 1. This is a question
        # 2. The verified text doesn't start with "Verified:" (which indicates it's a verification message)
        # 3. The text doesn't already end with a question mark
        if is_question and not verified_text.startswith("Verified:") and not verified_text.endswith('?'):
            verified_text += '?'
        # If this is not a question but the text ends with a question mark (and it's not a verification message)
        elif not is_question and not verified_text.startswith("Verified:") and verified_text.endswith('?'):
            verified_text = verified_text[:-1] + '.'
        
        return verified_text
            
    def clean_generated_content(self, text: str, is_question: bool) -> str:
        """
        Clean generated content by removing explanatory phrases, normalizing punctuation, etc.
        
        Args:
            text: Text to clean
            is_question: Whether the text is a question
            
        Returns:
            Cleaned text
        """
        # Remove any explanatory phrases or meta-information
        text = re.sub(r'^(Generated content:|Verified content:|Corrected version:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(Verification result:.*|Reference Command:.*|Note:.*|Verified Response:.*)$', '', text, flags=re.IGNORECASE)
        
        # Remove any remaining placeholder-like patterns
        text = re.sub(r'___[A-Za-z_]+___', '', text)
        
        # Remove any quotes that might have been added
        text = text.strip('"\'')
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Ensure the text starts with a capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure the text ends with proper punctuation
        if is_question and not text.endswith('?'):
            text += '?'
        elif not is_question and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def generate_conversations_from_paper(self, 
                                         paper_content: str, 
                                         conversation_generator,
                                         num_chunks: int = 5,
                                         num_turns: int = 3,
                                         expansion_factor: int = 2,
                                         static_fields: Dict[str, bool] = None,
                                         reference_fields: List[str] = None) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
        """
        Generate conversations from a paper and then expand the dataset.
        
        Args:
            paper_content: The content of the research paper
            conversation_generator: An instance of OllamaConversationGenerator
            num_chunks: Number of chunks to create from the paper
            num_turns: Number of turns per conversation
            expansion_factor: Number of variations to create per conversation
            static_fields: Dict mapping field names to boolean indicating if they should remain static
            reference_fields: List of fields to use as reference when generating paraphrases
            
        Returns:
            Tuple of (original conversations, expanded conversations)
        """
        self.logger.info("Generating conversations from paper content")
        
        # Chunk the paper content
        chunks = conversation_generator.chunk_text(
            paper_content, 
            chunk_size=2000, 
            overlap=200
        )
        
        # Limit to the requested number of chunks
        if len(chunks) > num_chunks:
            # Take evenly spaced chunks rather than just the first N
            indices = np.linspace(0, len(chunks)-1, num_chunks, dtype=int)
            chunks = [chunks[i] for i in indices]
        
        # Generate conversations for each chunk
        conversations = []
        for i, chunk in enumerate(tqdm(chunks, desc="Generating conversations")):
            self.logger.info(f"Generating conversation {i+1}/{len(chunks)}")
            conversation = conversation_generator.generate_conversation(
                content=chunk,
                num_turns=num_turns,
                conversation_context="research paper"
            )
            
            if conversation:
                conversations.append(conversation)
        
        # Expand the conversations
        if conversations:
            expanded_conversations = self.expand_conversation_dataset(
                conversations,
                expansion_factor=expansion_factor,
                static_fields=static_fields,
                reference_fields=reference_fields
            )
        else:
            expanded_conversations = []
        
        return conversations, expanded_conversations
    
    def save_conversations_to_jsonl(self, conversations: List[List[Dict[str, str]]], filename: str) -> str:
        """
        Save conversations to a JSONL file.
        
        Args:
            conversations: List of conversations to save
            filename: Name of the output file (without extension)
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.output_dir, f"{filename}.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + '\n')
                
        self.logger.info(f"Saved {len(conversations)} conversations to {output_path}")
        return output_path
    
    def save_conversations_to_parquet(self, conversations: List[List[Dict[str, str]]], filename: str) -> str:
        """
        Save conversations to a Parquet file.
        
        Args:
            conversations: List of conversations to save
            filename: Name of the output file (without extension)
            
        Returns:
            Path to the saved file
        """
        # Convert the conversations to a format suitable for Parquet
        data = []
        for i, conversation in enumerate(conversations):
            data.append({
                'conversation_id': i,
                'conversation': json.dumps(conversation)
            })
            
        df = pd.DataFrame(data)
        output_path = os.path.join(self.output_dir, f"{filename}.parquet")
        df.to_parquet(output_path, engine='pyarrow')
        
        self.logger.info(f"Saved {len(conversations)} conversations to {output_path}")
        return output_path
    
    def load_conversations_from_jsonl(self, file_path: str) -> List[List[Dict[str, str]]]:
        """
        Load conversations from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of conversations
        """
        conversations = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversation = json.loads(line.strip())
                conversations.append(conversation)
                
        self.logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
        return conversations
    
    def convert_conversations_to_dataframe(self, conversations: List[List[Dict[str, str]]]) -> pd.DataFrame:
        """
        Convert conversations to a DataFrame format for analysis.
        
        Args:
            conversations: List of conversations
            
        Returns:
            DataFrame with structured conversation data
        """
        data = []
        
        for conv_idx, conversation in enumerate(conversations):
            for turn_idx, turn in enumerate(conversation):
                source = turn.get('from', '')
                value = turn.get('value', '')
                
                # Create a row for this turn
                row = {
                    'conversation_id': conv_idx,
                    'turn_idx': turn_idx,
                    'source': source,
                    'content': value,
                    'content_length': len(value),
                    'word_count': len(value.split()),
                    'is_question': self._is_question(value)
                }
                
                data.append(row)
                
        # Create DataFrame
        if not data:
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                'conversation_id', 'turn_idx', 'source', 'content', 
                'content_length', 'word_count', 'is_question'
            ])
        
        return pd.DataFrame(data)
    
    def convert_to_multi_format(self, conversations: List[List[Dict[str, str]]], 
                              base_filename: str, 
                              formats: List[str] = ['jsonl', 'parquet', 'csv', 'df']):
        """
        Convert conversations to multiple formats and save them.
        
        Args:
            conversations: List of conversations
            base_filename: Base name for the output files
            formats: List of output formats to generate
            
        Returns:
            Dictionary mapping format names to output paths
        """
        output_files = {}
        
        if 'jsonl' in formats:
            output_files['jsonl'] = self.save_conversations_to_jsonl(conversations, base_filename)
            
        if 'parquet' in formats:
            output_files['parquet'] = self.save_conversations_to_parquet(conversations, base_filename)
            
        if 'df' in formats or 'csv' in formats:
            df = self.convert_conversations_to_dataframe(conversations)
            output_files['df'] = df
            
            if 'csv' in formats:
                csv_path = os.path.join(self.output_dir, f"{base_filename}.csv")
                df.to_csv(csv_path, index=False)
                output_files['csv'] = csv_path
                self.logger.info(f"Saved DataFrame to CSV: {csv_path}")
        
        return output_files
    
    def analyze_expanded_dataset(self, original_conversations: List[List[Dict[str, str]]], 
                               expanded_conversations: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        Analyze the expanded dataset in comparison to the original using natural language queries.
        
        Args:
            original_conversations: List of original conversations
            expanded_conversations: List of expanded conversations
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {
            "original_count": len(original_conversations),
            "expanded_count": len(expanded_conversations),
            "expansion_ratio": len(expanded_conversations) / len(original_conversations) if original_conversations else 0,
            "basic_statistics": {},
            "advanced_analysis": {}
        }
        
        # Convert conversations to DataFrames for analysis
        orig_df = self.convert_conversations_to_dataframe(original_conversations)
        expanded_df = self.convert_conversations_to_dataframe(expanded_conversations)
        
        # Calculate basic statistics
        analysis_results["basic_statistics"] = {
            "original": {
                "num_conversations": len(original_conversations),
                "avg_turns_per_conversation": orig_df.groupby('conversation_id').size().mean() if not orig_df.empty else 0,
                "avg_human_message_length": orig_df[orig_df['source'] == 'human']['content_length'].mean() if not orig_df.empty else 0,
                "avg_gpt_message_length": orig_df[orig_df['source'] == 'gpt']['content_length'].mean() if not orig_df.empty else 0
            },
            "expanded": {
                "num_conversations": len(expanded_conversations),
                "avg_turns_per_conversation": expanded_df.groupby('conversation_id').size().mean() if not expanded_df.empty else 0,
                "avg_human_message_length": expanded_df[expanded_df['source'] == 'human']['content_length'].mean() if not expanded_df.empty else 0,
                "avg_gpt_message_length": expanded_df[expanded_df['source'] == 'gpt']['content_length'].mean() if not expanded_df.empty else 0
            }
        }
        
        # If PandasQueryIntegration is available, use it for advanced analysis
        if self.pandas_query or self.ollama_query:
            analysis_results["advanced_analysis"] = self._perform_advanced_analysis(orig_df, expanded_df)
        
        return analysis_results
    
    def _perform_advanced_analysis(self, orig_df: pd.DataFrame, expanded_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform advanced analysis using PandasQueryIntegration or OllamaLlamaIndexIntegration.
        
        Args:
            orig_df: DataFrame with original conversations
            expanded_df: DataFrame with expanded conversations
            
        Returns:
            Dictionary with advanced analysis results
        """
        analysis_results = {}
        
        try:
            # If we have the pandas query integration available
            if self.pandas_query:
                try:
                    # Try to use the pandas query integration
                    orig_insights = self.pandas_query.generate_dataset_insights(orig_df, num_insights=3)
                    expanded_insights = self.pandas_query.generate_dataset_insights(expanded_df, num_insights=3)
                    
                    # Compare the datasets
                    comparison = self.pandas_query.compare_datasets(
                        orig_df, expanded_df,
                        df1_name="Original", df2_name="Expanded",
                        aspects=["shape", "statistics", "distributions"]
                    )
                    
                    analysis_results = {
                        "original_insights": orig_insights,
                        "expanded_insights": expanded_insights,
                        "comparison": comparison
                    }
                    
                    # Check if there was an API error in the insights
                    if any("Error code: 401" in insight.get("insight", "") for insight in orig_insights):
                        # Fall back to Ollama if we encounter API key errors
                        raise ValueError("API key error detected, falling back to Ollama")
                    
                except Exception as e:
                    self.logger.warning(f"Error using PandasQueryIntegration: {str(e)}. Falling back to OllamaLlamaIndexIntegration.")
                    # Fall back to Ollama integration
                    if self.ollama_query is None:
                        try:
                            self.ollama_query = OllamaLlamaIndexIntegration(
                                ollama_model=getattr(self.ollama_interface, 'model', "llama3"),
                                verbose=True
                            )
                            self.logger.info("Created OllamaLlamaIndexIntegration for DataFrame analysis")
                        except Exception as e2:
                            self.logger.error(f"Failed to create OllamaLlamaIndexIntegration: {str(e2)}")
            
            # Use Ollama query integration if available
            if self.ollama_query:
                # Form some basic analysis queries
                queries = [
                    "Compare the average content length between original and expanded conversations",
                    "Identify the main differences in word count and sentence structure",
                    "Summarize the key quality differences between original and expanded data"
                ]
                
                # Create a combined dataframe for comparison
                combined_df = pd.concat([
                    orig_df.assign(dataset="original"),
                    expanded_df.assign(dataset="expanded")
                ])
                
                # Run queries
                query_results = {}
                for query in queries:
                    result = self.ollama_query.query_dataframe_with_ollama(combined_df, query)
                    query_results[query] = result
                
                analysis_results["ollama_analysis"] = query_results
                    
        except Exception as e:
            self.logger.error(f"Error performing advanced analysis: {str(e)}")
            analysis_results["error"] = str(e)
            
        return analysis_results
    
    def _is_question(self, text: str) -> bool:
        """
        Determine if the text is a question.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if the text is a question, False otherwise
        """
        text = text.strip().lower()
        return (text.endswith('?') or 
                text.startswith(('what', 'when', 'where', 'who', 'why', 'how', 'can', 'could',
                                'would', 'should', 'is', 'are', 'do', 'does', 'will', 'may')))


# Example usage
if __name__ == "__main__":
    import ollama
    
    # Define a simple ollama_interface
    class OllamaInterface:
        def __init__(self, model_name="llama3"):
            self.model = model_name
            
        def chat(self, messages):
            return ollama.chat(model=self.model, messages=messages)
    
    # Initialize the expander
    ollama_interface = OllamaInterface(model_name="llama3")
    expander = DatasetExpander(ollama_interface, output_dir="./expanded_data")
    
    # Import the conversation generator
    from conversation_generator import OllamaConversationGenerator
    generator = OllamaConversationGenerator(model_name="llama3")
    
    # Sample paper content
    paper_content = """
    Attention mechanisms have become an integral part of compelling sequence modeling
    and transduction models in various tasks, allowing modeling of dependencies without
    regard to their distance in the input or output sequences. In this paper we present the
    Transformer, a model architecture eschewing recurrence and instead relying entirely
    on an attention mechanism to draw global dependencies between input and output.
    """
    
    # Generate and expand conversations
    orig_conversations, expanded_conversations = expander.generate_conversations_from_paper(
        paper_content=paper_content,
        conversation_generator=generator,
        num_chunks=2,
        num_turns=3,
        expansion_factor=2,
        static_fields={'human': True, 'gpt': False},  # Keep human questions static, vary gpt responses
        reference_fields=['human']  # Use human questions as reference when generating gpt responses
    )
    
    # Save in multiple formats
    output_files = expander.convert_to_multi_format(
        expanded_conversations, 
        "transformer_paper_conversations",
        formats=['jsonl', 'parquet', 'csv']
    )
    
    print(f"Generated files: {output_files}")