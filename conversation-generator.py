import json
import re
import ollama
import logging

class OllamaConversationGenerator:
    """
    A lightweight class to generate formatted conversations using Ollama models.
    Produces conversations in the format with alternating "from": "human" and "from": "gpt" entries.
    """
    
    def __init__(self, model_name="llama3"):
        """
        Initialize the conversation generator.
        
        Args:
            model_name (str): Name of the Ollama model to use for generating conversations.
        """
        self.model = model_name
        self.logger = logging.getLogger(__name__)
        
    def generate_conversation(self, content, num_turns=3, conversation_context="research"):
        """
        Generate a conversation about the given content.
        
        Args:
            content (str): The content to generate a conversation about.
            num_turns (int): Number of back-and-forth turns in the conversation.
            conversation_context (str): Context to guide the conversation topic.
            
        Returns:
            list: A list of conversation turns in the format [{"from": "human", "value": "..."},
                                                             {"from": "gpt", "value": "..."}]
            or None if generation fails.
        """
        # Limit content length for the prompt
        truncated_content = content[:2000] if len(content) > 2000 else content
        
        system_prompt = f"""You are an assistant helping to create synthetic training data. 
        Generate a realistic conversation between a human and an AI assistant about the following {conversation_context} content:
        
        {truncated_content}
        
        The conversation should:
        1. Include exactly {num_turns} turns (human question, AI response).
        2. Be related to the content provided.
        3. Show the human asking questions and the AI providing helpful responses.
        4. Format the output as a JSON list with "from" (either "human" or "gpt") and "value" fields.
        
        Return ONLY the JSON array without explanations or markdown formatting."""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}],
            )
            
            content = response['message']['content']
            
            # Extract JSON from the response
            json_match = re.search(r'\[\s*{\s*"from":.+}\s*\]', content, re.DOTALL)
            if json_match:
                conversation_json = json_match.group(0)
                # Validate and clean JSON
                try:
                    conversation = json.loads(conversation_json)
                    self._validate_conversation_format(conversation)
                    return conversation
                except json.JSONDecodeError:
                    self.logger.warning("Error parsing JSON response, trying to clean...")
                    # Try to clean common JSON format issues
                    cleaned_json = re.sub(r'(\w+):', r'"\1":', conversation_json)
                    cleaned_json = re.sub(r'\'', r'"', cleaned_json)
                    try:
                        conversation = json.loads(cleaned_json)
                        self._validate_conversation_format(conversation)
                        return conversation
                    except Exception as e:
                        self.logger.error(f"Failed to parse JSON after cleaning: {e}")
                        return None
            else:
                # If no JSON pattern found, try to extract from the whole content
                try:
                    conversation = json.loads(content)
                    self._validate_conversation_format(conversation)
                    return conversation
                except:
                    self.logger.error("JSON format not found in response")
                    return None
                
        except Exception as e:
            self.logger.error(f"Error generating conversation: {str(e)}")
            return None
    
    def _validate_conversation_format(self, conversation):
        """
        Validate that the conversation follows the expected format.
        Ensures each entry has "from" and "value" fields and corrects any issues.
        
        Args:
            conversation (list): The conversation to validate.
            
        Raises:
            ValueError: If the conversation format is invalid and can't be fixed.
        """
        if not isinstance(conversation, list):
            raise ValueError("Conversation is not a list")
            
        for i, turn in enumerate(conversation):
            # Ensure each turn has 'from' and 'value' keys
            if not isinstance(turn, dict):
                raise ValueError(f"Turn {i} is not a dictionary")
                
            # Normalize 'from' key, handle variations like 'role', 'speaker', etc.
            if 'from' not in turn:
                if 'role' in turn:
                    turn['from'] = 'human' if turn['role'] in ['user', 'human'] else 'gpt'
                elif 'speaker' in turn:
                    turn['from'] = 'human' if turn['speaker'] in ['user', 'human'] else 'gpt'
                else:
                    # Alternate based on position
                    turn['from'] = 'human' if i % 2 == 0 else 'gpt'
            
            # Normalize 'value' key, handle variations like 'content', 'message', 'text', etc.
            if 'value' not in turn:
                for key in ['content', 'message', 'text']:
                    if key in turn:
                        turn['value'] = turn[key]
                        break
                else:
                    raise ValueError(f"Turn {i} is missing 'value' or equivalent field")
            
            # Normalize "from" values
            if turn['from'].lower() in ['assistant', 'ai', 'bot', 'claude']:
                turn['from'] = 'gpt'
            elif turn['from'].lower() in ['user', 'human', 'person']:
                turn['from'] = 'human'
            
            # Ensure it's one of the two expected values
            if turn['from'] not in ['human', 'gpt']:
                turn['from'] = 'human' if i % 2 == 0 else 'gpt'
    
    def generate_conversations_batch(self, content_chunks, num_turns=3, context="research"):
        """
        Generate multiple conversations from a list of content chunks.
        
        Args:
            content_chunks (list): List of text chunks to generate conversations about.
            num_turns (int): Number of turns in each conversation.
            context (str): Context to guide the conversation topic.
            
        Returns:
            list: List of generated conversations.
        """
        conversations = []
        for i, chunk in enumerate(content_chunks):
            self.logger.info(f"Generating conversation {i+1}/{len(content_chunks)}...")
            conversation = self.generate_conversation(chunk, num_turns, context)
            if conversation:
                conversations.append(conversation)
        return conversations
    
    @staticmethod
    def chunk_text(content, chunk_size=2000, overlap=200):
        """
        Split text content into overlapping chunks of specified size.
        
        Args:
            content (str): Text to split into chunks.
            chunk_size (int): Maximum size of each chunk.
            overlap (int): Number of characters to overlap between chunks.
            
        Returns:
            list: List of text chunks.
        """
        chunks = []
        if len(content) <= chunk_size:
            return [content]
            
        start = 0
        while start < len(content):
            end = start + chunk_size
            if end >= len(content):
                chunks.append(content[start:])
                break
                
            # Try to end at a sentence or paragraph boundary
            for boundary in ['\n\n', '\n', '. ', '? ', '! ']:
                boundary_pos = content.rfind(boundary, start, end)
                if boundary_pos > start:
                    end = boundary_pos + len(boundary)
                    break
                    
            chunks.append(content[start:end])
            start = end - overlap
            
        return chunks


# Example usage:
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Sample content to generate conversation about
    sample_content = """
    Attention mechanisms have become an integral part of compelling sequence modeling
    and transduction models in various tasks, allowing modeling of dependencies without
    regard to their distance in the input or output sequences. In this paper we present the
    Transformer, a model architecture eschewing recurrence and instead relying entirely
    on an attention mechanism to draw global dependencies between input and output.
    """
    
    # Initialize the generator
    generator = OllamaConversationGenerator(model_name="llama3")
    
    # Generate a conversation
    conversation = generator.generate_conversation(
        sample_content, 
        num_turns=3,
        conversation_context="AI research"
    )
    
    # Print the formatted conversation
    if conversation:
        print(json.dumps(conversation, indent=2))
    else:
        print("Failed to generate conversation")
