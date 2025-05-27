def generate_conversation(self, content, num_turns=3, conversation_context="research",
                       hedging_level="balanced", conversation_history=None, 
                       template_format=None, required_elements=None):
    """
    Generate a conversation about the given content using separate human and AI prompts.
    
    Args:
        content (str): The content to generate a conversation about.
        num_turns (int): Number of back-and-forth turns in the conversation.
        conversation_context (str): Context to guide the conversation topic.
        hedging_level (str): Level of hedging to use in responses 
                             ("confident", "balanced", "cautious").
        conversation_history (list, optional): Previous conversation history to build upon.
        template_format (str, optional): Output template format:
            - "standard": Default {"from": "human/gpt", "value": "..."}
            - "instruction": {"instruction": "...", "input": "...", "output": "..."}
            - "reference": {"instruction": "...", "input": "...", "output": "...", "reference": "..."}
        required_elements (list, optional): Elements that must be present in specific fields
            (e.g., [("/command", "reference")] requires "/command" in reference field)
            
    Returns:
        list: A list of conversation turns formatted according to template_format
        or None if generation fails.
    """
    # Initialize conversation history
    if conversation_history is None:
        conversation_history = []
    else:
        # Deep copy to avoid modifying the original
        conversation_history = [turn.copy() for turn in conversation_history]
    
    # Limit content length for the prompt
    truncated_content = content[:2000] if len(content) > 2000 else content
    
    # Build the conversation turn by turn in standard format first
    standard_conversation = []
    
    try:
        for turn in range(num_turns):
            # Generate human question
            if turn == 0 and not conversation_history:
                # First question - start the conversation
                human_question = self._generate_human_question(
                    content=truncated_content,
                    conversation_context=conversation_context,
                    conversation_history=standard_conversation,
                    is_first_question=True
                )
            else:
                # Follow-up question based on conversation so far
                human_question = self._generate_human_question(
                    content=truncated_content,
                    conversation_context=conversation_context,
                    conversation_history=standard_conversation,
                    is_first_question=False
                )
            
            if not human_question:
                self.logger.warning(f"Failed to generate human question for turn {turn + 1}")
                break
                
            standard_conversation.append({"from": "human", "value": human_question})
            
            # Generate AI response
            ai_response = self._generate_ai_response(
                content=truncated_content,
                conversation_context=conversation_context,
                conversation_history=standard_conversation,
                hedging_level=hedging_level
            )
            
            if not ai_response:
                self.logger.warning(f"Failed to generate AI response for turn {turn + 1}")
                # Remove the human question if we can't generate a response
                standard_conversation.pop()
                break
                
            standard_conversation.append({"from": "gpt", "value": ai_response})
        
        # Validate the final conversation
        if standard_conversation:
            self._validate_conversation_format(standard_conversation)
            
            # Convert to requested template format if needed
            if template_format and template_format != "standard":
                formatted_conversation = self._convert_to_template_format(
                    standard_conversation, 
                    template_format=template_format,
                    required_elements=required_elements
                )
                return formatted_conversation
            else:
                return standard_conversation
        else:
            return None
            
    except Exception as e:
        self.logger.error(f"Error generating conversation: {str(e)}")
        return None
        
def _convert_to_template_format(self, standard_conversation, template_format, required_elements=None):
    """Convert standard conversation format to other templates."""
    formatted_conversation = []
    
    # Initialize the dataset cleaner if needed for regeneration
    cleaner = None
    if required_elements:
        try:
            from agentChef.core.classification.dataset_cleaner import DatasetCleaner
            cleaner = DatasetCleaner(ollama_interface=self.ollama)
        except ImportError:
            self.logger.warning("DatasetCleaner not available, can't enforce required elements")
    
    # Process conversation turns into the requested format
    if template_format == "instruction":
        # For instruction format, pair human and AI turns
        for i in range(0, len(standard_conversation), 2):
            if i+1 < len(standard_conversation):
                human_turn = standard_conversation[i]
                ai_turn = standard_conversation[i+1]
                
                formatted_turn = {
                    "instruction": "Answer the following question based on your knowledge",
                    "input": human_turn["value"],
                    "output": ai_turn["value"]
                }
                formatted_conversation.append(formatted_turn)
                
    elif template_format == "reference":
        for i in range(0, len(standard_conversation), 2):
            if i+1 < len(standard_conversation):
                human_turn = standard_conversation[i]
                ai_turn = standard_conversation[i+1]
                
                # Generate a reference field with commands
                reference = self._generate_reference_field(
                    human_question=human_turn["value"],
                    ai_response=ai_turn["value"],
                    content=truncated_content
                )
                
                formatted_turn = {
                    "instruction": "Answer the following question based on your knowledge",
                    "input": human_turn["value"],
                    "output": ai_turn["value"],
                    "reference": reference
                }
                
                # Check for required elements if specified
                if required_elements and cleaner:
                    for element, field in required_elements:
                        if field in formatted_turn and element not in formatted_turn[field]:
                            # Regenerate the field
                            self.logger.info(f"Regenerating {field} to include {element}")
                            regenerated = self._regenerate_field_with_element(
                                cleaner=cleaner,
                                field_name=field,
                                field_content=formatted_turn[field],
                                required_element=element,
                                context=formatted_turn
                            )
                            if regenerated:
                                formatted_turn[field] = regenerated
                
                formatted_conversation.append(formatted_turn)
                
    else:
        # Unknown format, return the standard conversation
        return standard_conversation
    
    return formatted_conversation

def _generate_reference_field(self, human_question, ai_response, content):
    """Generate a reference field that might contain commands or API calls."""
    prompt = f"""
    Based on this conversation:
    
    Human: {human_question}
    
    AI: {ai_response}
    
    Create a reference field that includes command-like structures (prefixed with /) 
    or API calls that would be relevant to this conversation. These could be commands 
    to retrieve information, perform actions, or analyze data.
    
    For example:
    - /search [query]
    - /lookup [term]
    - /api call [endpoint] [parameters]
    
    Return ONLY the reference text with commands, no explanation.
    """
    
    try:
        response = self.ollama.chat(
            messages=[{"role": "user", "content": prompt}]
        )
        reference = response['message']['content'].strip()
        return reference
    except Exception as e:
        self.logger.error(f"Error generating reference field: {str(e)}")
        return f"/search {human_question.replace('?', '')}"

def _regenerate_field_with_element(self, cleaner, field_name, field_content, required_element, context):
    """Use the DatasetCleaner to regenerate a field to include a required element."""
    try:
        # Create a cleanup prompt
        original = field_content
        requirement = f"The content MUST include '{required_element}' somewhere in the text."
        
        context_prompt = f"""
        Context:
        - Field: {field_name}
        - Instruction: {context.get('instruction', 'N/A')}
        - Input: {context.get('input', 'N/A')}
        - Output: {context.get('output', 'N/A')}
        
        Original content: {original}
        
        Requirement: {requirement}
        
        Create a new version that fulfills the requirement while maintaining the same meaning and purpose.
        """
        
        # Use DatasetCleaner's interface to regenerate the content
        response = cleaner.ollama_interface.chat(messages=[
            {'role': 'system', 'content': 'You are a specialized content editor that adds required elements.'},
            {'role': 'user', 'content': context_prompt}
        ])
        
        regenerated = response['message']['content'].strip()
        
        # Verify the required element is present
        if required_element in regenerated:
            return regenerated
        else:
            # Force include the element if still not present
            return regenerated + f"\n\n{required_element} additional_context"
            
    except Exception as e:
        self.logger.error(f"Error regenerating field: {str(e)}")
        # Force include the element as a fallback
        return field_content + f"\n\n{required_element} fallback"


====================================================

def save_conversations_to_parquet(self, conversations, filename_base, template_format=None):
    """
    Save conversations to a parquet file with flexible template formats.
    
    Args:
        conversations: List of conversations to save
        filename_base: Base name for the output file
        template_format: Format template to use ("standard", "instruction", "reference")
        
    Returns:
        Path to the saved file
    """
    import pandas as pd
    
    if not conversations:
        self.logger.error("No conversations to save")
        return None
        
    # Create output directory
    os.makedirs(self.output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_base}_{timestamp}.parquet"
    output_path = os.path.join(self.output_dir, filename)
    
    # Convert conversations to a pandas DataFrame based on template
    if template_format == "instruction":
        rows = []
        for conv in conversations:
            if isinstance(conv, list):
                # Standard format, needs conversion
                for i in range(0, len(conv), 2):
                    if i+1 < len(conv):
                        human_turn = conv[i]
                        ai_turn = conv[i+1]
                        
                        if human_turn.get('from') == 'human' and ai_turn.get('from') == 'gpt':
                            rows.append({
                                'instruction': 'Answer the following question based on your knowledge',
                                'input': human_turn.get('value', ''),
                                'output': ai_turn.get('value', '')
                            })
            elif isinstance(conv, dict):
                # Already in instruction format
                rows.append(conv)
    
    elif template_format == "reference":
        rows = []
        for conv in conversations:
            if isinstance(conv, list):
                # Standard format, needs conversion with reference generation
                for i in range(0, len(conv), 2):
                    if i+1 < len(conv):
                        human_turn = conv[i]
                        ai_turn = conv[i+1]
                        
                        if human_turn.get('from') == 'human' and ai_turn.get('from') == 'gpt':
                            # Generate a reference field with the OllamaConversationGenerator
                            if hasattr(self, 'conversation_generator'):
                                reference = self.conversation_generator._generate_reference_field(
                                    human_turn.get('value', ''),
                                    ai_turn.get('value', ''),
                                    "Content not available"
                                )
                            else:
                                reference = f"/search {human_turn.get('value', '').replace('?', '')}"
                            
                            rows.append({
                                'instruction': 'Answer the following question based on your knowledge',
                                'input': human_turn.get('value', ''),
                                'output': ai_turn.get('value', ''),
                                'reference': reference
                            })
            elif isinstance(conv, dict) and 'reference' in conv:
                # Already in reference format
                rows.append(conv)
    
    else:
        # Standard format - default to flat structure with all turns
        rows = []
        for conv_idx, conv in enumerate(conversations):
            if isinstance(conv, list):
                for turn_idx, turn in enumerate(conv):
                    row = {
                        'conversation_id': conv_idx,
                        'turn_idx': turn_idx,
                        'from': turn.get('from', 'unknown'),
                        'value': turn.get('value', '')
                    }
                    rows.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    
    try:
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved {len(df)} rows to {output_path}")
        return output_path
    except Exception as e:
        self.logger.error(f"Error saving to parquet: {str(e)}")
        return None

def generate_conversations_from_paper(self, paper_content, conversation_generator=None, 
                                    num_chunks=5, num_turns=3, expansion_factor=1,
                                    static_fields=None, reference_fields=None, 
                                    template_format=None, required_elements=None):
    """Generate conversations from a paper with support for templates and required elements."""
    if conversation_generator is None:
        if not hasattr(self, 'conversation_generator'):
            from agentChef.core.generation.conversation_generator import OllamaConversationGenerator
            self.conversation_generator = OllamaConversationGenerator(
                model_name=getattr(self.ollama_interface, 'model', 'llama3')
            )
        conversation_generator = self.conversation_generator
    
    # Chunk the content
    chunks = conversation_generator.chunk_text(
        paper_content, 
        chunk_size=2000, 
        overlap=200
    )
    
    # Limit chunks if needed
    chunks = chunks[:num_chunks]
    
    # Generate conversations from chunks
    all_conversations = []
    expanded_conversations = []
    
    for chunk in chunks:
        # Generate conversation with template format
        conversation = conversation_generator.generate_conversation(
            content=chunk,
            num_turns=num_turns,
            conversation_context="research paper",
            template_format=template_format,
            required_elements=required_elements
        )
        
        if conversation:
            all_conversations.append(conversation)
            
            # Expand if needed
            if expansion_factor > 1:
                variations = self.expand_conversation_dataset(
                    conversations=[conversation],
                    expansion_factor=expansion_factor,
                    static_fields=static_fields,
                    reference_fields=reference_fields,
                    template_format=template_format
                )
                expanded_conversations.extend(variations)
    
    return all_conversations, expanded_conversations

def convert_to_multi_format(self, conversations, output_base, formats=None, template_format=None):
    """
    Save conversations in multiple formats with template support.
    
    Args:
        conversations: List of conversations to save
        output_base: Base name for output files
        formats: List of formats to save ('jsonl', 'parquet', 'csv')
        template_format: Template format to use
        
    Returns:
        Dict mapping formats to output paths
    """
    if formats is None:
        formats = ['jsonl']
        
    output_files = {}
    
    for fmt in formats:
        if fmt == 'jsonl':
            path = self.save_conversations_to_jsonl(conversations, output_base)
            output_files['jsonl'] = path
        elif fmt == 'parquet':
            path = self.save_conversations_to_parquet(conversations, output_base, template_format)
            output_files['parquet'] = path
        elif fmt == 'csv':
            # For CSV, convert via pandas DataFrame
            try:
                import pandas as pd
                
                # Convert to DataFrame first using the parquet logic
                parquet_path = self.save_conversations_to_parquet(conversations, output_base, template_format)
                
                if parquet_path:
                    # Read the parquet and save as CSV
                    df = pd.read_parquet(parquet_path)
                    csv_path = parquet_path.replace('.parquet', '.csv')
                    df.to_csv(csv_path, index=False)
                    output_files['csv'] = csv_path
            except Exception as e:
                self.logger.error(f"Error saving to CSV: {str(e)}")
    
    return output_files

===============================================

async def generate_conversation_dataset(self, papers=None, num_turns=3, 
                                     expansion_factor=3, clean=True, callback=None,
                                     template_format=None, required_elements=None):
    """
    Generate a conversation dataset from research papers with template support.
    
    Args:
        papers: List of papers to process
        num_turns: Number of conversation turns to generate
        expansion_factor: Factor by which to expand the dataset
        clean: Whether to clean the expanded dataset
        callback: Optional callback function for progress updates
        template_format: Output template format ("standard", "instruction", "reference")
        required_elements: Elements that must be present in specific fields
            
    Returns:
        Dictionary with generated dataset information
    """
    # ... existing code ...
    
    # Generate conversations for each paper with template support
    all_conversations = []
    for i, content in enumerate(paper_contents):
        update_progress(f"Generating conversations for paper {i+1}/{len(paper_contents)}")
        
        # Chunk the content
        chunks = self.conversation_generator.chunk_text(content, chunk_size=2000, overlap=200)
        
        # Generate conversations from each chunk
        for j, chunk in enumerate(chunks[:5]):  # Limit to 5 chunks per paper
            update_progress(f"Processing chunk {j+1}/{min(5, len(chunks))} of paper {i+1}")
            
            conversation = self.conversation_generator.generate_conversation(
                content=chunk,
                num_turns=num_turns,
                conversation_context="research paper",
                template_format=template_format,
                required_elements=required_elements
            )
            
            if conversation:
                all_conversations.append(conversation)
    
    # ... rest of the method with template support for saving ...
    
    # Save with template support
    if template_format:
        output_path = self.dataset_expander.save_conversations_to_parquet(
            cleaned_conversations if clean and cleaned_conversations else expanded_conversations,
            f"conversation_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            template_format=template_format
        )
    else:
        # Use standard JSONL format
        output_path = self.dataset_expander.save_conversations_to_jsonl(
            cleaned_conversations if clean and cleaned_conversations else expanded_conversations,
            f"conversation_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # ... rest of the method ...
    
# Update process_paper_files to support templates
async def process_paper_files(self, paper_files, output_format='jsonl', 
                           num_turns=3, expansion_factor=3, clean=True, callback=None,
                           template_format=None, required_elements=None):
    """
    Process paper files with support for templates and required elements.
    """
    # ... existing code ...
    
    # Process each paper with template support
    all_conversations = []
    
    for i, content in enumerate(paper_contents):
        update_progress(f"Generating conversations for paper {i+1}/{len(paper_contents)}")
        
        # Generate conversations using DatasetExpander's helper function with template support
        orig_convs, expanded_convs = self.dataset_expander.generate_conversations_from_paper(
            paper_content=content,
            conversation_generator=self.conversation_generator,
            num_chunks=5,  # Process 5 chunks per paper
            num_turns=num_turns,
            expansion_factor=expansion_factor,
            static_fields={'human': True, 'gpt': False},  # Keep human questions static
            reference_fields=['human'],  # Use human questions as reference
            template_format=template_format,
            required_elements=required_elements
        )
        
        all_conversations.extend(orig_convs)
    
    # ... existing code ...
    
    # Save in multiple formats with template support
    output_base = f"paper_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_files = self.dataset_expander.convert_to_multi_format(
        all_conversations,
        output_base,
        formats=formats if output_format == 'all' else [output_format],
        template_format=template_format
    )
    
    # ... rest of the method ...

=========================================================

# Example usage:

# Standard conversation format
conversations = generator.generate_conversation(content, num_turns=3)

# Instruction format: {"instruction": "...", "input": "...", "output": "..."}
instruction_conversations = generator.generate_conversation(
    content, 
    num_turns=3, 
    template_format="instruction"
)

# Reference format with required commands
reference_conversations = generator.generate_conversation(
    content, 
    num_turns=3, 
    template_format="reference",
    required_elements=[
        ("/command", "reference"),  # Require /command in reference field
        ("/api", "reference")       # Require /api in reference field
    ]
)

# Save with specific template format
output_path = expander.save_conversations_to_parquet(
    conversations,
    "dataset",
    template_format="reference"
)

# Or via the research manager
result = await manager.generate_conversation_dataset(
    papers=papers,
    num_turns=3,
    template_format="reference",
    required_elements=[("/command", "reference")]
)

==================================================================

# Example usage:

# Standard conversation format
conversations = generator.generate_conversation(content, num_turns=3)

# Instruction format: {"instruction": "...", "input": "...", "output": "..."}
instruction_conversations = generator.generate_conversation(
    content, 
    num_turns=3, 
    template_format="instruction"
)

# Reference format with required commands
reference_conversations = generator.generate_conversation(
    content, 
    num_turns=3, 
    template_format="reference",
    required_elements=[
        ("/command", "reference"),  # Require /command in reference field
        ("/api", "reference")       # Require /api in reference field
    ]
)

# Save with specific template format
output_path = expander.save_conversations_to_parquet(
    conversations,
    "dataset",
    template_format="reference"
)

# Or via the research manager
result = await manager.generate_conversation_dataset(
    papers=papers,
    num_turns=3,
    template_format="reference",
    required_elements=[("/command", "reference")]
)