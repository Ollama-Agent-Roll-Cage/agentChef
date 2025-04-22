# AgentChef: How to Run the Examples

This guide explains how to run each example for the agentChef library and what to expect when running them. Before starting, make sure you have all the prerequisites installed and configured correctly.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running:
   - Install from [https://ollama.ai/](https://ollama.ai/)
   - Start the Ollama service before running examples
   - Pull the llama3 model: `ollama pull llama3`
3. **agentChef** installed:
   ```bash
   pip install agentChef
   ```

## Setting Up Your Environment

First, create a dedicated directory for the examples and set up a virtual environment:

```bash
# Create a directory
mkdir agentchef-examples
cd agentchef-examples

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install agentChef
pip install agentChef

# Create directories for output
mkdir -p output/{expanded_data,cleaned_output}
```

Now, you're ready to create and run the examples.

## 1. OllamaInterface Example

### Create the Script

Create a file named `ollama_interface_example.py`:

```python
from agentChef.ollama_interface import OllamaInterface

# Initialize the interface with a specific model
ollama = OllamaInterface(model_name="llama3")

# Check if Ollama is available and running
if ollama.is_available():
    print("Ollama is available and running.")
    
    # Generate a text response using chat
    print("Generating a response about transformer models...")
    response = ollama.chat(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what a transformer model is in simple terms."}
    ])
    
    # Print the response
    print("\nResponse:")
    print(response["message"]["content"])
    
    # Generate embeddings for text
    print("\nGenerating embeddings...")
    embeddings = ollama.embeddings("This is sample text for embedding.")
    print(f"Generated embedding with {len(embeddings)} dimensions")
else:
    print("Ollama is not available. Please ensure it's installed and running.")
```

### Run the Script

```bash
python ollama_interface_example.py
```

### What to Expect

You should see:
1. Confirmation that Ollama is running
2. A text response explaining transformer models in simple terms
3. Confirmation that embeddings were generated, showing the number of dimensions (typically 1024 or 4096 depending on the model)

If Ollama isn't running, you'll get an error message prompting you to start it.

## 2. ConversationGenerator Example

### Create the Script

Create a file named `conversation_generator_example.py`:

```python
from agentChef.ollama_interface import OllamaInterface
from agentChef.conversation_generator import OllamaConversationGenerator
import json

# Initialize the Ollama interface
print("Initializing Ollama interface...")
ollama = OllamaInterface(model_name="llama3")

# Initialize the conversation generator
print("Initializing conversation generator...")
generator = OllamaConversationGenerator(
    model_name="llama3", 
    ollama_interface=ollama,
    enable_hedging=True
)

# Sample content to generate a conversation about
paper_content = """
Attention mechanisms have become an integral part of compelling sequence modeling
and transduction models in various tasks, allowing modeling of dependencies without
regard to their distance in the input or output sequences. In this paper we present the
Transformer, a model architecture eschewing recurrence and instead relying entirely
on an attention mechanism to draw global dependencies between input and output.
"""

print("Generating a conversation with 3 turns...")
# Generate a conversation with 3 turns about the paper
conversation = generator.generate_conversation(
    content=paper_content,
    num_turns=3,
    conversation_context="AI research",
    hedging_level="balanced"  # Use balanced hedging for natural responses
)

# Print the formatted conversation
print("\nGenerated Conversation:")
print(json.dumps(conversation, indent=2))

# Split longer content into chunks for processing
print("\nSplitting content into chunks...")
chunks = generator.chunk_text(paper_content, chunk_size=100, overlap=20)
print(f"Split content into {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:50]}..." if len(chunk) > 50 else f"Chunk {i+1}: {chunk}")

# Generate a hedged response to a specific question
print("\nGenerating a hedged response...")
hedged_response = generator.generate_hedged_response(
    prompt="What are the advantages of transformer models over RNNs?",
    hedging_profile="balanced",
    knowledge_level="medium",
    subject_expertise="machine learning"
)

print("\nHedged Response:")
print(hedged_response)
```

### Run the Script

```bash
python conversation_generator_example.py
```

### What to Expect

You should see:
1. Initialization messages for Ollama interface and conversation generator
2. A generated conversation about transformer models with 3 turns (human-AI-human-AI-human-AI)
3. The paper content split into multiple overlapping chunks
4. A hedged response explaining the advantages of transformer models over RNNs

The conversation will be formatted as a JSON structure with alternating "human" and "gpt" roles. The hedged response will show natural language with appropriate uncertainty markers based on the balanced hedging profile.

## 3. DatasetExpander Example

### Create the Script

Create a file named `dataset_expander_example.py`:

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.dataset_expander import DatasetExpander

async def expand_dataset_example():
    # Initialize components
    print("Initializing Ollama interface and dataset expander...")
    ollama = OllamaInterface(model_name="llama3")
    expander = DatasetExpander(
        ollama_interface=ollama, 
        output_dir="./output/expanded_data"  # Directory to save outputs
    )
    
    # Sample conversations
    original_conversations = [
        [
            {"from": "human", "value": "What are attention mechanisms?"},
            {"from": "gpt", "value": "Attention mechanisms allow neural networks to focus on specific parts of input data."}
        ],
        [
            {"from": "human", "value": "How do transformers compare to RNNs?"},
            {"from": "gpt", "value": "Transformers process all tokens in parallel using attention, while RNNs process sequentially."}
        ]
    ]
    
    print(f"Starting with {len(original_conversations)} original conversations")
    
    # Expand the conversations with variations
    print("\nExpanding conversations...")
    print("This may take a minute as it generates multiple variations...")
    expanded_conversations = expander.expand_conversation_dataset(
        conversations=original_conversations,
        expansion_factor=2,  # Create 2 variations of each conversation
        static_fields={'human': True, 'gpt': False},  # Keep human questions static, vary AI responses
        reference_fields=['human']  # Use human questions as reference for variations
    )
    
    print(f"Original conversations: {len(original_conversations)}")
    print(f"Expanded conversations: {len(expanded_conversations)}")
    
    # Display a sample of original and expanded
    print("\nSample comparison:")
    print("Original: " + original_conversations[0][1]["value"])
    print("Expanded 1: " + expanded_conversations[0][1]["value"])
    print("Expanded 2: " + expanded_conversations[2][1]["value"])
    
    # Save the expanded conversations in different formats
    print("\nSaving expanded conversations...")
    jsonl_path = expander.save_conversations_to_jsonl(expanded_conversations, "expanded_dataset")
    parquet_path = expander.save_conversations_to_parquet(expanded_conversations, "expanded_dataset")
    
    print(f"Saved to JSONL: {jsonl_path}")
    print(f"Saved to Parquet: {parquet_path}")
    
    # Convert conversations to DataFrame for analysis
    print("\nConverting to DataFrame for analysis...")
    df = expander.convert_conversations_to_dataframe(expanded_conversations)
    print(f"DataFrame created with {len(df)} rows")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Analyze the expanded dataset compared to the original
    print("\nAnalyzing expanded dataset...")
    analysis = expander.analyze_expanded_dataset(original_conversations, expanded_conversations)
    print("\nExpansion Analysis:")
    print(f"Original count: {analysis['original_count']}")
    print(f"Expanded count: {analysis['expanded_count']}")
    print(f"Expansion ratio: {analysis['expansion_ratio']:.1f}x")
    print(f"Basic statistics: {analysis['basic_statistics']}")

# Run the async example
asyncio.run(expand_dataset_example())
```

### Run the Script

```bash
python dataset_expander_example.py
```

### What to Expect

You should see:
1. Initialization messages for the Ollama interface and dataset expander
2. Starting with 2 original conversations
3. Expansion process creating 4 total conversations (2 originals × 2 variations)
4. A comparison showing how the responses vary while questions remain the same
5. Confirmation of saved files in JSONL and Parquet formats
6. A summary of the DataFrame structure for analysis
7. Analysis results comparing original and expanded datasets

This example demonstrates how the expander creates variations of AI responses while keeping human questions the same, creating a more diverse dataset from limited original examples.

## 4. DatasetCleaner Example

### Create the Script

Create a file named `dataset_cleaner_example.py`:

```python
import asyncio
from agentChef.ollama_interface import OllamaInterface
from agentChef.dataset_cleaner import DatasetCleaner

async def clean_dataset_example():
    # Initialize the Ollama interface
    print("Initializing Ollama interface and dataset cleaner...")
    ollama = OllamaInterface(model_name="llama3")
    
    # Initialize the dataset cleaner
    cleaner = DatasetCleaner(
        ollama_interface=ollama,
        output_dir="./output/cleaned_output"  # Directory to save cleaned data
    )
    
    # Sample original conversations
    original_conversations = [
        [
            {"from": "human", "value": "What are attention mechanisms?"},
            {"from": "gpt", "value": "Attention mechanisms allow neural networks to focus on specific parts of input data."}
        ]
    ]
    
    # Sample expanded conversations with quality issues
    expanded_conversations = [
        [
            {"from": "human", "value": "What are attention mechanisms?"},
            {"from": "gpt", "value": "Attention mechanisms allow neural networks to focusing on specific parts of input data."}  # Grammar error
        ],
        [
            {"from": "human", "value": "What is the purpose of attention?"},
            {"from": "gpt", "value": "It helps models pay attention to important parts."}  # Too brief compared to original
        ]
    ]
    
    print("Original conversation:")
    print(f"Human: {original_conversations[0][0]['value']}")
    print(f"AI: {original_conversations[0][1]['value']}")
    
    print("\nExpanded conversations with issues:")
    print(f"1. Human: {expanded_conversations[0][0]['value']}")
    print(f"   AI: {expanded_conversations[0][1]['value']} (grammar error: 'to focusing')")
    print(f"2. Human: {expanded_conversations[1][0]['value']}")
    print(f"   AI: {expanded_conversations[1][1]['value']} (too brief)")
    
    # Analyze the dataset for quality issues
    print("\nAnalyzing dataset for quality issues...")
    analysis = await cleaner.analyze_dataset(original_conversations, expanded_conversations)
    
    print("\nDataset Analysis:")
    print(f"Total original conversations: {analysis['total_original']}")
    print(f"Total expanded conversations: {analysis['total_expanded']}")
    print(f"Issues by type: {analysis['issues_by_type']}")
    
    # Clean the dataset
    print("\nCleaning dataset...")
    print("This may take a minute as each conversation is analyzed and fixed...")
    cleaned_conversations = await cleaner.clean_dataset(
        original_conversations=original_conversations,
        expanded_conversations=expanded_conversations,
        cleaning_criteria={
            "fix_hallucinations": True,
            "normalize_style": True,
            "correct_grammar": True,
            "ensure_coherence": True
        }
    )
    
    print(f"\nCleaned {len(cleaned_conversations)} conversations")
    
    # Print a sample of the cleaning
    if cleaned_conversations:
        print("\nBefore cleaning (with grammar error):")
        print(expanded_conversations[0][1]["value"])
        
        print("\nAfter cleaning (grammar fixed):")
        print(cleaned_conversations[0][1]["value"])
        
        print("\nBefore cleaning (too brief):")
        print(expanded_conversations[1][1]["value"])
        
        print("\nAfter cleaning (more detailed):")
        print(cleaned_conversations[1][1]["value"])

# Run the async example
asyncio.run(clean_dataset_example())
```

### Run the Script

```bash
python dataset_cleaner_example.py
```

### What to Expect

You should see:
1. Initialization messages for Ollama interface and dataset cleaner
2. The original conversation and expanded conversations with issues highlighted
3. Analysis results showing the types of issues detected
4. Confirmation that the dataset has been cleaned
5. Before and after comparisons showing:
   - Grammar issues fixed (e.g., "to focusing" → "to focus")
   - Brief responses expanded to be more detailed and consistent with the original style

The cleaner preserves the original intent and meaning while fixing quality issues, resulting in a more consistent dataset.

## 5. WebCrawler Example

### Create the Script

Create a file named `web_crawler_example.py`:

```python
import asyncio
from agentChef.crawlers_module import WebCrawler

async def web_crawler_example():
    # URL to crawl
    url = "https://example.com"
    
    # Fetch content from the URL
    print(f"Fetching content from {url}...")
    html_content = await WebCrawler.fetch_url_content(url)
    
    if html_content:
        print(f"Successfully fetched {len(html_content)} bytes of HTML content")
        
        # Extract plain text from HTML
        text_content = await WebCrawler.extract_text_from_html(html_content)
        print("\nExtracted Text Content:")
        print(text_content[:300] + "..." if len(text_content) > 300 else text_content)
        
        # For PyPI packages, extract structured data
        pypi_url = "https://pypi.org/project/pandas/"
        print(f"\nFetching PyPI package info from {pypi_url}...")
        pypi_html = await WebCrawler.fetch_url_content(pypi_url)
        
        if pypi_html:
            print(f"Successfully fetched {len(pypi_html)} bytes of PyPI HTML content")
            
            package_data = await WebCrawler.extract_pypi_content(pypi_html, "pandas")
            if package_data:
                print("\nExtracted Package Data:")
                print(f"Name: {package_data['name']}")
                if 'metadata' in package_data:
                    for section, items in package_data['metadata'].items():
                        print(f"Metadata section: {section} ({len(items)} items)")
                print(f"Documentation length: {len(package_data['documentation'])} characters")
                
                # Format PyPI info into readable markdown
                formatted_info = await WebCrawler.format_pypi_info({"info": {
                    "name": "pandas",
                    "version": "2.0.0",
                    "summary": "Powerful data structures for data analysis, time series, and statistics",
                    "description": "pandas is a Python package providing fast, flexible, and expressive data structures",
                    "author": "The Pandas Development Team",
                    "author_email": "pandas-dev@python.org",
                    "home_page": "https://pandas.pydata.org",
                    "license": "BSD-3-Clause",
                    "project_urls": {
                        "Documentation": "https://pandas.pydata.org/docs/",
                        "Source": "https://github.com/pandas-dev/pandas"
                    },
                    "requires_dist": ["numpy>=1.20.0", "python-dateutil>=2.8.0"]
                }})
                
                print("\nFormatted Package Info (sample):")
                print(formatted_info[:500] + "..." if len(formatted_info) > 500 else formatted_info)
            else:
                print("Could not extract package data")
    else:
        print("Failed to fetch content from URL")

# Run the async example
asyncio.run(web_crawler_example())
```

### Run the Script

```bash
python web_crawler_example.py
```

### What to Expect

You should see:
1. Confirmation of fetching HTML content from example.com
2. The extracted plain text from the example.com page
3. Confirmation of fetching PyPI package information for pandas
4. Extracted structured data from the pandas PyPI page, including:
   - Package name
   - Metadata sections
   - Documentation content length
5. A formatted markdown representation of the pandas package information

This example demonstrates the WebCrawler's ability to fetch and extract content from web pages, with special handling for PyPI packages.

## 6. ArxivSearcher Example

### Create the Script

Create a file named `arxiv_searcher_example.py`:

```python
import asyncio
from agentChef.crawlers_module import ArxivSearcher

async def arxiv_searcher_example():
    # Initialize the ArXiv searcher
    arxiv = ArxivSearcher()
    
    # Look up a specific paper by ID
    arxiv_id = "1706.03762"  # "Attention Is All You Need" paper
    print(f"Fetching information for arXiv paper {arxiv_id}...")
    
    try:
        # Fetch paper information
        paper_info = await arxiv.fetch_paper_info(arxiv_id)
        
        # Print basic information
        print("\nPaper Information:")
        print(f"Title: {paper_info['title']}")
        print(f"Authors: {', '.join(paper_info['authors'])}")
        print(f"Categories: {', '.join(paper_info['categories'])}")
        print(f"Abstract: {paper_info['abstract'][:200]}...")
        print(f"Published: {paper_info['published']}")
        print(f"PDF Link: {paper_info['pdf_link']}")
        print(f"ArXiv URL: {paper_info['arxiv_url']}")
        
        # Format paper for learning
        print("\nFormatting paper for learning...")
        formatted_paper = await arxiv.format_paper_for_learning(paper_info)
        
        print("\nFormatted for Learning (first 500 characters):")
        print(formatted_paper[:500] + "..." if len(formatted_paper) > 500 else formatted_paper)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Can also extract arXiv ID from a URL
    urls = [
        "https://arxiv.org/abs/2201.08239",
        "https://arxiv.org/pdf/2201.08239.pdf",
        "2201.08239v1"
    ]
    
    print("\nExtracting arXiv IDs from different formats:")
    for url in urls:
        try:
            # Extract ID from URL
            extracted_id = arxiv.extract_arxiv_id(url)
            print(f"Input: {url} → Extracted ID: {extracted_id}")
        except ValueError as e:
            print(f"Error extracting ID from {url}: {str(e)}")

# Run the async example
asyncio.run(arxiv_searcher_example())
```

### Run the Script

```bash
python arxiv_searcher_example.py
```

### What to Expect

You should see:
1. Confirmation of fetching information for the "Attention Is All You Need" paper
2. Detailed paper information including:
   - Title, authors, categories, and publication date
   - A preview of the abstract
   - Links to the PDF and ArXiv page
3. A formatted version of the paper information for learning purposes
4. Examples of extracting ArXiv IDs from different URL formats and direct IDs

This example demonstrates how to retrieve and format information about academic papers from the ArXiv repository, which is useful for research and learning.

## 7. DuckDuckGoSearcher Example

### Create the Script

Create a file named `ddg_searcher_example.py`:

```python
import asyncio
from agentChef.crawlers_module import DuckDuckGoSearcher

async def ddg_searcher_example():
    # Initialize the DuckDuckGo searcher
    ddg = DuckDuckGoSearcher()
    
    # Perform searches with different queries
    queries = [
        "transformer neural networks",
        "attention mechanisms AI",
        "llama language model"
    ]
    
    for query in queries:
        # Perform a search
        print(f"\nSearching for '{query}' with max 3 results...")
        search_results = await ddg.text_search(query, max_results=3)
        
        # Print the formatted search results
        print("\nSearch Results:")
        print(search_results)
        print("-" * 50)

# Run the async example
asyncio.run(ddg_searcher_example())
```

### Run the Script

```bash
python ddg_searcher_example.py
```

### What to Expect

You should see:
1. Three separate searches for different queries
2. For each search, formatted results including:
   - A summary section if available
   - Related topics with links
   - The results are formatted in markdown with titles and bullet points

This example demonstrates how to use the DuckDuckGoSearcher to perform web searches programmatically, which is useful for research and information gathering.

## 8. GitHubCrawler Example

### Create the Script

Create a file named `github_crawler_example.py`:

```python
import asyncio
from agentChef.crawlers_module import GitHubCrawler

async def github_crawler_example():
    # Initialize the GitHub crawler
    github = GitHubCrawler()
    
    # Repository to analyze (use a smaller one for the example)
    repo_url = "https://github.com/agentchef/sample-repo"
    # Fallback to a popular repo if the example repo doesn't exist
    fallback_repo = "https://github.com/huggingface/transformers"
    
    try:
        print(f"Trying to analyze repository: {repo_url}")
        
        # Get a summary of the repository
        try:
            repo_summary = await github.get_repo_summary(repo_url)
        except Exception:
            print(f"Could not access {repo_url}, using fallback repo {fallback_repo}")
            repo_url = fallback_repo
            repo_summary = await github.get_repo_summary(repo_url)
        
        print(f"\nSuccessfully accessed: {repo_url}")
        print("\nRepository Summary (excerpt):")
        print(repo_summary[:500] + "..." if len(repo_summary) > 500 else repo_summary)
        
        # Query the repository content
        query = "Find Python files related to attention mechanisms"
        print(f"\nQuerying repository: '{query}'")
        print("This may take a minute to process...")
        
        query_result = await github.query_repo_content(repo_url, query)
        print("\nQuery Results (excerpt):")
        print(query_result[:300] + "..." if len(query_result) > 300 else query_result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: This example requires git to be installed and may take time to clone repositories.")
        print("If you're getting errors, try with a smaller repository or check your internet connection.")

# Run the async example
asyncio.run(github_crawler_example())
```

### Run the Script

```bash
python github_crawler_example.py
```

### What to Expect

You should see:
1. Confirmation of accessing a GitHub repository
2. A summary of the repository including:
   - Basic statistics like number of files, stars, and forks
   - Language distribution
   - Main directories
   - README preview
3. Results of querying the repository for Python files related to attention mechanisms

Note: This example requires git to be installed on your system and may take longer to run than the other examples because it needs to clone the repository. If you encounter any issues, try using a smaller repository or ensure you have a good internet connection.

## 9. PandasQuery Example

### Create the Script

Create a file named `pandas_query_example.py`:

```python
import pandas as pd
from agentChef.pandas_query import OllamaLlamaIndexIntegration
import asyncio

async def pandas_query_example():
    try:
        # Create a sample DataFrame
        df = pd.DataFrame({
            "city": ["Toronto", "Tokyo", "Berlin", "Sydney", "New York"],
            "population": [2930000, 13960000, 3645000, 5312000, 8419000],
            "country": ["Canada", "Japan", "Germany", "Australia", "USA"],
            "continent": ["North America", "Asia", "Europe", "Oceania", "North America"]
        })
        
        # Display the DataFrame
        print("Sample DataFrame:")
        print(df)
        
        # Initialize the Ollama-based query integration
        print("\nInitializing OllamaLlamaIndexIntegration...")
        ollama_query = OllamaLlamaIndexIntegration(ollama_model="llama3")
        
        # Define some natural language queries
        queries = [
            "What is the city with the highest population?",
            "How many cities are in North America?",
            "What's the average population of cities in Europe?",
            "Which country has the city with the smallest population?"
        ]
        
        # Execute each query
        for query in queries:
            print(f"\n\nQuery: {query}")
            print("Executing query...")
            try:
                result = ollama_query.query_dataframe_with_ollama(df, query)
                print(f"Result: {result['response']}")
                print(f"Pandas code used: {result['pandas_code']}")
            except Exception as e:
                print(f"Error executing query: {str(e)}")
    
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("This example requires additional dependencies:")
        print("pip install llama-index llama-index-experimental")

# Run the async example
asyncio.run(pandas_query_example())
```

### Run the Script

```bash
python pandas_query_example.py
```

### What to Expect

You should see:
1. The sample DataFrame with city information
2. Results for each query, including:
   - The natural language query
   - The generated response
   - The pandas code that was used to answer the query

For example, for the query "What is the city with the highest population?", you should see a response like "Tokyo" and the pandas code that was generated to find this answer.

Note: This example requires additional dependencies. If you get import errors, you may need to install:
```bash
pip install llama-index llama-index-experimental
```

## 10. UDRAGS Complete Example with oarc-crawlers

The UDRAGS system now leverages the oarc-crawlers package for enhanced research capabilities. Here's an updated example:

### Create the Script

Create a file named `oarc_udrags_example.py`:

```python
import asyncio
from pathlib import Path
from agentChef.udrags import ResearchManager

async def oarc_udrags_example():
    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the research manager
    print("Initializing ResearchManager with oarc-crawlers...")
    manager = ResearchManager(data_dir=str(output_dir), model_name="llama3")
    
    # Define a progress callback function
    def progress_callback(message):
        print(f"Progress: {message}")
    
    try:
        # Step 1: Research a topic using the enhanced crawlers
        print("\nStarting research on transformer neural networks...")
        research_results = await manager.research_topic(
            topic="Transformer neural networks",
            max_papers=1,
            max_search_results=2,
            callback=progress_callback
        )
        
        print(f"\nResearch completed! Found {len(research_results.get('processed_papers', []))} papers")
        
        # Step 2: Generate conversation dataset from research
        print("\nGenerating conversation dataset...")
        dataset_results = await manager.generate_conversation_dataset(
            num_turns=2,
            expansion_factor=2,
            clean=True,
            callback=progress_callback
        )
        
        print("\nDataset generation completed!")
        print(f"Generated {len(dataset_results.get('conversations', []))} original conversations")
        print(f"Generated {len(dataset_results.get('expanded_conversations', []))} expanded conversations")
        print(f"Generated {len(dataset_results.get('cleaned_conversations', []))} cleaned conversations")
        
        print(f"\nOutput saved to: {dataset_results.get('output_path', 'unknown')}")
        
    finally:
        # Clean up
        manager.cleanup()
        print("\nCleanup completed")

# Run the example
if __name__ == "__main__":
    asyncio.run(oarc_udrags_example())
```

### Run the Script

```bash
python oarc_udrags_example.py
```

### What to Expect

You should see:
1. Initialization of the ResearchManager
2. Progress updates as the system researches transformer neural networks
3. Summary of the research results, including papers and search results
4. Progress updates as the system generates conversations from the research
5. Details about the generated dataset, including counts of original, expanded, and cleaned conversations
6. A sample of one of the generated conversations
7. Confirmation of cleanup

This complete example demonstrates the full workflow from research to dataset generation, showing how all the components work together. Note that this example may take significantly longer to run than the others due to the multiple steps involved.

## 11. Using oarc-crawlers Integration

The agentChef package now integrates with the `oarc-crawlers` package, providing enhanced functionality for crawling and data extraction. This integration includes:

- Improved web crawling with BeautifulSoup (BSWebCrawler)
- Enhanced ArXiv paper fetching (ArxivFetcher)
- DuckDuckGo search capabilities (DuckDuckGoSearcher)
- GitHub repository analysis (GitHubCrawler)
- New YouTube downloading features (YouTubeDownloader)
- Parquet storage for all data (ParquetStorage)

### Example: Using YouTubeDownloader

Create a file named `youtube_example.py`:

```python
import asyncio
from oarc_crawlers import YouTubeDownloader

async def youtube_example():
    # Initialize the downloader
    downloader = YouTubeDownloader(data_dir="./data")
    
    # Search for videos
    search_results = await downloader.search_videos("transformer neural networks", limit=3)
    
    # Print results
    if "results" in search_results:
        for i, video in enumerate(search_results["results"]):
            print(f"{i+1}. {video['title']} by {video['author']}")
            print(f"   URL: {video['url']}")
    
    # Extract captions from a specific video
    video_url = "https://www.youtube.com/watch?v=UND34KcVVnM"  # Change to your video
    captions = await downloader.extract_captions(video_url)
    
    # Print caption languages
    if "captions" in captions:
        print(f"\nFound captions in {len(captions['captions'])} languages")
        for lang_code in captions['captions'].keys():
            print(f"- {lang_code}")

# Run the async function
asyncio.run(youtube_example())
```

### Run the Script

```bash
python youtube_example.py
```

### What to Expect

You should see:
1. Search results for videos related to "transformer neural networks"
2. Details of the videos found, including title, author, and URL
3. Captions extracted from a specific video, with the languages available

This example demonstrates how to use the YouTubeDownloader from the `oarc-crawlers` package to search for videos and extract captions.
