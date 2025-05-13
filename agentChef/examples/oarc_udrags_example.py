"""Example demonstrating ragchef with the new oarc-crawlers integration."""

import asyncio
import logging
from pathlib import Path
from agentChef.ragchef import ResearchManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def oarc_ragchef_example():
    """Demonstrate a complete ragchef workflow with oarc-crawlers."""
    
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the research manager
    print("Initializing ResearchManager...")
    manager = ResearchManager(data_dir=str(output_dir), model_name="llama3")
    
    # Define a progress callback function
    def progress_callback(message):
        print(f"Progress: {message}")
    
    # Step 1: Research a topic using the new crawlers
    try:
        print("\nStarting research on transformer neural networks...")
        research_results = await manager.research_topic(
            topic="Transformer neural networks",
            max_papers=1,  # Limit to 1 paper for the example
            max_search_results=2,  # Limit to 2 search results
            include_github=False,  # Skip GitHub to speed up the example
            callback=progress_callback
        )
        
        print("\nResearch completed!")
        print(f"Found {len(research_results.get('processed_papers', []))} papers")
        print(f"Found search results: {bool(research_results.get('search_results', ''))}")
        
        # Step 2: Generate conversation dataset from research
        print("\nGenerating conversation dataset...")
        dataset_results = await manager.generate_conversation_dataset(
            num_turns=2,  # Small number for example
            expansion_factor=2,
            clean=True,
            callback=progress_callback
        )
        
        # Show results
        print("\nDataset generation completed!")
        print(f"Generated {len(dataset_results.get('conversations', []))} original conversations")
        print(f"Generated {len(dataset_results.get('expanded_conversations', []))} expanded conversations")
        print(f"Generated {len(dataset_results.get('cleaned_conversations', []))} cleaned conversations")
        
        # If we have conversations, show a sample
        if dataset_results.get('conversations'):
            print("\nSample conversation:")
            first_conv = dataset_results['conversations'][0]
            for turn in first_conv[:2]:  # Show first two turns
                print(f"{turn['from']}: {turn['value'][:50]}...")
                
        print(f"\nOutput saved to: {dataset_results.get('output_path', 'unknown')}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
    finally:
        # Clean up
        manager.cleanup()
        print("\nCleanup completed")

if __name__ == "__main__":
    asyncio.run(oarc_ragchef_example())