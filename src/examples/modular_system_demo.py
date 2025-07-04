"""
Example demonstrating the new modular AgentChef system with agent-specific prompts,
conversation storage, and abstracted query engine.

This example shows how to:
1. Create and register agents with specific prompts
2. Use the agent data manager for crawling and storage
3. Query data with agent-specific prompts
4. Store conversations and knowledge for agents

Written By: @BorcherdingL
Date: 6/29/2025
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path

# Import the new modular components
from agentChef.core.prompts.agent_prompt_manager import AgentPromptManager, initialize_default_agents
from agentChef.core.storage.conversation_storage import ConversationStorage, KnowledgeEntry
from agentChef.core.llamaindex.pandas_query import PandasQueryIntegration
from agentChef.core.crawlers.crawlers_module import AgentDataManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Demonstrate the new modular AgentChef system."""
    
    print("🧪👨‍🍳 AgentChef Modular System Demo")
    print("=" * 50)
    
    # 1. Initialize the prompt manager and register agents
    print("\n1. Setting up Agent Prompt Manager...")
    prompt_manager = AgentPromptManager("demo_data/prompts")
    
    # Initialize default agents
    initialize_default_agents(prompt_manager)
    
    # Register a custom agent
    custom_agent_config = {
        "domain": "data_science",
        "personality": "analytical",
        "expertise": ["machine learning", "data analysis", "visualization"],
        "prompts": {
            "data_analysis": """
            You are DataScienceChef, specializing in data analysis and ML insights.
            Agent: {agent_name}
            
            Query: {query}
            
            Instructions for data analysis:
            - Focus on statistical significance and patterns
            - Identify potential ML features and target variables
            - Suggest appropriate algorithms and techniques
            - Provide actionable insights for model development
            
            DataFrame info: {df_info}
            
            Generate pandas code for ML-focused analysis:
            """
        }
    }
    
    prompt_manager.register_agent("datasciencechef", custom_agent_config)
    
    # List available agents
    agents = prompt_manager.list_agents()
    print(f"Available agents: {agents}")
    
    # 2. Create storage system and save some sample data
    print("\n2. Setting up Conversation Storage...")
    storage = ConversationStorage("demo_data/storage")
    
    # Save a sample conversation
    sample_conversation = [
        {"from": "human", "value": "What is the best approach for analyzing customer data?"},
        {"from": "gpt", "value": "For customer data analysis, I recommend starting with exploratory data analysis to understand the data distribution, then applying segmentation techniques like clustering to identify customer groups, and finally using predictive modeling to forecast customer behavior."}
    ]
    
    conv_id = storage.save_conversation("datasciencechef", sample_conversation)
    print(f"Saved sample conversation: {conv_id}")
    
    # Save some knowledge entries
    knowledge_entries = [
        KnowledgeEntry(
            agent_name="datasciencechef",
            entry_id="ml_001",
            topic="customer segmentation",
            content="Customer segmentation using RFM analysis (Recency, Frequency, Monetary) is effective for e-commerce data. K-means clustering often works well for this type of analysis.",
            knowledge_type="technique",
            source="expert_knowledge"
        ),
        KnowledgeEntry(
            agent_name="datasciencechef",
            entry_id="ml_002",
            topic="feature engineering",
            content="When working with customer data, create features like customer lifetime value, purchase frequency, average order value, and days since last purchase.",
            knowledge_type="best_practice",
            source="expert_knowledge"
        )
    ]
    
    storage.save_knowledge("datasciencechef", knowledge_entries)
    print(f"Saved {len(knowledge_entries)} knowledge entries")
    
    # 3. Create sample data and demonstrate agent-specific query engine
    print("\n3. Demonstrating Agent-Specific Query Engine...")
    
    # Create sample customer data
    sample_data = pd.DataFrame({
        "customer_id": range(1, 101),
        "age": [25 + i % 40 for i in range(100)],
        "total_spent": [100 + i * 10 + (i % 7) * 50 for i in range(100)],
        "num_orders": [1 + i % 10 for i in range(100)],
        "days_since_last_order": [i % 30 for i in range(100)],
        "customer_segment": [["Bronze", "Silver", "Gold"][i % 3] for i in range(100)]
    })
    
    print(f"Created sample dataset: {sample_data.shape}")
    
    # Initialize agent-specific query engine
    query_engine = PandasQueryIntegration(agent_name="datasciencechef")
    
    # Test different types of queries with agent-specific prompts
    queries = [
        ("What are the key characteristics of different customer segments?", "data_analysis"),
        ("How can we identify high-value customers?", "data_analysis"),
        ("What features would be most useful for predicting customer churn?", "data_analysis")
    ]
    
    for query, prompt_type in queries:
        print(f"\nQuery: {query}")
        print(f"Prompt type: {prompt_type}")
        
        result = query_engine.query_dataframe(
            sample_data, 
            query, 
            prompt_type=prompt_type,
            save_result=True
        )
        
        print(f"Response: {result['response'][:200]}...")
        if result.get('pandas_instructions'):
            print(f"Generated code: {result['pandas_instructions'][:100]}...")
    
    # 4. Generate agent insights
    print("\n4. Generating Agent-Specific Insights...")
    insights = query_engine.generate_agent_insights(sample_data, num_insights=3)
    
    for i, insight in enumerate(insights, 1):
        print(f"\nInsight {i}: {insight['query']}")
        print(f"Answer: {insight['insight'][:150]}...")
    
    # 5. Demonstrate agent data manager for crawling
    print("\n5. Demonstrating Agent Data Manager...")
    
    # Create agent data manager
    agent_manager = AgentDataManager("datasciencechef", "demo_data/agent_data")
    
    # Simulate crawling some web data (you can replace with real URLs)
    web_params = {
        "url": "https://example.com/customer-analysis-guide"
    }
    
    # Note: This would normally crawl real data
    # For demo purposes, we'll simulate the result
    simulated_crawl_result = {
        "source": "web",
        "url": web_params["url"],
        "content": "Customer analysis best practices include data cleaning, exploratory data analysis, segmentation, and predictive modeling. Key metrics to track include customer lifetime value, churn rate, and purchase frequency.",
        "timestamp": "2025-06-29T12:00:00"
    }
    
    # Store the simulated crawled data
    if agent_manager.storage:
        storage_result = agent_manager._store_crawled_data_for_agent(
            simulated_crawl_result, 
            "web", 
            web_params
        )
        print(f"Stored crawled data: {storage_result}")
    
    # 6. Show agent knowledge summary
    print("\n6. Agent Knowledge Summary...")
    summary = agent_manager.get_agent_knowledge_summary()
    print(f"Knowledge summary: {summary}")
    
    # 7. Show storage statistics
    print("\n7. Storage Statistics...")
    stats = storage.get_agent_stats("datasciencechef")
    print(f"Agent stats: {stats}")
    
    # 8. List all conversations for the agent
    conversations = storage.list_conversations("datasciencechef")
    print(f"\nStored conversations: {len(conversations)}")
    
    # Load and display a conversation
    if conversations:
        conv, metadata = storage.load_conversation("datasciencechef", conversations[0])
        print(f"Sample conversation: {len(conv)} turns")
        for turn in conv[:2]:  # Show first 2 turns
            print(f"  {turn['from']}: {turn['value'][:100]}...")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("\nKey improvements:")
    print("- Agent-specific prompts and knowledge management")
    print("- Modular storage system focused on conversations")
    print("- Abstracted query engine supporting different agent contexts")
    print("- Unified data management interface")

if __name__ == "__main__":
    asyncio.run(main())
