"""
Simple PandasRAG Usage Example (Minimal Dependencies)
=====================================================

This example demonstrates basic PandasRAG functionality without
requiring all the storage and prompt management components.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path so we can import agentChef
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentChef.core.chefs.pandas_rag import PandasRAG

def main():
    print("🍳 Simple AgentChef PandasRAG Example")
    print("=" * 40)
    
    # Initialize PandasRAG with minimal setup - use a model that exists
    try:
        rag = PandasRAG(
            data_dir="./simple_example_data", 
            model_name="llama3.2:3b",  # Use the correct model name
            log_level="INFO"
        )
        print(f"✅ PandasRAG initialized successfully")
        print(f"📁 Data directory: {rag.data_dir}")
        print(f"🤖 Model: {rag.model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize PandasRAG: {e}")
        # Try with a different model
        try:
            print("🔄 Trying with llama3.2:1b...")
            rag = PandasRAG(
                data_dir="./simple_example_data", 
                model_name="llama3.2:1b",
                log_level="INFO"
            )
            print(f"✅ PandasRAG initialized with fallback model")
        except Exception as e2:
            print(f"❌ Failed with fallback model: {e2}")
            print("💡 Please run: ollama pull llama3.2:3b")
            return
    
    # Register a simple agent
    try:
        agent_id = "data_analyst"
        rag.register_agent(
            agent_id=agent_id,
            system_prompt="You are a helpful data analyst. Analyze data and provide clear insights.",
            description="A data analyst focused on extracting insights from datasets."
        )
        print(f"✅ Registered agent: {agent_id}")
    except Exception as e:
        print(f"❌ Failed to register agent: {e}")
        return
    
    # Create sample data
    sample_data = pd.DataFrame({
        'product': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'] * 4,
        'price': [1200, 800, 400, 300, 150] * 4,
        'sales': [100, 150, 120, 80, 90, 110, 160, 130, 85, 95,
                 120, 170, 140, 90, 100, 130, 180, 150, 95, 105],
        'quarter': ['Q1'] * 5 + ['Q2'] * 5 + ['Q3'] * 5 + ['Q4'] * 5,
        'region': ['North', 'South', 'East', 'West', 'Central'] * 4
    })
    
    print(f"\n📊 Created sample dataset with {len(sample_data)} rows")
    print("Sample data:")
    print(sample_data.head(10))
    
    # Initialize conversation
    rag.create_empty_conversation(agent_id)
    
    # Test queries with simpler ones first
    questions = [
        "Calculate the average price for each product",
        "Find the product with highest total sales",
        "Show sales by quarter",
    ]
    
    print(f"\n🔍 Testing queries...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n" + "="*50)
        print(f"📝 Question {i}: {question}")
        
        try:
            # Test the query
            response = rag.query(
                dataframe=sample_data,
                question=question,
                agent_id=agent_id,
                save_conversation=True
            )
            print(f"🤖 Response: {response}")
            
        except Exception as e:
            print(f"❌ Error with query: {e}")
            # Still save the conversation for record keeping
            rag.save_conversation(agent_id, "user", question)
            rag.save_conversation(agent_id, "assistant", f"Error: {str(e)}")
    
    # Show conversation history
    print(f"\n💬 Conversation History:")
    try:
        conversations = rag.get_conversations(agent_id, limit=10)
        if not conversations.empty:
            print(f"Found {len(conversations)} conversation entries")
            for idx, conv in conversations.iterrows():
                role = conv.get('role', 'unknown')
                content = str(conv.get('content', ''))[:100]
                role_emoji = "👤" if role == "user" else "🤖"
                print(f"{role_emoji} {role}: {content}...")
        else:
            print("No conversations found in storage.")
            
        # Try in-memory conversations
        if agent_id in rag.conversation_history:
            print(f"\nIn-memory conversations: {len(rag.conversation_history[agent_id])}")
            for conv in rag.conversation_history[agent_id][-3:]:  # Show last 3
                role = conv.get('role', 'unknown')
                content = str(conv.get('content', ''))[:100]
                role_emoji = "👤" if role == "user" else "🤖"
                print(f"{role_emoji} {role}: {content}...")
                
    except Exception as e:
        print(f"❌ Error getting conversations: {e}")
    
    # Show summary
    print(f"\n📈 Summary:")
    try:
        summary = rag.get_summary()
        print(f"Total agents: {summary['total_agents']}")
        for agent, info in summary['agents'].items():
            print(f"🤖 {agent}: {info['conversation_turns']} conversations, {info['knowledge_items']} knowledge items")
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
    
    print(f"\n✅ Simple PandasRAG example completed!")

if __name__ == "__main__":
    main()