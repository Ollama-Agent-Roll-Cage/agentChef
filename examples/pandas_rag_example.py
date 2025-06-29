"""
Simple PandasRAG Usage Example
==============================

This example demonstrates how to use the PandasRAG class for 
script-based workflows with AgentChef.

The PandasRAG class provides a simple, pip-friendly interface for:
- Registering agents with specific prompts
- Querying pandas DataFrames with natural language
- Managing agent conversations and knowledge
- Working without the web UI
"""

import pandas as pd
from agentChef import PandasRAG

def main():
    print("🍳 AgentChef PandasRAG Example")
    print("=" * 40)
    
    # Initialize PandasRAG with a custom data directory
    rag = PandasRAG(data_dir="./example_data", model_name="llama3.2")
    
    print(f"📁 Data directory: {rag.data_dir}")
    print(f"🤖 Model: {rag.model_name}")
    
    # Register a research assistant agent
    agent_id = "research_assistant"
    rag.register_agent(
        agent_id=agent_id,
        system_prompt="You are a helpful research assistant specialized in data analysis. "
                     "Provide clear, concise insights about datasets and trends.",
        description="A research assistant focused on data analysis and insights."
    )
    
    print(f"\n✅ Registered agent: {agent_id}")
    
    # Create some sample data
    sample_data = pd.DataFrame({
        'product': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
        'sales': [100, 150, 120, 110, 160, 125, 95, 145, 115, 105, 
                 155, 130, 102, 148, 118, 108, 158, 128, 98, 142,
                 122, 112, 162, 135, 104, 152, 124, 106, 156, 132,
                 100, 146, 116, 110, 150, 126, 96, 144, 120, 114,
                 164, 138, 108, 154, 128, 118, 168, 142, 102, 148,
                 124, 116, 166, 140, 106, 156, 132, 120, 170, 144],
        'region': ['North', 'South', 'East', 'West'] * 15,
        'month': ['Jan', 'Feb', 'Mar'] * 20
    })
    
    print(f"\n📊 Created sample dataset with {len(sample_data)} rows")
    print("Sample data:")
    print(sample_data.head())
    
    # Initialize empty conversation for the agent
    rag.create_empty_conversation(agent_id)
    print(f"\n💬 Created empty conversation history for {agent_id}")
    
    # Query the data
    print("\n🔍 Querying the data...")
    questions = [
        "What are the average sales by product?",
        "Which region has the highest total sales?",
        "What trends do you see in the data?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        try:
            response = rag.query(
                dataframe=sample_data,
                question=question,
                agent_id=agent_id,
                save_conversation=True
            )
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error querying: {e}")
            # Add manual conversation entry for demonstration
            rag.save_conversation(agent_id, "user", question)
            rag.save_conversation(agent_id, "assistant", 
                                f"I encountered an error while analyzing the data: {e}")
    
    # Add some knowledge to the agent
    rag.add_knowledge(
        agent_id=agent_id,
        content="Product A typically performs well in Q1, Product B in Q2-Q3, and Product C year-round.",
        source="domain_knowledge",
        metadata={"category": "product_insights", "priority": "high"}
    )
    
    rag.add_knowledge(
        agent_id=agent_id,
        content="Regional variations are often due to seasonal patterns and local preferences.",
        source="market_research",
        metadata={"category": "regional_insights", "priority": "medium"}
    )
    
    print(f"\n📚 Added knowledge to {agent_id}")
    
    # Show agent summary
    print(f"\n📈 Agent Summary:")
    summary = rag.get_summary()
    print(f"Total agents: {summary['total_agents']}")
    
    for agent, info in summary['agents'].items():
        print(f"\n🤖 Agent: {agent}")
        print(f"  - Conversation turns: {info['conversation_turns']}")
        print(f"  - Knowledge items: {info['knowledge_items']}")
        print(f"  - Description: {info['profile'].get('description', 'N/A')}")
    
    # Show recent conversations
    print(f"\n💬 Recent conversations for {agent_id}:")
    conversations = rag.get_conversations(agent_id, limit=6)
    for _, conv in conversations.iterrows():
        role_emoji = "👤" if conv['role'] == "user" else "🤖"
        print(f"{role_emoji} {conv['role']}: {conv['content'][:100]}...")
    
    # Show knowledge base
    print(f"\n📚 Knowledge base for {agent_id}:")
    knowledge = rag.get_knowledge(agent_id)
    for _, kb in knowledge.iterrows():
        print(f"📄 [{kb['source']}] {kb['content'][:80]}...")
    
    # Export data
    print(f"\n💾 Exporting agent data...")
    exported = rag.export_data(agent_id, "./exports")
    for data_type, filepath in exported.items():
        print(f"  - {data_type}: {filepath}")
    
    print(f"\n✅ PandasRAG example completed!")
    print(f"📁 Data stored in: {rag.data_dir}")
    print(f"📤 Exports available in: ./exports")


if __name__ == "__main__":
    main()
