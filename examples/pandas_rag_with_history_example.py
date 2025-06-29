"""
Example demonstrating PandasRAG with conversation history support.
Shows how the system maintains context across multiple queries.
"""

from agentChef import PandasRAG
import pandas as pd

def main():
    # Initialize PandasRAG with conversation history support
    print("🚀 Initializing PandasRAG with conversation history...")
    rag = PandasRAG(
        data_dir="./conversation_history_demo",
        max_history_turns=5  # Keep last 5 conversation turns in context
    )
    
    # Create sample sales data
    sales_data = pd.DataFrame({
        "product": ["Laptop", "Phone", "Tablet", "Watch", "Headphones", "Camera"],
        "price": [1200, 800, 400, 300, 150, 600],
        "sales_q1": [100, 200, 150, 80, 120, 90],
        "sales_q2": [120, 180, 160, 90, 140, 85],
        "sales_q3": [110, 220, 140, 100, 130, 95],
        "sales_q4": [150, 250, 180, 120, 160, 110],
        "category": ["Electronics", "Electronics", "Electronics", "Wearables", "Audio", "Electronics"],
        "launch_year": [2020, 2021, 2019, 2022, 2020, 2021]
    })
    
    print("📊 Sample Data:")
    print(sales_data.head())
    
    # Register a sales analyst agent
    agent_id = rag.register_agent(
        "sales_analyst",
        system_prompt="You are a sales data analyst who provides insights on product performance, trends, and business recommendations.",
        description="Analyzes sales data and provides actionable business insights with conversation context"
    )
    
    print(f"\n🤖 Registered agent: {agent_id}")
    
    # Start a conversation with building context
    print("\n💬 Starting conversation with context building...")
    
    # First query - establish baseline
    response1 = rag.query(
        sales_data, 
        "What are the top 3 performing products by total annual sales?",
        agent_id=agent_id
    )
    print(f"\n❓ Q1: What are the top 3 performing products by total annual sales?")
    print(f"💡 A1: {response1}")
    
    # Second query - builds on first
    response2 = rag.query(
        sales_data,
        "For those top performers, what trends do you see across quarters?",
        agent_id=agent_id
    )
    print(f"\n❓ Q2: For those top performers, what trends do you see across quarters?")
    print(f"💡 A2: {response2}")
    
    # Third query - references previous analysis
    response3 = rag.query(
        sales_data,
        "Based on our discussion, which product category should we focus on next year?",
        agent_id=agent_id
    )
    print(f"\n❓ Q3: Based on our discussion, which product category should we focus on next year?")
    print(f"💡 A3: {response3}")
    
    # Fourth query - asks for specific recommendations
    response4 = rag.query(
        sales_data,
        "Given the patterns we've identified, what specific price adjustments would you recommend?",
        agent_id=agent_id
    )
    print(f"\n❓ Q4: Given the patterns we've identified, what specific price adjustments would you recommend?")
    print(f"💡 A4: {response4}")
    
    # Show conversation summary
    print("\n📋 Conversation Summary:")
    summary = rag.get_conversation_summary(agent_id, num_exchanges=4)
    print(summary)
    
    # Demonstrate chat session
    print("\n💬 Starting Interactive Chat Session...")
    chat_session = rag.chat_with_data(sales_data, agent_id)
    
    # Interactive queries
    interactive_queries = [
        "How do launch years correlate with current performance?",
        "Are there any seasonal patterns in the data we should consider?",
        "What would be the impact of a 10% price increase on Electronics?"
    ]
    
    for query in interactive_queries:
        response = chat_session.ask(query)
        print(f"\n❓ Interactive: {query}")
        print(f"💡 Response: {response}")
    
    # Save the session
    session_saved = chat_session.save_session("sales_analysis_session_2025")
    print(f"\n💾 Session saved: {session_saved}")
    
    # Get final session summary
    final_summary = chat_session.get_summary()
    print(f"\n📊 Final Session Summary:")
    print(final_summary)
    
    # Demonstrate conversation without history for comparison
    print("\n🔄 Comparison: Query without conversation history...")
    response_no_history = rag.query(
        sales_data,
        "Based on our previous analysis, what should we focus on?",
        agent_id=agent_id,
        include_history=False  # Disable history
    )