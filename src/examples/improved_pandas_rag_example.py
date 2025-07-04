"""
Improved PandasRAG Usage Example
===============================

This example demonstrates the PandasRAG system with better data analysis 
and conversation tracking.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path so we can import agentChef
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentChef.core.chefs.pandas_rag import PandasRAG

def main():
    print("🍳 Improved AgentChef PandasRAG Example")
    print("=" * 50)
    
    # Initialize PandasRAG
    try:
        rag = PandasRAG(
            data_dir="./improved_example_data", 
            model_name="llama3.2:3b",
            log_level="INFO"
        )
        print(f"✅ PandasRAG initialized successfully")
        print(f"📁 Data directory: {rag.data_dir}")
        print(f"🤖 Model: {rag.model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize PandasRAG: {e}")
        return
    
    # Register a sales analyst agent
    try:
        agent_id = "sales_analyst"
        rag.register_agent(
            agent_id=agent_id,
            system_prompt="You are a sales data analyst. Analyze data clearly and provide specific insights with actual numbers from the data.",
            description="A sales analyst focused on extracting actionable insights from sales datasets."
        )
        print(f"✅ Registered agent: {agent_id}")
    except Exception as e:
        print(f"❌ Failed to register agent: {e}")
        return
    
    # Create more comprehensive sample data
    sample_data = pd.DataFrame({
        'product': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'] * 4,
        'price': [1200, 800, 400, 300, 150] * 4,
        'sales': [100, 150, 120, 80, 90,    # Q1
                 110, 160, 130, 85, 95,    # Q2
                 120, 170, 140, 90, 100,   # Q3
                 130, 180, 150, 95, 105],  # Q4
        'quarter': ['Q1'] * 5 + ['Q2'] * 5 + ['Q3'] * 5 + ['Q4'] * 5,
        'region': ['North', 'South', 'East', 'West', 'Central'] * 4
    })
    
    print(f"\n📊 Created sample dataset with {len(sample_data)} rows")
    print("Sample data:")
    print(sample_data.head(10))
    
    print(f"\n📈 Data Summary:")
    print(f"Products: {sample_data['product'].unique()}")
    print(f"Quarters: {sample_data['quarter'].unique()}")
    print(f"Regions: {sample_data['region'].unique()}")
    
    # Initialize conversation
    rag.create_empty_conversation(agent_id)
    
    # Test queries with more specific questions
    questions = [
        "What is the average price for each product? Show me the exact numbers.",
        "Which product has the highest total sales across all quarters? Give me the specific numbers.",
        "Show me the total sales for each quarter. I want to see Q1, Q2, Q3, and Q4 figures.",
        "Which region performs best in total sales?",
        "What's the trend in sales from Q1 to Q4?"
    ]
    
    print(f"\n🔍 Testing queries...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n" + "="*70)
        print(f"📝 Question {i}: {question}")
        
        try:
            response = rag.query(
                dataframe=sample_data,
                question=question,
                agent_id=agent_id,
                save_conversation=True
            )
            print(f"🤖 Response: {response}")
            
        except Exception as e:
            print(f"❌ Error with query: {e}")
            rag.save_conversation(agent_id, "user", question)
            rag.save_conversation(agent_id, "assistant", f"Error: {str(e)}")
    
    # Show conversation history with better formatting
    print(f"\n💬 Conversation History:")
    try:
        conversations = rag.get_conversations(agent_id, limit=20)
        if not conversations.empty:
            print(f"Found {len(conversations)} conversation entries")
            
            # Group conversations by pairs (user question + assistant response)
            user_questions = conversations[conversations['role'] == 'user']
            print(f"\n📋 Question & Answer Summary:")
            
            for idx, user_conv in user_questions.iterrows():
                q_num = (idx // 2) + 1
                content = str(user_conv.get('content', ''))
                print(f"\n❓ Q{q_num}: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # Find corresponding assistant response
                assistant_responses = conversations[
                    (conversations['role'] == 'assistant') & 
                    (conversations.index > idx)
                ]
                if not assistant_responses.empty:
                    response = str(assistant_responses.iloc[0].get('content', ''))
                    print(f"💡 A{q_num}: {response[:200]}{'...' if len(response) > 200 else ''}")
        else:
            print("No conversations found in storage.")
            
        # Show in-memory conversations
        if agent_id in rag.conversation_history:
            print(f"\n🧠 In-memory conversations: {len(rag.conversation_history[agent_id])}")
            
    except Exception as e:
        print(f"❌ Error getting conversations: {e}")
    
    # Show detailed summary
    print(f"\n📈 Detailed Summary:")
    try:
        summary = rag.get_summary()
        print(f"📊 Total agents: {summary['total_agents']}")
        print(f"📁 Data directory: {summary['data_dir']}")
        
        for agent, info in summary['agents'].items():
            print(f"\n🤖 Agent: {agent}")
            print(f"   💬 Conversation turns: {info['conversation_turns']}")
            print(f"   📚 Knowledge items: {info['knowledge_items']}")
            print(f"   📝 Description: {info['profile'].get('description', 'N/A')}")
            print(f"   🎯 System prompt: {info['profile'].get('system_prompt', 'N/A')[:100]}...")
            
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
    
    # Test data analysis with actual pandas operations
    print(f"\n🔬 Direct Data Analysis (for comparison):")
    print(f"Average price by product:")
    avg_prices = sample_data.groupby('product')['price'].mean()
    for product, price in avg_prices.items():
        print(f"  - {product}: ${price:,.2f}")
    
    print(f"\nTotal sales by quarter:")
    quarterly_sales = sample_data.groupby('quarter')['sales'].sum()
    for quarter, sales in quarterly_sales.items():
        print(f"  - {quarter}: {sales:,} units")
    
    print(f"\nTop product by total sales:")
    product_sales = sample_data.groupby('product')['sales'].sum()
    top_product = product_sales.idxmax()
    top_sales = product_sales.max()
    print(f"  - {top_product}: {top_sales:,} units")
    
    print(f"\n✅ Improved PandasRAG example completed!")
    print(f"📁 Data stored in: {rag.data_dir}")

if __name__ == "__main__":
    main()