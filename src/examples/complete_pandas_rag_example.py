"""
Complete PandasRAG Example with Conversation History
==================================================

This comprehensive example demonstrates all PandasRAG features:
- Agent registration and customization
- Conversation history and context building
- Knowledge management
- Interactive chat sessions
- Data export and backup
- Multi-agent workflows
- Advanced analytics

Run this example to see the full power of PandasRAG!

Run with the following:
# 1. Install AgentChef
pip install agentchef

# 2. Install required dependencies
pip install pandas numpy ollama

# 3. Install and setup Ollama
# Visit: https://ollama.ai/download
# Then pull a model:
ollama pull llama3.2
ollama serve  # Make sure it's running

# Navigate to the examples directory
cd m:/oarc_repos_git/agentChef/src/examples

# Run the complete example
python complete_pandas_rag_example.py

Step 4: Expected Output Structure
After running, you'll have:

./business_intelligence_demo/
├── agents/
│   ├── sales_analyst_profile.json
│   ├── marketing_specialist_profile.json
│   ├── customer_analyst_profile.json
│   └── strategic_analyst_profile.json
├── conversations/
│   ├── sales_analyst_conversations.parquet
│   ├── marketing_specialist_conversations.parquet
│   ├── customer_analyst_conversations.parquet
│   └── strategic_analyst_conversations.parquet
├── knowledge/
│   ├── sales_analyst_knowledge.parquet
│   ├── marketing_specialist_knowledge.parquet
│   ├── customer_analyst_knowledge.parquet
│   └── strategic_analyst_knowledge.parquet
└── prompts/
    └── agent_prompts.json

./business_intelligence_backup/
├── sales_analyst/
├── marketing_specialist/
├── customer_analyst/
└── strategic_analyst/

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

try:
    from agentChef import PandasRAG
except ImportError:
    print("❌ AgentChef not installed. Install with: pip install agentchef")
    exit(1)

def create_comprehensive_business_dataset():
    """Create a realistic business dataset for demonstration."""
    print("📊 Creating comprehensive business dataset...")
    
    # Generate realistic business data
    np.random.seed(42)  # For reproducible results
    
    # Product data
    products = [
        "Gaming Laptop Pro", "Business Laptop", "Ultrabook Air", "Workstation Elite",
        "Smartphone X", "Smartphone Pro", "Smartphone Lite", "Tablet Plus",
        "Wireless Headphones", "Gaming Headset", "Bluetooth Speaker", "Smart Watch",
        "4K Monitor", "Ultrawide Monitor", "Gaming Monitor", "Webcam HD"
    ]
    
    categories = {
        "Gaming Laptop Pro": "Laptops", "Business Laptop": "Laptops", 
        "Ultrabook Air": "Laptops", "Workstation Elite": "Laptops",
        "Smartphone X": "Mobile", "Smartphone Pro": "Mobile", 
        "Smartphone Lite": "Mobile", "Tablet Plus": "Mobile",
        "Wireless Headphones": "Audio", "Gaming Headset": "Audio", 
        "Bluetooth Speaker": "Audio", "Smart Watch": "Wearables",
        "4K Monitor": "Displays", "Ultrawide Monitor": "Displays", 
        "Gaming Monitor": "Displays", "Webcam HD": "Accessories"
    }
    
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Generate 2 years of monthly data
    data = []
    for year in [2023, 2024]:
        for month in months:
            for product in products:
                for region in regions:
                    # Base sales with seasonal patterns
                    base_sales = random.randint(50, 300)
                    
                    # Seasonal multipliers
                    if month in ["Nov", "Dec"]:  # Holiday season
                        seasonal_multiplier = 1.5
                    elif month in ["Jun", "Jul", "Aug"]:  # Summer boost
                        seasonal_multiplier = 1.2
                    elif month in ["Jan", "Feb"]:  # Post-holiday dip
                        seasonal_multiplier = 0.8
                    else:
                        seasonal_multiplier = 1.0
                    
                    # Regional multipliers
                    region_multipliers = {
                        "North America": 1.3, "Europe": 1.1, "Asia Pacific": 1.4,
                        "Latin America": 0.8, "Middle East": 0.6
                    }
                    
                    # Product category multipliers
                    category_multipliers = {
                        "Laptops": 1.2, "Mobile": 1.5, "Audio": 0.9,
                        "Displays": 0.7, "Wearables": 1.1, "Accessories": 0.6
                    }
                    
                    # Calculate final sales
                    units_sold = int(base_sales * seasonal_multiplier * 
                                   region_multipliers[region] * 
                                   category_multipliers[categories[product]])
                    
                    # Price ranges by category
                    price_ranges = {
                        "Laptops": (800, 2500), "Mobile": (300, 1200), "Audio": (50, 400),
                        "Displays": (200, 800), "Wearables": (100, 500), "Accessories": (30, 150)
                    }
                    
                    base_price = random.randint(*price_ranges[categories[product]])
                    revenue = units_sold * base_price
                    
                    # Marketing spend (roughly 8-15% of revenue)
                    marketing_spend = int(revenue * random.uniform(0.08, 0.15))
                    
                    # Customer satisfaction (4.0-5.0 scale)
                    satisfaction = round(random.uniform(3.8, 4.9), 1)
                    
                    # Return rate (1-8%)
                    return_rate = round(random.uniform(0.01, 0.08), 3)
                    
                    # Customer acquisition cost
                    acquisition_cost = random.randint(20, 100)
                    
                    data.append({
                        "year": year,
                        "month": month,
                        "product": product,
                        "category": categories[product],
                        "region": region,
                        "units_sold": units_sold,
                        "unit_price": base_price,
                        "revenue": revenue,
                        "marketing_spend": marketing_spend,
                        "customer_satisfaction": satisfaction,
                        "return_rate": return_rate,
                        "customer_acquisition_cost": acquisition_cost,
                        "profit_margin": round(random.uniform(0.15, 0.35), 2),
                        "inventory_turnover": round(random.uniform(4, 12), 1),
                        "date": f"{year}-{months.index(month)+1:02d}-15"
                    })
    
    df = pd.DataFrame(data)
    print(f"✅ Created dataset with {len(df)} records covering:")
    print(f"   • {len(products)} products across {len(set(categories.values()))} categories")
    print(f"   • {len(regions)} regions")
    print(f"   • 24 months of data (2023-2024)")
    print(f"   • Total revenue: ${df['revenue'].sum():,}")
    
    return df

def register_specialized_agents(rag):
    """Register multiple specialized agents for different business functions."""
    print("\n🤖 Registering specialized business agents...")
    
    agents = {}
    
    # 1. Sales Performance Analyst
    agents["sales_analyst"] = rag.register_agent(
        "sales_analyst",
        system_prompt="""You are Sarah, a Senior Sales Performance Analyst with 8 years of experience in B2B technology sales.

Your expertise includes:
- Revenue trend analysis and forecasting
- Product performance optimization
- Regional market analysis
- Sales team performance metrics
- Conversion rate optimization

Your personality:
- Data-driven and analytical
- Proactive in identifying opportunities
- Excellent at explaining complex metrics simply
- Always provides actionable recommendations
- References specific numbers and trends

You always:
- Include specific revenue figures and percentages
- Compare performance across time periods
- Identify top and bottom performers
- Suggest concrete next steps
- Ask clarifying questions when helpful""",
        description="Expert in sales performance analysis, revenue optimization, and market insights"
    )
    
    # 2. Marketing ROI Specialist
    agents["marketing_specialist"] = rag.register_agent(
        "marketing_specialist", 
        system_prompt="""You are Marcus, a Marketing ROI Specialist and Customer Acquisition expert with 6 years of experience.

Your expertise includes:
- Marketing spend optimization
- Customer acquisition cost (CAC) analysis
- Campaign performance measurement
- Channel attribution and effectiveness
- Customer lifetime value calculations

Your personality:
- Creative yet analytical
- ROI-focused in all recommendations
- Great at connecting marketing metrics to business outcomes
- Enthusiastic about testing and optimization
- Clear communicator with stakeholders

You always:
- Calculate and discuss ROI metrics
- Compare marketing efficiency across channels/regions
- Identify cost-saving opportunities
- Recommend budget reallocation strategies
- Provide actionable campaign insights""",
        description="Specialist in marketing ROI, customer acquisition, and campaign optimization"
    )
    
    # 3. Customer Experience Analyst
    agents["customer_analyst"] = rag.register_agent(
        "customer_analyst",
        system_prompt="""You are Claire, a Customer Experience Analyst with expertise in satisfaction metrics and retention.

Your expertise includes:
- Customer satisfaction analysis
- Return rate optimization
- Customer journey mapping
- Retention strategy development
- Quality improvement recommendations

Your personality:
- Customer-obsessed and empathetic
- Detail-oriented about quality metrics
- Proactive about identifying pain points
- Solutions-focused and practical
- Great at translating data into customer stories

You always:
- Focus on customer satisfaction scores and trends
- Analyze return rates and quality issues
- Identify opportunities to improve customer experience
- Suggest specific quality improvements
- Connect satisfaction to business outcomes""",
        description="Expert in customer satisfaction, quality metrics, and experience optimization"
    )
    
    # 4. Strategic Business Analyst
    agents["strategic_analyst"] = rag.register_agent(
        "strategic_analyst",
        system_prompt="""You are Dr. Alex Chen, a Strategic Business Analyst with an MBA and 10 years of consulting experience.

Your expertise includes:
- Cross-functional business analysis
- Strategic planning and forecasting
- Competitive positioning
- Resource allocation optimization
- Long-term growth planning

Your personality:
- Big-picture thinker with attention to detail
- Strategic and forward-looking
- Excellent at synthesizing insights from multiple sources
- Collaborative and inclusive in analysis
- Clear about assumptions and limitations

You always:
- Provide comprehensive business context
- Connect operational metrics to strategic goals
- Identify cross-functional opportunities
- Recommend portfolio-level decisions
- Consider competitive and market dynamics""",
        description="Senior analyst specializing in strategic business insights and cross-functional analysis"
    )
    
    print(f"✅ Registered {len(agents)} specialized agents:")
    for agent_id, _ in agents.items():
        print(f"   • {agent_id}")
    
    return agents

def add_business_knowledge(rag, agents):
    """Add domain-specific knowledge to each agent."""
    print("\n📚 Adding business domain knowledge...")
    
    # Sales knowledge
    sales_knowledge = [
        ("Q4 is traditionally our strongest quarter, representing 35-40% of annual revenue due to enterprise budget cycles and holiday consumer spending.", "quarterly_patterns"),
        ("Our average sales cycle for enterprise customers is 90-120 days, while SMB customers typically close within 30-45 days.", "sales_cycles"),
        ("Product launches typically see a 60-80% increase in first-month sales, followed by 20-30% normalization in month 2.", "product_lifecycle"),
        ("North American and European markets show strong B2B seasonal patterns, while Asia Pacific maintains more consistent year-round demand.", "regional_patterns")
    ]
    
    # Marketing knowledge
    marketing_knowledge = [
        ("Our target customer acquisition cost (CAC) should be 3-4x lower than customer lifetime value (CLV) for sustainable growth.", "cac_targets"),
        ("Digital marketing channels show 40-60% higher ROI compared to traditional channels for our tech-savvy audience.", "channel_effectiveness"),
        ("Retargeting campaigns typically achieve 2-3x higher conversion rates compared to cold outreach campaigns.", "campaign_optimization"),
        ("Marketing qualified leads (MQLs) that receive follow-up within 5 minutes are 21x more likely to convert.", "lead_response")
    ]
    
    # Customer experience knowledge  
    customer_knowledge = [
        ("Customer satisfaction scores below 4.0 typically indicate product quality issues requiring immediate attention.", "satisfaction_thresholds"),
        ("Return rates above 5% usually signal either product defects or misaligned customer expectations.", "return_benchmarks"),
        ("Customers who rate their experience 4.5+ are 3x more likely to become repeat buyers and brand advocates.", "loyalty_drivers"),
        ("Response time to customer issues is the #1 driver of satisfaction, more important than resolution outcome.", "support_priorities")
    ]
    
    # Strategic knowledge
    strategic_knowledge = [
        ("Our competitive advantage lies in premium build quality and customer service, justifying 15-20% price premiums.", "competitive_positioning"),
        ("Market research shows 67% of our target customers prioritize product reliability over price for business-critical purchases.", "customer_priorities"),
        ("Industry trends indicate a 25% annual growth in remote work solutions, making mobility products strategic priorities.", "market_trends"),
        ("Our most profitable customer segments are mid-market companies (50-500 employees) with high technology adoption rates.", "target_segments")
    ]
    
    # Add knowledge to respective agents
    knowledge_sets = [
        ("sales_analyst", sales_knowledge),
        ("marketing_specialist", marketing_knowledge), 
        ("customer_analyst", customer_knowledge),
        ("strategic_analyst", strategic_knowledge)
    ]
    
    total_added = 0
    for agent_id, knowledge_items in knowledge_sets:
        for content, category in knowledge_items:
            rag.add_knowledge(
                agent_id=agent_id,
                content=content,
                source="business_intelligence",
                metadata={"category": category, "priority": "high", "domain": "business_operations"}
            )
            total_added += 1
    
    print(f"✅ Added {total_added} knowledge items across {len(agents)} agents")

def demonstrate_progressive_conversations(rag, agents, df):
    """Demonstrate how conversation history builds context across multiple queries."""
    print("\n💬 Demonstrating Progressive Conversations with Context Building")
    print("=" * 70)
    
    # Progressive Sales Analysis Conversation
    print("\n🔍 SALES ANALYSIS CONVERSATION")
    print("-" * 40)
    
    sales_questions = [
        "Hi Sarah! I need help understanding our overall sales performance. What are the key insights from our data?",
        "That's helpful! Based on those top performers you mentioned, what seasonal patterns do you see?", 
        "Interesting seasonal insight! Given these patterns, which product categories should we prioritize for Q1 2025?",
        "Great recommendation! What specific actions should our sales team take to capitalize on these insights?",
        "Perfect! Can you create a priority action plan with specific metrics we should track?"
    ]
    
    print("Sarah (Sales Analyst) will build context through this conversation...")
    for i, question in enumerate(sales_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        response = rag.query(df, question, "sales_analyst", save_conversation=True)
        print(f"🤖 Sarah: {response[:200]}...")
        
        if i < len(sales_questions):
            input("   Press Enter to continue to next question...")
    
    # Cross-Agent Collaboration
    print("\n\n🤝 CROSS-AGENT COLLABORATION")
    print("-" * 40)
    
    # Marketing builds on sales insights
    print("\n👥 Marketing Specialist builds on Sales insights...")
    marketing_question = "Marcus, I just spoke with Sarah about our top-performing products and seasonal patterns. How should we adjust our marketing strategy based on these sales insights?"
    
    # Get recent sales conversation for context
    sales_context = rag.get_conversation_summary("sales_analyst", num_exchanges=3)
    enhanced_question = f"Context from Sales Analysis: {sales_context}\n\nQuestion: {marketing_question}"
    
    marketing_response = rag.query(df, enhanced_question, "marketing_specialist", save_conversation=True)
    print(f"🤖 Marcus: {marketing_response[:200]}...")
    
    # Customer Experience builds on both
    print("\n👥 Customer Analyst considers Sales + Marketing insights...")
    customer_question = "Claire, considering the sales patterns and marketing strategies we've discussed, what customer experience improvements should we prioritize?"
    
    cx_response = rag.query(df, customer_question, "customer_analyst", save_conversation=True)
    print(f"🤖 Claire: {cx_response[:200]}...")

def demonstrate_interactive_sessions(rag, agents, df):
    """Demonstrate interactive chat sessions with different agents."""
    print("\n\n💻 INTERACTIVE CHAT SESSIONS")
    print("=" * 40)
    
    # Start interactive session with strategic analyst
    print("\n🎯 Starting Strategic Planning Session with Dr. Alex Chen...")
    
    strategic_session = rag.chat_with_data(df, "strategic_analyst")
    
    strategic_queries = [
        "Dr. Chen, based on all our business data, what are the top 3 strategic priorities for 2025?",
        "How do the insights from our sales, marketing, and customer experience teams align with these priorities?",
        "What specific KPIs should we track to measure success on these strategic initiatives?",
        "If we had to choose just one priority due to budget constraints, which would you recommend and why?"
    ]
    
    for i, query in enumerate(strategic_queries, 1):
        print(f"\n❓ Strategic Query {i}: {query}")
        response = strategic_session.ask(query)
        print(f"🧠 Dr. Chen: {response[:250]}...")
        
        if i < len(strategic_queries):
            input("   Press Enter for next strategic question...")
    
    # Save this important session
    session_saved = strategic_session.save_session("strategic_planning_2025")
    print(f"\n💾 Strategic session saved as: {session_saved}")
    
    # Get session summary
    session_summary = strategic_session.get_summary()
    print(f"\n📋 Session Summary: {session_summary[:200]}...")

def demonstrate_advanced_features(rag, agents, df):
    """Demonstrate advanced PandasRAG features."""
    print("\n\n🚀 ADVANCED FEATURES DEMONSTRATION")
    print("=" * 45)
    
    # 1. Memory comparison
    print("\n🧠 Memory vs No-Memory Comparison")
    print("-" * 35)
    
    test_question = "Based on our previous analysis, what's the most important insight for our business?"
    
    with_memory = rag.query(df, test_question, "strategic_analyst", include_history=True)
    without_memory = rag.query(df, test_question, "strategic_analyst", include_history=False)
    
    print("🧠 WITH conversation memory:")
    print(f"   {with_memory[:150]}...")
    print("\n🤖 WITHOUT conversation memory:")  
    print(f"   {without_memory[:150]}...")
    
    # 2. Knowledge base demonstration
    print("\n\n📚 Knowledge Base Integration")
    print("-" * 30)
    
    knowledge_question = "How do our current customer satisfaction scores compare to our internal benchmarks and what actions should we take?"
    knowledge_response = rag.query(df, knowledge_question, "customer_analyst", save_conversation=True)
    print(f"📖 Knowledge-Enhanced Response: {knowledge_response[:200]}...")
    
    # 3. Agent statistics
    print("\n\n📊 Agent Performance Statistics")
    print("-" * 32)
    
    summary = rag.get_summary()
    print(f"Total agents: {summary['total_agents']}")
    print(f"Data directory: {summary['data_directory']}")
    
    for agent_id, info in summary['agents'].items():
        print(f"\n🤖 {agent_id}:")
        print(f"   • Conversations: {info['conversation_turns']}")
        print(f"   • Knowledge items: {info['knowledge_items']}")
        if 'profile' in info:
            description = info['profile'].get('description', 'No description')[:50]
            print(f"   • Role: {description}...")

def demonstrate_data_export(rag, agents):
    """Demonstrate data export and backup capabilities."""
    print("\n\n💾 DATA EXPORT & BACKUP")
    print("=" * 25)
    
    export_dir = "./business_intelligence_backup"
    print(f"📤 Exporting all agent data to: {export_dir}")
    
    for agent_id in agents.keys():
        print(f"\n📋 Exporting {agent_id} data...")
        exported = rag.export_data(
            agent_id=agent_id,
            export_dir=f"{export_dir}/{agent_id}",
            include_conversations=True,
            include_knowledge=True
        )
        
        for data_type, filepath in exported.items():
            print(f"   • {data_type}: {filepath}")
    
    print(f"\n✅ All agent data exported to {export_dir}")

def run_comprehensive_analysis(rag, agents, df):
    """Run a comprehensive business analysis using all agents."""
    print("\n\n🎯 COMPREHENSIVE BUSINESS ANALYSIS")
    print("=" * 40)
    
    # Each agent analyzes from their perspective
    analysis_results = {}
    
    analysis_prompts = {
        "sales_analyst": "Provide a comprehensive sales performance analysis with specific recommendations for improving revenue.",
        "marketing_specialist": "Analyze our marketing efficiency and ROI. What changes would maximize our marketing impact?", 
        "customer_analyst": "Evaluate our customer satisfaction and quality metrics. What improvements would have the biggest impact?",
        "strategic_analyst": "Synthesize insights across all business functions. What are our top strategic recommendations?"
    }
    
    for agent_id, prompt in analysis_prompts.items():
        print(f"\n🤖 {agent_id} analysis...")
        analysis_results[agent_id] = rag.query(
            df, prompt, agent_id, save_conversation=True
        )
        print(f"   ✅ Analysis complete ({len(analysis_results[agent_id])} characters)")
    
    # Create final summary
    print(f"\n📋 FINAL BUSINESS INTELLIGENCE SUMMARY")
    print("-" * 40)
    
    summary_prompt = f"""Based on comprehensive analysis from our business intelligence team:

Sales Analysis: {analysis_results['sales_analyst'][:200]}...
Marketing Analysis: {analysis_results['marketing_specialist'][:200]}...  
Customer Analysis: {analysis_results['customer_analyst'][:200]}...

Please provide an executive summary with top 3 actionable recommendations."""
    
    executive_summary = rag.query(df, summary_prompt, "strategic_analyst", save_conversation=True)
    print(f"\n🎯 Executive Summary:\n{executive_summary}")
    
    return analysis_results, executive_summary

def main():
    """Run the complete PandasRAG demonstration."""
    print("🍳 COMPLETE PANDASRAG BUSINESS INTELLIGENCE DEMO")
    print("=" * 55)
    print("This demo showcases PandasRAG's full capabilities for business analysis")
    print("with conversation history, multi-agent collaboration, and knowledge management.")
    
    # Initialize PandasRAG
    print("\n🚀 Initializing PandasRAG...")
    rag = PandasRAG(
        data_dir="./business_intelligence_demo",
        max_history_turns=10,  # Remember more context
        model_name="llama3.2"
    )
    
    # Create comprehensive dataset
    df = create_comprehensive_business_dataset()
    
    # Register specialized agents
    agents = register_specialized_agents(rag)
    
    # Add business knowledge
    add_business_knowledge(rag, agents)
    
    # Demonstrate progressive conversations
    demonstrate_progressive_conversations(rag, agents, df)
    
    # Demonstrate interactive sessions
    demonstrate_interactive_sessions(rag, agents, df)
    
    # Demonstrate advanced features
    demonstrate_advanced_features(rag, agents, df)
    
    # Run comprehensive analysis
    analysis_results, executive_summary = run_comprehensive_analysis(rag, agents, df)
    
    # Demonstrate data export
    demonstrate_data_export(rag, agents)
    
    # Final summary
    print("\n\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 30)
    print(f"📁 All data saved in: {rag.data_dir}")
    print(f"🤖 Created {len(agents)} specialized agents")
    print(f"📊 Analyzed {len(df)} business records")
    print(f"💬 Generated comprehensive conversation histories")
    print(f"📚 Built agent-specific knowledge bases")
    print(f"💾 Exported all data for backup/analysis")
    
    print("\n🚀 Your PandasRAG business intelligence system is ready!")
    print("You can now continue conversations with any agent or start new analyses.")

if __name__ == "__main__":
    main()