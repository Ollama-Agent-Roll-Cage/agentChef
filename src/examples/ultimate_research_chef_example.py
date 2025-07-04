"""
Ultimate Research Chef - The Complete AgentChef Experience
=========================================================

This example combines the best of both worlds:
- Research capabilities from RAGChef (ArXiv, web search, GitHub, dataset generation)
- Conversational AI from PandasRAG (chat history, context building, knowledge management)
- File ingestion capabilities (LaTeX, code, documents)
- Advanced analytics and visualization

Features:
- Natural conversation with research context
- Dynamic research based on conversation needs
- File ingestion and analysis during chat
- Dataset generation and expansion
- Long-term memory and knowledge building
- Cross-domain research coordination

Perfect for researchers who want an intelligent research companion that can:
- Chat naturally about research topics
- Automatically research papers and web sources
- Analyze and discuss ingested files
- Generate datasets from research
- Maintain context across long research sessions
"""

import pandas as pd
import sys
import json
import asyncio
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Add the src directory to the path so we can import agentChef
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core components
from agentChef.core.chefs.pandas_rag import PandasRAG
from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.core.crawlers.crawlers_module import (
    WebCrawlerWrapper, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler
)

class UltimateResearchChef:
    """
    The ultimate research assistant that combines conversational AI with comprehensive research tools.
    
    Capabilities:
    - Natural conversation with full context and memory
    - Real-time research triggered by conversation topics
    - File ingestion and analysis during chat
    - Dataset generation and research augmentation
    - Cross-domain knowledge synthesis
    - Long-term research project management
    """
    
    def __init__(self, 
                 chef_name: str = "research_companion",
                 knowledge_dir: str = "./ultimate_research_data",
                 model_name: str = "llama3.2:3b"):
        """
        Initialize the Ultimate Research Chef.
        
        Args:
            chef_name: Name of your research companion
            knowledge_dir: Directory for storing all research data
            model_name: Ollama model to use
        """
        self.chef_name = chef_name
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # Initialize conversational AI system
        print(f"🤖 Initializing Ultimate Research Chef: {chef_name}")
        self.rag = PandasRAG(
            data_dir=str(self.knowledge_dir / "conversations"),
            model_name=model_name,
            log_level="INFO",
            max_history_turns=20  # Keep substantial conversation history
        )
        
        # Initialize research system
        self.research_manager = ResearchManager(
            data_dir=str(self.knowledge_dir / "research"),
            model_name=model_name
        )
        
        # Initialize individual crawlers for on-demand research
        self.web_crawler = WebCrawlerWrapper()
        self.arxiv_searcher = ArxivSearcher()
        self.ddg_searcher = DuckDuckGoSearcher()
        self.github_crawler = GitHubCrawler(str(self.knowledge_dir / "github"))
        
        # File ingestion configurations (similar to FileIngestorRAG)
        self.file_processors = {
            ".tex": {"processor": self._process_latex, "domain": "research", "type": "paper"},
            ".py": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".md": {"processor": self._process_markdown, "domain": "general", "type": "documentation"},
            ".txt": {"processor": self._process_text, "domain": "general", "type": "notes"},
            ".json": {"processor": self._process_json, "domain": "general", "type": "data"},
            ".csv": {"processor": self._process_csv, "domain": "general", "type": "data"},
            ".pdf": {"processor": self._process_pdf, "domain": "research", "type": "paper"}
        }
        
        # Research context and session management
        self.active_research_topics = set()
        self.current_session = {
            "topic": None,
            "papers": [],
            "web_results": [],
            "files_analyzed": [],
            "insights": [],
            "datasets_generated": []
        }
        
        # Initialize specialized research agents
        self._setup_research_agents()
        
        print(f"✅ Ultimate Research Chef initialized!")
        print(f"📁 Knowledge directory: {self.knowledge_dir}")
        print(f"🔬 Research tools: ArXiv, Web Search, GitHub, File Analysis")
        print(f"💬 Conversation AI: Context-aware with long-term memory")

    def _setup_research_agents(self):
        """Set up specialized research agents for different domains."""
        
        # Master Research Coordinator
        self.coordinator_agent = self.rag.register_agent(
            "research_coordinator",
            system_prompt="""You are the Master Research Coordinator, an advanced AI research assistant with access to comprehensive research tools.

Your capabilities include:
- Conversational interaction with full context awareness
- Real-time research using ArXiv, web search, and GitHub
- File analysis (LaTeX papers, code, documents)
- Dataset generation and research augmentation
- Cross-domain knowledge synthesis
- Long-term research project management

Your personality:
- Intellectually curious and proactive
- Excellent at identifying research opportunities in conversation
- Great at connecting ideas across different sources
- Always ready to dive deeper into interesting topics
- Maintains context across long research sessions

When users mention research topics, you should:
1. Engage conversationally about their interests
2. Proactively suggest relevant research to explore
3. Offer to search for papers, web resources, or code
4. Help analyze and synthesize information
5. Generate insights and connections
6. Suggest follow-up research directions

You have access to:
- ArXiv paper search and analysis
- Web search for current information
- GitHub repository analysis
- File ingestion and analysis
- Dataset generation tools
- Long-term conversation memory

Always ask clarifying questions and offer specific research assistance.""",
            description="Master research coordinator with conversational AI and comprehensive research tools"
        )
        
        # Specialized Paper Analysis Agent
        self.paper_agent = self.rag.register_agent(
            "paper_analyst",
            system_prompt="""You are a specialized academic paper analysis agent with deep expertise in research literature.

Your expertise:
- Academic paper comprehension and analysis
- Research methodology evaluation
- Citation network analysis
- Cross-paper synthesis and comparison
- Research gap identification
- Literature review generation

You excel at:
- Extracting key insights from papers
- Comparing methodologies across studies
- Identifying research trends and patterns
- Suggesting related work and extensions
- Evaluating research quality and impact
- Generating research summaries and reviews

Always provide detailed academic analysis with proper context.""",
            description="Expert in academic paper analysis and literature review"
        )
        
        # Code and Technical Analysis Agent
        self.code_agent = self.rag.register_agent(
            "code_analyst", 
            system_prompt="""You are a technical code analysis and software research agent.

Your expertise:
- Code architecture and implementation analysis
- Research code evaluation and understanding
- Software engineering best practices
- Algorithm implementation review
- Technical documentation analysis
- Research tool and framework assessment

You help with:
- Understanding research code implementations
- Analyzing software architectures in research
- Evaluating technical approaches
- Suggesting improvements and optimizations
- Connecting code to research papers
- Technical feasibility assessment

Provide detailed technical analysis with practical insights.""",
            description="Expert in code analysis and technical research evaluation"
        )
        
        # Data and Insights Synthesis Agent
        self.synthesis_agent = self.rag.register_agent(
            "synthesis_specialist",
            system_prompt="""You are a research synthesis and insights specialist.

Your role:
- Synthesizing information across multiple sources
- Identifying patterns and connections
- Generating novel insights and hypotheses
- Creating comprehensive research summaries
- Suggesting future research directions
- Building knowledge maps and frameworks

You excel at:
- Connecting disparate pieces of information
- Identifying research gaps and opportunities
- Generating actionable insights
- Creating structured knowledge representations
- Facilitating interdisciplinary connections
- Strategic research planning

Focus on high-level synthesis and strategic thinking.""",
            description="Expert in research synthesis and strategic insights generation"
        )
        
        print(f"✅ Set up {len([self.coordinator_agent, self.paper_agent, self.code_agent, self.synthesis_agent])} specialized research agents")

    async def chat(self, message: str, trigger_research: bool = True) -> str:
        """
        Main chat interface that combines conversation with intelligent research.
        
        Args:
            message: User message
            trigger_research: Whether to automatically trigger research for detected topics
            
        Returns:
            AI response with any research results incorporated
        """
        # Analyze message for research opportunities
        research_triggers = self._detect_research_opportunities(message)
        
        # Start with conversational response
        print(f"\n💬 User: {message}")
        
        # Get initial conversational response
        response = self.rag.query(
            dataframe=pd.DataFrame([{"conversation": "ongoing", "timestamp": datetime.now().isoformat()}]),
            question=message,
            agent_id=self.coordinator_agent,
            save_conversation=True,
            include_history=True
        )
        
        print(f"🤖 {self.chef_name}: {response}")
        
        # Trigger research if relevant topics detected and enabled
        if trigger_research and research_triggers:
            print(f"\n🔍 Research opportunities detected: {', '.join(research_triggers)}")
            
            research_results = await self._intelligent_research(research_triggers, message)
            
            if research_results:
                # Synthesize research into conversation
                synthesis_response = await self._synthesize_research_into_conversation(
                    original_message=message,
                    original_response=response,
                    research_results=research_results
                )
                
                print(f"🔬 Research-enhanced response: {synthesis_response}")
                
                # Save the enhanced response
                self.rag.save_conversation(
                    self.coordinator_agent, 
                    "assistant", 
                    f"[Research-Enhanced] {synthesis_response}"
                )
                
                return synthesis_response
        
        return response

    def _detect_research_opportunities(self, message: str) -> List[str]:
        """Detect research topics and opportunities in user messages."""
        research_triggers = []
        
        # Research-related keywords and phrases
        research_patterns = {
            "paper_search": [
                r"papers? (?:on|about|related to) (.+)",
                r"research (?:on|about|into) (.+)",
                r"literature (?:on|about|review of) (.+)",
                r"studies (?:on|about) (.+)",
                r"find (?:papers|research|studies) (?:on|about) (.+)"
            ],
            "web_search": [
                r"what.s (?:new|latest|current) (?:in|with|about) (.+)",
                r"current (?:state|status|trends) (?:of|in) (.+)",
                r"recent (?:developments|advances|progress) (?:in|with) (.+)",
                r"industry (?:trends|news|updates) (?:about|in) (.+)"
            ],
            "code_search": [
                r"(?:implementation|code|github) (?:for|of) (.+)",
                r"how to (?:implement|code) (.+)",
                r"(?:examples|samples) of (.+) (?:code|implementation)",
                r"repositories (?:for|about|with) (.+)"
            ],
            "general_research": [
                r"learn (?:about|more about) (.+)",
                r"understand (.+) better",
                r"deep dive (?:into|on) (.+)",
                r"comprehensive (?:analysis|study) (?:of|on) (.+)"
            ]
        }
        
        message_lower = message.lower()
        
        for trigger_type, patterns in research_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, message_lower)
                if matches:
                    for match in matches:
                        # Clean up the matched topic
                        topic = match.strip().rstrip('?.,!').strip()
                        if len(topic) > 3:  # Avoid very short matches
                            research_triggers.append(topic)
                            self.active_research_topics.add(topic)
        
        return list(set(research_triggers))  # Remove duplicates

    async def _intelligent_research(self, topics: List[str], context: str) -> Dict[str, Any]:
        """
        Perform intelligent research based on detected topics.
        
        Args:
            topics: List of research topics detected
            context: Original message context
            
        Returns:
            Dictionary containing research results
        """
        print(f"🔍 Starting intelligent research on: {', '.join(topics)}")
        
        research_results = {
            "topics": topics,
            "papers": [],
            "web_results": [],
            "github_repos": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for topic in topics[:2]:  # Limit to first 2 topics to avoid overwhelming
            print(f"\n📚 Researching: {topic}")
            
            # Search ArXiv papers
            try:
                papers = await self.arxiv_searcher.search_papers(topic, max_results=3)
                if papers:
                    research_results["papers"].extend(papers)
                    print(f"   📄 Found {len(papers)} papers")
            except Exception as e:
                print(f"   ❌ Paper search error: {e}")
            
            # Search web for current information
            try:
                web_results = await self.ddg_searcher.text_search(topic, max_results=3)
                if web_results:
                    research_results["web_results"].extend(web_results)
                    print(f"   🌐 Found {len(web_results)} web results")
            except Exception as e:
                print(f"   ❌ Web search error: {e}")
            
            # Brief delay to avoid overwhelming services
            await asyncio.sleep(1)
        
        # Update current session
        self.current_session["papers"].extend(research_results["papers"])
        self.current_session["web_results"].extend(research_results["web_results"])
        
        return research_results

    async def _synthesize_research_into_conversation(self, 
                                                   original_message: str,
                                                   original_response: str, 
                                                   research_results: Dict[str, Any]) -> str:
        """
        Synthesize research results into the ongoing conversation.
        
        Args:
            original_message: User's original message
            original_response: AI's initial response
            research_results: Research findings
            
        Returns:
            Enhanced response incorporating research
        """
        # Create a DataFrame with research data for analysis
        research_data = []
        
        for paper in research_results.get("papers", []):
            research_data.append({
                "type": "paper",
                "title": paper.get("title", ""),
                "content": paper.get("abstract", ""),
                "source": "arxiv",
                "relevance": "high"
            })
        
        for result in research_results.get("web_results", []):
            research_data.append({
                "type": "web",
                "title": result.get("title", ""),
                "content": result.get("snippet", ""),
                "source": "web",
                "relevance": "medium"
            })
        
        if not research_data:
            return original_response
        
        research_df = pd.DataFrame(research_data)
        
        # Use synthesis agent to create enhanced response
        synthesis_prompt = f"""Based on our conversation and the research I just conducted, provide an enhanced response that incorporates the relevant findings.

Original user message: {original_message}
My initial response: {original_response}

I found {len(research_results.get('papers', []))} relevant papers and {len(research_results.get('web_results', []))} web sources.

Please provide an enhanced response that:
1. Builds on my initial response
2. Incorporates key insights from the research
3. Mentions specific papers or sources when relevant
4. Maintains the conversational tone
5. Suggests follow-up research directions if appropriate

Keep the response natural and conversational while being informative."""
        
        enhanced_response = self.rag.query(
            dataframe=research_df,
            question=synthesis_prompt,
            agent_id=self.synthesis_agent,
            save_conversation=True
        )
        
        return enhanced_response

    async def ingest_file_for_conversation(self, file_path: Union[str, Path]) -> bool:
        """
        Ingest a file and make it available for the current conversation.
        
        Args:
            file_path: Path to file to ingest
            
        Returns:
            bool: Success status
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.file_processors:
            print(f"❌ Unsupported file type: {file_ext}")
            return False
        
        print(f"📄 Ingesting {file_path.name} into conversation context...")
        
        try:
            # Process the file
            processor_config = self.file_processors[file_ext]
            processor_func = processor_config["processor"]
            
            processed_content = processor_func(file_path)
            
            if not processed_content:
                print(f"❌ Failed to process file content")
                return False
            
            # Determine content text
            if isinstance(processed_content, dict):
                content_text = processed_content.get("content", "")
                metadata = processed_content.get("metadata", {})
            else:
                content_text = str(processed_content)
                metadata = {}
            
            # Add to knowledge base
            self.rag.add_knowledge(
                agent_id=self.coordinator_agent,
                content=f"File: {file_path.name}\n\n{content_text}",
                source="file_ingestion",
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": processor_config["type"],
                    **metadata
                }
            )
            
            # Update current session
            self.current_session["files_analyzed"].append({
                "file": file_path.name,
                "type": processor_config["type"],
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"✅ Successfully ingested {file_path.name}")
            
            # Provide conversational feedback about the file
            file_analysis_prompt = f"I just analyzed the file '{file_path.name}'. Please provide a brief, conversational summary of what you found and how it might be relevant to our discussion."
            
            analysis_response = self.rag.query(
                dataframe=pd.DataFrame([{"file_content": content_text[:1000]}]),
                question=file_analysis_prompt,
                agent_id=self.paper_agent if processor_config["type"] == "paper" else self.coordinator_agent,
                save_conversation=True
            )
            
            print(f"📋 File Analysis: {analysis_response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return False

    # File processors (simplified versions)
    
    def _process_latex(self, file_path: Path) -> Dict[str, Any]:
        """Process LaTeX files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract title and abstract
            title_match = re.search(r'\\title\{([^}]+)\}', content)
            abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
            
            title = title_match.group(1) if title_match else file_path.name
            abstract = abstract_match.group(1).strip() if abstract_match else ""
            
            formatted_content = f"LaTeX Paper: {title}\n\nAbstract:\n{abstract}\n\nFull Content:\n{content[:2000]}..."
            
            return {
                "content": formatted_content,
                "metadata": {
                    "title": title,
                    "has_abstract": bool(abstract),
                    "word_count": len(content.split())
                }
            }
        except Exception as e:
            print(f"Error processing LaTeX: {e}")
            return None

    def _process_code(self, file_path: Path) -> Dict[str, Any]:
        """Process code files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract functions and classes
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            
            formatted_content = f"Code File: {file_path.name}\nLanguage: {self._detect_language(file_path)}\n"
            formatted_content += f"Functions: {', '.join(functions[:10])}\n"
            formatted_content += f"Classes: {', '.join(classes[:10])}\n\n"
            formatted_content += f"Code:\n{content}"
            
            return {
                "content": formatted_content,
                "metadata": {
                    "functions": functions,
                    "classes": classes,
                    "language": self._detect_language(file_path),
                    "line_count": len(content.splitlines())
                }
            }
        except Exception as e:
            print(f"Error processing code: {e}")
            return None

    def _process_markdown(self, file_path: Path) -> str:
        """Process Markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error processing Markdown: {e}")
            return None

    def _process_text(self, file_path: Path) -> str:
        """Process text files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

    def _process_json(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = f"JSON Data from {file_path.name}:\n{json.dumps(data, indent=2)}"
            
            return {
                "content": content,
                "metadata": {
                    "data_type": type(data).__name__,
                    "keys": list(data.keys()) if isinstance(data, dict) else []
                }
            }
        except Exception as e:
            print(f"Error processing JSON: {e}")
            return None

    def _process_csv(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV files."""
        try:
            df = pd.read_csv(file_path)
            
            content = f"CSV Data from {file_path.name}:\nShape: {df.shape}\nColumns: {list(df.columns)}\n\n"
            content += df.head(10).to_string()
            
            return {
                "content": content,
                "metadata": {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            }
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None

    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files (placeholder - would need PyPDF2 or similar)."""
        return f"PDF file detected: {file_path.name}. PDF processing requires additional libraries."

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.cpp': 'C++', '.c': 'C', '.java': 'Java', '.rs': 'Rust',
            '.go': 'Go', '.r': 'R', '.sql': 'SQL', '.sh': 'Shell'
        }
        return ext_to_lang.get(file_path.suffix.lower(), 'Unknown')

    async def generate_dataset_from_conversation(self, 
                                               session_topic: str = None,
                                               num_turns: int = 3,
                                               expansion_factor: int = 2) -> Dict[str, Any]:
        """
        Generate a dataset from the current conversation and research.
        
        Args:
            session_topic: Topic for the dataset (auto-detected if None)
            num_turns: Number of conversation turns per dataset entry
            expansion_factor: Factor to expand the dataset
            
        Returns:
            Dictionary with dataset generation results
        """
        if not session_topic:
            session_topic = ", ".join(list(self.active_research_topics)[:3])
        
        print(f"📊 Generating dataset for session topic: {session_topic}")
        
        # Use research manager to generate dataset
        try:
            # Collect session papers and content
            papers = self.current_session.get("papers", [])
            
            if not papers:
                print("⚠️ No research papers available for dataset generation")
                return {"error": "No research content available"}
            
            # Generate conversation dataset from research
            dataset_results = await self.research_manager.generate_conversation_dataset(
                papers=papers,
                num_turns=num_turns,
                expansion_factor=expansion_factor,
                clean=True,
                callback=lambda msg: print(f"   {msg}")
            )
            
            # Update session tracking
            self.current_session["datasets_generated"].append({
                "topic": session_topic,
                "timestamp": datetime.now().isoformat(),
                "num_conversations": len(dataset_results.get("conversations", [])),
                "output_path": dataset_results.get("output_path", "")
            })
            
            print(f"✅ Generated dataset with {len(dataset_results.get('conversations', []))} conversations")
            
            return dataset_results
            
        except Exception as e:
            print(f"❌ Error generating dataset: {e}")
            return {"error": str(e)}

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current research session."""
        summary = {
            "session_start": self.current_session.get("start_time", "Unknown"),
            "active_topics": list(self.active_research_topics),
            "research_statistics": {
                "papers_found": len(self.current_session.get("papers", [])),
                "web_results": len(self.current_session.get("web_results", [])),
                "files_analyzed": len(self.current_session.get("files_analyzed", [])),
                "datasets_generated": len(self.current_session.get("datasets_generated", []))
            },
            "conversation_statistics": {},
            "knowledge_base_size": 0
        }
        
        # Get conversation statistics
        try:
            conv_summary = self.rag.get_summary()
            summary["conversation_statistics"] = conv_summary
        except Exception as e:
            print(f"Error getting conversation summary: {e}")
        
        return summary

    async def start_interactive_session(self):
        """Start an interactive research chat session."""
        print(f"\n🎓 Welcome to Ultimate Research Chef Interactive Session!")
        print(f"💬 Chat naturally about research topics")
        print(f"🔍 I'll automatically research papers, web sources, and code")
        print(f"📄 You can ingest files with 'ingest <filepath>'")
        print(f"📊 Generate datasets with 'generate dataset'")
        print(f"📋 Get session summary with 'summary'")
        print(f"❌ Type 'exit' to end the session")
        print(f"\n{'-'*60}\n")
        
        # Initialize session
        self.current_session["start_time"] = datetime.now().isoformat()
        
        while True:
            try:
                user_input = input("💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"\n👋 Ending research session. Here's your summary:")
                    summary = self.get_session_summary()
                    print(json.dumps(summary, indent=2))
                    break
                
                elif user_input.lower().startswith('ingest '):
                    file_path = user_input[7:].strip()
                    await self.ingest_file_for_conversation(file_path)
                    continue
                
                elif user_input.lower() == 'generate dataset':
                    await self.generate_dataset_from_conversation()
                    continue
                
                elif user_input.lower() == 'summary':
                    summary = self.get_session_summary()
                    print(f"\n📊 Session Summary:")
                    print(json.dumps(summary, indent=2))
                    continue
                
                # Regular chat with research
                response = await self.chat(user_input, trigger_research=True)
                
            except KeyboardInterrupt:
                print(f"\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def demo_ultimate_research_chef():
    """Demonstrate the Ultimate Research Chef capabilities."""
    print("🚀 Ultimate Research Chef Demo")
    print("=" * 60)
    
    # Initialize the chef
    chef = UltimateResearchChef(
        chef_name="Einstein",
        knowledge_dir="./ultimate_research_demo",
        model_name="llama3.2:3b"
    )
    
    # Demo conversation with automatic research
    demo_messages = [
        "Hi Einstein! I'm interested in transformer neural networks. What can you tell me about them?",
        "That's fascinating! I'd love to see some recent papers on attention mechanisms.",
        "Can you help me understand how transformers compare to RNNs?",
        "I'm thinking about implementing a transformer model. Are there good code examples available?"
    ]
    
    print(f"\n🎭 Demonstrating conversational research capabilities:")
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n" + "="*70)
        print(f"Demo Message {i}:")
        response = await chef.chat(message, trigger_research=True)
        
        # Small delay between messages
        await asyncio.sleep(2)
    
    # Demonstrate file ingestion (if sample files exist)
    sample_files = ["README.md", "setup.py", "requirements.txt"]
    for file_path in sample_files:
        if Path(file_path).exists():
            print(f"\n📄 Demonstrating file ingestion with {file_path}")
            await chef.ingest_file_for_conversation(file_path)
            break
    
    # Generate a dataset from the session
    print(f"\n📊 Generating dataset from research session...")
    dataset_result = await chef.generate_dataset_from_conversation(
        session_topic="transformer neural networks",
        num_turns=2,
        expansion_factor=2
    )
    
    if "error" not in dataset_result:
        print(f"✅ Dataset generated with {len(dataset_result.get('conversations', []))} conversations")
    
    # Show session summary
    print(f"\n📋 Final Session Summary:")
    summary = chef.get_session_summary()
    print(json.dumps(summary, indent=2))
    
    print(f"\n✅ Demo completed! Ultimate Research Chef is ready for interactive use.")
    print(f"💡 Run chef.start_interactive_session() for interactive mode.")


def main():
    """Main function - choose demo or interactive mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Research Chef")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                       help="Run mode: demo or interactive")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model to use")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        asyncio.run(demo_ultimate_research_chef())
    else:
        chef = UltimateResearchChef(
            chef_name="Research_Companion",
            model_name=args.model
        )
        asyncio.run(chef.start_interactive_session())


if __name__ == "__main__":
    main()