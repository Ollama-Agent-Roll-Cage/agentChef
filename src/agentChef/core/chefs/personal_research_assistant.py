"""
Personal Research Assistant Chef - Advanced File Ingestion and Research System
==============================================================================

This chef combines file ingestion capabilities with conversational AI and real-time research.
Based on the FileIngestorRAG example but integrated into the core chefs system.

Features:
- Multi-format file ingestion (LaTeX, code, documents, data)
- Conversational AI with context building
- Real-time research integration
- Knowledge management and domain classification
- Interactive chat sessions with file analysis
- Cross-domain research coordination
"""

import json
import asyncio
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Core AgentChef imports
from .base_chef import BaseChef
from .ragchef import ResearchManager
from .pandas_rag import PandasRAG
from ..crawlers.crawlers_module import WebCrawlerWrapper, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler
from ..ollama.ollama_interface import OllamaInterface
from ...logs.agentchef_logging import log, setup_file_logging

import logging
logger = logging.getLogger(__name__)

class PersonalResearchAssistant(BaseChef):
    """
    Advanced Personal Research Assistant that combines file ingestion, 
    conversational AI, and real-time research capabilities.
    
    This chef can:
    - Ingest and analyze various file formats
    - Conduct conversational AI sessions with context
    - Perform real-time research on detected topics
    - Build and manage knowledge bases
    - Generate datasets from research and conversations
    - Coordinate cross-domain research projects
    """
    
    def __init__(self, 
                 assistant_name: str = "research_assistant",
                 knowledge_dir: str = "./research_assistant_data",
                 model_name: str = "llama3.2:3b"):
        """
        Initialize the Personal Research Assistant.
        
        Args:
            assistant_name: Name of the assistant
            knowledge_dir: Directory for storing knowledge and data
            model_name: Ollama model to use
        """
        super().__init__(name=assistant_name)
        
        self.assistant_name = assistant_name
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # Initialize core systems
        logger.info(f"Initializing Personal Research Assistant: {assistant_name}")
        
        # Conversational AI system
        self.rag = PandasRAG(
            data_dir=str(self.knowledge_dir / "conversations"),
            model_name=model_name,
            max_history_turns=15
        )
        
        # Research system
        self.research_manager = ResearchManager(
            data_dir=str(self.knowledge_dir / "research"),
            model_name=model_name
        )
        
        # Ollama interface
        self.ollama = OllamaInterface(model_name=model_name)
        
        # File processing configurations
        self.file_processors = {
            ".tex": {"processor": self._process_latex, "domain": "research", "type": "paper"},
            ".latex": {"processor": self._process_latex, "domain": "research", "type": "paper"},
            ".py": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".js": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".ts": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".cpp": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".c": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".java": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".rs": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".go": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".r": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".sql": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".sh": {"processor": self._process_code, "domain": "computer_science", "type": "code"},
            ".yml": {"processor": self._process_code, "domain": "computer_science", "type": "config"},
            ".yaml": {"processor": self._process_code, "domain": "computer_science", "type": "config"},
            ".md": {"processor": self._process_markdown, "domain": "general", "type": "documentation"},
            ".rst": {"processor": self._process_text, "domain": "general", "type": "documentation"},
            ".txt": {"processor": self._process_text, "domain": "general", "type": "notes"},
            ".json": {"processor": self._process_json, "domain": "general", "type": "data"},
            ".csv": {"processor": self._process_csv, "domain": "general", "type": "data"},
            ".xlsx": {"processor": self._process_excel, "domain": "general", "type": "data"},
            ".parquet": {"processor": self._process_parquet, "domain": "general", "type": "data"},
            ".pdf": {"processor": self._process_pdf, "domain": "research", "type": "document"}
        }
        
        # Initialize specialized agents
        self._setup_specialized_agents()
        
        # Session management
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "active_topics": set(),
            "files_processed": [],
            "research_conducted": [],
            "insights_generated": [],
            "datasets_created": []
        }
        
        # Set up logging
        setup_file_logging(str(self.knowledge_dir / "logs"))
        
        logger.info("Personal Research Assistant initialized successfully")

    def _setup_specialized_agents(self):
        """Set up specialized agents for different research functions."""
        
        # Main Research Coordinator
        self.coordinator_agent = self.rag.register_agent(
            "research_coordinator",
            system_prompt=f"""You are {self.assistant_name}, an advanced Personal Research Assistant with comprehensive file analysis and research capabilities.

Your expertise includes:
- Multi-format file ingestion and analysis (LaTeX papers, code, documents, data)
- Conversational AI with full context awareness
- Real-time research using ArXiv, web search, and GitHub
- Cross-domain knowledge synthesis
- Research project management
- Dataset generation and analysis

Your personality:
- Intellectually curious and deeply analytical
- Excellent at connecting ideas across different sources and formats
- Proactive in identifying research opportunities
- Great at explaining complex concepts clearly
- Always ready to dive deeper into interesting topics
- Maintains context across long research sessions

Your capabilities:
- Analyze LaTeX papers and extract key research insights
- Understand and explain code implementations
- Process data files and generate insights
- Conduct real-time research on any topic mentioned
- Build comprehensive knowledge bases
- Generate research datasets from conversations and files
- Coordinate multi-domain research projects

You should:
1. Engage naturally and conversationally
2. Proactively analyze uploaded files and offer insights
3. Suggest relevant research when topics arise
4. Connect information across different sources
5. Offer to generate datasets or conduct further research
6. Maintain context and build on previous conversations
7. Ask clarifying questions when helpful

Always provide specific, actionable insights and offer concrete next steps.""",
            description="Advanced personal research assistant with file analysis and research capabilities"
        )
        
        # File Analysis Specialist
        self.file_agent = self.rag.register_agent(
            "file_analyst",
            system_prompt="""You are a specialized File Analysis Agent with expertise in understanding and extracting insights from various file formats.

Your expertise:
- LaTeX paper analysis and research insight extraction
- Code architecture and implementation analysis
- Document processing and content synthesis
- Data file analysis and pattern recognition
- Cross-format knowledge integration

You excel at:
- Extracting key insights from academic papers
- Understanding code structure and functionality
- Analyzing data patterns and trends
- Connecting information across different file types
- Generating comprehensive file summaries
- Identifying research opportunities from file content

Always provide detailed analysis with actionable insights.""",
            description="Expert in multi-format file analysis and content extraction"
        )
        
        # Research Synthesis Agent
        self.synthesis_agent = self.rag.register_agent(
            "synthesis_specialist",
            system_prompt="""You are a Research Synthesis Specialist focused on connecting and synthesizing information from multiple sources.

Your role:
- Synthesizing insights across files, conversations, and research
- Identifying patterns and connections
- Generating novel hypotheses and research directions
- Creating comprehensive knowledge maps
- Building strategic research recommendations

You excel at:
- Cross-domain knowledge integration
- Pattern recognition across different information types
- Strategic thinking and planning
- Research gap identification
- Innovation and creative thinking
- Long-term project coordination

Focus on high-level synthesis and strategic insights.""",
            description="Expert in research synthesis and strategic insight generation"
        )
        
        logger.info(f"Set up {len([self.coordinator_agent, self.file_agent, self.synthesis_agent])} specialized agents")

    async def ingest_file(self, file_path: Union[str, Path], 
                         auto_research: bool = True) -> Dict[str, Any]:
        """
        Ingest and analyze a file, optionally triggering automatic research.
        
        Args:
            file_path: Path to the file to ingest
            auto_research: Whether to automatically research detected topics
            
        Returns:
            Dictionary with ingestion results and analysis
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.file_processors:
            return {"error": f"Unsupported file type: {file_ext}"}
        
        logger.info(f"Ingesting file: {file_path}")
        
        try:
            # Process the file
            processor_config = self.file_processors[file_ext]
            processor_func = processor_config["processor"]
            
            processed_content = await asyncio.to_thread(processor_func, file_path)
            
            if not processed_content:
                return {"error": "Failed to process file content"}
            
            # Extract content and metadata
            if isinstance(processed_content, dict):
                content_text = processed_content.get("content", "")
                metadata = processed_content.get("metadata", {})
            else:
                content_text = str(processed_content)
                metadata = {}
            
            # Add to knowledge base
            self.rag.add_knowledge(
                agent_id=self.coordinator_agent,
                content=f"File: {file_path.name}\nType: {processor_config['type']}\nDomain: {processor_config['domain']}\n\n{content_text}",
                source="file_ingestion",
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": processor_config["type"],
                    "domain": processor_config["domain"],
                    "ingestion_time": datetime.now().isoformat(),
                    **metadata
                }
            )
            
            # Analyze the file content
            analysis_prompt = f"""I've just processed the {processor_config['type']} file '{file_path.name}' from the {processor_config['domain']} domain. 

Please provide:
1. A brief summary of the file's main content
2. Key insights or important points
3. Potential research topics or questions this raises
4. How this connects to any previous files or conversations
5. Suggested next steps or follow-up actions

Keep your analysis conversational but thorough."""
            
            # Create a DataFrame with file data for analysis
            file_df = pd.DataFrame([{
                "file_name": file_path.name,
                "file_type": processor_config["type"],
                "domain": processor_config["domain"],
                "content": content_text[:2000]  # First 2000 chars for analysis
            }])
            
            analysis = self.rag.query(
                dataframe=file_df,
                question=analysis_prompt,
                agent_id=self.file_agent,
                save_conversation=True
            )
            
            # Update session tracking
            self.current_session["files_processed"].append({
                "file": file_path.name,
                "type": processor_config["type"],
                "domain": processor_config["domain"],
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis[:200] + "..." if len(analysis) > 200 else analysis
            })
            
            result = {
                "success": True,
                "file_name": file_path.name,
                "file_type": processor_config["type"],
                "domain": processor_config["domain"],
                "analysis": analysis,
                "metadata": metadata
            }
            
            # Detect research topics if auto_research is enabled
            if auto_research:
                research_topics = self._detect_research_topics(content_text + " " + analysis)
                if research_topics:
                    logger.info(f"Detected research topics: {research_topics}")
                    research_results = await self._conduct_research(research_topics[:2])  # Limit to 2 topics
                    result["research_triggered"] = research_results
            
            logger.info(f"Successfully ingested and analyzed: {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"error": f"Processing error: {e}"}

    def _detect_research_topics(self, text: str) -> List[str]:
        """Detect potential research topics from text content."""
        research_topics = []
        
        # Research-related patterns
        patterns = [
            r"(?:research|study|analysis|investigation)(?:\s+(?:on|into|about|of))?\s+([a-zA-Z\s]{3,30})",
            r"(?:algorithm|method|approach|technique)(?:\s+for)?\s+([a-zA-Z\s]{3,30})",
            r"(?:neural|machine|deep)\s+learning\s+([a-zA-Z\s]{3,30})",
            r"(?:data|dataset|database)\s+(?:for|about|on)\s+([a-zA-Z\s]{3,30})",
            r"(?:model|framework|system)\s+(?:for|of)\s+([a-zA-Z\s]{3,30})"
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                topic = match.strip().rstrip('.,!?').strip()
                if len(topic) > 5 and topic not in research_topics:
                    research_topics.append(topic)
        
        return research_topics[:5]  # Limit to 5 topics

    async def _conduct_research(self, topics: List[str]) -> Dict[str, Any]:
        """Conduct research on detected topics."""
        logger.info(f"Conducting research on topics: {topics}")
        
        research_results = {
            "topics": topics,
            "papers": [],
            "web_results": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for topic in topics:
            try:
                # Search for papers
                papers = await self.research_manager.arxiv_searcher.search_papers(topic, max_results=2)
                if papers:
                    research_results["papers"].extend(papers)
                
                # Search web
                web_results = await self.research_manager.ddg_searcher.text_search(topic, max_results=2)
                if web_results:
                    research_results["web_results"].extend(web_results)
                
                # Small delay
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Research error for topic '{topic}': {e}")
        
        # Update session tracking
        self.current_session["research_conducted"].append({
            "topics": topics,
            "papers_found": len(research_results["papers"]),
            "web_results_found": len(research_results["web_results"]),
            "timestamp": datetime.now().isoformat()
        })
        
        return research_results

    async def chat(self, message: str, 
                   include_file_context: bool = True,
                   auto_research: bool = True) -> str:
        """
        Chat with the assistant, optionally including file context and triggering research.
        
        Args:
            message: User message
            include_file_context: Whether to include context from ingested files
            auto_research: Whether to automatically research detected topics
            
        Returns:
            Assistant response
        """
        logger.info(f"Chat message received: {message[:100]}...")
        
        # Detect research opportunities
        research_topics = []
        if auto_research:
            research_topics = self._detect_research_topics(message)
        
        # Create a DataFrame for the conversation context
        context_data = [{
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "session_files": len(self.current_session["files_processed"]),
            "session_research": len(self.current_session["research_conducted"])
        }]
        
        conversation_df = pd.DataFrame(context_data)
        
        # Get initial response
        response = self.rag.query(
            dataframe=conversation_df,
            question=message,
            agent_id=self.coordinator_agent,
            save_conversation=True,
            include_history=True
        )
        
        # Conduct research if topics detected
        if research_topics:
            logger.info(f"Research topics detected: {research_topics}")
            
            try:
                research_results = await self._conduct_research(research_topics[:2])
                
                if research_results["papers"] or research_results["web_results"]:
                    # Enhance response with research
                    enhanced_response = await self._enhance_response_with_research(
                        original_message=message,
                        original_response=response,
                        research_results=research_results
                    )
                    
                    # Save enhanced response
                    self.rag.save_conversation(
                        self.coordinator_agent,
                        "assistant",
                        f"[Research-Enhanced] {enhanced_response}"
                    )
                    
                    return enhanced_response
            
            except Exception as e:
                logger.error(f"Research enhancement error: {e}")
        
        return response

    async def _enhance_response_with_research(self, 
                                            original_message: str,
                                            original_response: str,
                                            research_results: Dict[str, Any]) -> str:
        """Enhance response with research findings."""
        
        # Create research data DataFrame
        research_data = []
        
        for paper in research_results.get("papers", []):
            research_data.append({
                "type": "paper",
                "title": paper.get("title", ""),
                "content": paper.get("abstract", ""),
                "source": "arxiv"
            })
        
        for result in research_results.get("web_results", []):
            research_data.append({
                "type": "web",
                "title": result.get("title", ""),
                "content": result.get("snippet", ""),
                "source": "web"
            })
        
        if not research_data:
            return original_response
        
        research_df = pd.DataFrame(research_data)
        
        enhancement_prompt = f"""Based on our conversation and the research I just conducted, provide an enhanced response.

Original message: {original_message}
My initial response: {original_response}

I found {len(research_results.get('papers', []))} relevant papers and {len(research_results.get('web_results', []))} web sources.

Please enhance my response by:
1. Building on my initial response
2. Incorporating key insights from the research
3. Mentioning specific papers or sources when relevant
4. Maintaining a natural conversational tone
5. Suggesting follow-up research or actions if appropriate

Keep the response natural and engaging while being informative."""
        
        enhanced_response = self.rag.query(
            dataframe=research_df,
            question=enhancement_prompt,
            agent_id=self.synthesis_agent,
            save_conversation=True
        )
        
        return enhanced_response

    # File processors implementation
    
    async def _process_latex(self, file_path: Path) -> Dict[str, Any]:
        """Process LaTeX files and extract research content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract key components
            title_match = re.search(r'\\title\{([^}]+)\}', content)
            abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
            author_matches = re.findall(r'\\author\{([^}]+)\}', content)
            
            title = title_match.group(1).strip() if title_match else file_path.stem
            abstract = abstract_match.group(1).strip() if abstract_match else ""
            authors = [author.strip() for author in author_matches]
            
            # Extract sections
            sections = re.findall(r'\\(?:section|subsection)\{([^}]+)\}', content)
            
            # Extract citations
            citations = re.findall(r'\\cite\{([^}]+)\}', content)
            
            formatted_content = f"""LaTeX Research Paper: {title}

Authors: {', '.join(authors) if authors else 'Not specified'}

Abstract:
{abstract if abstract else 'No abstract found'}

Sections: {', '.join(sections[:10]) if sections else 'No sections identified'}

Citations: {len(citations)} references found

Full Content:
{content[:3000]}{'...' if len(content) > 3000 else ''}"""
            
            return {
                "content": formatted_content,
                "metadata": {
                    "title": title,
                    "authors": authors,
                    "has_abstract": bool(abstract),
                    "section_count": len(sections),
                    "citation_count": len(citations),
                    "word_count": len(content.split())
                }
            }
        except Exception as e:
            logger.error(f"Error processing LaTeX file: {e}")
            return None

    async def _process_code(self, file_path: Path) -> Dict[str, Any]:
        """Process code files and extract structure."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Language detection
            language = self._detect_language(file_path)
            
            # Extract functions and classes (Python-specific patterns, extend for other languages)
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            imports = re.findall(r'(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content)
            
            # Extract comments and docstrings
            comments = re.findall(r'#\s*(.+)', content)
            docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)
            
            formatted_content = f"""Code File: {file_path.name}
Language: {language}
Lines of Code: {len(content.splitlines())}

Functions: {', '.join(functions[:15]) if functions else 'None identified'}
Classes: {', '.join(classes[:10]) if classes else 'None identified'}
Key Imports: {', '.join(imports[:10]) if imports else 'None identified'}

Documentation:
{docstrings[0][:500] + '...' if docstrings and len(docstrings[0]) > 500 else docstrings[0] if docstrings else 'No docstrings found'}

Code Content:
{content}"""
            
            return {
                "content": formatted_content,
                "metadata": {
                    "language": language,
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                    "line_count": len(content.splitlines()),
                    "has_docstrings": bool(docstrings),
                    "comment_count": len(comments)
                }
            }
        except Exception as e:
            logger.error(f"Error processing code file: {e}")
            return None

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.cpp': 'C++', '.c': 'C', '.java': 'Java', '.rs': 'Rust',
            '.go': 'Go', '.r': 'R', '.sql': 'SQL', '.sh': 'Shell',
            '.yml': 'YAML', '.yaml': 'YAML', '.json': 'JSON'
        }
        return ext_to_lang.get(file_path.suffix.lower(), 'Unknown')

    async def _process_markdown(self, file_path: Path) -> str:
        """Process Markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract headers
            headers = re.findall(r'^#+\s*(.+)$', content, re.MULTILINE)
            
            formatted_content = f"""Markdown Document: {file_path.name}

Headers: {', '.join(headers[:10]) if headers else 'No headers found'}

Content:
{content}"""
            
            return formatted_content
        except Exception as e:
            logger.error(f"Error processing Markdown file: {e}")
            return None

    async def _process_text(self, file_path: Path) -> str:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return f"Text Document: {file_path.name}\n\n{content}"
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            return None

    async def _process_json(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = f"""JSON Data: {file_path.name}
Data Type: {type(data).__name__}
Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}

Content:
{json.dumps(data, indent=2)[:2000]}{'...' if len(str(data)) > 2000 else ''}"""
            
            return {
                "content": content,
                "metadata": {
                    "data_type": type(data).__name__,
                    "keys": list(data.keys()) if isinstance(data, dict) else [],
                    "size": len(str(data))
                }
            }
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            return None

    async def _process_csv(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV files."""
        try:
            df = pd.read_csv(file_path)
            
            content = f"""CSV Dataset: {file_path.name}
Shape: {df.shape}
Columns: {list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Sample Data:
{df.head(10).to_string()}

Summary Statistics:
{df.describe().to_string()}"""
            
            return {
                "content": content,
                "metadata": {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "data_types": df.dtypes.to_dict()
                }
            }
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            return None

    async def _process_excel(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel files."""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            content = f"Excel File: {file_path.name}\nSheets: {', '.join(sheet_names)}\n\n"
            
            # Process first sheet or first few sheets
            for sheet_name in sheet_names[:3]:  # Limit to first 3 sheets
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content += f"Sheet '{sheet_name}':\nShape: {df.shape}\nColumns: {list(df.columns)}\n"
                content += df.head(5).to_string() + "\n\n"
            
            return {
                "content": content,
                "metadata": {
                    "sheet_names": sheet_names,
                    "sheet_count": len(sheet_names)
                }
            }
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return None

    async def _process_parquet(self, file_path: Path) -> Dict[str, Any]:
        """Process Parquet files."""
        try:
            df = pd.read_parquet(file_path)
            
            content = f"""Parquet Dataset: {file_path.name}
Shape: {df.shape}
Columns: {list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Sample Data:
{df.head(10).to_string()}"""
            
            return {
                "content": content,
                "metadata": {
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
            }
        except Exception as e:
            logger.error(f"Error processing Parquet file: {e}")
            return None

    async def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files (placeholder - would need PyPDF2 or similar)."""
        return f"PDF Document: {file_path.name}\n\nPDF processing requires additional libraries (PyPDF2, pdfplumber, etc.)"

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        return {
            "session_info": {
                "start_time": self.current_session["start_time"],
                "duration": str(datetime.now() - datetime.fromisoformat(self.current_session["start_time"])),
                "assistant_name": self.assistant_name
            },
            "files_processed": {
                "count": len(self.current_session["files_processed"]),
                "details": self.current_session["files_processed"]
            },
            "research_conducted": {
                "count": len(self.current_session["research_conducted"]),
                "details": self.current_session["research_conducted"]
            },
            "conversation_stats": self.rag.get_summary() if hasattr(self.rag, 'get_summary') else {},
            "knowledge_base": {
                "total_items": len(self.current_session["files_processed"]),
                "domains": list(set(item["domain"] for item in self.current_session["files_processed"]))
            }
        }

    async def generate_research_dataset(self, 
                                      topic: str = None,
                                      num_turns: int = 3,
                                      expansion_factor: int = 2) -> Dict[str, Any]:
        """Generate a research dataset from session content."""
        if not topic:
            # Auto-detect topic from session
            all_topics = []
            for research in self.current_session["research_conducted"]:
                all_topics.extend(research["topics"])
            topic = ", ".join(all_topics[:3]) if all_topics else "research session"
        
        logger.info(f"Generating research dataset for topic: {topic}")
        
        try:
            # Collect research papers from session
            papers = []
            for research in self.current_session["research_conducted"]:
                papers.extend(research.get("papers", []))
            
            if not papers:
                return {"error": "No research papers available for dataset generation"}
            
            # Generate dataset using research manager
            dataset_results = await self.research_manager.generate_conversation_dataset(
                papers=papers,
                num_turns=num_turns,
                expansion_factor=expansion_factor,
                clean=True
            )
            
            # Update session tracking
            self.current_session["datasets_created"].append({
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "conversations": len(dataset_results.get("conversations", [])),
                "output_path": dataset_results.get("output_path", "")
            })
            
            return dataset_results
            
        except Exception as e:
            logger.error(f"Dataset generation error: {e}")
            return {"error": str(e)}

    async def start_interactive_session(self):
        """Start an interactive research session."""
        print(f"\n🎓 Welcome to {self.assistant_name} - Personal Research Assistant!")
        print(f"📁 I can analyze files, conduct research, and maintain conversations")
        print(f"💬 Chat naturally - I'll maintain context and suggest research")
        print(f"📄 Upload files with: ingest <filepath>")
        print(f"🔍 Manual research with: research <topic>")
        print(f"📊 Generate dataset with: dataset")
        print(f"📋 Session summary with: summary")
        print(f"❌ Exit with: exit")
        print(f"\n{'-'*70}\n")
        
        while True:
            try:
                user_input = input(f"💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"\n👋 Ending session. Here's your summary:")
                    summary = self.get_session_summary()
                    print(json.dumps(summary, indent=2, default=str))
                    break
                
                elif user_input.lower().startswith('ingest '):
                    file_path = user_input[7:].strip()
                    result = await self.ingest_file(file_path)
                    print(f"📄 {result}")
                    continue
                
                elif user_input.lower().startswith('research '):
                    topic = user_input[9:].strip()
                    research_results = await self._conduct_research([topic])
                    print(f"🔍 Found {len(research_results['papers'])} papers and {len(research_results['web_results'])} web results")
                    continue
                
                elif user_input.lower() == 'dataset':
                    result = await self.generate_research_dataset()
                    print(f"📊 Dataset generation: {result}")
                    continue
                
                elif user_input.lower() == 'summary':
                    summary = self.get_session_summary()
                    print(f"\n📊 Session Summary:")
                    print(json.dumps(summary, indent=2, default=str))
                    continue
                
                # Regular chat
                response = await self.chat(user_input)
                print(f"\n🤖 {self.assistant_name}: {response}")
                
            except KeyboardInterrupt:
                print(f"\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Session error: {e}")
                print(f"❌ Error: {e}")


# Example usage and testing
async def demo_personal_research_assistant():
    """Demo the Personal Research Assistant."""
    print("🚀 Personal Research Assistant Demo")
    print("=" * 50)
    
    # Initialize assistant
    assistant = PersonalResearchAssistant(
        assistant_name="Ada",
        knowledge_dir="./personal_assistant_demo",
        model_name="llama3.2:3b"
    )
    
    # Demo file ingestion (if files exist)
    demo_files = ["README.md", "setup.py", "requirements.txt"]
    for file_path in demo_files:
        if Path(file_path).exists():
            print(f"\n📄 Ingesting {file_path}...")
            result = await assistant.ingest_file(file_path)
            print(f"Result: {result.get('analysis', 'Analysis not available')[:200]}...")
            break
    
    # Demo conversation
    demo_messages = [
        "Hi Ada! Can you help me understand transformer neural networks?",
        "That's interesting! I'd like to see some recent research on attention mechanisms.",
        "Can you explain how this relates to the files I've uploaded?"
    ]
    
    for message in demo_messages:
        print(f"\n💬 User: {message}")
        response = await assistant.chat(message)
        print(f"🤖 Ada: {response[:300]}...")
        await asyncio.sleep(1)
    
    # Demo dataset generation
    print(f"\n📊 Generating research dataset...")
    dataset_result = await assistant.generate_research_dataset()
    if "error" not in dataset_result:
        print(f"✅ Generated dataset with {len(dataset_result.get('conversations', []))} conversations")
    
    # Show session summary
    print(f"\n📋 Session Summary:")
    summary = assistant.get_session_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    print(f"\n✅ Demo completed!")


if __name__ == "__main__":
    # Demo mode
    asyncio.run(demo_personal_research_assistant())
