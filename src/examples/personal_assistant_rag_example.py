"""
Enhanced Personal Assistant RAG with File Ingestion
==================================================

Extends the personal assistant to ingest and chat with various file types:
- LaTeX papers (.tex files from arXiv)
- Code files (.py, .js, .cpp, .java, etc.)
- Markdown documentation (.md)
- Text files (.txt)
- JSON data files (.json)
- CSV/Excel data files

Perfect for researchers who want to chat with their papers, code, and documentation.
"""

import pandas as pd
import sys
import json
import asyncio
import re
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Add the src directory to the path so we can import agentChef
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentChef.core.chefs.pandas_rag import PandasRAG

class FileIngestorRAG:
    """
    Enhanced personal assistant with comprehensive file ingestion capabilities.
    Can process and chat with LaTeX papers, code files, documentation, and data.
    """
    
    def __init__(self, assistant_name: str = "file_assistant", 
                 knowledge_dir: str = "./file_knowledge", 
                 model_name: str = "llama3.2:3b"):
        """
        Initialize the file ingestion assistant.
        
        Args:
            assistant_name: Name of your assistant instance
            knowledge_dir: Directory for persistent knowledge storage
            model_name: Ollama model to use
        """
        self.assistant_name = assistant_name
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the core RAG system
        self.rag = PandasRAG(
            data_dir=str(self.knowledge_dir),
            model_name=model_name,
            log_level="INFO",
            max_history_turns=50
        )
        
        # File type configurations
        self.file_processors = {
            # LaTeX and academic papers
            ".tex": {"processor": self._process_latex, "domain": "research", "type": "paper"},
            ".latex": {"processor": self._process_latex, "domain": "research", "type": "paper"},
            
            # Code files
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
            ".yaml": {"processor": self._process_code, "domain": "computer_science", "type": "config"},  # Fixed: was "..yaml"
            
            # Documentation
            ".md": {"processor": self._process_markdown, "domain": "general", "type": "documentation"},
            ".rst": {"processor": self._process_text, "domain": "general", "type": "documentation"},
            ".txt": {"processor": self._process_text, "domain": "general", "type": "notes"},
            
            # Data files
            ".json": {"processor": self._process_json, "domain": "general", "type": "data"},
            ".csv": {"processor": self._process_csv, "domain": "general", "type": "data"},
            ".xlsx": {"processor": self._process_excel, "domain": "general", "type": "data"},
            ".parquet": {"processor": self._process_parquet, "domain": "general", "type": "data"},
        }
        
        # Initialize domain-specific agents
        self._setup_domain_agents()
        
        print(f"🤖 Enhanced File Assistant '{assistant_name}' initialized")
        print(f"📁 Knowledge directory: {self.knowledge_dir}")
        print(f"🗂️ Supported file types: {list(self.file_processors.keys())}")

    def _setup_domain_agents(self):
        """Set up specialized agents for different domains."""
        
        # Research agent for papers and LaTeX
        self.rag.register_agent(
            "research_agent",
            system_prompt="""You are a research assistant specializing in academic papers and LaTeX documents.

Your expertise includes:
- Academic paper analysis and summarization
- LaTeX document structure and content extraction
- Mathematical notation and equation interpretation
- Research methodology evaluation
- Citation and reference analysis
- Cross-paper connection identification

You help with:
- Summarizing research papers and their key contributions
- Explaining complex mathematical concepts and equations
- Identifying relationships between different papers
- Extracting key methodologies and results
- Answering questions about research content
- Finding specific information within papers

Always provide context from the specific papers when answering questions.
Use proper academic language and cite relevant sections when helpful.""",
            description="Specialist in academic papers, LaTeX documents, and research analysis"
        )
        
        # Code agent for programming files
        self.rag.register_agent(
            "code_agent", 
            system_prompt="""You are a programming assistant specializing in code analysis and documentation.

Your expertise includes:
- Code structure and architecture analysis
- Function and class documentation
- Algorithm explanation and optimization
- Debugging and error identification
- Code quality assessment
- Best practices recommendations
- Cross-file dependency analysis

You help with:
- Explaining how code works and its purpose
- Identifying potential bugs or improvements
- Documenting functions and classes
- Suggesting optimizations and refactoring
- Finding specific implementations or patterns
- Understanding code architecture and design

Always reference specific code sections and provide concrete examples.
Consider performance, readability, and maintainability in your analysis.""",
            description="Specialist in code analysis, documentation, and programming guidance"
        )
        
        # Documentation agent for markdown and text
        self.rag.register_agent(
            "docs_agent",
            system_prompt="""You are a documentation specialist focusing on technical writing and knowledge management.

Your expertise includes:
- Technical documentation analysis
- Markdown and text structure interpretation
- Information organization and retrieval
- Procedure and guide explanation
- Knowledge synthesis across documents
- Content summarization and extraction

You help with:
- Explaining procedures and instructions
- Finding specific information in documentation
- Summarizing long documents and guides
- Connecting information across multiple documents
- Answering questions about processes and procedures
- Organizing and structuring knowledge

Always provide clear, structured answers with references to specific sections.
Focus on practical, actionable information.""",
            description="Specialist in documentation, guides, and knowledge management"
        )
        
        print("✅ Set up specialized domain agents")

    def ingest_file(self, file_path: Union[str, Path], 
                   domain_override: str = None,
                   custom_metadata: Dict[str, Any] = None) -> bool:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to the file to ingest
            domain_override: Override the default domain for this file
            custom_metadata: Additional metadata for the file
            
        Returns:
            bool: True if ingestion was successful
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        # Get file extension and processor
        file_ext = file_path.suffix.lower()
        if file_ext not in self.file_processors:
            print(f"❌ Unsupported file type: {file_ext}")
            print(f"   Supported types: {list(self.file_processors.keys())}")
            return False
        
        processor_config = self.file_processors[file_ext]
        processor_func = processor_config["processor"]
        default_domain = processor_config["domain"]
        file_type = processor_config["type"]
        
        # Use domain override if provided
        domain = domain_override or default_domain
        
        print(f"\n📄 Processing {file_type} file: {file_path.name}")
        print(f"🎯 Target domain: {domain}")
        
        try:
            # Process the file
            processed_content = processor_func(file_path)
            
            if not processed_content:
                print(f"❌ Failed to process file content")
                return False
            
            # Prepare metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_type,
                "file_extension": file_ext,
                "file_size": file_path.stat().st_size,
                "ingested_at": datetime.now().isoformat(),
                "source": "file_ingestion",
                "tags": [file_type, file_ext.lstrip('.'), domain]
            }
            
            # Add custom metadata
            if custom_metadata:
                metadata.update(custom_metadata)
            
            # Add file-specific metadata from processing
            if isinstance(processed_content, dict):
                content_text = processed_content.get("content", "")
                metadata.update(processed_content.get("metadata", {}))
            else:
                content_text = str(processed_content)
            
            # Ingest into the appropriate domain
            success = self.rag.add_knowledge(
                agent_id=f"{self.assistant_name}_{domain}",
                content=f"File: {file_path.name}\n\n{content_text}",
                source="file_ingestion",
                metadata=metadata
            )
            
            if success:
                print(f"✅ Successfully ingested {file_path.name} into {domain} domain")
                print(f"   Content length: {len(content_text)} characters")
                return True
            else:
                print(f"❌ Failed to add to knowledge base")
                return False
                
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return False

    def ingest_directory(self, directory_path: Union[str, Path], 
                        recursive: bool = True,
                        file_patterns: List[str] = None,
                        domain_mapping: Dict[str, str] = None) -> Dict[str, int]:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory_path: Path to directory to scan
            recursive: Whether to scan subdirectories
            file_patterns: List of glob patterns to match (e.g., ['*.py', '*.md'])
            domain_mapping: Map file extensions to specific domains
            
        Returns:
            Dict with ingestion statistics
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            print(f"❌ Directory not found: {directory_path}")
            return {"error": "Directory not found"}
        
        print(f"\n📂 Scanning directory: {directory_path}")
        print(f"   Recursive: {recursive}")
        
        # Find files to process
        files_to_process = []
        
        if file_patterns:
            # Use specific patterns
            for pattern in file_patterns:
                if recursive:
                    files_to_process.extend(directory_path.rglob(pattern))
                else:
                    files_to_process.extend(directory_path.glob(pattern))
        else:
            # Find all supported files
            for ext in self.file_processors.keys():
                pattern = f"*{ext}"
                if recursive:
                    files_to_process.extend(directory_path.rglob(pattern))
                else:
                    files_to_process.extend(directory_path.glob(pattern))
        
        # Remove duplicates and sort
        files_to_process = sorted(set(files_to_process))
        
        print(f"📋 Found {len(files_to_process)} files to process")
        
        # Process files
        stats = {
            "total_files": len(files_to_process),
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "by_domain": {}
        }
        
        for file_path in files_to_process:
            print(f"\n  Processing: {file_path.relative_to(directory_path)}")
            
            # Get domain from mapping if provided
            domain_override = None
            if domain_mapping:
                domain_override = domain_mapping.get(file_path.suffix.lower())
            
            success = self.ingest_file(file_path, domain_override=domain_override)
            
            if success:
                stats["successful"] += 1
                
                # Track by type
                file_type = self.file_processors[file_path.suffix.lower()]["type"]
                stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1
                
                # Track by domain
                domain = domain_override or self.file_processors[file_path.suffix.lower()]["domain"]
                stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
            else:
                stats["failed"] += 1
        
        print(f"\n📊 Directory ingestion complete:")
        print(f"   ✅ Successful: {stats['successful']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   📁 By domain: {stats['by_domain']}")
        print(f"   📄 By type: {stats['by_type']}")
        
        return stats

    def chat_with_files(self, question: str, 
                       domain: str = "general",
                       file_filter: Dict[str, Any] = None) -> str:
        """
        Chat with your ingested files using the appropriate domain agent.
        
        Args:
            question: Your question about the files
            domain: Domain to query (research, computer_science, general)
            file_filter: Filter files by metadata (e.g., {"file_type": "code"})
            
        Returns:
            str: Assistant's response
        """
        # Map domains to agent IDs
        domain_agents = {
            "research": "research_agent",
            "computer_science": "code_agent", 
            "general": "docs_agent"
        }
        
        agent_id = domain_agents.get(domain, "docs_agent")
        
        # Add file context to the question if filtering is requested
        enhanced_question = question
        if file_filter:
            filter_desc = ", ".join([f"{k}={v}" for k, v in file_filter.items()])
            enhanced_question = f"[Filter: {filter_desc}] {question}"
        
        # Query using the appropriate agent
        return self.rag.query_assistant(
            question=enhanced_question,
            domain=domain,
            include_cross_domain=True,
            use_long_term_memory=True
        )

    # File processors for different types
    
    def _process_latex(self, file_path: Path) -> Dict[str, Any]:
        """Process LaTeX files (research papers)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract key sections from LaTeX
            sections = self._extract_latex_sections(content)
            
            # Extract metadata
            metadata = {
                "latex_sections": list(sections.keys()),
                "has_equations": bool(re.search(r'\\begin\{equation\}|\\begin\{align\}|\\\[|\$\$', content)),
                "has_figures": bool(re.search(r'\\begin\{figure\}|\\includegraphics', content)),
                "has_tables": bool(re.search(r'\\begin\{table\}|\\begin\{tabular\}', content)),
                "word_count": len(content.split()),
                "packages": self._extract_latex_packages(content)
            }
            
            # Format content for ingestion
            formatted_content = self._format_latex_content(content, sections)
            
            return {
                "content": formatted_content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing LaTeX file: {e}")
            return None

    def _extract_latex_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from LaTeX content."""
        sections = {}
        
        # Common LaTeX section patterns
        section_patterns = [
            (r'\\title\{([^}]+)\}', 'title'),
            (r'\\author\{([^}]+)\}', 'author'),
            (r'\\abstract\{([^}]+)\}', 'abstract'),
            (r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 'abstract'),
            (r'\\section\{([^}]+)\}', 'section'),
            (r'\\subsection\{([^}]+)\}', 'subsection'),
        ]
        
        for pattern, section_type in section_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                if section_type in sections:
                    sections[section_type].extend(matches)
                else:
                    sections[section_type] = matches
        
        return sections

    def _extract_latex_packages(self, content: str) -> List[str]:
        """Extract LaTeX packages used."""
        package_pattern = r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}'
        packages = re.findall(package_pattern, content)
        return list(set(packages))

    def _format_latex_content(self, content: str, sections: Dict[str, str]) -> str:
        """Format LaTeX content for better readability."""
        formatted = f"LaTeX Document Analysis\n{'='*50}\n\n"
        
        # Add title if found
        if 'title' in sections:
            formatted += f"Title: {sections['title'][0]}\n\n"
        
        # Add author if found
        if 'author' in sections:
            formatted += f"Author(s): {sections['author'][0]}\n\n"
        
        # Add abstract if found
        if 'abstract' in sections:
            formatted += f"Abstract:\n{sections['abstract'][0]}\n\n"
        
        # Clean up LaTeX commands for readability
        cleaned_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', content)
        cleaned_content = re.sub(r'\\[a-zA-Z]+', '', cleaned_content)
        cleaned_content = re.sub(r'\{|\}', '', cleaned_content)
        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
        
        formatted += f"Content:\n{cleaned_content[:3000]}..."
        
        return formatted

    def _process_code(self, file_path: Path) -> Dict[str, Any]:
        """Process code files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract code metadata
            metadata = {
                "language": self._detect_language(file_path),
                "line_count": len(content.splitlines()),
                "char_count": len(content),
                "functions": self._extract_functions(content, file_path.suffix),
                "classes": self._extract_classes(content, file_path.suffix),
                "imports": self._extract_imports(content, file_path.suffix),
                "comments": self._extract_comments(content, file_path.suffix)
            }
            
            # Format content for ingestion
            formatted_content = self._format_code_content(content, file_path, metadata)
            
            return {
                "content": formatted_content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing code file: {e}")
            return None

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript', 
            '.cpp': 'C++',
            '.c': 'C',
            '.java': 'Java',
            '.rs': 'Rust',
            '.go': 'Go',
            '.r': 'R',
            '.sql': 'SQL',
            '.sh': 'Shell',
            '.yml': 'YAML',
            '.yaml': 'YAML'
        }
        return ext_to_lang.get(file_path.suffix.lower(), 'Unknown')

    def _extract_functions(self, content: str, file_ext: str) -> List[str]:
        """Extract function names from code."""
        functions = []
        
        if file_ext == '.py':
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        elif file_ext in ['.js', '.ts']:
            functions = re.findall(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            functions.extend(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function', content))
        elif file_ext in ['.java', '.cpp', '.c']:
            functions = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
        
        return list(set(functions))

    def _extract_classes(self, content: str, file_ext: str) -> List[str]:
        """Extract class names from code."""
        classes = []
        
        if file_ext == '.py':
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        elif file_ext in ['.java', '.cpp']:
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        elif file_ext in ['.js', '.ts']:
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        
        return list(set(classes))

    def _extract_imports(self, content: str, file_ext: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        
        if file_ext == '.py':
            imports = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content)
            imports.extend(re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content))
        elif file_ext in ['.js', '.ts']:
            imports = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content)
        elif file_ext == '.java':
            imports = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content)
        
        return list(set(imports))

    def _extract_comments(self, content: str, file_ext: str) -> int:
        """Count comment lines in code."""
        lines = content.splitlines()
        comment_count = 0
        
        for line in lines:
            line = line.strip()
            if file_ext == '.py' and line.startswith('#'):
                comment_count += 1
            elif file_ext in ['.js', '.ts', '.java', '.cpp', '.c'] and line.startswith('//'):
                comment_count += 1
            elif file_ext == '.sql' and line.startswith('--'):
                comment_count += 1
        
        return comment_count

    def _format_code_content(self, content: str, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Format code content for ingestion."""
        formatted = f"Code File Analysis: {file_path.name}\n{'='*50}\n\n"
        formatted += f"Language: {metadata['language']}\n"
        formatted += f"Lines: {metadata['line_count']}\n"
        formatted += f"Functions: {len(metadata['functions'])}\n"
        formatted += f"Classes: {len(metadata['classes'])}\n"
        formatted += f"Imports: {len(metadata['imports'])}\n\n"
        
        if metadata['functions']:
            formatted += f"Functions found: {', '.join(metadata['functions'][:10])}\n\n"
        
        if metadata['classes']:
            formatted += f"Classes found: {', '.join(metadata['classes'][:10])}\n\n"
        
        formatted += f"Code Content:\n{'-'*20}\n{content}"
        
        return formatted

    def _process_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Process Markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract markdown metadata
            metadata = {
                "headers": self._extract_markdown_headers(content),
                "links": self._extract_markdown_links(content),
                "code_blocks": self._extract_code_blocks(content),
                "word_count": len(content.split()),
                "line_count": len(content.splitlines())
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing Markdown file: {e}")
            return None

    def _extract_markdown_headers(self, content: str) -> List[str]:
        """Extract headers from Markdown content."""
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        return headers

    def _extract_markdown_links(self, content: str) -> List[str]:
        """Extract links from Markdown content."""
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        return [f"{text}: {url}" for text, url in links]

    def _extract_code_blocks(self, content: str) -> int:
        """Count code blocks in Markdown."""
        code_blocks = re.findall(r'```[^`]*```', content, re.DOTALL)
        return len(code_blocks)

    def _process_text(self, file_path: Path) -> str:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error processing text file: {e}")
            return None

    def _process_json(self, file_path: Path) -> Dict[str, Any]:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable format
            content = f"JSON Data from {file_path.name}:\n"
            content += json.dumps(data, indent=2, ensure_ascii=False)
            
            metadata = {
                "json_keys": list(data.keys()) if isinstance(data, dict) else [],
                "data_type": type(data).__name__,
                "size": len(str(data))
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing JSON file: {e}")
            return None

    def _process_csv(self, file_path: Path) -> Dict[str, Any]:
        """Process CSV files."""
        try:
            df = pd.read_csv(file_path)
            
            # Create summary
            content = f"CSV Data from {file_path.name}:\n"
            content += f"Shape: {df.shape}\n"
            content += f"Columns: {list(df.columns)}\n\n"
            content += "Sample data:\n"
            content += df.head(10).to_string()
            
            metadata = {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_count": len(df.columns),
                "data_types": df.dtypes.to_dict()
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return None

    def _process_excel(self, file_path: Path) -> Dict[str, Any]:
        """Process Excel files."""
        try:
            df = pd.read_excel(file_path)
            
            # Create summary (similar to CSV)
            content = f"Excel Data from {file_path.name}:\n"
            content += f"Shape: {df.shape}\n"
            content += f"Columns: {list(df.columns)}\n\n"
            content += "Sample data:\n"
            content += df.head(10).to_string()
            
            metadata = {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_count": len(df.columns),
                "data_types": df.dtypes.to_dict()
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            return None

    def _process_parquet(self, file_path: Path) -> Dict[str, Any]:
        """Process Parquet files."""
        try:
            df = pd.read_parquet(file_path)
            
            # Create summary (similar to CSV)
            content = f"Parquet Data from {file_path.name}:\n"
            content += f"Shape: {df.shape}\n"
            content += f"Columns: {list(df.columns)}\n\n"
            content += "Sample data:\n"
            content += df.head(10).to_string()
            
            metadata = {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_count": len(df.columns),
                "data_types": df.dtypes.to_dict()
            }
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error processing Parquet file: {e}")
            return None

    def search_ingested_files(self, search_term: str, 
                             file_type: str = None,
                             domain: str = None) -> pd.DataFrame:
        """
        Search through ingested files.
        
        Args:
            search_term: Term to search for
            file_type: Filter by file type (code, paper, documentation, data)
            domain: Filter by domain (research, computer_science, general)
            
        Returns:
            DataFrame with search results
        """
        # Implementation would depend on having the search functionality in RAG
        # For now, return empty DataFrame
        return pd.DataFrame()

    def get_file_summary(self) -> Dict[str, Any]:
        """Get summary of all ingested files."""
        summary = self.rag.get_summary()
        
        # Add file-specific information
        file_summary = {
            "total_files": 0,
            "by_type": {},
            "by_domain": {},
            "by_extension": {},
            "agents": summary.get("agents", {})
        }
        
        # This would be enhanced with actual file tracking
        return file_summary


def main():
    """Demonstrate the enhanced file ingestion assistant."""
    print("🚀 Enhanced Personal Assistant with File Ingestion")
    print("=" * 60)
    
    # Initialize the assistant
    assistant = FileIngestorRAG(
        assistant_name="my_file_assistant",
        knowledge_dir="./my_file_knowledge",
        model_name="llama3.2:3b"
    )
    
    print(f"\n📊 File Assistant Ready!")
    print(f"Supported file types: {list(assistant.file_processors.keys())}")
    
    # Example usage
    print(f"\n📝 Example Usage:")
    print(f"")
    print(f"# Ingest a single file")
    print(f"assistant.ingest_file('paper.tex', domain_override='research')")
    print(f"")
    print(f"# Ingest a directory")
    print(f"stats = assistant.ingest_directory('./my_papers', recursive=True)")
    print(f"")
    print(f"# Chat with your files")
    print(f"response = assistant.chat_with_files(")
    print(f"    'What are the main contributions of this paper?',")
    print(f"    domain='research'")
    print(f")")
    print(f"")
    print(f"# Search through files")
    print(f"results = assistant.search_ingested_files('neural networks')")
    
    # Demonstrate with sample files if they exist
    sample_files = [
        "README.md",
        "setup.py", 
        "requirements.txt"
    ]
    
    found_files = [f for f in sample_files if Path(f).exists()]
    
    if found_files:
        print(f"\n🧪 Demonstrating with found files:")
        for file_path in found_files[:2]:  # Process max 2 files for demo
            success = assistant.ingest_file(file_path)
            if success:
                print(f"✅ Processed: {file_path}")
                
                # Demo chat
                response = assistant.chat_with_files(
                    f"What can you tell me about {file_path}?",
                    domain="general"
                )
                print(f"💬 Response: {response[:200]}...")
    
    print(f"\n✅ Enhanced File Assistant demonstration complete!")
    print(f"📁 Knowledge stored in: {assistant.knowledge_dir}")
    print(f"\n💡 Your assistant is ready to ingest and chat with your files!")


if __name__ == "__main__":
    main()