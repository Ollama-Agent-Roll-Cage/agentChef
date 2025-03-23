import sys
import os
import re
import json
import time
import argparse
import requests
import shutil
import tempfile
import gzip
import tarfile
import asyncio
import threading
import markdown
import concurrent.futures
from pathlib import Path
from datetime import datetime
from functools import partial

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
                            QComboBox, QSpinBox, QCheckBox, QFileDialog, QSplitter,
                            QMessageBox, QDialog, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl
from PyQt6.QtGui import QIcon, QFont, QDesktopServices, QTextCursor, QColor, QPalette
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Ollama
import ollama

# For ArXiv API
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# For DuckDuckGo search
from duckduckgo_search import DDGS

# Default paths
DEFAULT_DATA_DIR = os.path.join(Path.home(), '.research_assistant')
os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEFAULT_DATA_DIR, 'research_assistant.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ResearchAssistant")

class IterativeResearchThread(QThread):
    """Thread for managing the iterative research process."""
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)  # value, description
    result_signal = pyqtSignal(str)  # final markdown result
    error_signal = pyqtSignal(str)
    search_result_signal = pyqtSignal(list)  # search results
    paper_signal = pyqtSignal(dict)  # paper info
    
    def __init__(self, research_topic, model_name, max_iterations=10, time_limit_minutes=30):
        super().__init__()
        self.research_topic = research_topic
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_minutes * 60
        
        self.research_state = {
            "topic": research_topic,
            "search_results": [],
            "papers": [],
            "iterations": [],
            "final_report": ""
        }
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="research_assistant_"))
        self.stop_requested = False
        
    def run(self):
        """Main research workflow."""
        try:
            start_time = time.time()
            iteration_count = 0
            self.update_signal.emit(f"Starting research on: {self.research_topic}")
            
            # Initial research planning
            self.progress_signal.emit(5, "Planning research approach")
            research_plan = self.generate_research_plan()
            self.research_state["plan"] = research_plan
            self.update_signal.emit(f"Research Plan:\n{research_plan}\n")
            
            # Web search for relevant information
            self.progress_signal.emit(10, "Searching the web for relevant information")
            web_results = self.perform_web_search()
            self.research_state["search_results"] = web_results
            self.search_result_signal.emit(web_results)
            
            # Identify relevant papers on ArXiv
            self.progress_signal.emit(20, "Identifying relevant papers on ArXiv")
            arxiv_queries = self.generate_arxiv_queries()
            arxiv_papers = self.search_arxiv_papers(arxiv_queries)
            self.research_state["arxiv_papers"] = arxiv_papers
            
            # Download and process top papers
            self.progress_signal.emit(30, "Downloading and processing top papers")
            processed_papers = self.process_papers(arxiv_papers[:3])  # Process top 3 papers
            self.research_state["papers"] = processed_papers
            
            # Begin iterative research process
            while iteration_count < self.max_iterations and not self.stop_requested:
                # Check time limit
                elapsed_time = time.time() - start_time
                if elapsed_time > self.time_limit_seconds:
                    self.update_signal.emit(f"Time limit reached after {iteration_count} iterations.")
                    break
                
                # Update progress
                progress_value = 30 + (iteration_count / self.max_iterations) * 60
                self.progress_signal.emit(int(progress_value), f"Research iteration {iteration_count+1}/{self.max_iterations}")
                
                iteration_result = self.perform_research_iteration(iteration_count)
                self.research_state["iterations"].append(iteration_result)
                
                if iteration_result.get("status") == "complete":
                    self.update_signal.emit(f"Research completed after {iteration_count+1} iterations!")
                    break
                
                iteration_count += 1
            
            # Generate final report
            self.progress_signal.emit(90, "Generating final research report")
            final_report = self.generate_final_report()
            self.research_state["final_report"] = final_report
            
            self.progress_signal.emit(100, "Research complete!")
            self.result_signal.emit(final_report)
            
            # Cleanup temporary files
            self.cleanup()
            
        except Exception as e:
            logger.exception("Error in research process")
            self.error_signal.emit(f"Research error: {str(e)}")
    
    def stop(self):
        """Request the thread to stop."""
        self.stop_requested = True
    
    def generate_research_plan(self):
        """Generate a research plan using the LLM."""
        prompt = f"""
        I need to conduct in-depth research on the topic: "{self.research_topic}".
        
        Create a detailed research plan that includes:
        1. Key questions to investigate
        2. Important subtopics to explore
        3. Types of sources that would be valuable
        4. Specific search terms to use for finding relevant papers on ArXiv
        
        Format your response as a structured research plan with clear sections.
        """
        
        response = ollama.chat(model=self.model_name, messages=[
            {"role": "system", "content": "You are a research assistant that designs effective research plans."},
            {"role": "user", "content": prompt}
        ])
        
        return response['message']['content']
    
    def perform_web_search(self):
        """Perform web search using DuckDuckGo."""
        try:
            search_prompt = f"""
            Based on the research topic "{self.research_topic}",
            provide exactly 3 search queries that would yield the most relevant and 
            specific information. Format them as a numbered list only.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "system", "content": "You are a search expert. Respond with only the requested information."},
                {"role": "user", "content": search_prompt}
            ])
            
            # Extract queries from the response
            search_text = response['message']['content']
            queries = re.findall(r'\d+\.\s*(.*)', search_text)
            
            if not queries:
                queries = [self.research_topic]
            
            # Perform the searches
            all_results = []
            for query in queries[:3]:  # Limit to 3 queries
                self.update_signal.emit(f"Searching web for: {query}")
                try:
                    # Search for text
                    text_results = DDGS().text(
                        keywords=query,
                        region="wt-wt",
                        safesearch="off",
                        max_results=5
                    )
                    all_results.extend(text_results or [])
                    
                    # Search for news
                    news_results = DDGS().news(
                        keywords=query,
                        region="wt-wt",
                        safesearch="off",
                        max_results=3
                    )
                    all_results.extend(news_results or [])
                    
                except Exception as e:
                    self.update_signal.emit(f"Search error for query '{query}': {str(e)}")
            
            # Remove duplicates by URL
            unique_results = []
            seen_urls = set()
            for result in all_results:
                url = result.get('href') or result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
            
            self.update_signal.emit(f"Found {len(unique_results)} unique web results")
            return unique_results
            
        except Exception as e:
            self.update_signal.emit(f"Web search error: {str(e)}")
            return []
    
    def generate_arxiv_queries(self):
        """Generate specific queries for ArXiv based on the research topic."""
        prompt = f"""
        Based on the research topic "{self.research_topic}", generate 3 specific 
        search queries optimized for the ArXiv academic search engine.
        
        For each query:
        1. Focus on academic/scientific terminology relevant to this field
        2. Include key concepts and methodologies
        3. Use proper Boolean operators if helpful (AND, OR)
        
        Return only the 3 search queries as a numbered list, with no additional text.
        """
        
        response = ollama.chat(model=self.model_name, messages=[
            {"role": "system", "content": "You are a scientific research assistant specializing in academic literature searches."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract queries from the response
        queries_text = response['message']['content']
        queries = re.findall(r'\d+\.\s*(.*)', queries_text)
        
        if not queries:
            queries = [self.research_topic]
            
        self.update_signal.emit(f"Generated ArXiv queries: {', '.join(queries)}")
        return queries
    
    def search_arxiv_papers(self, queries):
        """Search for papers on ArXiv using the provided queries."""
        all_papers = []
        
        for query in queries:
            self.update_signal.emit(f"Searching ArXiv for: {query}")
            try:
                # URL encode the query
                encoded_query = urllib.parse.quote(query)
                base_url = 'http://export.arxiv.org/api/query'
                search_url = f"{base_url}?search_query=all:{encoded_query}&start=0&max_results=5"
                
                with urllib.request.urlopen(search_url) as response:
                    response_data = response.read().decode('utf-8')
                
                # Parse the XML response
                root = ET.fromstring(response_data)
                
                # Define namespace mapping
                namespaces = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                # Extract papers
                entries = root.findall('.//atom:entry', namespaces)
                
                for entry in entries:
                    paper_id_element = entry.find('.//arxiv:id', namespaces)
                    paper_id = paper_id_element.text.split('/')[-1] if paper_id_element is not None else None
                    
                    title = entry.find('atom:title', namespaces).text.strip().replace('\n', ' ')
                    authors = [author.find('atom:name', namespaces).text for author in entry.findall('atom:author', namespaces)]
                    abstract = entry.find('atom:summary', namespaces).text.strip().replace('\n', ' ')
                    published = entry.find('atom:published', namespaces).text
                    
                    paper_info = {
                        'arxiv_id': paper_id,
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'published': published,
                        'query': query
                    }
                    
                    all_papers.append(paper_info)
                    
            except Exception as e:
                self.update_signal.emit(f"ArXiv search error for query '{query}': {str(e)}")
        
        # Remove duplicates by paper ID
        unique_papers = []
        seen_ids = set()
        for paper in all_papers:
            if paper['arxiv_id'] not in seen_ids:
                seen_ids.add(paper['arxiv_id'])
                unique_papers.append(paper)
        
        self.update_signal.emit(f"Found {len(unique_papers)} unique ArXiv papers")
        
        # Rank papers by relevance
        ranked_papers = self.rank_papers_by_relevance(unique_papers)
        return ranked_papers
    
    def rank_papers_by_relevance(self, papers):
        """Rank papers by relevance to the research topic."""
        if not papers:
            return []
            
        try:
            # Create a prompt to rank papers
            papers_info = "\n\n".join([
                f"Paper {i+1}:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
                for i, paper in enumerate(papers)
            ])
            
            prompt = f"""
            Based on the research topic "{self.research_topic}", rank the following papers by relevance.
            
            {papers_info}
            
            Provide the papers in ranked order by listing only their numbers (e.g., "3, 1, 2").
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "system", "content": "You are a research assistant that ranks academic papers by relevance."},
                {"role": "user", "content": prompt}
            ])
            
            # Extract ranking from response
            ranking_text = response['message']['content']
            
            # Try to find a clear list of numbers
            ranking_match = re.search(r'(\d+(?:,\s*\d+)*)', ranking_text)
            if ranking_match:
                # Extract the ranked order
                paper_order = [int(num.strip()) for num in ranking_match.group(1).split(',')]
                
                # Check if the ranks are valid
                if max(paper_order) <= len(papers) and min(paper_order) >= 1:
                    # Reorder papers based on ranking (adjusting for 0-based indexing)
                    ranked_papers = []
                    for rank in paper_order:
                        if 1 <= rank <= len(papers):
                            ranked_papers.append(papers[rank-1])
                    
                    # Add any papers not explicitly ranked
                    ranked_indices = [i-1 for i in paper_order]
                    for i, paper in enumerate(papers):
                        if i not in ranked_indices:
                            ranked_papers.append(paper)
                            
                    return ranked_papers
            
            # If ranking failed, return the original list
            self.update_signal.emit("Could not parse paper ranking, using original order")
            return papers
            
        except Exception as e:
            self.update_signal.emit(f"Error ranking papers: {str(e)}")
            return papers
    
    def process_papers(self, papers):
        """Download and process ArXiv papers."""
        processed_papers = []
        
        for paper in papers:
            try:
                self.update_signal.emit(f"Processing paper: {paper['title']}")
                
                # Download paper source files
                paper_dir = self.download_paper_source(paper['arxiv_id'])
                
                if paper_dir:
                    # Extract LaTeX content
                    latex_content = self.extract_latex_content(paper_dir)
                    
                    # Update paper info
                    paper['source_dir'] = str(paper_dir)
                    paper['latex_content'] = latex_content
                    
                    # Generate a summary
                    paper['summary'] = self.summarize_paper(paper)
                    
                    processed_papers.append(paper)
                    self.paper_signal.emit(paper)
                    
            except Exception as e:
                self.update_signal.emit(f"Error processing paper {paper['arxiv_id']}: {str(e)}")
        
        self.update_signal.emit(f"Processed {len(processed_papers)} papers")
        return processed_papers
    
    def download_paper_source(self, arxiv_id):
        """Download source files for a paper from ArXiv."""
        try:
            # Create directory for paper
            paper_dir = self.temp_dir / arxiv_id
            paper_dir.mkdir(exist_ok=True)
            
            # Download source file
            source_url = f"https://arxiv.org/e-print/{arxiv_id}"
            temp_file = paper_dir / "source.tar.gz"
            
            with urllib.request.urlopen(source_url) as response:
                with open(temp_file, 'wb') as f:
                    f.write(response.read())
            
            # Try to extract as tar.gz
            try:
                with tarfile.open(temp_file, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                    self.update_signal.emit(f"Extracted tar.gz source for {arxiv_id}")
            except tarfile.ReadError:
                # If not tar.gz, try as gzip
                try:
                    with gzip.open(temp_file, 'rb') as gz:
                        with open(paper_dir / 'main.tex', 'wb') as f:
                            f.write(gz.read())
                    self.update_signal.emit(f"Extracted gzip source for {arxiv_id}")
                except Exception:
                    self.update_signal.emit(f"Source for {arxiv_id} is not in standard format")
            
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
                
            return paper_dir
            
        except Exception as e:
            self.update_signal.emit(f"Error downloading source for {arxiv_id}: {str(e)}")
            return None
    
    def extract_latex_content(self, paper_dir):
        """Extract and concatenate LaTeX content from source files."""
        try:
            latex_content = []
            
            # Find all .tex files
            tex_files = list(paper_dir.glob('**/*.tex'))
            
            if not tex_files:
                self.update_signal.emit(f"No .tex files found in {paper_dir}")
                return ""
            
            # Try to identify the main .tex file
            main_candidates = [f for f in tex_files if f.name.lower() in ('main.tex', 'paper.tex', 'manuscript.tex')]
            
            if main_candidates:
                main_file = main_candidates[0]
            else:
                # Find the largest .tex file or just use the first one
                main_file = max(tex_files, key=lambda f: f.stat().st_size)
            
            # Read the main file
            with open(main_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                latex_content.append(f"% Main file: {main_file.name}\n{content}")
            
            # Read other important files like abstract, intro, etc.
            important_sections = ['abstract', 'intro', 'introduction', 'method', 'approach', 
                                'result', 'conclusion', 'discussion']
            
            for tex_file in tex_files:
                if tex_file != main_file:
                    if any(section in tex_file.name.lower() for section in important_sections):
                        try:
                            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                latex_content.append(f"% From file: {tex_file.name}\n{content}")
                        except Exception as e:
                            self.update_signal.emit(f"Error reading {tex_file}: {str(e)}")
            
            return "\n\n".join(latex_content)
            
        except Exception as e:
            self.update_signal.emit(f"Error extracting LaTeX content: {str(e)}")
            return ""
    
    def summarize_paper(self, paper):
        """Generate a summary of the paper using the LLM."""
        try:
            # Create a context from title, authors, abstract and sample of LaTeX content
            latex_sample = paper.get('latex_content', '')[:5000] + "..." if len(paper.get('latex_content', '')) > 5000 else paper.get('latex_content', '')
            
            context = f"""
            Title: {paper['title']}
            Authors: {', '.join(paper['authors'])}
            Abstract: {paper['abstract']}
            
            LaTeX Content Sample:
            {latex_sample}
            """
            
            prompt = f"""
            Please provide a comprehensive summary of this research paper. Include:
            
            1. The main research question or problem addressed
            2. Key methodologies or approaches used
            3. Main findings or results
            4. Significant conclusions and implications
            5. How this relates to the research topic: "{self.research_topic}"
            
            Format your summary with clear sections and bullet points where appropriate.
            """
            
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "system", "content": "You are a research assistant specializing in creating concise but comprehensive summaries of academic papers."},
                {"role": "user", "content": context + "\n\n" + prompt}
            ])
            
            return response['message']['content']
            
        except Exception as e:
            self.update_signal.emit(f"Error summarizing paper: {str(e)}")
            return "Failed to generate summary due to an error."
    
    def perform_research_iteration(self, iteration_number):
        """Perform a single research iteration, building on previous work."""
        iteration_result = {
            "iteration": iteration_number,
            "timestamp": datetime.now().isoformat(),
            "thoughts": "",
            "findings": "",
            "next_steps": "",
            "status": "in_progress"
        }
        
        try:
            # Construct the prompt based on current research state
            research_summary = self.get_research_summary()
            
            prompt = f"""
            # Research Iteration {iteration_number + 1}
            
            ## Current Research Topic
            {self.research_topic}
            
            ## Research Progress Summary
            {research_summary}
            
            ## Your Task For This Iteration
            Based on the research so far, please:
            
            1. Analyze the information collected
            2. Identify key insights and connections
            3. Determine what's missing or needs further investigation
            4. Develop the next section of the research report
            
            Structure your response with these headings:
            
            ### Thought Process
            (Your analysis of the current state and reasoning)
            
            ### Key Findings
            (New insights and connections you've identified)
            
            ### Next Steps
            (What should be investigated in the next iteration, if needed)
            
            ### Report Section
            (A new section to add to the final report, formatted in Markdown)
            
            ### Status Assessment
            (Either "CONTINUE" if more research is needed, or "COMPLETE" if you have sufficient information for a comprehensive report)
            """
            
            self.update_signal.emit(f"Starting research iteration {iteration_number + 1}")
            
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "system", "content": "You are an advanced research assistant working on developing a comprehensive research report on an important topic."},
                {"role": "user", "content": prompt}
            ])
            
            # Parse the structured response
            content = response['message']['content']
            
            # Extract thought process
            thought_match = re.search(r'#+\s*Thought Process\s*(.*?)(?=#+\s*Key Findings|$)', content, re.DOTALL)
            iteration_result["thoughts"] = thought_match.group(1).strip() if thought_match else ""
            
            # Extract key findings
            findings_match = re.search(r'#+\s*Key Findings\s*(.*?)(?=#+\s*Next Steps|$)', content, re.DOTALL)
            iteration_result["findings"] = findings_match.group(1).strip() if findings_match else ""
            
            # Extract next steps
            next_steps_match = re.search(r'#+\s*Next Steps\s*(.*?)(?=#+\s*Report Section|$)', content, re.DOTALL)
            iteration_result["next_steps"] = next_steps_match.group(1).strip() if next_steps_match else ""
            
            # Extract report section
            report_match = re.search(r'#+\s*Report Section\s*(.*?)(?=#+\s*Status Assessment|$)', content, re.DOTALL)
            iteration_result["report_section"] = report_match.group(1).strip() if report_match else ""
            
            # Extract status assessment
            status_match = re.search(r'#+\s*Status Assessment\s*(.*?)(?=$)', content, re.DOTALL)
            status_text = status_match.group(1).strip() if status_match else ""
            
            # Determine if research is complete
            if "COMPLETE" in status_text.upper():
                iteration_result["status"] = "complete"
                self.update_signal.emit("Research process has determined it has sufficient information.")
            else:
                iteration_result["status"] = "in_progress"
            
            # Add formatted summary to update signal
            update_text = f"""
            --- Iteration {iteration_number + 1} Summary ---
            
            Thought Process: {iteration_result['thoughts'][:100]}...
            
            Key Findings: {iteration_result['findings'][:100]}...
            
            Status: {iteration_result['status'].upper()}
            """
            
            self.update_signal.emit(update_text)
            
            return iteration_result
            
        except Exception as e:
            self.update_signal.emit(f"Error in research iteration: {str(e)}")
            iteration_result["error"] = str(e)
            return iteration_result
    
    def get_research_summary(self):
        """Generate a summary of the research progress so far."""
        # Summarize web search results
        web_summary = f"Found {len(self.research_state.get('search_results', []))} web sources."
        
        # Summarize papers
        papers = self.research_state.get('papers', [])
        paper_summary = f"Analyzed {len(papers)} academic papers:"
        for i, paper in enumerate(papers, 1):
            paper_summary += f"\n{i}. {paper['title']} - {paper.get('summary', '')[:150]}..."
        
        # Summarize previous iterations
        iterations = self.research_state.get('iterations', [])
        iteration_summary = ""
        if iterations:
            for i, iteration in enumerate(iterations, 1):
                key_points = iteration.get('findings', '')[:200] + "..."
                iteration_summary += f"\nIteration {i} key findings: {key_points}"
        
        # Combine all summaries
        full_summary = f"""
        {web_summary}
        
        {paper_summary}
        
        Previous research progress: {iteration_summary}
        """
        
        return full_summary
    
    def generate_final_report(self):
        try:
            # First, let's compile research sections
            report_sections = []
            for iteration in self.research_state.get('iterations', []):
                if 'report_section' in iteration and iteration['report_section']:
                    report_sections.append(iteration['report_section'])
                
            compiled_sections = "\n\n".join(report_sections)

            # Get all sources
            web_sources = []
            for i, result in enumerate(self.research_state.get('search_results', []), 1):
                source = {
                    "id": f"web{i}",
                    "type": "web",
                    "title": result.get('title', ''),
                    "url": result.get('link') or result.get('url', ''),
                    "publisher": result.get('source', ''),
                    "date": result.get('date', datetime.now().strftime('%Y-%m-%d'))
                }
                web_sources.append(source)

            paper_sources = []
            for i, paper in enumerate(self.research_state.get('papers', []), 1):
                source = {
                    "id": f"paper{i}",
                    "type": "paper",
                    "title": paper.get('title', ''),
                    "authors": paper.get('authors', []),
                    "url": f"https://arxiv.org/abs/{paper.get('arxiv_id', '')}",
                    "arxiv_id": paper.get('arxiv_id', ''),
                    "date": paper.get('published', '').split('T')[0] if paper.get('published') else ''
                }
                paper_sources.append(source)

            # Combine all sources
            all_sources = web_sources + paper_sources
            sources_json = json.dumps(all_sources, indent=2)

            # Create citation instructions with actual sources
            citation_examples = "\n".join([
                f"- [{source['id']}] refers to: {source['title']}"
                for source in all_sources[:3]
            ])

            prompt = f"""
            # Research Report Generation Task

            ## Research Topic
            {self.research_topic}

            ## Available Sources
            {sources_json}

            ## Citation Instructions
            Use these actual sources for citations. Examples:
            {citation_examples}

            ## Content to Include
            {compiled_sections}

            ## Output Format Requirements
            1. Create a well-structured research report
            2. Use the actual sources provided for citations in [id] format
            3. Every major claim should have a citation
            4. Include a proper bibliography listing ALL sources used
            5. Format in Markdown with proper headings
            6. Include a table of contents

            Only use the sources provided. Do not invent or make up citations.
            """

            self.update_signal.emit("Generating final report with citations...")
            
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "system", "content": "You are a research report writer. Use ONLY the provided sources for citations."},
                {"role": "user", "content": prompt}
            ])

            final_report = response['message']['content']

            # Verify citations
            used_citations = set(re.findall(r'\[(.*?)\]', final_report))
            valid_citations = set(source['id'] for source in all_sources)
            
            if not used_citations.issubset(valid_citations):
                self.update_signal.emit("Warning: Found invalid citations, regenerating report...")
                # Could add recursive retry here

            return final_report

        except Exception as e:
            self.update_signal.emit(f"Error generating final report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def generate_bibliography(self, sources):
        """Generate a bibliography section from the sources list."""
        bibliography = "# Bibliography\n\n"
        
        # Sort sources by type and then id
        web_sources = [s for s in sources if s['type'] == 'web']
        paper_sources = [s for s in sources if s['type'] == 'paper']
        
        # Add papers first
        if paper_sources:
            for source in paper_sources:
                authors_text = ", ".join(source.get('authors', ['Unknown Author']))
                title = source.get('title', 'Untitled Paper')
                date = source.get('date', '')
                arxiv_id = source.get('arxiv_id', '')
                url = source.get('url', '')
                
                entry = f"[{source['id']}] {authors_text}. \"{title}\". "
                if date:
                    entry += f"({date}). "
                entry += f"arXiv:{arxiv_id}. Available at: {url}\n\n"
                bibliography += entry
        
        # Then add web sources
        if web_sources:
            for source in web_sources:
                title = source.get('title', 'Untitled Web Page')
                publisher = source.get('publisher', 'Unknown Publisher')
                date = source.get('date', '')
                url = source.get('url', '#')
                
                entry = f"[{source['id']}] \"{title}\". {publisher}. "
                if date:
                    entry += f"({date}). "
                entry += f"Available at: {url}\n\n"
                bibliography += entry
        
        return bibliography
        
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")

class FloatingResearchWindow(QMainWindow):
    """Main window for the research assistant."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Research Assistant")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set initial size
        self.resize(400, 600)
        
        # For window dragging
        self.dragging = False
        self.offset = None
        
        # Initialize UI
        self.setup_ui()
        
        # Research thread
        self.research_thread = None
        
        # Available models from Ollama
        self.available_models = []
        self.fetch_available_models()
        
        # Settings
        self.max_iterations = 10
        self.time_limit_minutes = 30
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main container widget with background
        self.container = QWidget()
        self.container.setObjectName("container")
        self.container.setStyleSheet("""
            #container {
                background-color: rgba(30, 30, 40, 0.95);
                border-radius: 15px;
                border: 1px solid #4444ff;
            }
        """)
        
        self.setCentralWidget(self.container)
        
        # Main layout
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header with title and close button
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🧠 Research Assistant")
        title_label.setStyleSheet("""
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
        """)
        
        close_button = QPushButton("×")
        close_button.setFixedSize(30, 30)
        close_button.setStyleSheet("""
            QPushButton {
                color: #ffffff;
                background-color: transparent;
                border: none;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff4444;
            }
        """)
        close_button.clicked.connect(self.close)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(close_button)
        
        layout.addLayout(header_layout)
        
        # Research prompt input
        prompt_label = QLabel("Research Topic:")
        prompt_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(prompt_label)
        
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Describe the research topic or question...")
        self.prompt_input.setMinimumHeight(80)
        self.prompt_input.setMaximumHeight(120)
        self.prompt_input.setStyleSheet("""
            QTextEdit {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.prompt_input)
        
        # Model selection
        model_layout = QHBoxLayout()
        
        model_label = QLabel("Ollama Model:")
        model_label.setStyleSheet("color: #ffffff;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 4px 8px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
            }
        """)
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)
        
        # Research parameters
        params_layout = QHBoxLayout()
        
        iteration_label = QLabel("Max Iterations:")
        iteration_label.setStyleSheet("color: #ffffff;")
        params_layout.addWidget(iteration_label)
        
        self.iteration_spin = QSpinBox()
        self.iteration_spin.setRange(1, 50)
        self.iteration_spin.setValue(10)
        self.iteration_spin.setStyleSheet("""
            QSpinBox {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 4px 8px;
            }
        """)
        self.iteration_spin.valueChanged.connect(self.update_settings)
        params_layout.addWidget(self.iteration_spin)
        
        time_label = QLabel("Time Limit (mins):")
        time_label.setStyleSheet("color: #ffffff;")
        params_layout.addWidget(time_label)
        
        self.time_spin = QSpinBox()
        self.time_spin.setRange(1, 120)
        self.time_spin.setValue(30)
        self.time_spin.setStyleSheet("""
            QSpinBox {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 4px 8px;
            }
        """)
        self.time_spin.valueChanged.connect(self.update_settings)
        params_layout.addWidget(self.time_spin)
        
        layout.addLayout(params_layout)
        
        # Start research button
        self.research_button = QPushButton("🔍 Start Research")
        self.research_button.setStyleSheet("""
            QPushButton {
                background-color: #4444ff;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5555ff;
            }
            QPushButton:pressed {
                background-color: #3333cc;
            }
        """)
        self.research_button.clicked.connect(self.start_research)
        layout.addWidget(self.research_button)
        
        # Progress bar
        self.progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(50, 50, 60, 0.8);
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4444ff;
                border-radius: 7px;
            }
        """)
        
        self.cancel_button = QPushButton("✕")
        self.cancel_button.setFixedSize(25, 25)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: #ffffff;
                border: none;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_research)
        self.cancel_button.setVisible(False)
        
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_layout.addWidget(self.cancel_button)
        
        layout.addLayout(self.progress_layout)
        self.progress_bar.setVisible(False)
        
        # Console output
        console_label = QLabel("Research Progress:")
        console_label.setStyleSheet("color: #ffffff;")
        layout.addWidget(console_label)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMinimumHeight(200)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: rgba(20, 20, 30, 0.9);
                color: #33ff33;
                border: 1px solid #555555;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas, Monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.console_output)
        
        # View Results button
        self.view_results_button = QPushButton("📋 View Results")
        self.view_results_button.setStyleSheet("""
            QPushButton {
                background-color: #44aa44;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #55bb55;
            }
            QPushButton:pressed {
                background-color: #338833;
            }
        """)
        self.view_results_button.clicked.connect(self.view_results)
        self.view_results_button.setVisible(False)
        layout.addWidget(self.view_results_button)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.offset = event.position().toPoint()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.offset:
            new_pos = event.globalPosition().toPoint() - self.offset
            self.move(new_pos)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            
    def fetch_available_models(self):
        """Fetch available models from Ollama using the Python API."""
        self.log_message("Fetching available Ollama models...")
        self.model_combo.clear()
        
        try:
            # Use the Python API instead of subprocess
            models_list = ollama.list()
            
            if hasattr(models_list, 'models') and models_list.models:
                model_names = [model.model for model in models_list.models]
                
                if model_names:
                    self.model_combo.addItems(model_names)
                    self.available_models = model_names
                    self.log_message(f"Found {len(model_names)} Ollama models")
                else:
                    self.model_combo.addItem("No models found")
                    self.log_message("No Ollama models found")
            else:
                self.model_combo.addItem("No models found")
                self.log_message("No models found in response")
                    
        except Exception as e:
            self.log_message(f"Error fetching models: {str(e)}")
            self.model_combo.addItem("Error fetching models")
    
    def update_settings(self):
        """Update research settings from UI."""
        self.max_iterations = self.iteration_spin.value()
        self.time_limit_minutes = self.time_spin.value()
    
    def log_message(self, message):
        """Add a message to the console output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Set cursor to end of document
        cursor = self.console_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_output.setTextCursor(cursor)
        
        # Add the new message
        self.console_output.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_research(self):
        """Start the research process."""
        # Get research prompt
        research_prompt = self.prompt_input.toPlainText().strip()
        if not research_prompt:
            QMessageBox.warning(self, "Input Required", "Please enter a research topic.")
            return
        
        # Get selected model
        selected_model = self.model_combo.currentText()
        if selected_model in ["No models found", "Error fetching models"]:
            QMessageBox.warning(self, "Model Required", "Please select a valid Ollama model.")
            return
        
        # Update UI
        self.research_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        self.view_results_button.setVisible(False)
        self.console_output.clear()
        
        self.log_message(f"Starting research on: {research_prompt}")
        self.log_message(f"Using model: {selected_model}")
        self.log_message(f"Max iterations: {self.max_iterations}")
        self.log_message(f"Time limit: {self.time_limit_minutes} minutes")
        
        # Create and start research thread
        self.research_thread = IterativeResearchThread(
            research_topic=research_prompt,
            model_name=selected_model,
            max_iterations=self.max_iterations,
            time_limit_minutes=self.time_limit_minutes
        )
        
        # Connect signals
        self.research_thread.update_signal.connect(self.log_message)
        self.research_thread.progress_signal.connect(self.update_progress)
        self.research_thread.result_signal.connect(self.handle_research_result)
        self.research_thread.error_signal.connect(self.handle_research_error)
        self.research_thread.search_result_signal.connect(self.handle_search_results)
        self.research_thread.paper_signal.connect(self.handle_paper_info)
        
        # Start the thread
        self.research_thread.start()
    
    def cancel_research(self):
        """Cancel the current research process."""
        if self.research_thread and self.research_thread.isRunning():
            self.log_message("Cancelling research...")
            self.research_thread.stop()
            
            # Wait for thread to finish
            if not self.research_thread.wait(3000):  # 3 second timeout
                self.research_thread.terminate()
                self.log_message("Research thread terminated.")
            
            self.reset_ui_after_research()
    
    def update_progress(self, value, message):
        """Update the progress bar."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")
    
    def handle_research_result(self, result):
        """Handle the final research result."""
        self.log_message("Research complete!")
        self.final_report = result
        self.reset_ui_after_research()
        self.view_results_button.setVisible(True)
    
    def handle_research_error(self, error_message):
        """Handle research errors."""
        self.log_message(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Research Error", error_message)
        self.reset_ui_after_research()
    
    def handle_search_results(self, results):
        """Handle search results update."""
        self.log_message(f"Found {len(results)} search results")
    
    def handle_paper_info(self, paper):
        """Handle paper information update."""
        self.log_message(f"Processed paper: {paper['title']}")
    
    def reset_ui_after_research(self):
        """Reset the UI after research completes or is cancelled."""
        self.research_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
    
    def view_results(self):
        """Show the research results in a separate window."""
        if hasattr(self, 'final_report'):
            result_dialog = ResearchResultDialog(self.final_report, parent=self)
            result_dialog.exec()
        else:
            QMessageBox.warning(self, "No Results", "No research results available.")

class ResearchResultDialog(QDialog):
    """Dialog for displaying research results."""
    def __init__(self, markdown_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Research Report")
        self.resize(800, 600)
        
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)
        
        # Add CSS for styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 20px;
                    padding: 0;
                    color: #333;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #2c3e50;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                h1 {{ font-size: 28px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ font-size: 24px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                p {{ margin: 15px 0; }}
                a {{ color: #3498db; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                code {{ 
                    background: #f8f8f8; 
                    padding: 3px 5px; 
                    border-radius: 3px; 
                    font-family: Consolas, monospace;
                }}
                pre {{ 
                    background: #f8f8f8; 
                    padding: 15px; 
                    border-radius: 5px; 
                    overflow-x: auto;
                    border: 1px solid #ddd;
                }}
                blockquote {{
                    margin: 15px 0;
                    padding: 10px 15px;
                    border-left: 4px solid #3498db;
                    background: #f8f8f8;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                ul, ol {{
                    margin: 15px 0;
                    padding-left: 30px;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create splitter for preview and raw markdown
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Create tabs for different views
        tab_widget = QTabWidget()
        
        # HTML Preview tab
        html_view = QWebEngineView()
        html_view.setHtml(styled_html)
        tab_widget.addTab(html_view, "Rendered Preview")
        
        # Raw markdown tab
        markdown_edit = QTextEdit()
        markdown_edit.setPlainText(markdown_content)
        markdown_edit.setReadOnly(False)  # Allow editing
        tab_widget.addTab(markdown_edit, "Markdown Source")
        
        splitter.addWidget(tab_widget)
        layout.addWidget(splitter)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Report")
        save_button.clicked.connect(lambda: self.save_report(markdown_content))
        button_layout.addWidget(save_button)
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: self.copy_to_clipboard(markdown_content))
        button_layout.addWidget(copy_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def save_report(self, content):
        """Save the report to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Research Report", "", "Markdown Files (*.md);;HTML Files (*.html);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.html'):
                # Convert to HTML and save
                html_content = markdown.markdown(content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            else:
                # If not .md extension, add it
                if not file_path.endswith('.md'):
                    file_path += '.md'
                
                # Save as markdown
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            QMessageBox.information(self, "Success", f"Report saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
    
    def copy_to_clipboard(self, content):
        """Copy the report to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(content)
        QMessageBox.information(self, "Success", "Report copied to clipboard")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Research Assistant")
    parser.add_argument("--topic", type=str, help="Research topic to analyze")
    parser.add_argument("--model", type=str, default="llama3:latest", help="Ollama model to use")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--time-limit", type=int, default=30, help="Time limit in minutes")
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Research Assistant")
    
    # Set global stylesheet
    app.setStyleSheet("""
        QLabel, QPushButton, QComboBox, QSpinBox, QTextEdit {
            font-family: Arial;
        }
    """)
    
    # Create main window
    window = FloatingResearchWindow()
    window.show()
    
    # If topic provided via command line, start research automatically
    if args.topic:
        window.prompt_input.setPlainText(args.topic)
        
        # Set model if valid
        if args.model:
            index = window.model_combo.findText(args.model)
            if index >= 0:
                window.model_combo.setCurrentIndex(index)
        
        # Set iterations and time limit
        window.iteration_spin.setValue(args.iterations)
        window.time_spin.setValue(args.time_limit)
        
        # Wait a bit for UI to initialize, then start research
        QTimer.singleShot(1000, window.start_research)
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()