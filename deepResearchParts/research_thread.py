import time
import json
import re
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import logging
import tempfile

# PyQt6 imports
from PyQt6.QtCore import QThread, pyqtSignal

# Ollama
import ollama

# Import research utilities
from research_utils import perform_web_search, search_arxiv_papers, download_paper_source, extract_latex_content

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
            arxiv_papers = search_arxiv_papers(arxiv_queries, self.update_signal.emit)
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
            
            # Call the web search function from research_utils
            web_results = perform_web_search(queries, self.update_signal.emit)
            return web_results
            
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
                paper_dir = download_paper_source(paper['arxiv_id'], self.temp_dir, self.update_signal.emit)
                
                if paper_dir:
                    # Extract LaTeX content
                    latex_content = extract_latex_content(paper_dir, self.update_signal.emit)
                    
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
                # Make sure we have a valid URL and title
                url = result.get('href') or result.get('url', '')
                if not url.startswith('http'):
                    self.update_signal.emit(f"Skipping invalid URL: {url}")
                    continue  # Skip sources without valid URLs
                    
                title = result.get('title', '')
                if not title:
                    title = "Untitled Web Page"
                    
                source = {
                    "id": f"web{i}",
                    "type": "web",
                    "title": title,
                    "url": url,
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

            # Ensure unique citation IDs
            seen_ids = set()
            for source in all_sources:
                original_id = source['id']
                counter = 1
                while source['id'] in seen_ids:
                    source['id'] = f"{original_id}_{counter}"
                    counter += 1
                seen_ids.add(source['id'])

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

            invalid_citations = used_citations - valid_citations
            if invalid_citations:
                self.update_signal.emit(f"Warning: Found invalid citations: {', '.join(invalid_citations)}")
                
                # Attempt to fix common issues in the report - replace made-up citations with valid ones
                if valid_citations:
                    for invalid_cite in invalid_citations:
                        if invalid_cite.startswith('web') or invalid_cite.startswith('paper'):
                            # Find a valid ID of the same type to substitute
                            valid_replacements = [v_id for v_id in valid_citations 
                                                if v_id.startswith(invalid_cite[0:3])]
                            if valid_replacements:
                                replacement = valid_replacements[0]
                                final_report = final_report.replace(f"[{invalid_cite}]", f"[{replacement}]")
                                self.update_signal.emit(f"Replaced invalid citation [{invalid_cite}] with [{replacement}]")

            # Debug: Log the final report content
            self.update_signal.emit(f"Final report content:\n{final_report}")

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
