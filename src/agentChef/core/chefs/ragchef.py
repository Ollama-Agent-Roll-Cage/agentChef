#!/usr/bin/env python3
"""
RAGChef - Research, Augmentation, & Generation Chef
==================================================

Complete research pipeline that combines:
- ArXiv, web, and GitHub research (via crawlers)
- Conversation generation and dataset creation
- Dataset expansion and cleaning
- Natural language data analysis (via PandasRAG)

This is an example of a complete custom chef that users can learn from.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Core AgentChef components
from .base_chef import BaseChef
from .pandas_rag import PandasRAG
from ..crawlers.crawlers_module import (
    WebCrawlerWrapper, ArxivSearcher, DuckDuckGoSearcher, GitHubCrawler
)
from ..generation.conversation_generator import OllamaConversationGenerator
from ..augmentation.dataset_expander import DatasetExpander
from ..classification.dataset_cleaner import DatasetCleaner
from ..ollama.ollama_interface import OllamaInterface

# Set up logging
logger = logging.getLogger(__name__)

class ResearchManager(BaseChef):
    """
    Complete research pipeline chef that demonstrates the full AgentChef capability.
    
    This chef shows how to combine:
    - Research capabilities (crawlers)
    - Conversation management (PandasRAG)
    - Dataset generation and cleaning
    - Natural language data analysis
    """
    
    def __init__(self, data_dir: str = "./ragchef_data", model_name: str = "llama3.2:3b"):
        """Initialize the RAGChef with all components."""
        super().__init__(
            name="ragchef",
            model_name=model_name,
            data_dir=data_dir,
            enable_ui=False
        )
        
        # Initialize PandasRAG for conversation and data analysis
        self.rag = PandasRAG(
            data_dir=f"{data_dir}/conversations",
            model_name=model_name,
            max_history_turns=10
        )
        
        # Register research assistant agent
        self.research_agent = self.rag.register_agent(
            "research_assistant",
            system_prompt="""You are a research assistant specializing in academic research, 
            dataset analysis, and scientific insights. You help analyze research papers, 
            generate insights from datasets, and provide comprehensive research summaries.""",
            description="Expert research assistant for academic and scientific analysis"
        )
        
        # Initialize research components
        self._setup_research_components()
        
        # Initialize generation and augmentation components
        self._setup_generation_components()
        
        # Research state
        self.research_state = {
            "topic": "",
            "papers": [],
            "search_results": [],
            "github_repos": [],
            "conversations": [],
            "expanded_conversations": [],
            "cleaned_conversations": []
        }
        
        logger.info(f"RAGChef initialized with data directory: {data_dir}")
    
    def _setup_research_components(self):
        """Initialize research crawlers."""
        try:
            self.web_crawler = WebCrawlerWrapper()
            self.arxiv_searcher = ArxivSearcher()
            self.ddg_searcher = DuckDuckGoSearcher()
            self.github_crawler = GitHubCrawler(data_dir=str(self.data_dir))
            logger.info("Research components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize research components: {e}")
            # Create dummy components to prevent None errors
            self.web_crawler = None
            self.arxiv_searcher = None
            self.ddg_searcher = None
            self.github_crawler = None
    
    def _setup_generation_components(self):
        """Initialize generation and processing components."""
        try:
            self.conversation_generator = OllamaConversationGenerator(
                model_name=self.model_name,
                ollama_interface=self.ollama
            )
            
            self.dataset_expander = DatasetExpander(
                ollama_interface=self.ollama,
                output_dir=str(self.data_dir / "expanded")
            )
            
            self.dataset_cleaner = DatasetCleaner(
                ollama_interface=self.ollama,
                output_dir=str(self.data_dir / "cleaned")
            )
            
            logger.info("Generation components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize generation components: {e}")
    
    async def research_topic(self, 
                           topic: str, 
                           max_papers: int = 5, 
                           max_search_results: int = 10,
                           include_github: bool = False,
                           github_repos: List[str] = None,
                           callback=None) -> Dict[str, Any]:
        """
        Research a topic comprehensively using multiple sources.
        
        Args:
            topic: Research topic
            max_papers: Maximum ArXiv papers to retrieve
            max_search_results: Maximum web search results
            include_github: Whether to search GitHub
            github_repos: Specific GitHub repos to analyze
            callback: Progress callback function
            
        Returns:
            Dictionary with comprehensive research results
        """
        def update_progress(message):
            logger.info(message)
            if callback:
                callback(message)
        
        self.research_state["topic"] = topic
        update_progress(f"Starting comprehensive research on: {topic}")
        
        # 1. ArXiv Research
        papers = []
        if self.arxiv_searcher:
            try:
                update_progress("Searching ArXiv for academic papers...")
                papers = await self.arxiv_searcher.search_papers(topic, max_results=max_papers)
                update_progress(f"Found {len(papers)} ArXiv papers")
            except Exception as e:
                logger.error(f"ArXiv search failed: {e}")
        
        # 2. Web Search
        search_results = []
        if self.ddg_searcher:
            try:
                update_progress("Performing web search...")
                search_results = await self.ddg_searcher.text_search(topic, max_results=max_search_results)
                update_progress(f"Found web search results")
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        # 3. GitHub Research (optional)
        github_results = []
        if include_github and self.github_crawler and github_repos:
            try:
                update_progress("Analyzing GitHub repositories...")
                for repo_url in github_repos:
                    summary = await self.github_crawler.get_repo_summary(repo_url)
                    github_results.append({"url": repo_url, "summary": summary})
                update_progress(f"Analyzed {len(github_results)} repositories")
            except Exception as e:
                logger.error(f"GitHub analysis failed: {e}")
        
        # Store research results
        self.research_state.update({
            "papers": papers,
            "search_results": search_results,
            "github_repos": github_results
        })
        
        # Add research to agent knowledge
        research_summary = self._generate_research_summary()
        self.rag.add_knowledge(
            agent_id=self.research_agent,
            content=research_summary,
            source=f"research:{topic}",
            metadata={
                "research_date": datetime.now().isoformat(),
                "papers_count": len(papers),
                "search_results_count": len(search_results) if isinstance(search_results, list) else 0,
                "github_repos_count": len(github_results)
            }
        )
        
        update_progress("Research completed successfully")
        return self.research_state.copy()
    
    async def generate_conversation_dataset(self,
                                          papers: List[Dict] = None,
                                          num_turns: int = 3,
                                          expansion_factor: int = 2,
                                          clean: bool = True,
                                          callback=None) -> Dict[str, Any]:
        """
        Generate conversation datasets from research papers with expansion and cleaning.
        
        Args:
            papers: Papers to process (uses research_state if None)
            num_turns: Conversation turns to generate
            expansion_factor: Dataset expansion factor
            clean: Whether to clean the dataset
            callback: Progress callback
            
        Returns:
            Dictionary with generated datasets
        """
        def update_progress(message):
            logger.info(message)
            if callback:
                callback(message)
        
        if papers is None:
            papers = self.research_state.get("papers", [])
        
        if not papers:
            return {"error": "No papers available for dataset generation"}
        
        update_progress(f"Generating conversations from {len(papers)} papers...")
        
        # Generate original conversations
        all_conversations = []
        for i, paper in enumerate(papers):
            update_progress(f"Processing paper {i+1}/{len(papers)}")
            
            # Extract paper content
            content = paper.get("abstract", "") or paper.get("content", "")
            if content:
                # Generate conversation
                conversation = self.conversation_generator.generate_conversation(
                    content=content,
                    num_turns=num_turns,
                    conversation_context="research paper analysis"
                )
                if conversation:
                    all_conversations.append(conversation)
        
        update_progress(f"Generated {len(all_conversations)} original conversations")
        
        # Expand conversations
        expanded_conversations = all_conversations
        if expansion_factor > 1:
            update_progress(f"Expanding dataset by factor of {expansion_factor}")
            expanded_conversations = self.dataset_expander.expand_conversation_dataset(
                conversations=all_conversations,
                expansion_factor=expansion_factor,
                static_fields={'human': True, 'gpt': False}
            )
            update_progress(f"Generated {len(expanded_conversations)} expanded conversations")
        
        # Clean conversations
        cleaned_conversations = expanded_conversations
        if clean and len(expanded_conversations) > 0:
            update_progress("Cleaning expanded conversations")
            cleaned_conversations = self.dataset_cleaner.clean_dataset(
                original_conversations=all_conversations,
                expanded_conversations=expanded_conversations,
                cleaning_criteria={
                    "fix_hallucinations": True,
                    "normalize_style": True,
                    "correct_grammar": True,
                    "ensure_coherence": True
                }
            )
            update_progress(f"Cleaned {len(cleaned_conversations)} conversations")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.dataset_expander.save_conversations_to_jsonl(
            cleaned_conversations,
            f"research_conversations_{timestamp}"
        )
        
        # Store in research state
        self.research_state.update({
            "conversations": all_conversations,
            "expanded_conversations": expanded_conversations,
            "cleaned_conversations": cleaned_conversations
        })
        
        update_progress(f"Dataset generation completed! Saved to: {output_path}")
        
        return {
            "original_count": len(all_conversations),
            "expanded_count": len(expanded_conversations),
            "cleaned_count": len(cleaned_conversations),
            "output_path": output_path,
            "conversations": all_conversations,
            "expanded_conversations": expanded_conversations,
            "cleaned_conversations": cleaned_conversations
        }
    
    async def analyze_research_with_rag(self, question: str) -> str:
        """
        Use PandasRAG to analyze research results with natural language.
        
        Args:
            question: Natural language question about the research
            
        Returns:
            Analysis response from the research agent
        """
        # Convert research data to DataFrame for analysis
        import pandas as pd
        
        research_data = []
        for paper in self.research_state.get("papers", []):
            research_data.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": ", ".join(paper.get("authors", [])),
                "categories": ", ".join(paper.get("categories", [])),
                "published": paper.get("published", ""),
                "url": paper.get("url", "")
            })
        
        if not research_data:
            return "No research data available for analysis. Please run research_topic() first."
        
        df = pd.DataFrame(research_data)
        
        # Query using PandasRAG with conversation history
        response = self.rag.query(
            dataframe=df,
            question=question,
            agent_id=self.research_agent,
            save_conversation=True
        )
        
        return response
    
    def _generate_research_summary(self) -> str:
        """Generate a comprehensive summary of research results."""
        papers = self.research_state.get("papers", [])
        search_results = self.research_state.get("search_results", [])
        github_repos = self.research_state.get("github_repos", [])
        topic = self.research_state.get("topic", "Unknown")
        
        summary = f"""# Research Summary: {topic}

## Overview
- Topic: {topic}
- ArXiv Papers: {len(papers)}
- Web Search Results: {len(search_results) if isinstance(search_results, list) else 1 if search_results else 0}
- GitHub Repositories: {len(github_repos)}
- Research Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Papers Found"""
        
        for i, paper in enumerate(papers[:5], 1):
            title = paper.get("title", "Unknown Title")
            authors = ", ".join(paper.get("authors", [])[:3])
            if len(paper.get("authors", [])) > 3:
                authors += " et al."
            summary += f"\n{i}. {title} - {authors}"
        
        if len(papers) > 5:
            summary += f"\n... and {len(papers) - 5} more papers"
        
        summary += f"\n\nThis research provides comprehensive coverage of {topic} from academic and web sources."
        
        return summary
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input through the research pipeline."""
        if isinstance(input_data, str):
            # Treat as research topic
            return await self.research_topic(input_data)
        elif isinstance(input_data, dict):
            # Treat as research parameters
            topic = input_data.get("topic")
            if topic:
                return await self.research_topic(
                    topic=topic,
                    max_papers=input_data.get("max_papers", 5),
                    max_search_results=input_data.get("max_search_results", 10),
                    include_github=input_data.get("include_github", False),
                    github_repos=input_data.get("github_repos")
                )
        
        return {"error": "Invalid input. Provide a topic string or research parameters dict."}
    
    async def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate conversation dataset from research."""
        return await self.generate_conversation_dataset(**kwargs)
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            # Clean up any temporary directories
            temp_dirs = [d for d in Path(self.data_dir).glob("tmp*") if d.is_dir()]
            for temp_dir in temp_dirs:
                import shutil
                shutil.rmtree(temp_dir)
            logger.info("RAGChef cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage and testing
async def main():
    """Example usage of RAGChef."""
    chef = ResearchManager(data_dir="./example_ragchef_data")
    
    # Research a topic
    print("🔬 Researching transformer models...")
    research_results = await chef.research_topic(
        topic="transformer neural networks",
        max_papers=2,
        max_search_results=3
    )
    
    print(f"Found {len(research_results['papers'])} papers")
    
    # Analyze research with natural language
    print("\n💬 Analyzing research with RAG...")
    analysis = await chef.analyze_research_with_rag(
        "What are the main themes across these transformer papers?"
    )
    print(f"Analysis: {analysis}")
    
    # Generate dataset
    print("\n📊 Generating conversation dataset...")
    dataset_results = await chef.generate_conversation_dataset(
        num_turns=2,
        expansion_factor=2,
        clean=True
    )
    
    print(f"Generated {dataset_results['cleaned_count']} cleaned conversations")
    
    # Cleanup
    chef.cleanup()

if __name__ == "__main__":
    asyncio.run(main())