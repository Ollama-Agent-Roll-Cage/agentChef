from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import logging

from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.core.generation.conversation_generator import OllamaConversationGenerator
from agentChef.core.augmentation.dataset_expander import DatasetExpander
from agentChef.core.classification.dataset_cleaner import DatasetCleaner
from agentChef.core.classification.classification import Classifier
from agentChef.core.llamaindex.pandas_query import PandasQueryIntegration
from agentChef.core.ollama.ollama_interface import OllamaInterface
from agentChef.core.crawlers.crawlers_module import (
    WebCrawlerWrapper,
    ArxivSearcher,
    DuckDuckGoSearcher,
    GitHubCrawler,
    ParquetStorageWrapper
)

# Initialize FastAPI app
app = FastAPI(
    title="AgentChef API",
    description="Comprehensive API for AI research, dataset generation and management",
    version="1.0.0"
)

# Initialize shared components
ollama_interface = OllamaInterface(model_name="llama3")
research_manager = ResearchManager(model_name="llama3")
conversation_gen = OllamaConversationGenerator(ollama_interface=ollama_interface)
dataset_expander = DatasetExpander(ollama_interface=ollama_interface)
dataset_cleaner = DatasetCleaner(ollama_interface=ollama_interface)
classifier = Classifier()
pandas_query = PandasQueryIntegration()
storage = ParquetStorageWrapper()

# Request/Response Models
class ResearchRequest(BaseModel):
    topic: str
    max_papers: int = 5
    max_search_results: int = 10
    include_github: bool = False
    github_repos: Optional[List[str]] = None

class GenerateRequest(BaseModel):
    content: str
    num_turns: int = 3
    conversation_context: str = "general"
    hedging_level: str = "balanced"

class ExpandRequest(BaseModel):
    conversations: List[Dict[str, Any]]
    expansion_factor: int = 2
    static_fields: Dict[str, bool] = None
    reference_fields: List[str] = None

class CleanRequest(BaseModel):
    conversations: List[Dict[str, Any]]
    criteria: Dict[str, bool] = {
        "fix_hallucinations": True,
        "normalize_style": True,
        "correct_grammar": True,
        "ensure_coherence": True
    }

class ClassificationRequest(BaseModel):
    text: str
    categories: List[str] = ["harm", "bias", "quality"]

class QueryRequest(BaseModel):
    query: str
    data: List[Dict[str, Any]]

class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 1
    extract_code: bool = False

class StorageRequest(BaseModel):
    data: List[Dict[str, Any]]
    format: str = "parquet"
    filename: str = "output"

# API Routes

# Research Endpoints
@app.post("/research/topic")
async def research_topic(request: ResearchRequest):
    """Research a specific topic using multiple sources."""
    try:
        result = await research_manager.research_topic(**request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generation Endpoints
@app.post("/generate/conversation")
async def generate_conversation(request: GenerateRequest):
    """Generate a conversation from provided content."""
    try:
        result = await conversation_gen.generate_conversation(**request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dataset Expansion Endpoints
@app.post("/expand/dataset")
async def expand_dataset(request: ExpandRequest):
    """Expand a conversation dataset with variations."""
    try:
        result = await dataset_expander.expand_conversation_dataset(**request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dataset Cleaning Endpoints
@app.post("/clean/dataset")
async def clean_dataset(request: CleanRequest):
    """Clean and validate a conversation dataset."""
    try:
        result = await dataset_cleaner.clean_dataset(**request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Classification Endpoints
@app.post("/classify/content")
async def classify_content(request: ClassificationRequest):
    """Classify content across multiple categories."""
    try:
        results = {}
        for category in request.categories:
            is_flagged = await classifier.classify_content(request.text, category)
            results[category] = is_flagged
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query Endpoints
@app.post("/query/data")
async def query_data(request: QueryRequest):
    """Query data using natural language."""
    try:
        import pandas as pd
        df = pd.DataFrame(request.data)
        result = pandas_query.query_dataframe(df, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Crawler Endpoints
@app.post("/crawl/web")
async def crawl_web(request: CrawlRequest):
    """Crawl and extract content from a web page."""
    try:
        crawler = WebCrawlerWrapper()
        content = await crawler.fetch_url_content(request.url)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/arxiv/{paper_id}")
async def get_arxiv_paper(paper_id: str):
    """Fetch and process an ArXiv paper."""
    try:
        searcher = ArxivSearcher()
        paper = await searcher.fetch_paper_info(paper_id)
        return paper
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl/github")
async def analyze_github(repo_url: str):
    """Analyze a GitHub repository."""
    try:
        crawler = GitHubCrawler()
        summary = await crawler.get_repo_summary(repo_url)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Storage Endpoints
@app.post("/storage/save")
async def save_data(request: StorageRequest):
    """Save data in specified format."""
    try:
        if request.format == "parquet":
            path = storage.save_to_parquet(request.data, f"{request.filename}.parquet")
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        return {"path": str(path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility Endpoints
@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models = await ollama_interface.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}")
async def set_model(model_name: str):
    """Set the active Ollama model."""
    try:
        ollama_interface.set_model(model_name)
        conversation_gen.set_model(model_name)
        dataset_expander.set_model(model_name)
        dataset_cleaner.set_model(model_name)
        return {"message": f"Model set to {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))