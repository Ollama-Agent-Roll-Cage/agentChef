import click
import asyncio
import json
from pathlib import Path

from agentChef.core.chefs.ragchef import ResearchManager
from agentChef.cli.help_texts import RESEARCH_GROUP_HELP
from agentChef.utils.const import SUCCESS

@click.group()
def research():
    """Research operations for finding and analyzing content."""
    pass

@research.command()
@click.option("--topic", required=True, help="Research topic")
@click.option("--max-papers", default=5, help="Maximum number of papers")
@click.option("--include-github/--no-github", default=False, help="Include GitHub repositories")
@click.option("--output", type=Path, help="Output file for research results")
def topic(topic: str, max_papers: int, include_github: bool, output: Path):
    """Research a specific topic."""
    manager = ResearchManager()
    
    # Run async research in event loop
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        manager.research_topic(
            topic=topic,
            max_papers=max_papers,
            include_github=include_github,
            callback=lambda msg: click.echo(msg)
        )
    )
    
    # Save results if output specified
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Results saved to {output}")
    
    return SUCCESS

@research.command()
@click.option("--turns", default=3, help="Number of conversation turns")
@click.option("--expand", default=2, help="Expansion factor")
@click.option("--clean/--no-clean", default=True, help="Clean output")
@click.option("--output", type=Path, help="Output file")
def generate(turns: int, expand: int, clean: bool, output: Path):
    """Generate conversation dataset."""
    manager = ResearchManager()
    
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        manager.generate_conversation_dataset(
            num_turns=turns,
            expansion_factor=expand,
            clean=clean,
            callback=lambda msg: click.echo(msg)
        )
    )
    
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Dataset saved to {output}")
    
    return SUCCESS

@research.command()
@click.option("--query", required=True, help="Natural language query")
@click.option("--dataset", type=Path, help="Dataset file to query")
@click.option("--output", type=Path, help="Output file for query results")
def query(query: str, dataset: Path, output: Path):
    """Query a dataset using natural language."""
    from agentChef.core.llamaindex.pandas_query import PandasQueryIntegration
    
    query_engine = PandasQueryIntegration()
    
    # Load dataset
    import pandas as pd
    df = pd.read_parquet(dataset) if dataset.suffix == '.parquet' else pd.read_json(dataset)
    
    # Run query
    result = query_engine.query_dataframe(df, query)
    
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"Query results saved to {output}")
    else:
        click.echo(json.dumps(result, indent=2))
    
    return SUCCESS

@research.command()
@click.option("--text", required=True, help="Text to classify")
@click.option("--categories", "-c", multiple=True, help="Categories to check")
def classify(text: str, categories: List[str]):
    """Classify content using the Granite Guardian model."""
    from agentChef.core.classification.classification import Classifier
    
    classifier = Classifier()
    results = {}
    
    # Run async classification in event loop
    loop = asyncio.get_event_loop()
    for category in categories:
        result = loop.run_until_complete(
            classifier.classify_content(text, category)
        )
        results[category] = result
    
    click.echo(json.dumps(results, indent=2))
    return SUCCESS
