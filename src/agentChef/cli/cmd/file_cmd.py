"""
File management commands for AgentChef File Ingestion system.
Provides CLI interface for storing files, managing knowledge, and chatting with ingested content.
"""

import click
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Import the FileIngestorRAG class
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentChef.examples.personal_assistant_rag_example import FileIngestorRAG

@click.group()
def files():
    """File ingestion and management operations for AgentChef."""
    pass

@files.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--assistant-name', '-n', default='file_assistant', 
              help='Name of the assistant instance')
@click.option('--knowledge-dir', '-d', default='./agentchef_files',
              help='Directory to store knowledge base')
@click.option('--model', '-m', default='llama3.2:3b',
              help='Ollama model to use')
@click.option('--domain', help='Override domain for this file (research, computer_science, general)')
@click.option('--tags', help='Comma-separated tags for the file')
@click.option('--copy-file/--no-copy-file', default=True,
              help='Copy file to knowledge directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def store(file_path: str, assistant_name: str, knowledge_dir: str, model: str, 
          domain: Optional[str], tags: Optional[str], copy_file: bool, verbose: bool):
    """Store a single file in the AgentChef knowledge base.
    
    Examples:
        agentchef files store paper.pdf --domain research
        agentchef files store code.py --tags "ml,pytorch" 
        agentchef files store README.md --assistant-name my_assistant
    """
    try:
        # Initialize assistant
        if verbose:
            click.echo(f"🤖 Initializing assistant '{assistant_name}'...")
            click.echo(f"📁 Knowledge directory: {knowledge_dir}")
            click.echo(f"🧠 Model: {model}")
        
        assistant = FileIngestorRAG(
            assistant_name=assistant_name,
            knowledge_dir=knowledge_dir,
            model_name=model
        )
        
        # Prepare metadata
        custom_metadata = {}
        if tags:
            custom_metadata['user_tags'] = [tag.strip() for tag in tags.split(',')]
        
        # Copy file to knowledge directory if requested
        source_path = Path(file_path)
        target_path = source_path
        
        if copy_file:
            knowledge_path = Path(knowledge_dir)
            files_dir = knowledge_path / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = files_dir / source_path.name
            
            # Handle duplicate filenames
            counter = 1
            while target_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                target_path = files_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file
            import shutil
            shutil.copy2(source_path, target_path)
            custom_metadata['original_path'] = str(source_path)
            custom_metadata['stored_path'] = str(target_path)
            
            if verbose:
                click.echo(f"📋 Copied file to: {target_path}")
        
        # Ingest the file
        success = assistant.ingest_file(
            target_path, 
            domain_override=domain,
            custom_metadata=custom_metadata
        )
        
        if success:
            click.echo(f"✅ Successfully stored {source_path.name}")
            if verbose:
                click.echo(f"   Domain: {domain or 'auto-detected'}")
                click.echo(f"   Tags: {tags or 'none'}")
                click.echo(f"   Storage path: {target_path}")
        else:
            click.echo(f"❌ Failed to store {source_path.name}")
            return 1
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1

@files.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--assistant-name', '-n', default='file_assistant',
              help='Name of the assistant instance')
@click.option('--knowledge-dir', '-d', default='./agentchef_files',
              help='Directory to store knowledge base')
@click.option('--model', '-m', default='llama3.2:3b',
              help='Ollama model to use')
@click.option('--recursive/--no-recursive', '-r', default=True,
              help='Scan subdirectories recursively')
@click.option('--patterns', help='Comma-separated file patterns (e.g., "*.py,*.md")')
@click.option('--domain-mapping', help='JSON string mapping extensions to domains')
@click.option('--copy-files/--no-copy-files', default=True,
              help='Copy files to knowledge directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without actually doing it')
def bulk_store(directory_path: str, assistant_name: str, knowledge_dir: str, model: str,
               recursive: bool, patterns: Optional[str], domain_mapping: Optional[str],
               copy_files: bool, verbose: bool, dry_run: bool):
    """Store multiple files from a directory in the AgentChef knowledge base.
    
    Examples:
        agentchef files bulk-store ./papers --patterns "*.pdf,*.tex"
        agentchef files bulk-store ./code --recursive --domain-mapping '{"py": "computer_science"}'
        agentchef files bulk-store ./docs --dry-run --verbose
    """
    try:
        # Parse patterns
        file_patterns = None
        if patterns:
            file_patterns = [p.strip() for p in patterns.split(',')]
        
        # Parse domain mapping
        domain_map = None
        if domain_mapping:
            try:
                domain_map = json.loads(domain_mapping)
            except json.JSONDecodeError:
                click.echo(f"❌ Invalid JSON in domain-mapping: {domain_mapping}")
                return 1
        
        if dry_run:
            click.echo(f"🧪 DRY RUN - No files will be actually processed")
        
        # Initialize assistant
        if verbose:
            click.echo(f"🤖 Initializing assistant '{assistant_name}'...")
            click.echo(f"📁 Knowledge directory: {knowledge_dir}")
            click.echo(f"🧠 Model: {model}")
        
        assistant = FileIngestorRAG(
            assistant_name=assistant_name,
            knowledge_dir=knowledge_dir,
            model_name=model
        )
        
        # Scan directory first to show what will be processed
        directory = Path(directory_path)
        files_to_process = []
        
        if file_patterns:
            for pattern in file_patterns:
                if recursive:
                    files_to_process.extend(directory.rglob(pattern))
                else:
                    files_to_process.extend(directory.glob(pattern))
        else:
            # Find all supported files
            for ext in assistant.file_processors.keys():
                pattern = f"*{ext}"
                if recursive:
                    files_to_process.extend(directory.rglob(pattern))
                else:
                    files_to_process.extend(directory.glob(pattern))
        
        files_to_process = sorted(set(files_to_process))
        
        click.echo(f"📂 Scanning: {directory}")
        click.echo(f"🔍 Found {len(files_to_process)} files to process")
        
        if verbose or dry_run:
            click.echo("\nFiles to process:")
            for file_path in files_to_process:
                relative_path = file_path.relative_to(directory)
                file_ext = file_path.suffix.lower()
                domain = domain_map.get(file_ext.lstrip('.'), 'auto') if domain_map else 'auto'
                click.echo(f"  📄 {relative_path} (domain: {domain})")
        
        if dry_run:
            click.echo(f"\n✅ Dry run complete. {len(files_to_process)} files would be processed.")
            return 0
        
        # Process files
        if not files_to_process:
            click.echo("ℹ️ No files found to process")
            return 0
        
        if not click.confirm(f"Process {len(files_to_process)} files?"):
            click.echo("Cancelled.")
            return 0
        
        # Setup file copying if requested
        files_dir = None
        if copy_files:
            knowledge_path = Path(knowledge_dir)
            files_dir = knowledge_path / "files"
            files_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files with progress
        successful = 0
        failed = 0
        
        with click.progressbar(files_to_process, label='Processing files') as files:
            for file_path in files:
                try:
                    # Determine domain
                    domain_override = None
                    if domain_map:
                        file_ext = file_path.suffix.lower().lstrip('.')
                        domain_override = domain_map.get(file_ext)
                    
                    # Handle file copying
                    target_path = file_path
                    custom_metadata = {}
                    
                    if copy_files and files_dir:
                        target_path = files_dir / file_path.name
                        
                        # Handle duplicates
                        counter = 1
                        while target_path.exists():
                            stem = file_path.stem
                            suffix = file_path.suffix
                            target_path = files_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                        
                        # Copy file
                        import shutil
                        shutil.copy2(file_path, target_path)
                        custom_metadata['original_path'] = str(file_path)
                        custom_metadata['stored_path'] = str(target_path)
                    
                    # Ingest file
                    success = assistant.ingest_file(
                        target_path,
                        domain_override=domain_override,
                        custom_metadata=custom_metadata
                    )
                    
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        if verbose:
                            click.echo(f"\n❌ Failed: {file_path.name}")
                
                except Exception as e:
                    failed += 1
                    if verbose:
                        click.echo(f"\n❌ Error processing {file_path.name}: {e}")
        
        # Summary
        click.echo(f"\n📊 Bulk storage complete:")
        click.echo(f"   ✅ Successful: {successful}")
        click.echo(f"   ❌ Failed: {failed}")
        click.echo(f"   📁 Knowledge stored in: {knowledge_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1

@files.command()
@click.option('--assistant-name', '-n', default='file_assistant',
              help='Name of the assistant instance')
@click.option('--knowledge-dir', '-d', default='./agentchef_files',
              help='Directory with knowledge base')
@click.option('--model', '-m', default='llama3.2:3b',
              help='Ollama model to use')
@click.option('--domain', type=click.Choice(['research', 'computer_science', 'general']),
              default='general', help='Domain to chat with')
@click.option('--interactive/--single', '-i', default=False,
              help='Start interactive chat session')
@click.option('--question', '-q', help='Single question to ask')
@click.option('--history/--no-history', default=True,
              help='Include conversation history')
def chat(assistant_name: str, knowledge_dir: str, model: str, domain: str,
         interactive: bool, question: Optional[str], history: bool):
    """Chat with your ingested files using natural language.
    
    Examples:
        agentchef files chat -q "What papers did I ingest about transformers?"
        agentchef files chat --domain research --interactive
        agentchef files chat -q "Show me the main functions in my Python files" --domain computer_science
    """
    try:
        # Check if knowledge directory exists
        if not Path(knowledge_dir).exists():
            click.echo(f"❌ Knowledge directory not found: {knowledge_dir}")
            click.echo("   Use 'agentchef files store' or 'agentchef files bulk-store' first")
            return 1
        
        # Initialize assistant
        click.echo(f"🤖 Loading assistant '{assistant_name}'...")
        click.echo(f"📁 Knowledge directory: {knowledge_dir}")
        click.echo(f"🧠 Model: {model}")
        click.echo(f"🎯 Domain: {domain}")
        
        assistant = FileIngestorRAG(
            assistant_name=assistant_name,
            knowledge_dir=knowledge_dir,
            model_name=model
        )
        
        if interactive:
            # Interactive chat mode
            click.echo(f"\n💬 Starting interactive chat with your files")
            click.echo(f"   Type 'exit', 'quit', or 'bye' to end the session")
            click.echo(f"   Type '/domain <name>' to switch domains")
            click.echo(f"   Type '/help' for more commands")
            click.echo(f"\n{'-'*60}")
            
            current_domain = domain
            
            while True:
                try:
                    user_input = click.prompt(f"\n[{current_domain}] You", type=str)
                    
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        click.echo("👋 Goodbye!")
                        break
                    
                    if user_input.startswith('/domain '):
                        new_domain = user_input[8:].strip()
                        if new_domain in ['research', 'computer_science', 'general']:
                            current_domain = new_domain
                            click.echo(f"🎯 Switched to domain: {current_domain}")
                        else:
                            click.echo(f"❌ Invalid domain. Use: research, computer_science, general")
                        continue
                    
                    if user_input == '/help':
                        click.echo(f"""
💡 Available commands:
   /domain <name>     - Switch domain (research, computer_science, general)
   /help             - Show this help
   exit/quit/bye     - End chat session

💬 Just type your questions naturally, like:
   - "What papers did I ingest about neural networks?"
   - "Show me the Python functions in my code files"
   - "Summarize the main findings from my research papers"
                        """)
                        continue
                    
                    # Get response
                    click.echo(f"🤔 Thinking...")
                    response = assistant.chat_with_files(
                        user_input,
                        domain=current_domain
                    )
                    
                    click.echo(f"\n🤖 Assistant: {response}")
                    
                except KeyboardInterrupt:
                    click.echo(f"\n👋 Chat interrupted. Goodbye!")
                    break
                except Exception as e:
                    click.echo(f"\n❌ Error: {e}")
        
        elif question:
            # Single question mode
            click.echo(f"\n💭 Question: {question}")
            click.echo(f"🤔 Thinking...")
            
            response = assistant.chat_with_files(question, domain=domain)
            
            click.echo(f"\n🤖 Assistant: {response}")
        
        else:
            click.echo(f"❌ Either provide --question or use --interactive mode")
            return 1
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1

@files.command()
@click.option('--assistant-name', '-n', default='file_assistant',
              help='Name of the assistant instance')
@click.option('--knowledge-dir', '-d', default='./agentchef_files',
              help='Directory with knowledge base')
@click.option('--format', type=click.Choice(['table', 'json', 'summary']),
              default='summary', help='Output format')
def list(assistant_name: str, knowledge_dir: str, format: str):
    """List all files in the knowledge base.
    
    Examples:
        agentchef files list
        agentchef files list --format table
        agentchef files list --format json
    """
    try:
        knowledge_path = Path(knowledge_dir)
        
        if not knowledge_path.exists():
            click.echo(f"❌ Knowledge directory not found: {knowledge_dir}")
            return 1
        
        # Look for stored files
        files_info = []
        
        # Check for copied files
        files_dir = knowledge_path / "files"
        if files_dir.exists():
            for file_path in files_dir.iterdir():
                if file_path.is_file():
                    files_info.append({
                        'name': file_path.name,
                        'path': str(file_path),
                        'size': file_path.stat().st_size,
                        'extension': file_path.suffix,
                        'modified': file_path.stat().st_mtime
                    })
        
        if not files_info:
            click.echo(f"ℹ️ No files found in knowledge base: {knowledge_dir}")
            return 0
        
        if format == 'json':
            click.echo(json.dumps(files_info, indent=2))
        elif format == 'table':
            df = pd.DataFrame(files_info)
            click.echo(df.to_string(index=False))
        else:  # summary
            click.echo(f"📁 Knowledge Base: {knowledge_dir}")
            click.echo(f"📊 Total files: {len(files_info)}")
            
            # Group by extension
            by_ext = {}
            for info in files_info:
                ext = info['extension']
                by_ext[ext] = by_ext.get(ext, 0) + 1
            
            click.echo(f"📄 By file type:")
            for ext, count in sorted(by_ext.items()):
                click.echo(f"   {ext or 'no extension'}: {count}")
            
            # Total size
            total_size = sum(info['size'] for info in files_info)
            size_mb = total_size / (1024 * 1024)
            click.echo(f"💾 Total size: {size_mb:.2f} MB")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1

@files.command()
@click.argument('search_term')
@click.option('--assistant-name', '-n', default='file_assistant',
              help='Name of the assistant instance')
@click.option('--knowledge-dir', '-d', default='./agentchef_files',
              help='Directory with knowledge base')
@click.option('--domain', type=click.Choice(['research', 'computer_science', 'general']),
              help='Filter by domain')
@click.option('--file-type', help='Filter by file type (code, paper, documentation, data)')
@click.option('--limit', '-l', default=10, help='Maximum results to show')
def search(search_term: str, assistant_name: str, knowledge_dir: str,
           domain: Optional[str], file_type: Optional[str], limit: int):
    """Search through ingested files for specific content.
    
    Examples:
        agentchef files search "neural networks"
        agentchef files search "transformer" --domain research
        agentchef files search "function" --file-type code --limit 5
    """
    try:
        if not Path(knowledge_dir).exists():
            click.echo(f"❌ Knowledge directory not found: {knowledge_dir}")
            return 1
        
        # Initialize assistant for search
        assistant = FileIngestorRAG(
            assistant_name=assistant_name,
            knowledge_dir=knowledge_dir,
            model_name="llama3.2:3b"  # Use fast model for search
        )
        
        click.echo(f"🔍 Searching for: '{search_term}'")
        if domain:
            click.echo(f"🎯 Domain filter: {domain}")
        if file_type:
            click.echo(f"📄 File type filter: {file_type}")
        
        # Use assistant to search (this would need to be implemented in the FileIngestorRAG class)
        # For now, we'll use a simple question format
        search_question = f"Find all content related to '{search_term}'"
        if file_type:
            search_question += f" in {file_type} files"
        if domain:
            search_question += f" in the {domain} domain"
        
        results = assistant.chat_with_files(
            search_question,
            domain=domain or 'general'
        )
        
        click.echo(f"\n🔍 Search Results:")
        click.echo(f"{'-'*60}")
        click.echo(results)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1

@files.command()
@click.option('--knowledge-dir', '-d', default='./agentchef_files',
              help='Directory with knowledge base')
@click.option('--backup-dir', '-b', help='Directory to backup to (default: knowledge_dir_backup)')
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
def clean(knowledge_dir: str, backup_dir: Optional[str], force: bool):
    """Clean up the knowledge base (with backup option).
    
    Examples:
        agentchef files clean --backup-dir ./backup
        agentchef files clean --force
    """
    try:
        knowledge_path = Path(knowledge_dir)
        
        if not knowledge_path.exists():
            click.echo(f"ℹ️ Knowledge directory not found: {knowledge_dir}")
            return 0
        
        # Backup if requested
        if backup_dir:
            backup_path = Path(backup_dir)
            click.echo(f"💾 Creating backup in: {backup_path}")
            
            import shutil
            if backup_path.exists():
                if not click.confirm(f"Backup directory exists. Overwrite?"):
                    click.echo("Cancelled.")
                    return 0
                shutil.rmtree(backup_path)
            
            shutil.copytree(knowledge_path, backup_path)
            click.echo(f"✅ Backup created")
        
        # Confirm cleanup
        if not force:
            if not click.confirm(f"Delete all knowledge in {knowledge_dir}?"):
                click.echo("Cancelled.")
                return 0
        
        # Clean up
        import shutil
        shutil.rmtree(knowledge_path)
        click.echo(f"🗑️ Knowledge base cleaned: {knowledge_dir}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    files()