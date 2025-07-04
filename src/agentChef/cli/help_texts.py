"""
Help text constants for the AgentChef CLI.

This module centralizes all help and usage text for CLI commands and options,
ensuring consistent and maintainable documentation across the toolkit.
"""

# Command group descriptions
RESEARCH_GROUP_HELP = "Research and dataset generation operations using AgentChef."
FILES_GROUP_HELP = "File ingestion and management operations for AgentChef knowledge bases."
BUILD_GROUP_HELP = "Build operations for AgentChef package management."
DATA_GROUP_HELP = "Data management operations for viewing and manipulating AgentChef data files."
CONFIG_GROUP_HELP = "Manage configuration settings for AgentChef."
MCP_GROUP_HELP = "Model Context Protocol (MCP) server operations for AgentChef."

# Command option descriptions
ARGS_HELP = "Show this help message and exit."
ARGS_VERBOSE_HELP = "Enable verbose output and debug logging"
ARGS_CONFIG_HELP = "Path to custom configuration file"
ARGS_TOPIC_HELP = "Research topic to investigate"
ARGS_MAX_PAPERS_HELP = "Maximum number of papers to research"
ARGS_MAX_RESULTS_HELP = "Maximum number of results to return"
ARGS_MODEL_HELP = "Ollama model to use (e.g., llama3.2:3b)"
ARGS_DATA_DIR_HELP = "Directory to store AgentChef data"
ARGS_OUTPUT_PATH_HELP = "Directory to save the output"
ARGS_OUTPUT_FILE_HELP = "File to save the output"
ARGS_AGENT_NAME_HELP = "Name of the agent to use"
ARGS_KNOWLEDGE_DIR_HELP = "Directory for agent knowledge storage"
ARGS_DOMAIN_HELP = "Domain for the agent (research, computer_science, general)"
ARGS_QUESTION_HELP = "Question to ask the agent"
ARGS_INTERACTIVE_HELP = "Start interactive chat session"
ARGS_FILE_PATH_HELP = "Path to the file to process"
ARGS_PATTERNS_HELP = "File patterns to match (e.g., '*.py,*.md')"
ARGS_RECURSIVE_HELP = "Process directories recursively"
ARGS_CLEAN_HELP = "Clean build directories first"
ARGS_PORT_HELP = "Port to run the server on"
ARGS_FORMAT_HELP = "Output format (json, table, summary)"

# --- Main CLI Help ---
MAIN_HELP = """AgentChef - AI Agent Development Framework

A comprehensive toolkit for building, training, and deploying AI agents with natural language capabilities.

Create specialized agents that can analyze data, conduct research, and maintain context across conversations - all powered by local LLMs.

Core features:
• PandasRAG - Conversational data analysis with agent memory
• ResearchManager - Automated research and dataset generation  
• FileIngestor - Chat with documents, code, and data files
• Custom Chefs - Build specialized agents for any domain

Examples:
    # Store files for chatting
    agentchef files store paper.pdf --domain research
    agentchef files bulk-store ./code --patterns "*.py"
    
    # Chat with your files
    agentchef files chat -q "What are the main findings?"
    agentchef files chat --interactive --domain research
    
    # Research workflows
    agentchef research topic "transformer models" --max-papers 5
    agentchef research generate input.txt --turns 3

Get started: https://github.com/yourusername/agentChef
"""

# Research command help
RESEARCH_HELP = f"""
Research and dataset generation operations using AgentChef.

USAGE:
  agentchef research COMMAND [OPTIONS]

Commands:
  topic                 Research a specific topic using multiple sources
  generate              Generate conversation datasets from content
  query                 Query datasets using natural language
  classify              Classify content using AI models

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  agentchef research topic "transformer models" --max-papers 5
  agentchef research generate input.txt --turns 3 --expand 2
  agentchef research query "Find conversations about AI" --dataset data.parquet
"""

# Files command help  
FILES_HELP = f"""
File ingestion and management operations for AgentChef knowledge bases.

USAGE:
  agentchef files COMMAND [OPTIONS]

Commands:
  store                 Store a single file in knowledge base
  bulk-store            Store multiple files from directory
  chat                  Chat with ingested files
  list                  List files in knowledge base
  search                Search through ingested files
  clean                 Clean up knowledge base

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}  
  --help                {ARGS_HELP}

Examples:
  agentchef files store paper.pdf --domain research
  agentchef files bulk-store ./docs --patterns "*.md,*.txt"
  agentchef files chat -q "What are the main points?" --domain research
"""

__all__ = [
    'MAIN_HELP', 'RESEARCH_HELP', 'FILES_HELP', 'RESEARCH_GROUP_HELP', 
    'FILES_GROUP_HELP', 'BUILD_GROUP_HELP', 'DATA_GROUP_HELP', 'CONFIG_GROUP_HELP',
    'MCP_GROUP_HELP', 'ARGS_VERBOSE_HELP', 'ARGS_CONFIG_HELP', 'ARGS_TOPIC_HELP',
    'ARGS_MAX_PAPERS_HELP', 'ARGS_MODEL_HELP', 'ARGS_DATA_DIR_HELP'
]