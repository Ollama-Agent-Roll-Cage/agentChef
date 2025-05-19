"""
Help text constants for the OARC Crawlers CLI.

This module centralizes all help and usage text for CLI commands and options,
ensuring consistent and maintainable documentation across the toolkit.
"""

# Command group descriptions
YOUTUBE_GROUP_HELP = "YouTube operations for downloading videos and extracting information."
GH_GROUP_HELP = "GitHub operations for cloning, analyzing and extracting from repositories."
ARXIV_GROUP_HELP = "ArXiv operations for downloading papers and extracting content."
WEB_GROUP_HELP = "Web crawler operations for extracting content from websites."
DDG_GROUP_HELP = "DuckDuckGo search operations for finding information online."
BUILD_GROUP_HELP = "Build operations for package management."
PUBLISH_GROUP_HELP = "Publish operations for distributing packages."
DATA_GROUP_HELP = "Data management operations for viewing and manipulating data files."
CONFIG_GROUP_HELP = "Manage configuration settings for OARC Crawlers."
MCP_GROUP_HELP = "Model Context Protocol (MCP) server operations."

# Command option descriptions
ARGS_HELP = "Show this help message and exit."
ARGS_VERBOSE_HELP = "Enable verbose output and debug logging"
ARGS_CONFIG_HELP = "Path to custom configuration file"
ARGS_URL_HELP = "URL to process"
ARGS_REPO_URL_HELP = "GitHub repository URL"
ARGS_VIDEO_URL_HELP = "YouTube video URL"
ARGS_VIDEO_ID_HELP = "YouTube video ID or URL"
ARGS_PLAYLIST_URL_HELP = "YouTube playlist URL"
ARGS_QUERY_HELP = "Search query"
ARGS_MAX_RESULTS_HELP = "Maximum number of results to return"
ARGS_LIMIT_HELP = "Maximum number of results to return"
ARGS_ID_HELP = "arXiv paper ID"
ARGS_OUTPUT_PATH_HELP = "Directory to save the output"
ARGS_OUTPUT_FILE_HELP = "File to save the output"
ARGS_LANGUAGE_HELP = "Programming language of the code"
ARGS_LANGUAGES_HELP = "Comma-separated language codes (e.g. \"en,es,fr\")"
ARGS_FORMAT_HELP = "Output format"
ARGS_CODE_HELP = "Code snippet to search for"
ARGS_CLEAN_HELP = "Clean build directories first"
ARGS_TEST_HELP = "Upload to TestPyPI instead of PyPI"
ARGS_BUILD_HELP = "Build the package before publishing"
ARGS_PORT_HELP = "Port to run the server on"
ARGS_TRANSPORT_HELP = "Transport method to use"
ARGS_DATA_DIR_HELP = "Directory to store data"
ARGS_PACKAGE_HELP = "PyPI package name"
ARGS_RESOLUTION_HELP = "Video resolution (\"highest\", \"lowest\", or specific like \"720p\")"
ARGS_EXTRACT_AUDIO_HELP = "Extract audio only"
ARGS_FILENAME_HELP = "Custom filename for the downloaded file"
ARGS_MAX_VIDEOS_HELP = "Maximum number of videos to download"
ARGS_MAX_MESSAGES_HELP = "Maximum number of messages to collect"
ARGS_DURATION_HELP = "Duration in seconds to collect messages"
ARGS_MCP_NAME_HELP = "Custom name for the MCP server in VS Code"
ARGS_PYPI_USERNAME_HELP = "PyPI username (if not using keyring)'"
ARGS_PYPI_PASSWORD_HELP = "PyPI password (if not using keyring)"
ARGS_PYPI_CONFIG_FILE_HELP = "Path to PyPI config file (.pypirc)"
ARGS_FILE_PATH_HELP = "Path to the file"
ARGS_MAX_ROWS_HELP = "Maximum number of rows to display"
ARGS_CATEGORY_HELP = "ArXiv category to fetch papers from"
ARGS_IDS_HELP = "Comma-separated list of arXiv paper IDs"
ARGS_MAX_DEPTH_HELP = "Maximum depth for citation network generation"

# --- Main CLI Help ---
# This shorter version is meant to be used as a docstring with Click's built-in help formatting
MAIN_HELP = """OARC Crawlers Command-Line Interface."""

# This is the original longer version, renamed so you can choose which to use in your CLI code
MAIN_HELP_DETAILED = f"""
OARC Crawlers Command-Line Interface.

USAGE:
  oarc-crawlers [OPTIONS] COMMAND [ARGS]...

  For detailed information about any command:
    oarc-crawlers <command> --help

Options:
  --version             Show the version and exit.
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Commands:
  arxiv                 {ARXIV_GROUP_HELP}
  build                 {BUILD_GROUP_HELP}
  config                Manage configuration settings for OARC Crawlers.
  data                  {DATA_GROUP_HELP}
  ddg                   {DDG_GROUP_HELP}
  gh                    {GH_GROUP_HELP}
  mcp                   Model Context Protocol (MCP) server operations.
  publish               {PUBLISH_GROUP_HELP}
  web                   {WEB_GROUP_HELP}
  yt                    {YOUTUBE_GROUP_HELP}
"""

# --- Command Group Help Texts ---

BUILD_HELP = f"""
Build operations for package management.

USAGE:
  oarc-crawlers build COMMAND [OPTIONS]

Commands:
  package               Build the OARC Crawlers package.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers build package
  oarc-crawlers build package --clean
  oarc-crawlers build package --config ~/.oarc/config.ini

"""

PUBLISH_HELP = f"""
Publish operations for distributing packages.

USAGE:
  oarc-crawlers publish COMMAND [OPTIONS]

Commands:
  pypi                  Publish to PyPI or TestPyPI.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers publish pypi
  oarc-crawlers publish pypi --test
  oarc-crawlers publish pypi --config ~/.oarc/config.ini

"""

YOUTUBE_HELP = f"""
YouTube operations for downloading videos and extracting information.

USAGE:
  oarc-crawlers youtube COMMAND [OPTIONS]

Commands:
  download              Download a YouTube video.
  playlist              Download videos from a YouTube playlist.
  captions              Extract captions/subtitles from a YouTube video.
  search                Search for videos on YouTube.
  chat                  Fetch chat messages from a YouTube live stream.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers yt download --url https://youtube.com/watch?v=example
  oarc-crawlers yt search --query "python tutorials"
  oarc-crawlers yt download --url https://youtube.com/watch?v=example --config ~/.oarc/config.ini

"""

GH_HELP = f"""
GitHub operations for cloning, analyzing and extracting from repositories.

USAGE:
  oarc-crawlers gh COMMAND [OPTIONS]

Commands:
  clone                 Clone a GitHub repository.
  analyze               Analyze a GitHub repository's content.
  find-similar          Find code similar to a snippet in a repository.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers gh clone --url https://github.com/username/repo
  oarc-crawlers gh analyze --url https://github.com/username/repo
  oarc-crawlers gh clone --url https://github.com/username/repo --config ~/.oarc/config.ini

"""

ARXIV_HELP = f"""
ArXiv operations for downloading papers and extracting content.

USAGE:
  oarc-crawlers arxiv COMMAND [OPTIONS]

Commands:
  download              Download LaTeX source files for a paper.
  search                Search for papers on arXiv.
  latex                 Download and extract LaTeX content from a paper.
  keywords              Extract keywords from an arXiv paper.
  references            Extract bibliography references from an arXiv paper.
  equations             Extract mathematical equations from an arXiv paper.
  category              Fetch recent papers from an arXiv category.
  batch                 Process multiple papers in batch.
  citation-network      Generate a citation network from seed papers.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers arxiv download --id 2310.12123
  oarc-crawlers arxiv latex --id 1909.11065
  oarc-crawlers arxiv download --id 2310.12123 --config ~/.oarc/config.ini

"""

WEB_HELP = f"""
Web crawler operations for extracting content from websites.

USAGE:
  oarc-crawlers web COMMAND [OPTIONS]

Commands:
  crawl                 Extract content from a webpage.
  docs                  Extract content from a documentation site.
  pypi                  Extract information about a PyPI package.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers web crawl --url https://example.com
  oarc-crawlers web pypi --package requests
  oarc-crawlers web crawl --url https://example.com --config ~/.oarc/config.ini

"""

DDG_HELP = f"""
DuckDuckGo search operations for finding information online.

USAGE:
  oarc-crawlers ddg COMMAND [OPTIONS]

Commands:
  text                  Perform a DuckDuckGo text search.
  images                Perform a DuckDuckGo image search.
  news                  Perform a DuckDuckGo news search.

Options:
  --verbose             {ARGS_VERBOSE_HELP}
  --config TEXT         {ARGS_CONFIG_HELP}
  --help                {ARGS_HELP}

Examples:
  oarc-crawlers ddg text --query "quantum computing"
  oarc-crawlers ddg images --query "cute cats"
  oarc-crawlers ddg text --query "quantum computing" --config ~/.oarc/config.ini

"""