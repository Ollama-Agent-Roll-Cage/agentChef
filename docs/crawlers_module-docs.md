# Crawlers Module Documentation

This module contains classes for crawling and extracting information from different sources:
- WebCrawler: General web page crawling
- ArxivSearcher: ArXiv paper lookup and parsing
- DuckDuckGoSearcher: DuckDuckGo search API integration
- GitHubCrawler: GitHub repository cloning and extraction

## Class: `WebCrawler`

Class for crawling web pages and extracting content.

### Methods

#### `fetch_url_content`

```python
@staticmethod
async def fetch_url_content(url)
```

Fetch content from a URL.

**Parameters:**
- `url` (str): The URL to fetch content from

**Returns:**
- str: HTML content of the page or None if failed

#### `extract_text_from_html`

```python
@staticmethod
async def extract_text_from_html(html)
```

Extract main text content from HTML using BeautifulSoup.

**Parameters:**
- `html` (str): HTML content

**Returns:**
- str: Extracted text content

#### `extract_pypi_content`

```python
@staticmethod
async def extract_pypi_content(html, package_name)
```

Specifically extract PyPI package documentation from HTML.

**Parameters:**
- `html` (str): HTML content from PyPI page
- `package_name` (str): Name of the package

**Returns:**
- dict: Structured package data or None if failed

#### `format_pypi_info`

```python
@staticmethod
async def format_pypi_info(package_data)
```

Format PyPI package data into a readable markdown format.

**Parameters:**
- `package_data` (dict): Package data from PyPI API

**Returns:**
- str: Formatted markdown text

## Class: `ArxivSearcher`

Class for searching and retrieving ArXiv papers.

### Methods

#### `extract_arxiv_id`

```python
@staticmethod
def extract_arxiv_id(url_or_id)
```

Extract arXiv ID from a URL or direct ID string.

**Parameters:**
- `url_or_id` (str): ArXiv URL or direct ID

**Returns:**
- str: Extracted ArXiv ID

**Raises:**
- ValueError: If ID cannot be extracted

#### `fetch_paper_info`

```python
@staticmethod
async def fetch_paper_info(arxiv_id)
```

Fetch paper metadata from arXiv API.

**Parameters:**
- `arxiv_id` (str): ArXiv paper ID

**Returns:**
- dict: Paper metadata

**Raises:**
- ValueError: If paper cannot be found
- ConnectionError: If connection to ArXiv fails

#### `format_paper_for_learning`

```python
@staticmethod
async def format_paper_for_learning(paper_info)
```

Format paper information for learning.

**Parameters:**
- `paper_info` (dict): Paper metadata

**Returns:**
- str: Formatted markdown text

## Class: `DuckDuckGoSearcher`

Class for performing searches using DuckDuckGo API.

### Methods

#### `text_search`

```python
@staticmethod
async def text_search(search_query, max_results=5)
```

Perform an async text search using DuckDuckGo.

**Parameters:**
- `search_query` (str): Query to search for
- `max_results` (int): Maximum number of results to return

**Returns:**
- str: Formatted search results in markdown

## Class: `GitHubCrawler`

Class for crawling and extracting content from GitHub repositories.

### Constructor

```python
def __init__(self, data_dir=None)
```

Initialize the GitHub Crawler.

**Parameters:**
- `data_dir` (str, optional): Directory to store data. Defaults to DATA_DIR.

### Methods

#### `extract_repo_info_from_url`

```python
@staticmethod
def extract_repo_info_from_url(url)
```

Extract repository owner and name from GitHub URL.

**Parameters:**
- `url` (str): GitHub repository URL

**Returns:**
- Tuple[str, str, str]: Repository owner, name, and branch (if available)

**Raises:**
- ValueError: If URL is not a valid GitHub repository URL

#### `get_repo_dir_path`

```python
def get_repo_dir_path(self, owner, repo_name)
```

Get the directory path for storing repository data.

**Parameters:**
- `owner` (str): Repository owner
- `repo_name` (str): Repository name

**Returns:**
- Path: Directory path

#### `clone_repo`

```python
async def clone_repo(self, repo_url, temp_dir=None)
```

Clone a GitHub repository to a temporary directory.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `temp_dir` (str, optional): Temporary directory path. If None, creates one.

**Returns:**
- Path: Path to the cloned repository

**Raises:**
- Exception: If cloning fails

#### `is_binary_file`

```python
def is_binary_file(self, file_path)
```

Check if a file is binary.

**Parameters:**
- `file_path` (str): Path to the file

**Returns:**
- bool: True if file is binary, False otherwise

#### `process_repo_to_dataframe`

```python
async def process_repo_to_dataframe(self, repo_path, max_file_size_kb=500)
```

Process repository files and convert to DataFrame.

**Parameters:**
- `repo_path` (Path): Path to cloned repository
- `max_file_size_kb` (int): Maximum file size in KB to process

**Returns:**
- pd.DataFrame: DataFrame containing file information

#### `get_language_from_extension`

```python
@staticmethod
def get_language_from_extension(extension)
```

Get programming language name from file extension.

**Parameters:**
- `extension` (str): File extension with leading dot

**Returns:**
- str: Language name or 'Unknown'

#### `clone_and_store_repo`

```python
async def clone_and_store_repo(self, repo_url)
```

Clone a GitHub repository and store its data in Parquet format.

**Parameters:**
- `repo_url` (str): GitHub repository URL

**Returns:**
- str: Path to the Parquet file containing repository data

**Raises:**
- Exception: If cloning or processing fails

#### `query_repo_content`

```python
async def query_repo_content(self, repo_url, query)
```

Query repository content using natural language.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `query` (str): Natural language query about the repository

**Returns:**
- str: Query result formatted as markdown

**Raises:**
- Exception: If querying fails

#### `get_repo_summary`

```python
async def get_repo_summary(self, repo_url)
```

Get a summary of the repository.

**Parameters:**
- `repo_url` (str): GitHub repository URL

**Returns:**
- str: Repository summary formatted as markdown

#### `find_similar_code`

```python
async def find_similar_code(self, repo_url, code_snippet)
```

Find similar code in the repository.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `code_snippet` (str): Code snippet to find similar code for

**Returns:**
- str: Similar code findings formatted as markdown
