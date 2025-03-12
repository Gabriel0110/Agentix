"""
Firecrawl tools for Agentix agents.

This module provides tools to scrape and crawl websites using the Firecrawl API.
It offers URL scraping, website crawling, site mapping, and async crawling capabilities.
"""

import os
from typing import Any, Dict, List, Optional

from .tools import Tool
from .tool_metadata import ToolParameter, ToolDocumentation

try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


class FirecrawlScrapeTool(Tool):
    """Tool for scraping individual URLs using Firecrawl."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl scrape tool.
        
        Args:
            api_key: Firecrawl API key. If not provided, will look for FIRECRAWL_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key and FIRECRAWL_AVAILABLE:
            import warnings
            warnings.warn(
                "No Firecrawl API key provided. Please provide an API key or set the FIRECRAWL_API_KEY environment variable."
            )
    
    @property
    def name(self) -> str:
        return "FirecrawlScrape"
    
    @property
    def description(self) -> str:
        return "Scrape content from a single URL using Firecrawl's scraping capabilities."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                description="The URL to scrape content from",
                type="string",
                required=True
            ),
            ToolParameter(
                name="formats",
                description="List of output formats: 'markdown', 'html', 'text'",
                type="array",
                required=False
            ),
            ToolParameter(
                name="include_images",
                description="Whether to include images in the scraped content",
                type="boolean",
                required=False
            ),
            ToolParameter(
                name="include_links",
                description="Whether to include links in the scraped content",
                type="boolean",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: FirecrawlScrape {"url": "https://example.com", "formats": ["markdown", "html"]}',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Scrape content from a URL using Firecrawl.
        
        Args:
            input_str: The URL as a string (used if args not provided)
            args: Dictionary with URL and optional parameters
        
        Returns:
            String with scraped content or error message
        """
        if not FIRECRAWL_AVAILABLE:
            return "Error: The firecrawl package is not installed. Please install it with 'pip install firecrawl-py'"
        
        if not self.api_key:
            return "Error: No Firecrawl API key provided. Please provide an API key or set the FIRECRAWL_API_KEY environment variable."
        
        try:
            # Extract parameters
            url = input_str
            formats = ["markdown"]
            include_images = True
            include_links = True
            
            if args:
                if "url" in args:
                    url = args["url"]
                if "formats" in args:
                    formats = args["formats"]
                if "include_images" in args:
                    include_images = args["include_images"]
                if "include_links" in args:
                    include_links = args["include_links"]
            
            if not url:
                return "Error: No URL provided. Please provide a URL to scrape."
            
            # Initialize Firecrawl client
            app = FirecrawlApp(api_key=self.api_key)
            
            # Prepare scrape parameters
            params = {
                'formats': formats,
                'includeImages': include_images,
                'includeLinks': include_links
            }
            
            # Perform the scrape
            result = await app.scrape_url(url, params=params)
            
            if not result:
                return f"No content scraped from URL: {url}"
            
            # Format the output
            output_parts = [f"Content scraped from: {url}\n"]
            
            # Add content for each requested format
            for fmt in formats:
                if fmt in result:
                    output_parts.append(f"\n{fmt.upper()} Content:")
                    output_parts.append(result[fmt])
            
            # Add metadata if available
            if 'metadata' in result:
                output_parts.append("\nMetadata:")
                for key, value in result['metadata'].items():
                    output_parts.append(f"{key}: {value}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error scraping URL: {str(e)}"


class FirecrawlCrawlTool(Tool):
    """Tool for crawling websites using Firecrawl."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl crawl tool.
        
        Args:
            api_key: Firecrawl API key. If not provided, will look for FIRECRAWL_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key and FIRECRAWL_AVAILABLE:
            import warnings
            warnings.warn(
                "No Firecrawl API key provided. Please provide an API key or set the FIRECRAWL_API_KEY environment variable."
            )
    
    @property
    def name(self) -> str:
        return "FirecrawlCrawl"
    
    @property
    def description(self) -> str:
        return "Crawl a website and extract content from multiple pages using Firecrawl."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                description="The starting URL to crawl from",
                type="string",
                required=True
            ),
            ToolParameter(
                name="limit",
                description="Maximum number of pages to crawl",
                type="integer",
                required=False
            ),
            ToolParameter(
                name="formats",
                description="List of output formats: 'markdown', 'html', 'text'",
                type="array",
                required=False
            ),
            ToolParameter(
                name="exclude_paths",
                description="List of paths to exclude from crawling (e.g., ['blog/*', 'docs/*'])",
                type="array",
                required=False
            ),
            ToolParameter(
                name="allowed_domains",
                description="List of domains allowed for crawling",
                type="array",
                required=False
            ),
            ToolParameter(
                name="async_crawl",
                description="Whether to use async crawling (returns job ID for status checking)",
                type="boolean",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: FirecrawlCrawl {"url": "https://example.com", "limit": 100, "formats": ["markdown"]}',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Crawl a website using Firecrawl.
        
        Args:
            input_str: The URL as a string (used if args not provided)
            args: Dictionary with URL and optional parameters
        
        Returns:
            String with crawl results or error message
        """
        if not FIRECRAWL_AVAILABLE:
            return "Error: The firecrawl package is not installed. Please install it with 'pip install firecrawl-py'"
        
        if not self.api_key:
            return "Error: No Firecrawl API key provided. Please provide an API key or set the FIRECRAWL_API_KEY environment variable."
        
        try:
            # Extract parameters
            url = input_str
            limit = 100
            formats = ["markdown"]
            exclude_paths = []
            allowed_domains = []
            async_crawl = False
            
            if args:
                if "url" in args:
                    url = args["url"]
                if "limit" in args:
                    limit = int(args["limit"])
                if "formats" in args:
                    formats = args["formats"]
                if "exclude_paths" in args:
                    exclude_paths = args["exclude_paths"]
                if "allowed_domains" in args:
                    allowed_domains = args["allowed_domains"]
                if "async_crawl" in args:
                    async_crawl = args["async_crawl"]
            
            if not url:
                return "Error: No URL provided. Please provide a URL to crawl."
            
            # Initialize Firecrawl client
            app = FirecrawlApp(api_key=self.api_key)
            
            # Prepare crawl parameters
            params = {
                'limit': limit,
                'scrapeOptions': {
                    'formats': formats
                },
                'excludePaths': exclude_paths
            }
            
            if allowed_domains:
                params['allowedDomains'] = allowed_domains
            
            # Perform the crawl
            if async_crawl:
                result = await app.async_crawl_url(url, params=params)
                return f"Async crawl started with ID: {result['id']}\nUse FirecrawlStatus tool to check progress."
            else:
                result = await app.crawl_url(url, params=params, poll_interval=30)
                
                if not result or not result.get('pages'):
                    return f"No content crawled from URL: {url}"
                
                # Format the output
                output_parts = [
                    f"Crawl results for: {url}",
                    f"Pages crawled: {len(result['pages'])}",
                    f"Time taken: {result.get('duration', 'N/A')} seconds\n"
                ]
                
                # Add summary of crawled pages
                output_parts.append("Crawled pages:")
                for page in result['pages']:
                    output_parts.append(f"\n{page['url']}")
                    if 'title' in page:
                        output_parts.append(f"Title: {page['title']}")
                    if 'contentLength' in page:
                        output_parts.append(f"Content length: {page['contentLength']} bytes")
                
                return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error crawling website: {str(e)}"


class FirecrawlStatusTool(Tool):
    """Tool for checking the status of async crawl jobs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl status tool.
        
        Args:
            api_key: Firecrawl API key. If not provided, will look for FIRECRAWL_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
    
    @property
    def name(self) -> str:
        return "FirecrawlStatus"
    
    @property
    def description(self) -> str:
        return "Check the status of an asynchronous crawl job or cancel it."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="crawl_id",
                description="The ID of the crawl job to check",
                type="string",
                required=True
            ),
            ToolParameter(
                name="cancel",
                description="Whether to cancel the crawl job",
                type="boolean",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: FirecrawlStatus {"crawl_id": "abc123"}',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Check or cancel a crawl job.
        
        Args:
            input_str: The crawl ID as a string (used if args not provided)
            args: Dictionary with crawl ID and optional cancel flag
        
        Returns:
            String with crawl status or cancellation result
        """
        if not FIRECRAWL_AVAILABLE:
            return "Error: The firecrawl package is not installed. Please install it with 'pip install firecrawl-py'"
        
        if not self.api_key:
            return "Error: No Firecrawl API key provided. Please provide an API key or set the FIRECRAWL_API_KEY environment variable."
        
        try:
            # Extract parameters
            crawl_id = input_str
            cancel = False
            
            if args:
                if "crawl_id" in args:
                    crawl_id = args["crawl_id"]
                if "cancel" in args:
                    cancel = args["cancel"]
            
            if not crawl_id:
                return "Error: No crawl ID provided. Please provide a crawl ID to check."
            
            # Initialize Firecrawl client
            app = FirecrawlApp(api_key=self.api_key)
            
            if cancel:
                result = await app.cancel_crawl(crawl_id)
                return f"Crawl job {crawl_id} cancelled. Status: {result['status']}"
            else:
                result = await app.check_crawl_status(crawl_id)
                
                # Format the status output
                output_parts = [
                    f"Status for crawl {crawl_id}:",
                    f"State: {result.get('state', 'Unknown')}",
                    f"Progress: {result.get('progress', 0)}%"
                ]
                
                if 'pagesProcessed' in result:
                    output_parts.append(f"Pages processed: {result['pagesProcessed']}")
                if 'errors' in result and result['errors']:
                    output_parts.append("\nErrors:")
                    for error in result['errors']:
                        output_parts.append(f"- {error}")
                
                return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error checking crawl status: {str(e)}"


class FirecrawlMapTool(Tool):
    """Tool for mapping website URLs using Firecrawl."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl map tool.
        
        Args:
            api_key: Firecrawl API key. If not provided, will look for FIRECRAWL_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
    
    @property
    def name(self) -> str:
        return "FirecrawlMap"
    
    @property
    def description(self) -> str:
        return "Generate a list of URLs from a website using Firecrawl's mapping capabilities."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                description="The URL of the website to map",
                type="string",
                required=True
            ),
            ToolParameter(
                name="exclude_subdomains",
                description="Whether to exclude subdomains from the map",
                type="boolean",
                required=False
            ),
            ToolParameter(
                name="use_sitemap",
                description="Whether to use the website's sitemap for mapping",
                type="boolean",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: FirecrawlMap {"url": "https://example.com", "use_sitemap": true}',
            parameters=self.parameters
        )
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Map a website using Firecrawl.
        
        Args:
            input_str: The URL as a string (used if args not provided)
            args: Dictionary with URL and optional parameters
        
        Returns:
            String with mapped URLs or error message
        """
        if not FIRECRAWL_AVAILABLE:
            return "Error: The firecrawl package is not installed. Please install it with 'pip install firecrawl-py'"
        
        if not self.api_key:
            return "Error: No Firecrawl API key provided. Please provide an API key or set the FIRECRAWL_API_KEY environment variable."
        
        try:
            # Extract parameters
            url = input_str
            exclude_subdomains = False
            use_sitemap = True
            
            if args:
                if "url" in args:
                    url = args["url"]
                if "exclude_subdomains" in args:
                    exclude_subdomains = args["exclude_subdomains"]
                if "use_sitemap" in args:
                    use_sitemap = args["use_sitemap"]
            
            if not url:
                return "Error: No URL provided. Please provide a URL to map."
            
            # Initialize Firecrawl client
            app = FirecrawlApp(api_key=self.api_key)
            
            # Prepare map parameters
            params = {
                'excludeSubdomains': exclude_subdomains,
                'useSitemap': use_sitemap
            }
            
            # Perform the mapping
            result = await app.map_url(url, params=params)
            
            if not result or not result.get('urls'):
                return f"No URLs mapped for website: {url}"
            
            # Format the output
            output_parts = [
                f"URL map for: {url}",
                f"Total URLs found: {len(result['urls'])}\n",
                "URLs:"
            ]
            
            # Add URLs with their metadata if available
            for url_info in result['urls']:
                if isinstance(url_info, dict):
                    output_parts.append(f"\n{url_info['url']}")
                    if 'lastmod' in url_info:
                        output_parts.append(f"Last modified: {url_info['lastmod']}")
                    if 'priority' in url_info:
                        output_parts.append(f"Priority: {url_info['priority']}")
                else:
                    output_parts.append(f"\n{url_info}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error mapping website: {str(e)}"


class FirecrawlToolkit:
    """
    Toolkit for Firecrawl tools.
    
    This toolkit provides tools for scraping, crawling, and mapping websites
    using the Firecrawl API.
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 enable_scrape: bool = True,
                 enable_crawl: bool = True,
                 enable_status: bool = True,
                 enable_map: bool = True):
        """
        Initialize the Firecrawl toolkit with selected tools.
        
        Args:
            api_key: Firecrawl API key. If not provided, will look for FIRECRAWL_API_KEY environment variable.
            enable_scrape: Whether to enable the scrape tool
            enable_crawl: Whether to enable the crawl tool
            enable_status: Whether to enable the status tool
            enable_map: Whether to enable the map tool
        """
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.enable_scrape = enable_scrape
        self.enable_crawl = enable_crawl
        self.enable_status = enable_status
        self.enable_map = enable_map
        
        # Install warning message
        if not FIRECRAWL_AVAILABLE:
            import warnings
            warnings.warn(
                "Firecrawl package not found. Please install with: pip install firecrawl-py"
            )
    
    def get_tools(self) -> List[Tool]:
        """
        Get all enabled Firecrawl tools.
        
        Returns:
            List of enabled Firecrawl tools
        """
        tools = []
        
        if self.enable_scrape:
            tools.append(FirecrawlScrapeTool(api_key=self.api_key))
        
        if self.enable_crawl:
            tools.append(FirecrawlCrawlTool(api_key=self.api_key))
        
        if self.enable_status:
            tools.append(FirecrawlStatusTool(api_key=self.api_key))
        
        if self.enable_map:
            tools.append(FirecrawlMapTool(api_key=self.api_key))
        
        return tools 