"""
Tavily API tools for Agentix agents.

This module provides a set of tools to perform web searches and content extraction
using the Tavily API. It offers text search with various parameters, image search,
content extraction, and hybrid RAG capabilities.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .tools import Tool, ToolParameter, ToolDocumentation

from dotenv import load_dotenv
load_dotenv()

try:
    from tavily import TavilyClient, AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class TavilySearchTool(Tool):
    """Tool for searching the web using Tavily's search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily search tool.
        
        Args:
            api_key: Tavily API key. If not provided, will look for TAVILY_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key and TAVILY_AVAILABLE:
            import warnings
            warnings.warn(
                "No Tavily API key provided. Please provide an API key or set the TAVILY_API_KEY environment variable."
            )
    
    @property
    def name(self) -> str:
        return "TavilySearch"
    
    @property
    def description(self) -> str:
        return "Search the web for information using Tavily's search API. Can search general information or news with various parameters."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The search query to find information about",
                type="string",
                required=True
            ),
            ToolParameter(
                name="search_depth",
                description="The depth of search to perform: 'basic' (faster) or 'advanced' (more thorough)",
                type="string",
                required=False
            ),
            ToolParameter(
                name="topic",
                description="The topic category: 'general' or 'news'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of search results to return (1-20)",
                type="integer",
                required=False
            ),
            ToolParameter(
                name="include_domains",
                description="List of domains to include in search results",
                type="array",
                required=False
            ),
            ToolParameter(
                name="exclude_domains",
                description="List of domains to exclude from search results",
                type="array",
                required=False
            ),
            ToolParameter(
                name="include_answer",
                description="Whether to include an LLM-generated answer based on search results: true, false, 'basic', or 'advanced'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="include_images",
                description="Whether to include images in the search results",
                type="boolean",
                required=False
            ),
            ToolParameter(
                name="time_range",
                description="How far back to search: 'day', 'week', 'month', 'year' (or 'd', 'w', 'm', 'y')",
                type="string",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: TavilySearch {"query": "latest developments in quantum computing", "max_results": 5}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: TavilySearch {"query": "latest developments in quantum computing", "max_results": 5}',
            'TOOL REQUEST: TavilySearch {"query": "climate change news", "topic": "news", "time_range": "week", "max_results": 3}',
            'TOOL REQUEST: TavilySearch {"query": "electric vehicles", "search_depth": "advanced", "include_answer": "advanced"}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run a Tavily search query.
        
        Args:
            input_str: The search query as a string (used if args not provided)
            args: Dictionary with query and optional parameters
        
        Returns:
            String with search results or error message
        """
        if not TAVILY_AVAILABLE:
            return "Error: The tavily package is not installed. Please install it with 'pip install tavily'."
        
        if not self.api_key:
            return "Error: No Tavily API key provided. Please provide an API key or set the TAVILY_API_KEY environment variable."
        
        try:
            # Extract parameters
            query = input_str
            search_depth = "basic"
            topic = "general"
            max_results = 5
            include_domains = []
            exclude_domains = []
            include_answer = False
            include_images = False
            time_range = None
            include_raw_content = False
            
            if args:
                if "query" in args:
                    query = args["query"]
                if "search_depth" in args:
                    search_depth = args["search_depth"]
                if "topic" in args:
                    topic = args["topic"]
                if "max_results" in args:
                    max_results = int(args["max_results"])
                if "include_domains" in args:
                    include_domains = args["include_domains"]
                if "exclude_domains" in args:
                    exclude_domains = args["exclude_domains"]
                if "include_answer" in args:
                    include_answer = args["include_answer"]
                if "include_images" in args:
                    include_images = args["include_images"]
                if "time_range" in args:
                    time_range = args["time_range"]
                if "include_raw_content" in args:
                    include_raw_content = args["include_raw_content"]
            
            if not query:
                return "Error: No search query provided. Please provide a search term."
            
            # Initialize the Tavily client
            client = AsyncTavilyClient(api_key=self.api_key)
            
            # Prepare parameters for the search
            search_params = {
                "query": query,
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_images": include_images,
                "include_raw_content": include_raw_content
            }
            
            # Add optional parameters if provided
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            if include_answer:
                search_params["include_answer"] = include_answer
            if time_range:
                search_params["time_range"] = time_range
            
            # Perform the search
            result = await client.search(**search_params)
            
            # Format results
            if not result or not result.get("results"):
                return f"No results found for query: '{query}'"
            
            # Initialize output components
            output_parts = [f"Search results for: '{query}'\n"]
            
            # Add the answer if it was requested and provided
            if include_answer and "answer" in result:
                output_parts.append(f"AI-Generated Answer:\n{result['answer']}\n\n")
            
            # Add images if they were requested and provided
            if include_images and "images" in result:
                image_section = ["Images:"]
                
                images = result["images"]
                for i, image in enumerate(images, 1):
                    if isinstance(image, dict) and "url" in image:  # For image with description
                        image_text = f"{i}. {image['url']}"
                        if "description" in image:
                            image_text += f"\n   Description: {image['description']}"
                        image_section.append(image_text)
                    elif isinstance(image, str):  # For image URL only
                        image_section.append(f"{i}. {image}")
                
                output_parts.append("\n".join(image_section) + "\n\n")
            
            # Add the search results
            results_section = ["Search Results:"]
            for i, res in enumerate(result["results"], 1):
                result_text = [
                    f"{i}. {res.get('title', 'No title')}",
                    f"   URL: {res.get('url', 'No URL')}"
                ]
                
                # Add publication date for news results
                if topic == "news" and "published_date" in res:
                    result_text.append(f"   Published: {res['published_date']}")
                
                # Add content snippet
                result_text.append(f"   {res.get('content', 'No content')[:250]}...")
                
                # Add raw content if requested
                if include_raw_content and "raw_content" in res:
                    raw_content_snippet = res["raw_content"][:500] + "..." if len(res["raw_content"]) > 500 else res["raw_content"]
                    result_text.append(f"   Raw Content: {raw_content_snippet}")
                
                results_section.append("\n".join(result_text))
            
            output_parts.append("\n".join(results_section))
            
            # Add metadata about response time
            if "response_time" in result:
                output_parts.append(f"\nResponse time: {result['response_time']:.2f} seconds")
            
            return "\n\n".join(output_parts)
            
        except Exception as e:
            return f"Error performing Tavily search: {str(e)}"


class TavilyExtractTool(Tool):
    """Tool for extracting content from web URLs using Tavily's extract API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily extract tool.
        
        Args:
            api_key: Tavily API key. If not provided, will look for TAVILY_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key and TAVILY_AVAILABLE:
            import warnings
            warnings.warn(
                "No Tavily API key provided. Please provide an API key or set the TAVILY_API_KEY environment variable."
            )
    
    @property
    def name(self) -> str:
        return "TavilyExtract"
    
    @property
    def description(self) -> str:
        return "Extract content from web URLs using Tavily's extraction API."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="urls",
                description="The URL or list of URLs to extract content from (max 20)",
                type="string",
                required=True
            ),
            ToolParameter(
                name="include_images",
                description="Whether to include images in the extracted content",
                type="boolean",
                required=False
            ),
            ToolParameter(
                name="extract_depth",
                description="The depth of extraction: 'basic' (faster) or 'advanced' (more thorough)",
                type="string",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: TavilyExtract {"urls": "https://example.com", "extract_depth": "basic"}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: TavilyExtract {"urls": "https://example.com", "extract_depth": "basic"}',
            'TOOL REQUEST: TavilyExtract {"urls": ["https://example.com", "https://example.org"], "include_images": true, "extract_depth": "advanced"}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract content from URLs using Tavily.
        
        Args:
            input_str: The URL as a string (used if args not provided)
            args: Dictionary with URLs and optional parameters
        
        Returns:
            String with extracted content or error message
        """
        if not TAVILY_AVAILABLE:
            return "Error: The tavily package is not installed. Please install it with 'pip install tavily'."
        
        if not self.api_key:
            return "Error: No Tavily API key provided. Please provide an API key or set the TAVILY_API_KEY environment variable."
        
        try:
            # Extract parameters
            urls = input_str
            include_images = False
            extract_depth = "basic"
            
            if args:
                if "urls" in args:
                    urls = args["urls"]
                if "include_images" in args:
                    include_images = args["include_images"]
                if "extract_depth" in args:
                    extract_depth = args["extract_depth"]
            
            if not urls:
                return "Error: No URLs provided. Please provide at least one URL to extract content from."
            
            # Initialize the Tavily client
            client = AsyncTavilyClient(api_key=self.api_key)
            
            # Prepare the URLs (ensure it's a list)
            if isinstance(urls, str):
                # Check if it might be a JSON string representing a list
                if urls.startswith("[") and urls.endswith("]"):
                    try:
                        urls = json.loads(urls)
                    except json.JSONDecodeError:
                        # Not a valid JSON, treat as a single URL
                        urls = [urls]
                else:
                    urls = [urls]
            
            # Prepare parameters for extraction
            extract_params = {
                "urls": urls,
                "include_images": include_images,
                "extract_depth": extract_depth
            }
            
            # Perform the extraction
            result = await client.extract(**extract_params)
            
            # Format results
            if not result:
                return f"No content extracted from the provided URLs."
            
            # Initialize output
            output_parts = ["Content extracted from URLs:"]
            
            # Add successful results
            if "results" in result and result["results"]:
                output_parts.append("\nSuccessful extractions:")
                
                for i, res in enumerate(result["results"], 1):
                    result_text = [
                        f"{i}. URL: {res.get('url', 'No URL')}",
                        f"   Content length: {len(res.get('raw_content', ''))}"
                    ]
                    
                    # Add snippet of content
                    content = res.get('raw_content', 'No content')
                    snippet = content[:500] + "..." if len(content) > 500 else content
                    result_text.append(f"   Content snippet: {snippet}")
                    
                    # Add images if requested and available
                    if include_images and "images" in res and res["images"]:
                        images_text = ["   Images:"]
                        for j, img in enumerate(res["images"][:5], 1):  # Limit to 5 images
                            images_text.append(f"     {j}. {img}")
                        if len(res["images"]) > 5:
                            images_text.append(f"     ... and {len(res['images']) - 5} more images")
                        result_text.append("\n".join(images_text))
                    
                    output_parts.append("\n".join(result_text))
            
            # Add failed results
            if "failed_results" in result and result["failed_results"]:
                output_parts.append("\nFailed extractions:")
                
                for i, res in enumerate(result["failed_results"], 1):
                    output_parts.append(
                        f"{i}. URL: {res.get('url', 'No URL')}\n"
                        f"   Error: {res.get('error', 'Unknown error')}"
                    )
            
            # Add metadata about response time
            if "response_time" in result:
                output_parts.append(f"\nResponse time: {result['response_time']:.2f} seconds")
            
            return "\n\n".join(output_parts)
            
        except Exception as e:
            return f"Error performing Tavily extraction: {str(e)}"


class TavilyToolkit:
    """
    Toolkit for Tavily API tools.
    
    This toolkit provides tools for searching the web and extracting content
    from web pages using the Tavily API.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 enable_search: bool = True,
                 enable_extract: bool = True):
        """
        Initialize the Tavily toolkit with selected tools.
        
        Args:
            api_key: Tavily API key. If not provided, will look for TAVILY_API_KEY environment variable.
            enable_search: Whether to enable the search tool
            enable_extract: Whether to enable the extract tool
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.enable_search = enable_search
        self.enable_extract = enable_extract
        
        # Install warning message
        if not TAVILY_AVAILABLE:
            import warnings
            warnings.warn(
                "Tavily package not found. Please install with: pip install tavily"
            )
        
        if not self.api_key and TAVILY_AVAILABLE:
            import warnings
            warnings.warn(
                "No Tavily API key provided. Please provide an API key or set the TAVILY_API_KEY environment variable."
            )
    
    def get_tools(self) -> List[Tool]:
        """
        Get all enabled Tavily tools.
        
        Returns:
            List of enabled Tavily tools
        """
        tools = []
        
        if self.enable_search:
            tools.append(TavilySearchTool(api_key=self.api_key))
        
        if self.enable_extract:
            tools.append(TavilyExtractTool(api_key=self.api_key))
        
        return tools 