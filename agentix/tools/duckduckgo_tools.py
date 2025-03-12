"""
DuckDuckGo search tools for Agentix agents.

This module provides a set of tools to perform various types of searches using the 
DuckDuckGo search engine. It offers text search, image search, video search, news search,
and access to DuckDuckGo's AI chat capabilities.
"""

from typing import Any, Dict, List, Optional

from .tools import Tool, ToolParameter, ToolDocumentation

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException, TimeoutException
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False


class DuckDuckGoTextSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo text search."""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 10, verify: bool = True, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the DuckDuckGo text search tool.
        
        Args:
            proxy: Optional proxy for HTTP client (e.g. "tb" for Tor Browser, 
                  or "http://user:pass@example.com:3128")
            timeout: Timeout value for HTTP client in seconds
            verify: SSL verification when making requests
            headers: Optional dictionary of headers for the HTTP client (e.g. custom User-Agent)
        """
        super().__init__()
        self.proxy = proxy
        self.timeout = timeout
        self.verify = verify
        self.headers = headers
    
    @property
    def name(self) -> str:
        return "DuckDuckGoTextSearch"
    
    @property
    def description(self) -> str:
        return "Search the web for information using DuckDuckGo's text search capabilities."
    
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
                name="region",
                description="Region code for regional search results (e.g., 'wt-wt', 'us-en', 'uk-en')",
                type="string",
                required=False
            ),
            ToolParameter(
                name="safesearch",
                description="Safety level for results: 'on', 'moderate', or 'off'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="timelimit",
                description="Time limit for results: 'd' (day), 'w' (week), 'm' (month), 'y' (year)",
                type="string",
                required=False
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results to return (default: 5)",
                type="integer",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: DuckDuckGoTextSearch {"query": "climate change solutions", "max_results": 5}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: DuckDuckGoTextSearch {"query": "climate change solutions", "max_results": 5}',
            'TOOL REQUEST: DuckDuckGoTextSearch {"query": "AI ethics", "timelimit": "m", "max_results": 3}',
            'TOOL REQUEST: DuckDuckGoTextSearch {"query": "programming languages filetype:pdf", "max_results": 3}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run a text search using DuckDuckGo.
        
        Args:
            input_str: The search query as a string (used if args not provided)
            args: Dictionary with query and optional parameters
        
        Returns:
            String with search results or error message
        """
        if not DDGS_AVAILABLE:
            return "Error: The duckduckgo-search package is not installed. Please install it with 'pip install duckduckgo-search'."
        
        try:
            # Extract parameters
            query = input_str
            region = "wt-wt"
            safesearch = "moderate"
            timelimit = None
            max_results = 5
            
            if args:
                if "query" in args:
                    query = args["query"]
                if "region" in args:
                    region = args["region"]
                if "safesearch" in args:
                    safesearch = args["safesearch"]
                if "timelimit" in args:
                    timelimit = args["timelimit"]
                if "max_results" in args:
                    max_results = int(args["max_results"])
            
            if not query:
                return "Error: No search query provided. Please provide a search term."
            
            # Initialize the DDGS client with proxy settings if provided
            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify, headers=self.headers)
            
            # Perform the search
            results = ddgs.text(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            )
            
            # Format results
            if not results:
                return f"No results found for query: '{query}'"
            
            formatted_results = []
            for idx, result in enumerate(results, 1):
                formatted_result = (
                    f"{idx}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )
                formatted_results.append(formatted_result)
            
            # Combine and return results
            output = f"Search results for: '{query}'\n\n"
            output += "\n".join(formatted_results)
            return output
            
        except RatelimitException:
            return "Error: DuckDuckGo search rate limit reached. Please try again later."
        except TimeoutException:
            return "Error: DuckDuckGo search timed out. Please try again or simplify your query."
        except DuckDuckGoSearchException as e:
            return f"Error: DuckDuckGo search failed: {str(e)}"
        except Exception as e:
            return f"Error performing DuckDuckGo search: {str(e)}"


class DuckDuckGoImageSearchTool(Tool):
    """Tool for searching images using DuckDuckGo."""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 10, verify: bool = True, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the DuckDuckGo image search tool.
        
        Args:
            proxy: Optional proxy for HTTP client (e.g. "tb" for Tor Browser, 
                  or "http://user:pass@example.com:3128")
            timeout: Timeout value for HTTP client in seconds
            verify: SSL verification when making requests
            headers: Optional dictionary of headers for the HTTP client (e.g. custom User-Agent)
        """
        super().__init__()
        self.proxy = proxy
        self.timeout = timeout
        self.verify = verify
        self.headers = headers
    
    @property
    def name(self) -> str:
        return "DuckDuckGoImageSearch"
    
    @property
    def description(self) -> str:
        return "Search for images on the web using DuckDuckGo's image search capabilities."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The search query to find images about",
                type="string",
                required=True
            ),
            ToolParameter(
                name="region",
                description="Region code for regional search results (e.g., 'wt-wt', 'us-en', 'uk-en')",
                type="string",
                required=False
            ),
            ToolParameter(
                name="safesearch",
                description="Safety level for results: 'on', 'moderate', or 'off'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="size",
                description="Image size: 'Small', 'Medium', 'Large', 'Wallpaper'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="color",
                description="Image color: 'color', 'Monochrome', 'Red', 'Orange', etc.",
                type="string",
                required=False
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results to return (default: 5)",
                type="integer",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: DuckDuckGoImageSearch {"query": "sunset beach", "max_results": 3}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: DuckDuckGoImageSearch {"query": "sunset beach", "max_results": 3}',
            'TOOL REQUEST: DuckDuckGoImageSearch {"query": "mountain landscape", "size": "Large", "max_results": 5}',
            'TOOL REQUEST: DuckDuckGoImageSearch {"query": "flowers", "color": "Red", "max_results": 3}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run an image search using DuckDuckGo.
        
        Args:
            input_str: The search query as a string (used if args not provided)
            args: Dictionary with query and optional parameters
        
        Returns:
            String with image search results or error message
        """
        if not DDGS_AVAILABLE:
            return "Error: The duckduckgo-search package is not installed. Please install it with 'pip install duckduckgo-search'."
        
        try:
            # Extract parameters
            query = input_str
            region = "wt-wt"
            safesearch = "moderate"
            size = None
            color = None
            max_results = 5
            
            if args:
                if "query" in args:
                    query = args["query"]
                if "region" in args:
                    region = args["region"]
                if "safesearch" in args:
                    safesearch = args["safesearch"]
                if "size" in args:
                    size = args["size"]
                if "color" in args:
                    color = args["color"]
                if "max_results" in args:
                    max_results = int(args["max_results"])
            
            if not query:
                return "Error: No search query provided. Please provide a search term."
            
            # Initialize the DDGS client with proxy settings if provided
            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify, headers=self.headers)
            
            # Perform the search
            results = ddgs.images(
                keywords=query,
                region=region,
                safesearch=safesearch,
                size=size,
                color=color,
                max_results=max_results
            )
            
            # Format results
            if not results:
                return f"No image results found for query: '{query}'"
            
            formatted_results = []
            for idx, result in enumerate(results, 1):
                formatted_result = (
                    f"{idx}. {result.get('title', 'No title')}\n"
                    f"   Image URL: {result.get('image', 'No image URL')}\n"
                    f"   Source: {result.get('url', 'No source URL')}\n"
                    f"   Dimensions: {result.get('height', 'N/A')}x{result.get('width', 'N/A')}\n"
                )
                formatted_results.append(formatted_result)
            
            # Combine and return results
            output = f"Image search results for: '{query}'\n\n"
            output += "\n".join(formatted_results)
            return output
            
        except RatelimitException:
            return "Error: DuckDuckGo search rate limit reached. Please try again later."
        except TimeoutException:
            return "Error: DuckDuckGo search timed out. Please try again or simplify your query."
        except DuckDuckGoSearchException as e:
            return f"Error: DuckDuckGo search failed: {str(e)}"
        except Exception as e:
            return f"Error performing DuckDuckGo image search: {str(e)}"


class DuckDuckGoVideoSearchTool(Tool):
    """Tool for searching videos using DuckDuckGo."""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 10, verify: bool = True, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the DuckDuckGo video search tool.
        
        Args:
            proxy: Optional proxy for HTTP client (e.g. "tb" for Tor Browser, 
                  or "http://user:pass@example.com:3128")
            timeout: Timeout value for HTTP client in seconds
            verify: SSL verification when making requests
            headers: Optional dictionary of headers for the HTTP client (e.g. custom User-Agent)
        """
        super().__init__()
        self.proxy = proxy
        self.timeout = timeout
        self.verify = verify
        self.headers = headers
    
    @property
    def name(self) -> str:
        return "DuckDuckGoVideoSearch"
    
    @property
    def description(self) -> str:
        return "Search for videos on the web using DuckDuckGo's video search capabilities."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The search query to find videos about",
                type="string",
                required=True
            ),
            ToolParameter(
                name="region",
                description="Region code for regional search results (e.g., 'wt-wt', 'us-en', 'uk-en')",
                type="string",
                required=False
            ),
            ToolParameter(
                name="safesearch",
                description="Safety level for results: 'on', 'moderate', or 'off'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="timelimit",
                description="Time limit for results: 'd' (day), 'w' (week), 'm' (month)",
                type="string",
                required=False
            ),
            ToolParameter(
                name="resolution",
                description="Video resolution: 'high', 'standard'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="duration",
                description="Video duration: 'short', 'medium', 'long'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results to return (default: 5)",
                type="integer",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: DuckDuckGoVideoSearch {"query": "python tutorial", "max_results": 3}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: DuckDuckGoVideoSearch {"query": "python tutorial", "max_results": 3}',
            'TOOL REQUEST: DuckDuckGoVideoSearch {"query": "cooking recipes", "resolution": "high", "duration": "medium", "max_results": 5}',
            'TOOL REQUEST: DuckDuckGoVideoSearch {"query": "space exploration", "timelimit": "m", "max_results": 3}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run a video search using DuckDuckGo.
        
        Args:
            input_str: The search query as a string (used if args not provided)
            args: Dictionary with query and optional parameters
        
        Returns:
            String with video search results or error message
        """
        if not DDGS_AVAILABLE:
            return "Error: The duckduckgo-search package is not installed. Please install it with 'pip install duckduckgo-search'."
        
        try:
            # Extract parameters
            query = input_str
            region = "wt-wt"
            safesearch = "moderate"
            timelimit = None
            resolution = None
            duration = None
            max_results = 5
            
            if args:
                if "query" in args:
                    query = args["query"]
                if "region" in args:
                    region = args["region"]
                if "safesearch" in args:
                    safesearch = args["safesearch"]
                if "timelimit" in args:
                    timelimit = args["timelimit"]
                if "resolution" in args:
                    resolution = args["resolution"]
                if "duration" in args:
                    duration = args["duration"]
                if "max_results" in args:
                    max_results = int(args["max_results"])
            
            if not query:
                return "Error: No search query provided. Please provide a search term."
            
            # Initialize the DDGS client with proxy settings if provided
            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify, headers=self.headers)
            
            # Perform the search
            results = ddgs.videos(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                resolution=resolution,
                duration=duration,
                max_results=max_results
            )
            
            # Format results
            if not results:
                return f"No video results found for query: '{query}'"
            
            formatted_results = []
            for idx, result in enumerate(results, 1):
                # Extract and format publishing date if available
                published = result.get('published', 'No date')
                if published and published != 'No date':
                    try:
                        # Convert to more readable format if possible
                        from datetime import datetime
                        dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                        published = dt.strftime('%Y-%m-%d')
                    except:
                        # Keep original if parsing fails
                        pass
                
                # Get view count if available
                view_count = "N/A"
                if 'statistics' in result and 'viewCount' in result['statistics']:
                    view_count = f"{result['statistics']['viewCount']:,}"
                
                formatted_result = (
                    f"{idx}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('content', 'No URL')}\n"
                    f"   Duration: {result.get('duration', 'N/A')}\n"
                    f"   Published: {published}\n"
                    f"   Publisher: {result.get('publisher', 'Unknown')}\n"
                    f"   Views: {view_count}\n"
                    f"   {result.get('description', 'No description')[:200]}...\n"
                )
                formatted_results.append(formatted_result)
            
            # Combine and return results
            output = f"Video search results for: '{query}'\n\n"
            output += "\n".join(formatted_results)
            return output
            
        except RatelimitException:
            return "Error: DuckDuckGo search rate limit reached. Please try again later."
        except TimeoutException:
            return "Error: DuckDuckGo search timed out. Please try again or simplify your query."
        except DuckDuckGoSearchException as e:
            return f"Error: DuckDuckGo search failed: {str(e)}"
        except Exception as e:
            return f"Error performing DuckDuckGo video search: {str(e)}"


class DuckDuckGoNewsSearchTool(Tool):
    """Tool for searching news using DuckDuckGo."""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 10, verify: bool = True, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the DuckDuckGo news search tool.
        
        Args:
            proxy: Optional proxy for HTTP client (e.g. "tb" for Tor Browser, 
                  or "http://user:pass@example.com:3128")
            timeout: Timeout value for HTTP client in seconds
            verify: SSL verification when making requests
            headers: Optional dictionary of headers for the HTTP client (e.g. custom User-Agent)
        """
        super().__init__()
        self.proxy = proxy
        self.timeout = timeout
        self.verify = verify
        self.headers = headers
    
    @property
    def name(self) -> str:
        return "DuckDuckGoNewsSearch"
    
    @property
    def description(self) -> str:
        return "Search for recent news articles using DuckDuckGo's news search capabilities."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The search query to find news about",
                type="string",
                required=True
            ),
            ToolParameter(
                name="region",
                description="Region code for regional search results (e.g., 'wt-wt', 'us-en', 'uk-en')",
                type="string",
                required=False
            ),
            ToolParameter(
                name="safesearch",
                description="Safety level for results: 'on', 'moderate', or 'off'",
                type="string",
                required=False
            ),
            ToolParameter(
                name="timelimit",
                description="Time limit for results: 'd' (day), 'w' (week), 'm' (month)",
                type="string",
                required=False
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results to return (default: 5)",
                type="integer",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: DuckDuckGoNewsSearch {"query": "artificial intelligence", "timelimit": "w", "max_results": 5}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: DuckDuckGoNewsSearch {"query": "artificial intelligence", "timelimit": "w", "max_results": 5}',
            'TOOL REQUEST: DuckDuckGoNewsSearch {"query": "climate summit", "timelimit": "d", "max_results": 3}',
            'TOOL REQUEST: DuckDuckGoNewsSearch {"query": "economic forecast", "region": "us-en", "max_results": 5}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Run a news search using DuckDuckGo.
        
        Args:
            input_str: The search query as a string (used if args not provided)
            args: Dictionary with query and optional parameters
        
        Returns:
            String with news search results or error message
        """
        if not DDGS_AVAILABLE:
            return "Error: The duckduckgo-search package is not installed. Please install it with 'pip install duckduckgo-search'."
        
        try:
            # Extract parameters
            query = input_str
            region = "wt-wt"
            safesearch = "moderate"
            timelimit = None
            max_results = 5
            
            if args:
                if "query" in args:
                    query = args["query"]
                if "region" in args:
                    region = args["region"]
                if "safesearch" in args:
                    safesearch = args["safesearch"]
                if "timelimit" in args:
                    timelimit = args["timelimit"]
                if "max_results" in args:
                    max_results = int(args["max_results"])
            
            if not query:
                return "Error: No search query provided. Please provide a search term."
            
            # Initialize the DDGS client with proxy settings if provided
            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify, headers=self.headers)
            
            # Perform the search
            results = ddgs.news(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results
            )
            
            # Format results
            if not results:
                return f"No news results found for query: '{query}'"
            
            formatted_results = []
            for idx, result in enumerate(results, 1):
                date = result.get('date', 'No date')
                if date and date != 'No date':
                    # Format date if it exists and is not the placeholder
                    try:
                        # Convert to more readable format if possible
                        from datetime import datetime
                        dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
                        date = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
                    except:
                        # Keep original if parsing fails
                        pass
                
                formatted_result = (
                    f"{idx}. {result.get('title', 'No title')}\n"
                    f"   Date: {date}\n"
                    f"   Source: {result.get('source', 'Unknown source')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                    f"   {result.get('body', 'No description')}\n"
                )
                formatted_results.append(formatted_result)
            
            # Combine and return results
            output = f"News search results for: '{query}'\n\n"
            output += "\n".join(formatted_results)
            return output
            
        except RatelimitException:
            return "Error: DuckDuckGo search rate limit reached. Please try again later."
        except TimeoutException:
            return "Error: DuckDuckGo search timed out. Please try again or simplify your query."
        except DuckDuckGoSearchException as e:
            return f"Error: DuckDuckGo search failed: {str(e)}"
        except Exception as e:
            return f"Error performing DuckDuckGo news search: {str(e)}"


class DuckDuckGoChatTool(Tool):
    """Tool for using DuckDuckGo's AI chat capabilities."""
    
    def __init__(self, proxy: Optional[str] = None, timeout: int = 30, verify: bool = True, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the DuckDuckGo chat tool.
        
        Args:
            proxy: Optional proxy for HTTP client (e.g. "tb" for Tor Browser, 
                  or "http://user:pass@example.com:3128")
            timeout: Timeout value for HTTP client in seconds (defaults to 30 for chat)
            verify: SSL verification when making requests
            headers: Optional dictionary of headers for the HTTP client (e.g. custom User-Agent)
        """
        super().__init__()
        self.proxy = proxy
        self.timeout = timeout  # Use longer timeout for chat by default
        self.verify = verify
        self.headers = headers
    
    @property
    def name(self) -> str:
        return "DuckDuckGoChat"
    
    @property
    def description(self) -> str:
        return "Ask questions to DuckDuckGo's AI assistant, which can provide information and summaries."
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description="The question or prompt to send to the AI assistant",
                type="string",
                required=True
            ),
            ToolParameter(
                name="model",
                description="AI model to use: 'gpt-4o-mini', 'llama-3.3-70b', 'claude-3-haiku', 'o3-mini', 'mistral-small-3'",
                type="string",
                required=False
            )
        ]
    
    @property
    def docs(self) -> ToolDocumentation:
        return ToolDocumentation(
            name=self.name,
            description=self.description,
            usage_example='TOOL REQUEST: DuckDuckGoChat {"query": "Explain quantum computing in simple terms", "model": "gpt-4o-mini"}',
            parameters=self.parameters
        )
    
    @property
    def usage_examples(self) -> List[str]:
        return [
            'TOOL REQUEST: DuckDuckGoChat {"query": "Explain quantum computing in simple terms", "model": "gpt-4o-mini"}',
            'TOOL REQUEST: DuckDuckGoChat {"query": "What are the main causes of climate change?", "model": "claude-3-haiku"}',
            'TOOL REQUEST: DuckDuckGoChat {"query": "Summarize the history of artificial intelligence"}'
        ]
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask a question to DuckDuckGo's AI assistant.
        
        Args:
            input_str: The question as a string (used if args not provided)
            args: Dictionary with query and optional model parameter
        
        Returns:
            String with AI assistant's response or error message
        """
        if not DDGS_AVAILABLE:
            return "Error: The duckduckgo-search package is not installed. Please install it with 'pip install duckduckgo-search'."
        
        try:
            # Extract parameters
            query = input_str
            model = "gpt-4o-mini"  # Default model
            
            if args:
                if "query" in args:
                    query = args["query"]
                if "model" in args:
                    model = args["model"]
            
            if not query:
                return "Error: No question provided. Please provide a question for the AI assistant."
            
            # Validate model
            valid_models = ["gpt-4o-mini", "llama-3.3-70b", "claude-3-haiku", "o3-mini", "mistral-small-3"]
            if model not in valid_models:
                model = "gpt-4o-mini"  # Fallback to default if invalid
            
            # Initialize the DDGS client with proxy settings if provided
            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify, headers=self.headers)
            
            # Get response from AI
            response = ddgs.chat(keywords=query, model=model, timeout=self.timeout)
            
            if not response:
                return f"No response received from DuckDuckGo AI for query: '{query}'"
            
            # Return formatted response
            output = f"DuckDuckGo AI ({model}) response:\n\n{response}"
            return output
            
        except RatelimitException:
            return "Error: DuckDuckGo AI chat rate limit reached. Please try again later."
        except TimeoutException:
            return "Error: DuckDuckGo AI chat timed out. Please try again or simplify your query."
        except DuckDuckGoSearchException as e:
            return f"Error: DuckDuckGo AI chat failed: {str(e)}"
        except Exception as e:
            return f"Error using DuckDuckGo AI chat: {str(e)}"


class DuckDuckGoToolkit:
    """
    Toolkit for DuckDuckGo search tools.
    
    This toolkit provides various tools for searching the web, images, videos, news,
    and interacting with DuckDuckGo's AI chat capabilities.
    """
    
    def __init__(self, 
                 enable_text_search: bool = True,
                 enable_image_search: bool = True,
                 enable_video_search: bool = True,
                 enable_news_search: bool = True,
                 enable_chat: bool = True,
                 proxy: Optional[str] = None,
                 timeout: int = 10,
                 verify: bool = True,
                 headers: Optional[Dict[str, str]] = None):
        """
        Initialize the DuckDuckGo toolkit with selected tools.
        
        Args:
            enable_text_search: Whether to enable the text search tool
            enable_image_search: Whether to enable the image search tool
            enable_video_search: Whether to enable the video search tool
            enable_news_search: Whether to enable the news search tool
            enable_chat: Whether to enable the AI chat tool
            proxy: Optional proxy for HTTP client (e.g. "tb" for Tor Browser, 
                  or "http://user:pass@example.com:3128")
            timeout: Timeout value for HTTP client in seconds
            verify: SSL verification when making requests
            headers: Optional dictionary of headers for the HTTP client (e.g. custom User-Agent)
        """
        self.enable_text_search = enable_text_search
        self.enable_image_search = enable_image_search
        self.enable_video_search = enable_video_search
        self.enable_news_search = enable_news_search
        self.enable_chat = enable_chat
        self.proxy = proxy
        self.timeout = timeout
        self.verify = verify
        self.headers = headers
        
        # Install warning message
        if not DDGS_AVAILABLE:
            import warnings
            warnings.warn(
                "DuckDuckGo Search package not found. Please install with: pip install duckduckgo-search"
            )
    
    def get_tools(self) -> List[Tool]:
        """
        Get all enabled DuckDuckGo search tools.
        
        Returns:
            List of enabled DuckDuckGo search tools
        """
        tools = []
        
        if self.enable_text_search:
            tools.append(DuckDuckGoTextSearchTool(
                proxy=self.proxy, 
                timeout=self.timeout, 
                verify=self.verify,
                headers=self.headers
            ))
        
        if self.enable_image_search:
            tools.append(DuckDuckGoImageSearchTool(
                proxy=self.proxy, 
                timeout=self.timeout, 
                verify=self.verify,
                headers=self.headers
            ))
        
        if self.enable_video_search:
            tools.append(DuckDuckGoVideoSearchTool(
                proxy=self.proxy, 
                timeout=self.timeout, 
                verify=self.verify,
                headers=self.headers
            ))
        
        if self.enable_news_search:
            tools.append(DuckDuckGoNewsSearchTool(
                proxy=self.proxy, 
                timeout=self.timeout, 
                verify=self.verify,
                headers=self.headers
            ))
        
        if self.enable_chat:
            tools.append(DuckDuckGoChatTool(
                proxy=self.proxy, 
                timeout=self.timeout, 
                verify=self.verify,
                headers=self.headers
            ))
        
        return tools 