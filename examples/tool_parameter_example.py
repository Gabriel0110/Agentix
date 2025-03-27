#!/usr/bin/env python
"""
tool_parameter_example.py

Demonstrates how to use tools with parameters in Agentix.
This example shows a search tool that accepts parameters like filters and limits.
"""
import os
import asyncio
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

from agentix.agents import Agent, AgentHooks
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import Tool

from dotenv import load_dotenv
load_dotenv()

@dataclass
class SearchParams:
    """Parameters for the search tool."""
    query: str
    limit: Optional[int] = 5
    filter: Optional[str] = None
    sort_by: Optional[str] = "relevance"


class ParameterizedSearchTool(Tool):
    """A search tool that accepts structured parameters."""
    
    name = "search"
    description = "Search for information with optional parameters for filtering and sorting"
    
    # Define the parameter schema that LLM will see
    parameter_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)"
            },
            "filter": {
                "type": "string",
                "description": "Optional filter criteria (e.g., 'recent', 'verified')"
            },
            "sort_by": {
                "type": "string",
                "description": "How to sort results (e.g., 'relevance', 'date')"
            }
        },
        "required": ["query"]
    }
    
    async def run(self, params: Dict[str, Any]) -> str:
        """
        Run the search tool with the provided parameters.
        
        Args:
            params: Dictionary containing search parameters
            
        Returns:
            String containing search results
        """
        # Convert the parameters dictionary to our dataclass for type safety
        search_params = SearchParams(
            query=params.get("query", ""),
            limit=params.get("limit", 5),
            filter=params.get("filter"),
            sort_by=params.get("sort_by", "relevance")
        )
        
        # Print what we received for debugging
        print(f"Search called with parameters: {search_params}")
        
        # In a real implementation, this would query a search API
        # For this example, we'll return fake results
        results = [
            {"title": f"Result {i} for '{search_params.query}'", "snippet": f"This is result {i} content..."}
            for i in range(1, min(search_params.limit + 1, 10))
        ]
        
        if search_params.filter:
            # Apply simple filtering (in a real implementation, this would be more sophisticated)
            results = [r for r in results if search_params.filter.lower() in r["title"].lower()]
            
        if search_params.sort_by == "date":
            # Sort by "date" (simulated by reversing the list)
            results.reverse()
        
        # Format results as a nice string
        formatted_results = "\n\n".join([
            f"📄 {r['title']}\n{r['snippet']}"
            for r in results
        ])
        
        return f"Search results for '{search_params.query}':\n\n{formatted_results}"


async def main():
    """Main function demonstrating the parameterized tool."""
    # Create the chat model
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.2
    )
    
    # Create memory
    memory = ShortTermMemory(max_messages=10)
    
    # Create our search tool
    search_tool = ParameterizedSearchTool()

    # Create hooks
    hooks = AgentHooks(
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end
    )
    
    # Create the agent with the search tool
    agent = Agent.create(
        name="ToolParameterAgent",
        model=chat_model,
        memory=memory,
        tools=[search_tool],
        instructions=[
            "You are a helpful assistant with access to a search tool.",
            "The search tool accepts parameters for customizing the search.",
            "Always try to use appropriate parameters when searching."
        ],
        hooks=hooks
    )
    
    # Set up hooks to see what's happening
    def on_tool_start(tool_name: str, params: Dict[str, Any]):
        print(f"\nStarting tool: {tool_name}")
        print(f"Parameters: {json.dumps(params, indent=2)}\n")
    
    def on_tool_end(tool_name: str, result: str):
        print(f"\nTool {tool_name} completed")
        print(f"Result: {result[:100]}...\n")
    
    # Example 1: Basic search
    print("\n\n=== Example 1: Basic search ===")
    await agent.run("Search for information about artificial intelligence")
    
    # Example 2: Search with parameters
    print("\n\n=== Example 2: Search with parameters ===")
    await agent.run("Search for recent news about climate change, but limit to 3 results and sort by date")
    
    # Example 3: Complex query with filtering
    print("\n\n=== Example 3: Search with filtering ===")
    await agent.run("I need verified information about renewable energy, can you search for me?")


if __name__ == "__main__":
    asyncio.run(main()) 