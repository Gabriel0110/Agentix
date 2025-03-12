"""
Example demonstrating how to use function-based tools in Agentix.

This example shows:
1. Converting sync and async functions to tools
2. Automatic parameter extraction from type hints and docstrings
3. Using function tools with an agent
"""

import os
import asyncio
from typing import Dict, List, Optional

from agentix.agents import Agent
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import function_tool

# Example 1: Simple synchronous function tool
@function_tool(
    name="GetWeather",
    description="Get current weather for a city",
    usage_example='TOOL REQUEST: GetWeather {"city": "San Francisco"}'
)
def get_weather(city: str) -> str:
    """
    Get the current weather for the specified city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the current weather
    """
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny"

# Example 2: Async function with multiple parameters
@function_tool(
    name="SearchDatabase",
    description="Search a database for records"
)
async def search_database(
    query: str,
    limit: Optional[int] = 5,
    category: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Search the database for matching records.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        category: Optional category to filter by
    
    Returns:
        List of matching records
    """
    # Simulate async database query
    await asyncio.sleep(0.5)
    
    # Mock results
    results = [
        {"id": "1", "title": f"Result {i} for {query}", "category": category or "general"}
        for i in range(limit)
    ]
    
    return results

# Example 3: Function with complex return type
@function_tool(
    name="ProcessData",
    description="Process data and return statistics"
)
def process_data(data: List[float], include_advanced: bool = False) -> Dict[str, float]:
    """
    Calculate statistics for a list of numbers.
    
    Args:
        data: List of numbers to process
        include_advanced: Whether to include advanced statistics
        
    Returns:
        Dictionary of calculated statistics
    """
    import statistics
    
    stats = {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "std_dev": statistics.stdev(data) if len(data) > 1 else 0
    }
    
    if include_advanced:
        stats.update({
            "variance": statistics.variance(data) if len(data) > 1 else 0,
            "mode": statistics.mode(data) if data else None
        })
    
    return stats

async def main():
    print("🔧 Function Tools Demo")
    print("=" * 50)
    
    # Initialize the agent
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.8
    )
    
    memory = ShortTermMemory(max_messages=10)
    
    # Create agent with our function tools
    agent = Agent.create(
        name="FunctionToolAgent",
        model=chat_model,
        memory=memory,
        tools=[
            get_weather,  # Note: decorator returns a Tool instance
            search_database,
            process_data
        ],
        instructions=[
            "You are a helpful assistant with access to various function-based tools.",
            "Use the tools to help answer user queries.",
            "Always analyze tool results before making additional tool calls."
        ]
    )
    
    # Example 1: Weather query
    print("\n🌤️ Weather Query Example")
    print("-" * 30)
    response = await agent.run("What's the weather like in Tokyo?")
    print(f"Response: {response}")
    
    # Example 2: Database search
    print("\n🔍 Database Search Example")
    print("-" * 30)
    response = await agent.run(
        "Search the database for 'machine learning' articles, limit to 3 results in the 'tech' category"
    )
    print(f"Response: {response}")
    
    # Example 3: Data processing
    print("\n📊 Data Processing Example")
    print("-" * 30)
    response = await agent.run(
        "Process this data with advanced statistics: [1.5, 2.5, 3.5, 4.5, 5.5]"
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 