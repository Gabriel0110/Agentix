#!/usr/bin/env python
"""
Firecrawl web scraping and crawling agent example.

This example demonstrates how to use the Firecrawl tools in an Agentix agent.
The agent can scrape individual URLs, crawl entire websites, generate site maps,
and manage asynchronous crawling jobs using the Firecrawl API.
"""

import os
import sys
import asyncio

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import FirecrawlToolkit

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def main():
    """
    Example demonstrating a web scraping and crawling agent with Firecrawl tools.
    """
    
    # 1) Create a chat model with appropriate settings
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Using a capable model for web scraping tasks
        temperature=0.2,      # Lower temperature for more factual responses
        #stream=True,          # Stream output to console
        #on_token=on_token     # Hook to process tokens
    )

    # 2) Create a memory system
    memory = ShortTermMemory(max_messages=20)
    
    # 3) Set up the Firecrawl toolkit with all tools enabled
    firecrawl_toolkit = FirecrawlToolkit(
        api_key=os.environ.get("FIRECRAWL_API_KEY"),
        enable_scrape=True,
        enable_crawl=True,
        enable_status=True,
        enable_map=True
    )
    
    # 4) Create agent options
    agent_options = AgentOptions(
        max_steps=10,         # Allow up to 10 steps before requiring final answer
        usage_limit=15,       # Allow up to 15 tool calls
        time_to_live=120000,  # 2 minute timeout
        debug=True            # Enable debug mode for more verbose output
    )
    
    # 5) Create the agent with instructions for Firecrawl tools
    agent_instructions = [
        "You are a web scraping and crawling assistant that can extract content from websites.",
        "You have access to Firecrawl tools to help answer user questions.",
        "BEST PRACTICES:",
        "- For single pages, use FirecrawlScrape",
        "- For entire sites, use FirecrawlCrawl",
        "- For site structure analysis, use FirecrawlMap",
        "- For large sites, use async crawling and FirecrawlStatus",
        "- Always check rate limits and respect robots.txt",
        "- Provide clear progress updates for long-running tasks",
        "",
        "Always structure your responses with clear sections and cite the sources of extracted content."
    ]
    
    # Create the agent
    agent = Agent.create(
        name="Web Scraping Assistant",
        model=chat_model,
        memory=memory,
        tools=firecrawl_toolkit.get_tools(),
        instructions=agent_instructions,
        options=agent_options
    )
    
    # Example questions to test
    example_questions = [
        "Scrape the content from https://example.com and format it in markdown",
        "Crawl https://docs.python.org/3/ but limit to 10 pages and exclude the /download/ section",
        "Generate a site map for https://httpbin.org/ including subdomains",
        "Start an async crawl of https://quotes.toscrape.com/ and monitor its progress",
        "Extract all blog posts from https://blog.python.org/ in the last month"
    ]
    
    print("\n" + "="*80)
    print("Firecrawl Web Scraping Agent Example")
    print("="*80 + "\n")
    
    # Choose a question to test 
    # Uncomment the question you want to test or add your own
    user_question = example_questions[0]
    # user_question = "Your custom question here"
    
    print(f"User: {user_question}\n")
    print("Assistant: ", end="")
    
    # Run the agent with the selected question
    response = await agent.run(user_question)
    
    # Print the final response if not streaming
    if not chat_model.stream:
        print(response)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main()) 