#!/usr/bin/env python
"""
DuckDuckGo search agent example.

This example demonstrates how to use the DuckDuckGo search tools in an Agentix agent.
The agent can search the web, find images, get news, use DuckDuckGo's AI chat, and search for videos.
It also shows how to use proxy settings and custom headers for enhanced privacy or
accessing from restricted networks.
"""

import os
import sys
import asyncio

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import DuckDuckGoToolkit

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()

async def main():
    """
    Example demonstrating a web search agent with DuckDuckGo tools.
    """
    
    # 1) Create a chat model with appropriate settings
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Using a capable model for search tasks
        temperature=0.2,      # Lower temperature for more factual responses
        #stream=True,          # Stream output to console
        #on_token=on_token     # Hook to process tokens
    )

    # 2) Create a memory system
    memory = ShortTermMemory(max_messages=20)
    
    # 3) Set up the DuckDuckGo toolkit with all tools enabled
    # Uncomment proxy settings if needed:
    # For Tor Browser: proxy="tb", timeout=20
    # For custom proxy: proxy="socks5h://user:pass@proxy.example.com:1234", timeout=15
    
    # Example custom headers (uncomment to use)
    # custom_headers = {
    #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    #     "Accept-Language": "en-US,en;q=0.9",
    #     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    # }
    
    ddg_toolkit = DuckDuckGoToolkit(
        enable_text_search=True,
        enable_image_search=False,
        enable_video_search=False,
        enable_news_search=True,
        enable_chat=False,
        # proxy="tb",                    # Optional: Use Tor Browser as proxy
        # proxy=os.environ.get("DDGS_PROXY"), # Or read from env variable
        # timeout=20,                    # Increase timeout for proxy connections
        # verify=True,                   # SSL verification (set to False only if necessary)
        # headers=custom_headers         # Optional: Set custom HTTP headers
    )
    
    # 4) Create agent options
    agent_options = AgentOptions(
        max_steps=10,         # Allow up to 10 steps before requiring final answer
        usage_limit=15,       # Allow up to 15 tool calls
        time_to_live=120000,  # 2 minute timeout
        debug=True            # Enable debug mode for more verbose output
    )
    
    # 5) Create the agent with instructions for DuckDuckGo search
    agent_instructions = [
        "You are a helpful web search assistant that can find information, news, images, videos, and more.",
        "You have access to DuckDuckGo search tools to help answer user questions.",
        "For each question, think about which search tool would be most appropriate:",
        "- Use DuckDuckGoTextSearch for general information queries",
        "- Use DuckDuckGoNewsSearch for recent news and events",
        #"- Use DuckDuckGoImageSearch when the user wants to find images",
        #"- Use DuckDuckGoVideoSearch when the user wants to find videos",
        #"- Use DuckDuckGoChat to get AI-generated summaries or explanations",
        "",
        "IMPORTANT SEARCH TIPS:",
        "- Be specific with search terms to get better results",
        "- Use search operators like filetype:pdf, site:example.com when appropriate",
        "- For complex queries, break them down into multiple simple searches",
        "- When quoting sources, always include the URL from search results",
        "- Limit search results to 3-5 entries to avoid overwhelming responses",
        "",
        "Always provide a final answer based on search results, with proper citations."
    ]
    
    # Create the agent
    agent = Agent.create(
        name="Web Search Assistant",
        model=chat_model,
        memory=memory,
        tools=ddg_toolkit.get_tools(),
        instructions=agent_instructions,
        options=agent_options
    )
    
    # Example questions to test
    example_questions = [
        "What are the latest developments in quantum computing?",
        #"Show me information about electric vehicles and their environmental impact.",
        #"Find recent news about climate change policies.",
        #"Search for images of aurora borealis. What causes this phenomenon?",
        #"Find videos about machine learning tutorials.",
        #"What are the main arguments for and against artificial general intelligence development?"
    ]
    
    print("\n" + "="*80)
    print("DuckDuckGo Search Agent Example")
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