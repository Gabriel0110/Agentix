#!/usr/bin/env python
"""
Tavily web search and extraction agent example.

This example demonstrates how to use the Tavily API tools in an Agentix agent.
The agent can search the web for information and extract content from web pages,
providing comprehensive and accurate responses based on real-time web data.
"""

import os
import sys
import asyncio

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import TavilyToolkit

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def main():
    """
    Example demonstrating a web search and extraction agent with Tavily tools.
    """
    
    # 1) Create a chat model with appropriate settings
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Using a capable model for search and content tasks
        temperature=0.2,      # Lower temperature for more factual responses
        #stream=True,          # Stream output to console
        #on_token=on_token     # Hook to process tokens
    )

    # 2) Create a memory system
    memory = ShortTermMemory(max_messages=20)
    
    # 3) Set up the Tavily toolkit with all tools enabled
    tavily_toolkit = TavilyToolkit(
        api_key=os.environ.get("TAVILY_API_KEY"),
        enable_search=True,
        enable_extract=True
    )
    
    # 4) Create agent options
    agent_options = AgentOptions(
        max_steps=10,         # Allow up to 10 steps before requiring final answer
        usage_limit=15,       # Allow up to 15 tool calls
        time_to_live=120000,  # 2 minute timeout
        debug=True            # Enable debug mode for more verbose output
    )
    
    # 5) Create the agent with instructions for Tavily tools
    agent_instructions = [
        "You are a helpful web research assistant that can find information and extract content from websites.",
        "You have access to Tavily tools to help answer user questions:",
        "1. TavilySearch - Use this to search the web for information",
        "   - You can search for 'general' topics or specifically request 'news'",
        "   - You can request different search depths: 'basic' (faster) or 'advanced' (more thorough)",
        "   - You can include or exclude specific domains",
        "   - You can limit results by time range: 'day', 'week', 'month', 'year'",
        "   - You can request images by setting 'include_images': true",
        "   - You can request an AI-generated answer with 'include_answer': 'advanced'",
        "",
        "2. TavilyExtract - Use this to extract content from specific web pages",
        "   - Provide one or more URLs to extract content from",
        "   - You can request different extraction depths: 'basic' or 'advanced'",
        "   - You can include images in the extraction with 'include_images': true",
        "",
        "SEARCH WORKFLOW TIPS:",
        "- For general questions, start with TavilySearch",
        "- If you find interesting links that need more detail, use TavilyExtract on those URLs",
        "- For complex research, combine both tools: search first, then extract details",
        "- Always cite your sources by including URLs",
        "- For news queries, specifically set 'topic': 'news' and use appropriate time ranges",
        "",
        "Always provide a well-structured, comprehensive final answer based on the information you find."
    ]
    
    # Create the agent
    agent = Agent.create(
        name="Web Research Assistant",
        model=chat_model,
        memory=memory,
        tools=tavily_toolkit.get_tools(),
        instructions=agent_instructions,
        options=agent_options
    )
    
    # Example questions to test
    example_questions = [
        "What are the latest developments in quantum computing? Give me a comprehensive overview.",
        "Compare and contrast the environmental impacts of electric vehicles versus traditional combustion engine vehicles.",
        "Find recent news about climate change policies in the EU and summarize the key points.",
        "What are the top recommendations for cybersecurity best practices in 2023?",
        "Research the impact of artificial intelligence on healthcare and provide examples of current applications.",
        "Explain how the James Webb Space Telescope works and what discoveries it has made so far."
    ]
    
    print("\n" + "="*80)
    print("Tavily Web Research Agent Example")
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