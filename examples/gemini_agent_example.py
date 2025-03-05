#!/usr/bin/env python
"""
Google Gemini model agent example.

This example demonstrates how to use Google's Gemini models with an Agentix agent.
The example supports both text-only and multimodal (text + images) inputs.
"""

import os
import sys
import asyncio

from agentix.llms import GeminiChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import TavilyToolkit

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def run_text_only_example():
    """Run a simple text-only example with Gemini."""
    
    print("\n" + "="*80)
    print("Gemini Text-Only Example")
    print("="*80 + "\n")
    
    # 1) Create a Gemini model with appropriate settings
    gemini_model = GeminiChat(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model="gemini-2.0-flash",     # Use flash for faster responses
        temperature=0.2,              # Lower temperature for more factual responses
        debug=True,                    # Enable debug logging
        #max_output_tokens=1024,       # Limit the output length
        #stream=True,                  # Stream output to console
        #on_token=on_token,            # Hook to process tokens
        #system_instruction="You are a helpful and knowledgeable AI assistant that provides accurate and concise information.",
    )

    # 2) Create a memory system
    memory = ShortTermMemory(max_messages=10)
    
    # 3) Set up the Tavily toolkit for search capabilities
    search_toolkit = TavilyToolkit(
        api_key=os.environ.get("TAVILY_API_KEY"),
        enable_search=True,
        enable_extract=True
    )
    
    # 4) Create agent options
    agent_options = AgentOptions(
        max_steps=5,          # Allow up to 5 steps before requiring final answer
        usage_limit=10,       # Allow up to 10 tool calls
        time_to_live=60000,   # 1 minute timeout
        debug=True            # Enable debug mode for more verbose output
    )
    
    # 5) Create the agent
    agent = Agent.create(
        name="Gemini Agent",
        model=gemini_model,
        memory=memory,
        tools=search_toolkit.get_tools(),
        instructions=[
            "You are a helpful assistant powered by Google's Gemini model.",
            "When answering questions, be concise, accurate, and helpful.",
            "You have access to search tools to help you find up-to-date information."
        ],
        options=agent_options
    )
    
    # Example question
    user_question = "What are some recent advancements in renewable energy technology?"
    
    print(f"User: {user_question}\n")
    print("Assistant: ", end="")
    
    # Run the agent with the question
    response = await agent.run(user_question)
    
    # Print the final response if not streaming
    if not gemini_model.stream:
        print(response)
    
    print("\n")


async def run_multimodal_example():
    """Run a multimodal example with Gemini that analyzes images."""
    
    # Check if the example image exists
    image_path = "examples/data/solar_panels.jpg"
    if not os.path.exists(image_path):
        print(f"Warning: Example image not found at {image_path}")
        print("Skipping multimodal example.\n")
        return
    
    print("\n" + "="*80)
    print("Gemini Multimodal Example")
    print("="*80 + "\n")
    
    # Create a Gemini model with appropriate settings
    gemini_model = GeminiChat(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        model="gemini-2.0-pro",       # Use pro model for multimodal capabilities
        temperature=0.2,              # Lower temperature for more factual responses
        max_output_tokens=1024,       # Limit the output length
        stream=True,                  # Stream output to console
        on_token=on_token,            # Hook to process tokens
        debug=True                    # Enable debug logging
    )
    
    # Format the messages including the image
    print(f"Analyzing image: {image_path}\n")
    print("Assistant: ", end="")
    
    # Use the chat_with_images method directly
    response = await gemini_model.chat_with_images(
        text_prompt="Describe this image in detail and explain how it relates to renewable energy.",
        image_paths=[image_path],
        system_instruction="You are a visual analysis expert who can describe images in detail and explain their significance."
    )
    
    # Print the final response if not streaming
    if not gemini_model.stream:
        print(response)
    
    print("\n")


async def main():
    """Run all Gemini examples."""
    
    # Check if the API key is available
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set your Google API key in the .env file or environment variables.")
        return
    
    # Run the text-only example
    await run_text_only_example()
    
    # Run the multimodal example
    await run_multimodal_example()
    
    print("="*80)
    print("Examples completed.")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main()) 