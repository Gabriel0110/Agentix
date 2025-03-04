#!/usr/bin/env python
"""
Example showcasing the AgentPromptBuilder for automated prompt generation.

This example demonstrates how to use the AgentPromptBuilder to automatically
generate effective system prompts for agents with different tools and purposes.
"""

import os
import asyncio
from dotenv import load_dotenv

from agentix import (
    OpenAIChat,
    Agent,
    ShortTermMemory,
    AgentOptions
)
from agentix.tools import YFinanceToolkit
from agentix.agents.prompt_builder import AgentPromptBuilder


async def main():
    """
    Example demonstrating the AgentPromptBuilder for automated prompt generation.
    """
    # Load environment variables
    load_dotenv()
    
    # 1) Create a model for the prompt builder
    builder_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",  # Use a strong model for prompt generation
        temperature=0.2  # Lower temperature for more predictable outputs
    )
    
    # 2) Initialize the prompt builder
    prompt_builder = AgentPromptBuilder(model=builder_model)
    
    # 3) Get some tools to use in our example
    yfinance_toolkit = YFinanceToolkit(
        enable_stock_price=True,
        enable_company_info=True,
        enable_historical_prices=True
    )
    finance_tools = yfinance_toolkit.get_tools()
    
    # 4) Generate a prompt for a stock analysis agent
    print("\nGenerating a prompt for a stock analysis agent...\n")
    stock_agent_prompt = await prompt_builder.preview_prompt(
        agent_name="MarketInsightGPT",
        tools=finance_tools,
        task_description=(
            "Help users analyze stocks and make informed investment decisions "
            "by providing current stock prices, company information, and historical performance data."
        ),
        additional_instructions=[
            "Use clear, jargon-free explanations suitable for both novice and experienced investors.",
            "When analyzing historical data, focus on meaningful patterns rather than just raw numbers.",
            "Always consider the broader market context when providing stock analysis."
        ],
        output_file="generated_stock_agent_prompt",
        output_format="json"
    )
    
    # 5) Create an agent using the generated prompt
    print("\nCreating an agent with the generated prompt...\n")
    
    chat_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    memory = ShortTermMemory(max_messages=20)
    
    agent_options = AgentOptions(
        max_steps=5,
        usage_limit=10,
        debug=True
    )
    
    # Split the generated prompt into instruction lines
    instruction_lines = [line.strip() for line in stock_agent_prompt.split('\n') if line.strip()]
    
    agent = Agent.create(
        name="MarketInsightGPT",
        model=chat_model,
        memory=memory,
        tools=finance_tools,
        instructions=instruction_lines,
        options=agent_options
    )
    
    # 6) Use the agent with a sample query
    print("\nTesting the agent with a sample query...\n")
    query = "What's the current stock price of Microsoft and how has it performed over the last week?"
    print(f"User Query: {query}")
    
    answer = await agent.run(query)
    print(f"\nAgent Response:\n{answer}\n")
    
    # 7) Generate a prompt with different tools/purpose to show flexibility
    print("\nGenerating a prompt for an agent with different tools/purpose...\n")
    # In a real scenario, you would use different tools here
    # For example purposes, we're using the same tools but with a different task
    
    await prompt_builder.preview_prompt(
        agent_name="InvestmentAdvisorGPT",
        tools=finance_tools,
        task_description=(
            "Provide personalized investment advice and portfolio recommendations "
            "based on financial data and user investment goals."
        ),
        additional_instructions=[
            "Always emphasize that you are providing educational information, not financial advice.",
            "Consider user's risk tolerance in your recommendations.",
            "Explain investment concepts in simple terms for beginners.",
            "Focus on long-term investment strategies rather than short-term market timing."
        ],
        output_file="generated_investment_advisor_prompt",
        output_format="json"
    )
    
    print("\nPrompt generation example completed.\n")
    print(f"The generated prompts have been saved to the current directory.")


if __name__ == "__main__":
    asyncio.run(main()) 