#!/usr/bin/env python
"""
Example showcasing agent creation with auto-generated prompts.

This example demonstrates how to create an agent with automatically generated
instructions using the built-in create_with_generated_prompt method.
"""

import os
import asyncio
from dotenv import load_dotenv

from agentix.agents import Agent, AgentOptions, AgentPromptBuilder
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import YFinanceToolkit


async def main():
    """
    Example demonstrating agent creation with auto-generated prompts.
    """
    # Load environment variables
    load_dotenv()
    
    # 1) Create LLM models
    agent_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    # This model will be used just for prompt generation
    prompt_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",  # Use a stronger model for prompt generation
        temperature=0.2
    )
    
    # 2) Set up tools
    yfinance_toolkit = YFinanceToolkit(
        enable_stock_price=True,
        enable_company_info=True,
        enable_historical_prices=True,
        enable_fundamentals=True
    )
    finance_tools = yfinance_toolkit.get_tools()
    
    # 3) Create memory and options
    memory = ShortTermMemory(max_messages=20)
    
    agent_options = AgentOptions(
        max_steps=5,
        usage_limit=10,
        debug=True
    )
    
    # 3.5) Optionally save the prompt to a JSON file first
    print("\nGenerating and saving the prompt to a JSON file...\n")
    
    prompt_builder = AgentPromptBuilder(model=prompt_model)
    
    await prompt_builder.preview_prompt(
        agent_name="InvestmentAssistantGPT",
        tools=finance_tools,
        task_description=(
            "Help users make informed investment decisions by analyzing stocks, "
            "providing financial data, and explaining market trends in simple terms."
        ),
        additional_instructions=[
            "Always verify stock symbols before analyzing multiple data points.",
            "Explain financial terms and metrics for users who may not be experts.",
            "When presenting historical data, focus on meaningful trends rather than just numbers.",
            "Acknowledge market uncertainties rather than making definitive predictions."
        ],
        output_file="auto_prompt_agent_prompt",
        output_format="json"
    )
    
    # 4) Create an agent with automatically generated prompt
    print("\nCreating an agent with auto-generated system prompt...\n")
    
    auto_prompt_agent = await Agent.create_with_generated_prompt(
        name="InvestmentAssistantGPT",
        model=agent_model,
        memory=memory,
        tools=finance_tools,
        task_description=(
            "Help users make informed investment decisions by analyzing stocks, "
            "providing financial data, and explaining market trends in simple terms."
        ),
        additional_instructions=[
            "Always verify stock symbols before analyzing multiple data points.",
            "Explain financial terms and metrics for users who may not be experts.",
            "When presenting historical data, focus on meaningful trends rather than just numbers.",
            "Acknowledge market uncertainties rather than making definitive predictions."
        ],
        prompt_builder_model=prompt_model,
        use_json_format=True,  # Use JSON format for prompt generation
        options=agent_options
    )
    
    # 5) Test the agent with a user query
    print("\nTesting the agent with a sample query...\n")
    
    query = "How has Tesla stock performed over the last month, and what are its key financial metrics?"
    print(f"User Query: {query}")
    
    answer = await auto_prompt_agent.run(query)
    print(f"\nAgent Response:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main()) 