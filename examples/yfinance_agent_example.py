#!/usr/bin/env python
"""
Example showcasing the use of YFinance tools with an Agentix agent.

This example demonstrates how to create an agent with stock analysis capabilities
using the YFinance toolkit. The agent can retrieve current stock prices, company
information, historical data, fundamentals, and financial statements.
"""

import os
import asyncio
import sys

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import YFinanceToolkit

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def main():
    """
    Example demonstrating a stock analysis agent with YFinance tools.
    """
    
    # 1) Create a chat model with appropriate settings
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Using a more capable model for financial analysis
        temperature=0.2,  # Lower temperature for more factual/precise responses
        #stream=True,  # Stream output to console
        #on_token=on_token  # Hook to process tokens
    )

    # 2) Create a memory system
    memory = ShortTermMemory(max_messages=20)
    
    # 3) Set up the YFinance toolkit with all tools enabled
    yfinance_toolkit = YFinanceToolkit(
        enable_stock_price=True,
        enable_company_info=True,
        enable_historical_prices=True,
        enable_fundamentals=True,
        enable_financials=True
    )
    
    # 4) Create agent options
    agent_options = AgentOptions(
        max_steps=10,  # Allow multiple steps for financial analysis
        usage_limit=15,
        time_to_live=120000,  # 120 seconds timeout
        debug=True  # Enable debugging
    )
    
    # 5) Create the agent with YFinance tools
    stock_agent = Agent.create(
        name="StockAnalysisAgent",
        model=chat_model,
        memory=memory,
        tools=yfinance_toolkit.get_tools(),
        instructions=[
            "You are a stock market analysis assistant that helps users analyze stocks and financial data.",
            "When analyzing stocks, follow these best practices:",
            "1. Verify stock symbols exist before making multiple requests by checking basic information first",
            "2. Always use valid stock symbols - common ones include AAPL (Apple), MSFT (Microsoft), GOOG (Google), AMZN (Amazon), TSLA (Tesla)",
            "3. Handle missing data or invalid symbols gracefully, providing alternatives when possible",
            "4. Provide clear explanations of financial metrics and what they mean for investors",
            "5. Present data in a structured format with summary statistics",
            "",
            "For historical data analysis:",
            "- Use appropriate period values: '1d', '5d', '1mo', '3mo', '6mo', '1y', etc.",
            "- Interval options: '1d', '1wk', '1mo'",
            "",
            "If you encounter errors:",
            "1. Verify the stock symbol is correct and try alternative symbols if needed",
            "2. Try different parameter values if the first attempt fails",
            "3. If a tool consistently fails, use a different approach or provide the best answer with available information",
            "4. Always use FINAL ANSWER: to complete your response"
        ],
        options=agent_options,
    )
    
    # 6) Example user questions to demonstrate the agent's capabilities
    questions = [
        #"What's the current stock price of Apple?",
        #"Can you analyze Tesla stock and give me a summary of its fundamentals?",
        "How has Amazon's stock performed over the last month?",
        #"What are Microsoft's most recent income statements?",
        # Test error handling with an invalid symbol
        #"Tell me about the stock BBC (should gracefully handle invalid symbol)"
    ]
    
    # 7) Run the agent with each question
    for i, question in enumerate(questions):
        print(f"\n\n--- Question {i+1} ---")
        print(f"User: {question}")
        print("Agent: ", end="")
        answer = await stock_agent.run(question)
        print(answer)
        print("\n\n")  # Extra spacing between questions


if __name__ == "__main__":
    asyncio.run(main()) 