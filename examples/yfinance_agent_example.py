#!/usr/bin/env python
"""
Example showcasing the use of YFinance tools with an Agentix agent for complex multi-task analysis.

This example demonstrates how to create an agent with stock analysis capabilities
that can handle complex queries requiring multiple tasks and planning.
The agent uses the YFinance toolkit and planning capabilities to break down
and execute sophisticated financial analysis requests.
"""

import os
import asyncio
import sys

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import YFinanceToolkit
from agentix.planner import SimpleLLMPlanner
from agentix.metrics.workflow_metrics import BaseWorkflowMetrics

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def main():
    """
    Example demonstrating a stock analysis agent handling complex multi-task queries.
    """
    
    # 1) Create a chat model with appropriate settings
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Using GPT-4 for complex analysis
        temperature=0.2,  # Lower temperature for factual/precise responses
        #stream=True,  # Stream output to console
        #on_token=on_token  # Hook to process tokens
    )

    # 2) Create a memory system with increased capacity for complex tasks
    memory = ShortTermMemory(max_messages=50)
    
    # 3) Set up the YFinance toolkit with all tools enabled
    yfinance_toolkit = YFinanceToolkit(
        enable_stock_price=True,
        enable_company_info=True,
        enable_historical_prices=True,
        enable_fundamentals=True,
        enable_financials=True
    )
    
    # 4) Create a planner for handling complex queries
    planner = SimpleLLMPlanner(chat_model)
    
    # 5) Create metrics tracker
    metrics = BaseWorkflowMetrics()
    
    # 6) Create agent options with increased limits for complex analysis
    agent_options = AgentOptions(
        max_steps=15,  # Increased steps for complex analysis
        usage_limit=20,  # Increased usage limit
        time_to_live=180000,  # 180 seconds timeout
        debug=True,  # Enable debugging
    )
    
    # 7) Create the agent with YFinance tools and planning capabilities
    stock_agent = Agent.create(
        name="StockAnalysisAgent",
        model=chat_model,
        memory=memory,
        tools=yfinance_toolkit.get_tools(),
        planner=planner,
        metrics=metrics,
        instructions=[
            "You are an advanced stock market analysis assistant that can handle complex multi-task queries.",
            "",
            "When analyzing stocks, follow these best practices:",
            "1. Break down complex queries into logical sub-tasks",
            "2. Verify stock symbols exist before making multiple requests",
            "3. Handle missing data or invalid symbols gracefully",
            "4. Provide clear explanations of financial metrics",
            "5. Present data in a structured format with summary statistics",
            "6. Compare metrics across companies when relevant",
            "7. Consider both historical and current data for analysis",
            "",
            "For historical data analysis:",
            "- Use appropriate period values: '1d', '5d', '1mo', '3mo', '6mo', '1y'",
            "- Interval options: '1d', '1wk', '1mo'",
            "",
            "For complex analysis:",
            "1. Start with basic information before diving into details",
            "2. Organize findings by category (price, fundamentals, financials)",
            "3. Provide comparative analysis when multiple stocks are involved",
            "4. Summarize key findings at the end",
            "",
            "If you encounter errors:",
            "1. Verify stock symbols and try alternative approaches",
            "2. Adapt the analysis based on available data",
            "3. Clearly communicate any limitations in the analysis",
            "4. Always use FINAL ANSWER: to complete your response"
        ],
        options=agent_options,
    )
    
    # 8) Complex multi-task query demonstrating planning and execution
    complex_query = """
    I need a comprehensive analysis comparing Apple (AAPL) and Microsoft (MSFT). Specifically:
    1. Compare their current market positions (stock prices, market cap, and basic financials)
    2. Analyze their performance over the last 6 months and identify key trends
    3. Evaluate their financial health by comparing recent income statements and fundamental metrics
    
    Provide a structured analysis with clear comparisons and your insights on their relative strengths.
    """
    
    print("\n=== Complex Multi-Task Analysis ===")
    print(f"User Query: {complex_query}")
    print("\nAgent: ", end="")
    
    # Execute the complex query
    answer = await stock_agent.run(complex_query)
    print(answer)
    
    # Print metrics report
    print("\n=== Performance Metrics ===")
    print(metrics.report())


if __name__ == "__main__":
    asyncio.run(main()) 