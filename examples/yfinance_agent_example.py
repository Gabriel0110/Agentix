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
        api_key=os.getenv("OPENAI_API_KEY"),
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
        max_steps=5,  # Allow multiple steps for financial analysis
        usage_limit=10,
        time_to_live=60000,  # 60 seconds timeout
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
            "You have access to the following YFinance tools that you must use correctly:",
            "- StockPrice: Get current price with 'TOOL REQUEST: StockPrice \"SYMBOL\"'",
            "- CompanyInfo: Get company details with 'TOOL REQUEST: CompanyInfo \"SYMBOL\"'", 
            "- StockHistoricalPrices: Get price history with 'TOOL REQUEST: StockHistoricalPrices {\"symbol\": \"SYMBOL\", \"period\": \"1mo\", \"interval\": \"1d\"}'",
            "- StockFundamentals: Get key metrics with 'TOOL REQUEST: StockFundamentals \"SYMBOL\"'",
            "- FinancialStatements: Get statements with 'TOOL REQUEST: FinancialStatements {\"symbol\": \"SYMBOL\", \"statement_type\": \"income\"}'",
            "",
            "When analyzing stocks:",
            "1. Always start by getting the current stock price using StockPrice",
            "2. Follow up with CompanyInfo to understand the business",
            "3. Use StockFundamentals to analyze key financial metrics",
            "4. For historical analysis, use StockHistoricalPrices with appropriate period/interval",
            "5. For detailed financials, use FinancialStatements with the right statement type",
            "",
            "After retrieving the necessary information with tools, ALWAYS provide your final answer to the user with the format:",
            "FINAL ANSWER: <Your comprehensive answer>",
            "",
            "Remember: You must use FINAL ANSWER: to conclude your response after using tools.",
            "Do not keep using tools repeatedly if you already have the information needed to answer the question.",
            "",
            "Provide clear explanations of all data retrieved and what it means for investors.",
            "Format tool requests exactly as shown in the examples above.",
            "Always verify the stock symbol exists before making multiple requests."
        ],
        options=agent_options,
    )
    
    # 6) Example user questions to demonstrate the agent's capabilities
    questions = [
        "What's the current stock price of Apple?",
        #"Can you analyze Tesla stock and give me a summary of its fundamentals?",
        #"How has Amazon's stock performed over the last month?",
        #"What are Microsoft's most recent income statements?",
        #"Compare Google (GOOGL) and Facebook (META) based on their financial ratios and performance.",
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