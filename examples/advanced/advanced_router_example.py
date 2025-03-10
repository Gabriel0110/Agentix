#!/usr/bin/env python
"""
advanced_router_example.py

This example demonstrates the usage of the AdvancedAgentRouter for intelligent
routing to specialized agents based on query content and agent capabilities.
It shows how to:
1. Create specialized agents for different domains
2. Define agent capabilities with keywords and examples
3. Use both keyword-based and LLM-powered routing
4. Set up fallback mechanisms for unmatched queries
5. Track and analyze routing decisions
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import AgentCapability, AdvancedAgentRouter, RouterOptions
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    """Main function demonstrating the advanced router."""
    # 1. Create specialized agents
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    finance_agent = Agent.create(
        name="FinanceAgent",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a financial expert specialized in investments, budgeting, and financial planning.",
            "Provide clear, actionable financial advice.",
            "Always consider risk management in your recommendations."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    legal_agent = Agent.create(
        name="LegalAgent",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a legal expert specialized in general law and regulations.",
            "Provide clear explanations of legal concepts and implications.",
            "Always include disclaimers about seeking professional legal counsel when appropriate."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    general_agent = Agent.create(
        name="GeneralAgent",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a general knowledge assistant.",
            "Provide helpful information on a wide range of topics.",
            "Direct specialized queries to appropriate experts."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    # 2. Define agent capabilities
    capabilities = {
        0: AgentCapability(
            name="Finance Expert",
            description="Specialized in financial advice, investments, and budgeting",
            keywords=["money", "invest", "budget", "stock", "finance", "savings"],
            examples=[
                "What stocks should I invest in?",
                "How do I create a monthly budget?",
                "Should I invest in cryptocurrency?"
            ]
        ),
        1: AgentCapability(
            name="Legal Expert",
            description="Specialized in legal advice and regulations",
            keywords=["legal", "law", "court", "rights", "contract", "sue"],
            examples=[
                "Is it legal to break my lease early?",
                "What are my rights as an employee?",
                "How do I handle a contract dispute?"
            ]
        )
        # Index 2 is our fallback general agent
    }
    
    # 3. Create the advanced router
    router = AdvancedAgentRouter(
        [finance_agent, legal_agent, general_agent],
        capabilities,
        options=RouterOptions(
            use_llm=True,  # LLM-powered routing,
            router_llm=model,
            debug=True,
            fallback_index=2,  # General agent as fallback
            confidence_threshold=0.8  # Minimum confidence for routing
        )
    )
    
    # 4. Test queries
    queries = [
        "What's the best way to invest $10,000?",
        "Is it legal to record a conversation without consent?",
        "How do I create a budget for my small business?",
        "What are my rights as a tenant?",
        "Tell me about the history of pizza",  # Should go to general agent
        "I need advice about a contract dispute with my financial advisor"  # Complex case
    ]
    
    print("=== Testing Advanced Agent Router ===\n")
    
    for query in queries:
        print(f"\nUser Query: \"{query}\"")
        print("-" * 50)
        
        try:
            response = await router.run(query)
            print("Response:", response)
        except Exception as error:
            print("Error processing query:", error)
    
    # 5. Display routing history
    print("\n=== Routing History ===\n")
    history = router.get_routing_history()
    
    for i, entry in enumerate(history):
        print(f"\nQuery {i + 1}:")
        print(f"Query: \"{entry.query}\"")
        print(f"Routed to: {entry.selected_agent}")
        print(f"Confidence: {entry.confidence:.2f}")
        if entry.reasoning:
            print(f"Reasoning: {entry.reasoning}")
        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(main()) 