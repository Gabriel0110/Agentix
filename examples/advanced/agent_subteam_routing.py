#!/usr/bin/env python
"""
agent_subteam_routing.py

This example demonstrates how to route user queries to either a single agent or a team of agents
based on the query content. It shows:
1. Creating a router that directs to a single specialized agent for simple queries
2. Detecting when multiple opinions are needed and running a full team
3. Processing and combining team results for complex queries
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import AgentRouter, AgentTeam
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Main function demonstrating agent subteam routing.
    Suppose we have 5 agents, but we want to route queries to subsets.
    """
    # Create the base model
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    # 5 specialized agents
    agents = [
        Agent.create(
            name="FinanceAgent",
            model=model,
            memory=ShortTermMemory(max_messages=5),
            instructions=["You are specialized in finance, investments, and budgeting."],
            options=AgentOptions(use_reflection=False, max_steps=1)
        ),  # Agent #0: Finance
        
        Agent.create(
            name="LegalAgent",
            model=model,
            memory=ShortTermMemory(max_messages=5),
            instructions=["You are specialized in legal advice and regulations."],
            options=AgentOptions(use_reflection=False, max_steps=1)
        ),  # Agent #1: Legal
        
        Agent.create(
            name="TechAgent",
            model=model,
            memory=ShortTermMemory(max_messages=5),
            instructions=["You are specialized in technology and programming."],
            options=AgentOptions(use_reflection=False, max_steps=1)
        ),  # Agent #2: Tech
        
        Agent.create(
            name="TravelAgent",
            model=model,
            memory=ShortTermMemory(max_messages=5),
            instructions=["You are specialized in travel and hospitality advice."],
            options=AgentOptions(use_reflection=False, max_steps=1)
        ),  # Agent #3: Travel
        
        Agent.create(
            name="GeneralAgent",
            model=model,
            memory=ShortTermMemory(max_messages=5),
            instructions=["You are a general knowledge assistant."],
            options=AgentOptions(use_reflection=False, max_steps=1)
        )  # Agent #4: General
    ]
    
    # Then create an AgentRouter that picks which subset to run
    def router_fn(query: str) -> int:
        """Determine which agent to route to based on the query."""
        lower = query.lower()
        if "stock" in lower or "budget" in lower:
            return 0  # finance
        if "law" in lower or "sue" in lower:
            return 1  # legal
        if "programming" in lower or "nodejs" in lower:
            return 2  # tech
        if "travel" in lower or "hotel" in lower:
            return 3  # travel
        return 4  # general
    
    # Basic router that picks 1 agent from the 5
    main_router = AgentRouter(agents, router_fn)
    
    # Advanced run function that decides between team and individual agent
    async def advanced_run(query: str) -> str:
        """
        Decide whether to run a full team or single agent based on the query.
        
        Args:
            query: The user query
            
        Returns:
            Response from either a single agent or combined team response
        """
        if "multiple opinions" in query:
            # Run all agents in parallel
            print(f"Running full team for query: {query}")
            team = AgentTeam("AllAgentsTeam", agents)
            results = await team.run_in_parallel(query)
            
            # Combine results with agent names
            combined = ""
            for i, result in enumerate(results):
                agent_name = agents[i].name if hasattr(agents[i], "name") else f"Agent {i}"
                combined += f"--- {agent_name} Opinion ---\n{result}\n\n"
            return combined
        else:
            # Otherwise route to just one agent
            print(f"Routing to single agent for query: {query}")
            return await main_router.run(query)
    
    # Example usage
    user_queries = [
        "What's a good stock to invest in?",
        "I want multiple opinions on NodeJS frameworks",
        "Is it legal to break a lease early?",
    ]
    
    for query in user_queries:
        print(f"\nUser: \"{query}\"")
        response = await advanced_run(query)
        print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    asyncio.run(main()) 