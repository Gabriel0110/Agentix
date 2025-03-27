#!/usr/bin/env python
"""
llm_convergence_example.py

This demo showcases usage of the LLM convergence checker for multi-agent analysis.
It showcases the benefits of having an LLM check for convergence criteria vs more simple/basic hardcoded checks.

The LLM convergence checker uses an LLM - this means your prompt(s) should be very well-structured
to ensure the LLM can accurately determine if the content meets the criteria. Prompt quality
is key to the success of the LLM convergence checker, just like any other LLM-based system.
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory, CompositeMemory
from agentix.agents.multi_agent import LLMConvergenceChecker, AgentTeam

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Example demonstrating how to use LLM convergence checker with a team of agents.
    """
    # Initialize our LLM
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    # Create short-term memories for each agent
    pros_agent_memory = ShortTermMemory(max_messages=5)
    cons_agent_memory = ShortTermMemory(max_messages=5)
    
    # Create shared memory
    shared_memory = CompositeMemory(
        pros_agent_memory,
        cons_agent_memory
    )
    
    # Create two simple agents: one for pros, one for cons
    pros_agent = Agent.create(
        name="ProsAgent",
        model=model,
        memory=pros_agent_memory,
        instructions=[
            "You are an analyst focused on identifying and explaining advantages and benefits.",
            "Always structure your response with clear sections.",
            "Include specific examples and evidence.",
            "End with clear recommendations.",
            "Format your final response with 'FINAL ANSWER:' prefix."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    cons_agent = Agent.create(
        name="ConsAgent",
        model=model,
        memory=cons_agent_memory,
        instructions=[
            "You are an analyst focused on identifying and explaining disadvantages and risks.",
            "Always structure your response with clear sections.",
            "Include specific examples and evidence.",
            "End with risk mitigation strategies.",
            "Format your final response with 'FINAL ANSWER:' prefix."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    # Define team roles
    team_config = {
        "roles": {
            "ProsAgent": {
                "name": "Advantages Analyst",
                "description": "Focuses on benefits and opportunities",
                "query_transform": lambda query: f"{query}\nAnalyze the advantages and benefits of this situation."
            },
            "ConsAgent": {
                "name": "Risks Analyst",
                "description": "Focuses on disadvantages and risks",
                "query_transform": lambda query: f"{query}\nAnalyze the disadvantages and risks of this situation."
            }
        }
    }
    
    # Create convergence criteria
    analysis_convergence_criteria = {
        "custom_instructions": [
            "Check if the analysis provides detailed explanations",
            "Verify that examples are specific and relevant",
            "Ensure recommendations are actionable",
            "Confirm the response is well-structured with clear sections"
        ]
    }
    
    # Create new model for the LLM convergence checker with temp of 0 (not required but recommended)
    convergence_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Create LLM convergence checker
    convergence_checker = LLMConvergenceChecker(
        convergence_model,
        analysis_convergence_criteria,
        True  # Enable debug logging
    )
    
    # Create the team
    team = AgentTeam(
        "ProsConsTeam",
        [pros_agent, cons_agent],
        {
            "shared_memory": shared_memory,
            "team_config": team_config,
            "debug": True
        }
    )
    
    # Enable shared memory
    team.enable_shared_memory()
    
    # Test queries
    queries = [
        "Should a small business invest in artificial intelligence technology?",
        "What are the implications of switching to a remote-first work policy?",
        "Should a company expand internationally or focus on domestic growth?"
    ]
    
    print("=== Testing LLM Convergence with Pros/Cons Analysis ===\n")
    
    for query in queries:
        print(f"\nAnalyzing Query: \"{query}\"")
        print("=" * 50)
        
        try:
            result = await team.run_interleaved(
                query,
                3,  # max rounds
                convergence_checker.has_converged,
                True  # require all agents
            )
            
            print("\nFinal Team Analysis:")
            print(result)
            
            # Add separator between queries
            print("\n" + "-" * 80 + "\n")
            
        except Exception as error:
            print("Error processing query:", error)
    
    # Display some statistics
    print("\n=== Analysis Complete ===")
    print("Queries processed:", len(queries))


if __name__ == "__main__":
    asyncio.run(main()) 