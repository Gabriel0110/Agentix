#!/usr/bin/env python
"""
agent_team_safe_example.py

Demonstrates how to use "safe" multi-agent orchestration functions
(like run_sequential_safe) with different error-handling strategies.
Includes:
  - Agents that might throw errors
  - 'stop_on_error' parameter usage
  - TeamHooks for advanced debugging/logging
  - Optional aggregator logic if you want a final step
"""
import os
import asyncio
from typing import List

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import AgentTeam, TeamHooks
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory

class SafeAgentTeam(AgentTeam):
    """
    A safe agent team that can handle errors and continue execution.
    """
    def __init__(self, name: str, agents: List[Agent]):
        super().__init__(name, agents)


async def main():
    """Main function demonstrating safe team execution."""
    # 1) Create LLM(s)
    model1 = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
    )
    model2 = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
    )
    model3 = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    # 2) Create memory for each agent
    mem_a = ShortTermMemory(max_messages=5)
    mem_b = ShortTermMemory(max_messages=5)
    mem_c = ShortTermMemory(max_messages=5)
    
    # 3) Create agents
    agent_a = Agent.create(
        name="AgentA",
        model=model1,
        memory=mem_a,
        instructions=["Respond politely. (No error here)"],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )
    
    # AgentB intentionally might throw an error
    agent_b = Agent.create(
        name="AgentB",
        model=model2,
        memory=mem_b,
        instructions=["Pretend to attempt the user query but throw an error for demonstration."],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )
    
    # Force an error for AgentB to demonstrate safe run
    class ErrorAgent:
        """Agent that always throws an error for demonstration purposes."""
        name = "AgentB"
        
        async def run(self, input_str: str) -> str:
            """Raises an error when called."""
            raise Exception("Intentional error from AgentB for demonstration!")
    
    # Replace the normal agent with our error agent
    agent_b = ErrorAgent()
    
    agent_c = Agent.create(
        name="AgentC",
        model=model3,
        memory=mem_c,
        instructions=["Provide a short helpful answer. (No error)"],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )
    
    # 4) Create our SafeAgentTeam
    team = SafeAgentTeam("DemoTeam", [agent_a, agent_b, agent_c])
    
    # 5) Define some hooks to see what happens behind the scenes
    hooks = TeamHooks(
        on_agent_start=lambda agent_name, input_str: 
            print(f"[START] {agent_name} with input: \"{input_str}\""),
        
        on_agent_end=lambda agent_name, output: 
            print(f"[END] {agent_name}: output => \"{output}\""),
        
        on_error=lambda agent_name, error: 
            print(f"[ERROR] in {agent_name}: {error}"),
        
        on_final=lambda outputs: 
            print("Final outputs from the entire sequential run =>", outputs)
    )
    
    # 6a) Demonstrate run_sequential_safe with stop_on_error=True
    #     - With stop_on_error=True, the loop breaks immediately after AgentB throws an error,
    #       so AgentC never runs.
    print("\n--- run_sequential_safe (stop_on_error = true) ---")
    user_prompt = "Hello from the user!"
    results_stop_on_error = await team.run_sequential_safe(user_prompt, True, hooks)
    print("\nResults (stop_on_error=true):", results_stop_on_error)
    
    # 6b) Demonstrate run_sequential_safe with stop_on_error=False
    #     - With stop_on_error=False, AgentB's error is logged, but AgentC still gets a chance to run,
    #       producing its output as the final step.
    print("\n--- run_sequential_safe (stop_on_error = false) ---")
    user_prompt2 = "Another user query - let's see if we continue after errors."
    results_continue = await team.run_sequential_safe(user_prompt2, False, hooks)
    print("\nResults (stop_on_error=false):", results_continue)


if __name__ == "__main__":
    asyncio.run(main()) 