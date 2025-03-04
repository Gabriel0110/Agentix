#!/usr/bin/env python
import os
import asyncio

from agentix.agents import Agent, AgentTeam, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory

async def main():
    """
    Example demonstrating how to use AgentTeam with parallel and sequential execution.
    """
    # 1) Create base LLMs
    agent1_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    agent2_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 2) Memory
    mem1 = ShortTermMemory(max_messages=5)
    mem2 = ShortTermMemory(max_messages=5)

    # 3) Agent #1: "GreetingAgent"
    greeting_agent = Agent.create(
        name="GreetingAgent",
        model=agent1_model,
        memory=mem1,
        instructions=["Greet the user in a friendly way."],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )

    # 4) Agent #2: "MotivationAgent"
    motivation_agent = Agent.create(
        name="MotivationAgent",
        model=agent2_model,
        memory=mem2,
        instructions=["Provide a short motivational statement or advice to the user."],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )

    # 5) Create an AgentTeam
    team = AgentTeam("Greeting+MotivationTeam", [greeting_agent, motivation_agent])

    # 6) Use run_in_parallel
    user_prompt = "I could use some positivity today!"
    print("User Prompt:", user_prompt)

    parallel_results = await team.run_in_parallel(user_prompt)
    print("\nParallel Results:\n", parallel_results)

    # 7) Use run_sequential
    sequential_result = await team.run_sequential(user_prompt)
    print("\nSequential Result:\n", sequential_result)


if __name__ == "__main__":
    asyncio.run(main()) 