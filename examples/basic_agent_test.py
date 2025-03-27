#!/usr/bin/env python
import asyncio
import os

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Example demonstrating a minimal agent setup.
    """
    # 1) Create a minimal LLM
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
    )

    # 2) Instantiate an Agent
    agent = Agent.create(
        model=chat_model,
        memory=ShortTermMemory(max_messages=5),
        instructions=["You are a simple, helpful agent. Help users with their requests."],
        options=AgentOptions(enable_planning=False, debug=True)
    )

    # 3) Run the agent with a simple question
    user_question = "What's a quick tip for staying productive at work?"
    print("User Question:", user_question)

    answer = await agent.run(user_question)
    print("\n\nAgent's Answer:", answer)


if __name__ == "__main__":
    asyncio.run(main()) 