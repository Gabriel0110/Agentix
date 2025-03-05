#!/usr/bin/env python
import asyncio
import os
import sys

from agentix.llms import OpenAIChat, TogetherChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions

from dotenv import load_dotenv
load_dotenv()

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def main():
    """
    Example demonstrating a minimal agent setup.
    """
    # 1) Create a minimal LLM
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
        #stream=True,  # Stream output to console
        #on_token=on_token  # Hook to process tokens
    )

    # chat_model = TogetherChat(
    #     api_key=os.environ.get("TOGETHER_API_KEY"),
    #     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    #     temperature=0.7,
    #     #stream=True,  # Stream output to console
    #     #on_token=on_token  # Hook to process tokens
    # )

    # 2) Create a simple short-term memory
    short_term_memory = ShortTermMemory(max_messages=5)

    # 2.1) Create agent options
    agent_options = AgentOptions(
        use_reflection=False,
        max_steps=1,
        usage_limit=2,
        time_to_live=5000,
    )

    # 3) Instantiate an Agent with NO reflection or tools
    agent = Agent.create(
        model=chat_model,
        memory=short_term_memory,
        instructions=[
            "You are a simple agent. Answer only in one short sentence."
        ],
        options=agent_options,
    )

    # 4) Run the agent with a simple question
    user_question = "What's a quick tip for staying productive at work?"
    print("User Question:", user_question)

    answer = await agent.run(user_question)
    print("\n\nAgent's Answer:", answer)


if __name__ == "__main__":
    asyncio.run(main()) 