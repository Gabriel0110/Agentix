#!/usr/bin/env python
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.memory import (
    CompositeMemory,
    ShortTermMemory,
    SummarizingMemory,
    LongTermMemory,
)
from agentix.llms import OpenAIChat, OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Example demonstrating a RAG (Retrieval-Augmented Generation) agent with
    composite memory including short-term, summarizing, and long-term memory.
    """
    # 1) Chat model
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 2) Summarizer model
    summarizer_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 3) Embeddings for long-term
    embeddings_model = OpenAIEmbeddings(
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # 4) Memory instances
    short_mem = ShortTermMemory(max_messages=10)
    summarizing_mem = SummarizingMemory(
        summarizer_model=summarizer_model,
        threshold=5,
        summary_prompt="Summarize earlier conversation:",
        max_summary_tokens=200
    )
    long_term_mem = LongTermMemory(
        embeddings=embeddings_model,
        max_messages=100,
        top_k=3
    )

    # 5) Composite memory
    composite_mem = CompositeMemory(short_mem, summarizing_mem, long_term_mem)

    # 6) Agent
    agent_options = AgentOptions(
        max_steps=5,
        usage_limit=10,
        time_to_live=60000,
        use_reflection=True,
        debug=True
    )

    agent = Agent.create(
        name="RAGAgent",
        model=chat_model,
        memory=composite_mem,
        instructions=[
            "If the user asks about older content, recall from memory. If uncertain, say so politely."
        ],
        options=agent_options
    )

    # 7) Simulate a user adding data, then later asking about it
    # First: user provides some info
    print("User: I'm planning a road trip from LA to Vegas next month, maybe around the 15th.")
    await agent.run("I'm planning a road trip from LA to Vegas next month, maybe around the 15th.")
    
    print("\nUser: I want to remember that I'll have a budget of $500 total.")
    await agent.run("I want to remember that I'll have a budget of $500 total.")

    # Later: user asks
    question = "Hey, do you recall how much money I budgeted for my LA to Vegas trip?"
    print(f"\nUser: {question}")
    answer = await agent.run(question)
    print("\nFinal Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 