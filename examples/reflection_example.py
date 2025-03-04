#!/usr/bin/env python
"""
reflection_example.py

Demonstrates an agent with two memory objects:
   1) A public short-term memory (shared with the user).
   2) A ReflectionMemory that stores chain-of-thought or self-critique.

The agent:
 - Adds "reflection" messages after each generation.
 - Optionally can see those reflections if we set include_reflections=True.
 - Otherwise, the reflection is purely for developer debugging.
"""
import os
import asyncio
from typing import Dict, Any

from agentix.agents import Agent, AgentOptions, AgentHooks
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory, ReflectionMemory, CompositeMemory

async def main():
    """
    Example demonstrating an agent that uses reflection memory.
    """
    # 1) Create the main model
    chat_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )

    # 2) Create a public ShortTermMemory for the user conversation
    public_memory = ShortTermMemory(max_messages=5)

    # 3) Create a ReflectionMemory that we do NOT show to the user by default
    #    If you pass `True` to ReflectionMemory constructor, it would include
    #    the reflection messages in the next prompt, letting the agent see them.
    #    If `False`, it effectively hides them from the agent's own prompt calls.
    reflection_mem = ReflectionMemory(include_reflections=False)

    # 4) Combine them in a CompositeMemory if you want a single `memory` object
    #    for the agent. Or you could keep them separate if you prefer to pass them
    #    manually. Let's combine them so agent sees only short-term memory by default.
    combined_mem = CompositeMemory(public_memory, reflection_mem)

    # 5) Create an Agent
    agent = Agent.create(
        name="ReflectiveAgent",
        model=chat_model,
        memory=combined_mem,
        instructions=[
            "You are a reflective agent who tries to self-critique after each answer.",
            "However, do NOT reveal your chain-of-thought or reflections to the user."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=True  # so agent can do multi-step if needed
        )
    )

    # 6) We'll override (or wrap) the final answer logic to store reflection
    #    messages after each run or step. Alternatively, you can do it with hooks.
    #    For simplicity, let's do it in a hook (on_step or on_final_answer).
    def on_final_answer(answer: str):
        print("Agent's Final Answer:", answer)
        # Now let's add a reflection message about that final answer
        reflection_mem.add_message({
            "role": "reflection",
            "content": f'I gave an answer: "{answer}". My quick critique: might want to verify correctness next time.'
        })

    hooks = AgentHooks(
        on_final_answer=on_final_answer
    )

    agent.hooks = hooks

    # 7) Use the Agent with a question
    print("\n--- Asking the agent a question ---")
    user_question = "What is the approximate radius of the Earth in kilometers?"
    final_output = await agent.run(user_question)
    print("Final Output =>", final_output)

    # 8) Print out reflection memory for debugging
    #    It's not appended to the user conversation by default (include_reflections = False).
    reflection_context = await reflection_mem.get_context()
    print("\n[DEBUG] Reflection Memory =>", reflection_context)

    # If we want to let the agent see its reflections next time, we can set:
    # reflection_mem.include_reflections = True
    # Then agent's next call might incorporate them in the prompt for deeper self-critique.


if __name__ == "__main__":
    asyncio.run(main()) 