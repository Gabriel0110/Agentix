#!/usr/bin/env python
import os
import asyncio
from typing import Dict, Any, List, Optional, Union, Awaitable

from agentix.agents import Agent, AgentOptions, AgentHooks
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat
from agentix.tools import Tool

# Dummy tool
class DummyMathTool(Tool):
    """A dummy math tool that always returns 42."""
    
    @property
    def name(self) -> str:
        return "DummyMath"
    
    @property
    def description(self) -> Optional[str]:
        return "Performs fake math calculations (dummy)."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f'FAKE MATH RESULT for "{input_str}": 42 (always).'


async def main():
    """
    Example demonstrating how to use hooks with an Agent.
    """
    # 1) Create LLM
    model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.6
    )

    # 2) Memory
    memory = ShortTermMemory(max_messages=5)

    # 3) Hooks
    async def on_tool_call(tool_name: str, query: str) -> bool:
        print(f'[Hook: on_tool_call] About to call "{tool_name}" with query="{query}"')
        # Could confirm or deny usage. If we return False, the call is canceled
        return True
    
    def on_step(messages: List[Dict[str, Any]]):
        print("[Hook: on_step] Current conversation so far:", messages)
    
    def on_final_answer(answer: str):
        print("[Hook: on_final_answer] The final answer is:", answer)
    
    hooks = AgentHooks(
        on_tool_call=on_tool_call,
        on_step=on_step,
        on_final_answer=on_final_answer
    )

    # 4) Create Agent
    math_tool = DummyMathTool()
    agent = Agent.create(
        name="HookedAgent",
        model=model,
        memory=memory,
        tools=[math_tool],
        instructions=['Use DummyMath if the user needs a calculation. Request it in the format "TOOL REQUEST: DummyMath {"num1": "123", "num2": "456"}"'],
        hooks=hooks,
        options=AgentOptions(
            use_reflection=True,
            max_steps=5,
            usage_limit=5,
            time_to_live=60000,
            debug=True,
        )
    )

    # 5) Run agent
    question = "What is 123 + 456, approximately?"
    print("User asks:", question)

    answer = await agent.run(question)
    print("\nFinal Answer from Agent:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 