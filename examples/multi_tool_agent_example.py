#!/usr/bin/env python

import os
import asyncio
from typing import Dict, Any, Optional

from agentix.agents import Agent, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import Tool

# Dummy tool #1
class FakeSearchTool(Tool):
    """A dummy search tool that returns fake results."""
    
    @property
    def name(self) -> str:
        return "FakeSearch"
    
    @property
    def description(self) -> str:
        return "Simulates a search engine lookup (dummy)."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f'FAKE SEARCH RESULTS for "{input_str}" (no real search done).'


# Dummy tool #2
class FakeTranslatorTool(Tool):
    """A dummy translator tool that returns fake French translations."""
    
    @property
    def name(self) -> str:
        return "FakeTranslator"
    
    @property
    def description(self) -> str:
        return "Pretends to translate input text into French."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f'FAKE TRANSLATION to French of: "{input_str}" => [Ceci est une traduction factice]'


async def main():
    """
    Example demonstrating an agent with multiple tools.
    """
    # 1) Create LLM
    chat_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )

    # 2) Memory
    mem = ShortTermMemory(max_messages=10)

    # 3) Tools
    search_tool = FakeSearchTool()
    translator_tool = FakeTranslatorTool()

    # 4) Agent Options
    options = AgentOptions(
        max_steps=5,
        usage_limit=5,
        time_to_live=60000,
        use_reflection=True,
        debug=True,
    )

    # 5) Create Agent with multiple tools
    agent = Agent.create(
        name="MultiToolAgent",
        model=chat_model,
        memory=mem,
        tools=[search_tool, translator_tool],
        instructions=[
            "You can use FakeSearch to look up information.",
            "You can use FakeTranslator to convert text to French.",
            "Use tools by responding EXACTLY in the format: TOOL REQUEST: <ToolName> \"<Query>\"",
            "Integrate tool results before proceeding to the next step.",
        ],
        options=options,
    )

    # 6) User question
    user_question = "Search for today's top news and then translate the summary into French."
    print("\nUser Question:", user_question)

    # 7) Run agent
    answer = await agent.run(user_question)
    print("\nAgent's Final Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 