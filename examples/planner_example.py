#!/usr/bin/env python
import os
import asyncio
from typing import Dict, Any, Optional

from agentix.agents import Agent, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import Tool
from agentix.planner import SimpleLLMPlanner

from dotenv import load_dotenv
load_dotenv()

# Dummy tool
class DummyCalendarTool(Tool):
    """A dummy calendar tool that simulates scheduling events."""
    
    @property
    def name(self) -> str:
        return "Calendar"
    
    @property
    def description(self) -> str:
        return "Manages event scheduling and date lookups (dummy)."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f"FAKE CALENDAR ACTION: {input_str}"


async def main():
    """
    Example demonstrating how to use a planner with an agent.
    """
    # 1) Create an LLM for both agent & planner
    main_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    planner_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.3
    )

    # 2) Planner
    planner = SimpleLLMPlanner(planner_model=planner_model)

    # 3) Memory
    memory = ShortTermMemory(max_messages=5)

    # 4) Tool
    calendar = DummyCalendarTool()

    # 5) Create Agent with Planner
    def on_plan_generated(plan):
        print("[PLAN GENERATED]\n", plan)
    
    agent = Agent.create(
        name="PlannerAgent",
        model=main_model,
        memory=memory,
        tools=[calendar],
        planner=planner,
        instructions=[
            "You can plan tasks first, then execute them. If a plan step references 'Calendar', call the Calendar tool."
        ],
        options=AgentOptions(
            max_steps=5,
            usage_limit=10,
            time_to_live=30000,
            use_reflection=True,
            debug=True,
        ),
        hooks={
            "on_plan_generated": on_plan_generated
        }
    )

    # 6) User request
    user_query = "Schedule a meeting next Friday to discuss project updates."
    print("User Query:", user_query)

    # 7) Run agent
    answer = await agent.run(user_query)
    print("\nFinal Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 