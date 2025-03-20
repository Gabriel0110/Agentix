#!/usr/bin/env python
"""
content_creation_team.py

A simple example demonstrating how to use autonomous agents in a team
for content creation. The team consists of:
1. Research Agent: Gathers information on the topic
2. Outline Agent: Creates a structured outline
3. Writer Agent: Generates the content
4. Editor Agent: Refines and improves the content
"""

import os
import asyncio
from typing import Dict, Any

from agentix.agents import AutonomousAgent, AutoAgentOptions
from agentix.agents.multi_agent import (
    AdvancedAgentTeam,
    AdvancedTeamOptions,
    AgentRole,
    TeamConfiguration
)
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory, CompositeMemory
from agentix.tools import DuckDuckGoToolkit

from dotenv import load_dotenv
load_dotenv()

async def main():
    """Main function demonstrating the content creation team."""
    
    # Initialize shared model and memory
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )

    # Agent memories
    research_mem = ShortTermMemory(max_messages=20)
    outline_mem = ShortTermMemory(max_messages=20)
    writer_mem = ShortTermMemory(max_messages=20)
    editor_mem = ShortTermMemory(max_messages=20)
    
    # Create shared memory for the team
    shared_memory = CompositeMemory(research_mem, outline_mem, writer_mem, editor_mem)
    
    # Initialize search tools
    search_tools = DuckDuckGoToolkit(
        enable_text_search=True,
        enable_news_search=True,
        enable_image_search=False,
        enable_video_search=False,
        enable_chat=False
    ).get_tools()
    
    # Create specialized agents
    
    # 1. Research Agent
    research_agent = AutonomousAgent(
        name="Researcher",
        model=model,
        memory=research_mem,
        tools=search_tools,
        instructions=[
            "You are a research specialist who gathers comprehensive information on topics.",
            "You must use your search tools to find key facts, statistics, and expert opinions.",
            "Always verify information from multiple sources when possible.",
            "Organize findings into clear categories.",
            "Include sources for important claims.",
            "Your research will be used by an Outline Agent, so format your output clearly.",
            "NEVER proceed without using your search tools - real research is essential."
        ],
        options=AutoAgentOptions(
            enable_planning=True,
            max_steps=15,
            debug=True
        )
    )
    
    # 2. Outline Agent
    outline_agent = AutonomousAgent(
        name="Outliner",
        model=model,
        memory=outline_mem,
        instructions=[
            "You are an expert at creating structured content outlines.",
            "Your input will be research gathered by the Research Agent.",
            "Break down topics into logical sections and subsections.",
            "Ensure a clear flow of ideas based on the research provided.",
            "Include key points to be covered in each section.",
            "Consider the target audience and purpose.",
            "Your outline will be used by a Writer Agent, so make it comprehensive and clear."
        ],
        options=AutoAgentOptions(
            enable_planning=False,
            max_steps=15,
            debug=True
        )
    )
    
    # 3. Writer Agent
    writer_agent = AutonomousAgent(
        name="Writer",
        model=model,
        memory=writer_mem,
        instructions=[
            "You are a skilled content writer.",
            "Your input will be an outline created by the Outline Agent, based on research.",
            "Follow the provided outline closely.",
            "Use engaging and clear language.",
            "Include relevant examples and explanations from the research.",
            "Maintain consistent tone and style.",
            "Your content will be edited by an Editor Agent, so focus on comprehensive content creation."
        ],
        options=AutoAgentOptions(
            enable_planning=False,
            max_steps=15,
            debug=True
        )
    )
    
    # 4. Editor Agent
    editor_agent = AutonomousAgent(
        name="Editor",
        model=model,
        memory=editor_mem,
        instructions=[
            "You are an expert content editor.",
            "Your input will be content written by the Writer Agent, based on an outline and research.",
            "Improve clarity and flow.",
            "Fix grammar and style issues.",
            "Ensure consistency throughout.",
            "Enhance readability while maintaining the original meaning.",
            "Add a final conclusion and summary if missing from the original content."
        ],
        options=AutoAgentOptions(
            enable_planning=False,
            max_steps=15,
            debug=True
        )
    )
    
    # Define team roles with specialized query transformations
    def research_transform(query: str) -> str:
        return f"Research this topic thoroughly: {query}\n\nFind key facts, statistics, and expert opinions. Organize your findings clearly and cite sources."
    
    def outline_transform(query: str) -> str:
        return f"Create a structured outline for this topic: {query}\n\nUse the research provided and organize into clear sections and subsections."
    
    def write_transform(query: str) -> str:
        return f"Write content following this outline: {query}\n\nCreate engaging, well-structured content that follows the outline and incorporates the research."
    
    def edit_transform(query: str) -> str:
        return f"Edit and improve this content: {query}\n\nEnhance clarity, fix any issues, and improve overall quality while maintaining the original message."
    
    # Create team configuration
    team_config = TeamConfiguration(
        roles={
            "Researcher": AgentRole(
                name="Research Specialist",
                description="Gathers and organizes information",
                query_transform=research_transform
            ),
            "Outliner": AgentRole(
                name="Outline Creator",
                description="Structures content organization",
                query_transform=outline_transform
            ),
            "Writer": AgentRole(
                name="Content Writer",
                description="Generates engaging content",
                query_transform=write_transform
            ),
            "Editor": AgentRole(
                name="Content Editor",
                description="Refines and improves content",
                query_transform=edit_transform
            )
        }
    )
    
    # Create the team
    team = AdvancedAgentTeam(
        name="ContentCreationTeam",
        agents=[research_agent, outline_agent, writer_agent, editor_agent],
        options=AdvancedTeamOptions(
            shared_memory=shared_memory,
            team_config=team_config,
            debug=True
        )
    )
    
    # Enable shared memory
    team.enable_shared_memory()
    
    # Example content creation task
    topic = "The Impact of Artificial Intelligence on Healthcare"
    
    print(f"\n🚀 Creating content about: {topic}")
    print("=" * 50)
    
    # Run the team sequentially (research -> outline -> write -> edit)
    result = await team.run_sequential(topic)
    
    print("\n✨ Final Content:")
    print("=" * 50)
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 