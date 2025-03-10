#!/usr/bin/env python
"""
advanced_team_collaboration_example.py

This example demonstrates the use of AdvancedAgentTeam for collaborative problem-solving where:
1. Agents have specialized roles with custom query transformations
2. Agents communicate with each other through shared memory
3. The team runs in an interleaved manner, where each agent builds on others' insights
4. The process continues until convergence criteria are met
5. Advanced hooks track the progress of the collaboration

This pattern is particularly useful for complex problem-solving that requires
iterative refinement from different perspectives working together.
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import (
    AdvancedAgentTeam, 
    AdvancedTeamOptions, 
    AdvancedTeamHooks,
    AgentRole, 
    TeamConfiguration, 
)
from agentix.memory import ShortTermMemory, CompositeMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Main function demonstrating an advanced agent team collaboration.
    """
    # Create shared model for all agents
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create a shared memory for the team
    shared_memory = ShortTermMemory(max_messages=20)
    
    # Create agent roles with specialized query transformations
    
    # Analyst role focuses on breaking down problems and identifying key components
    def analyst_transform(query: str) -> str:
        return f"As a strategic analyst, break down this problem into key components: {query}\n\nConsider what information we already have and what we still need to determine."
    
    # Critic role focuses on identifying potential issues or weaknesses
    def critic_transform(query: str) -> str:
        return f"As a critical thinker, evaluate the current approach to this problem: {query}\n\nIdentify any logical flaws, missing information, or alternative perspectives that should be considered."
    
    # Innovator role focuses on creative solutions and novel approaches
    def innovator_transform(query: str) -> str:
        return f"As an innovative thinker, suggest creative approaches to this problem: {query}\n\nBuild upon the team's current insights and propose solutions that might not be immediately obvious."
    
    # Synthesizer role focuses on combining insights and creating a cohesive solution
    def synthesizer_transform(query: str) -> str:
        return f"As a synthesizing expert, combine our collective insights on this problem: {query}\n\nCreate a cohesive solution that addresses the key points raised by the team."
    
    # Define team configuration with specialized roles
    team_config = TeamConfiguration(
        roles={
            "Analyst": AgentRole(
                name="Analyst",
                description="Breaks down problems and identifies key components",
                query_transform=analyst_transform
            ),
            "Critic": AgentRole(
                name="Critic", 
                description="Identifies potential issues or weaknesses",
                query_transform=critic_transform
            ),
            "Innovator": AgentRole(
                name="Innovator",
                description="Proposes creative solutions and novel approaches",
                query_transform=innovator_transform
            ),
            "Synthesizer": AgentRole(
                name="Synthesizer",
                description="Combines insights and creates cohesive solutions",
                query_transform=synthesizer_transform
            )
        },
        default_role=None  # No default role; each agent must have a specific role
    )
    
    # Create the specialized agents
    analyst_agent = Agent.create(
        name="Analyst",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are a strategic analyst who excels at breaking down complex problems.",
            "Identify key components, available information, and knowledge gaps.",
            "Create structured analyses that help the team understand the problem space."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    critic_agent = Agent.create(
        name="Critic",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are a critical thinker who evaluates proposed approaches and solutions.",
            "Identify logical flaws, missing information, and unexplored alternatives.",
            "Your role is not to be negative, but to strengthen the team's thinking."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    innovator_agent = Agent.create(
        name="Innovator",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are an innovative thinker who generates creative solutions.",
            "Build upon the team's analysis to propose novel approaches.",
            "Don't hesitate to suggest unconventional ideas that might lead to breakthroughs."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    synthesizer_agent = Agent.create(
        name="Synthesizer",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are a synthesis expert who combines diverse perspectives into cohesive solutions.",
            "Integrate the team's insights, addressing conflicts and finding common ground.",
            "Create comprehensive solutions that reflect the collective intelligence of the team."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    # Create advanced team hooks for monitoring the collaboration
    hooks = AdvancedTeamHooks(
        # Basic team hooks
        on_agent_start=lambda agent_name, query: print(f"\n🚀 {agent_name} starting work..."),
        on_agent_end=lambda agent_name, result: print(f"✅ {agent_name} contributed"),
        on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
        on_final=lambda results: print(f"🏁 Team process completed with {len(results)} contributions"),
        
        # Advanced hooks for round-based collaboration
        on_round_start=lambda round_num, max_rounds: print(f"\n📊 Starting collaboration round {round_num}/{max_rounds}"),
        on_round_end=lambda round_num, contributions: print(f"📝 Round {round_num} complete with {len(contributions)} contributions"),
        on_convergence=lambda agent, content: print(f"🎯 {agent.name} proposed a solution that meets convergence criteria"),
        on_aggregation=lambda final_result: print(f"🧩 Final solution synthesized from {len(final_result.split())} words")
    )
    
    # Configure the advanced team
    team_options = AdvancedTeamOptions(
        shared_memory=shared_memory,
        team_config=team_config,
        hooks=hooks,
        debug=True
    )
    
    # Create the advanced agent team
    team = AdvancedAgentTeam(
        name="ProblemSolvingTeam",
        agents=[analyst_agent, critic_agent, innovator_agent, synthesizer_agent],
        options=team_options
    )
    
    # Enable shared memory so all agents can see each other's contributions
    team.enable_shared_memory()
    
    # Define a convergence check function to determine when the team has reached a solution
    def check_convergence(content: str) -> bool:
        """
        Check if the content represents a converged solution.
        
        A solution is converged when:
        1. It contains "FINAL SOLUTION:" indicating the team believes they've solved it
        
        Args:
            content: The content to check
            
        Returns:
            True if convergence criteria are met, False otherwise
        """
        # Check for explicit final solution marker
        has_final_marker = "FINAL SOLUTION:" in content.upper()
        
        return has_final_marker
    
    async def solve_problem_collaboratively(query: str, max_rounds: int = 5) -> str:
        """
        Use the advanced agent team to solve a complex problem collaboratively.
        
        Args:
            query: The problem to solve
            max_rounds: Maximum number of collaboration rounds
            
        Returns:
            The team's final solution
        """
        print(f"\n🔍 Team tackling problem: '{query}'")
        
        # Run the team in interleaved mode until convergence or max rounds
        # Each agent will see others' contributions through shared memory
        final_solution = await team.run_interleaved(
            user_query=query,
            max_rounds=max_rounds,
            is_converged=check_convergence
        )
        
        return final_solution
    
    # Example complex problems that benefit from collaborative problem-solving
    problems = [
        "Design a sustainable urban transportation system that reduces carbon emissions while improving accessibility for all residents.",
        
        #"Develop a strategy for a community to prepare for and adapt to increasing climate-related disasters with limited resources.",
        
        #"Create an education system that better prepares students for the rapidly changing job market of the future."
    ]
    
    for problem in problems:
        print("\n" + "=" * 100)
        print(f"COMPLEX PROBLEM: {problem}")
        print("=" * 100)
        
        solution = await solve_problem_collaboratively(problem)
        
        print("\n" + "=" * 40 + " COLLABORATIVE SOLUTION " + "=" * 40)
        print(solution)
        print("=" * 100)
        
        # Add a pause between problems
        if problem != problems[-1]:
            print("\nMoving to next problem in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main()) 