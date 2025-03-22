import time
from typing import List, Dict, Any, Optional, Callable, Union, Awaitable, TypeVar
from dataclasses import dataclass, field

from ..agent import Agent
from .agent_team import AgentTeam, TeamHooks
from ...memory.memory import Memory
from ...utils.debug_logger import DebugLogger


@dataclass
class AdvancedTeamHooks(TeamHooks):
    """
    AdvancedTeamHooks extends the basic TeamHooks with:
    1) Round start and end events
    2) Convergence event when an agent's output meets criteria
    3) Aggregation event when all agents have contributed
    """
    on_round_start: Optional[Callable[[int, int], None]] = None
    on_round_end: Optional[Callable[[int, Dict[str, 'AgentContribution']], None]] = None
    on_convergence: Optional[Callable[[Agent, str], None]] = None
    on_aggregation: Optional[Callable[[str], None]] = None


@dataclass
class AgentRole:
    """Defines a specialization for an agent in a team."""
    name: str
    description: str
    query_transform: Callable[[str], str]


@dataclass
class TeamConfiguration:
    """Defines the roles and specializations for a team."""
    roles: Dict[str, AgentRole]
    default_role: Optional[AgentRole] = None


@dataclass
class AgentContribution:
    """Tracks an agent's contribution with metadata."""
    agent: Agent
    content: str
    has_final_answer: bool
    timestamp: Optional[int] = None


@dataclass
class AdvancedTeamOptions:
    """
    AdvancedTeamOptions extends the basic TeamOptions with:
    1) Shared memory for agents to see the same conversation context
    2) Team configuration with roles and specializations
    3) Debug flag for verbose logging
    """
    shared_memory: Optional[Memory] = None
    team_config: Optional[TeamConfiguration] = None
    hooks: Optional[AdvancedTeamHooks] = None
    debug: bool = False


class AdvancedAgentTeam(AgentTeam):
    """
    AdvancedAgentTeam extends the basic AgentTeam with:
    1) Shared memory (optional) so each agent sees the same conversation context
    2) Configurable agent roles and specializations
    3) Interleaved run method with role-based query transformation
    """
    
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        options: AdvancedTeamOptions
    ):
        """
        Initialize an advanced agent team.
        
        Args:
            name: The name of the team
            agents: List of agents in the team
            options: Advanced team options
        """
        super().__init__(name, agents)
        self.shared_memory = options.shared_memory
        self.team_config = options.team_config
        self.hooks = options.hooks
        self.logger = DebugLogger(options.debug)
    
    def set_team_configuration(self, config: TeamConfiguration) -> None:
        """
        Configure team roles and specializations.
        
        Args:
            config: Team configuration with roles
        """
        self.team_config = config
        self.logger.log("Team configuration updated", {
            "roles": list(config.roles.keys())
        })
    
    def enable_shared_memory(self) -> None:
        """
        If a sharedMemory is provided, each Agent's memory references
        the same memory object. This preserves agent tools and other configurations.
        """
        if not self.shared_memory:
            self.logger.warn("No shared memory set. Nothing to enable.")
            return
        
        for agent in self.agents:
            # Store the agent's tools before replacing the memory
            agent_tools = agent.tools
            
            # Set the shared memory
            agent.memory = self.shared_memory
            
            # Ensure the agent's tools are preserved 
            if agent_tools:
                # Make sure the tools are properly registered with the agent
                agent.tools = agent_tools
                self.logger.log(f"Preserved {len(agent_tools)} tools for agent {agent.name}")
        
        self.logger.log("Shared memory enabled for all agents")
    
    def get_agent_role(self, agent: Agent) -> Optional[AgentRole]:
        """
        Get the role for a specific agent.
        
        Args:
            agent: The agent to get role for
            
        Returns:
            The agent's role, or None if not found
        """
        if not self.team_config:
            return None
        
        # Check for specific role assignment
        role = self.team_config.roles.get(agent.name)
        if role:
            return role
        
        # Fall back to default role if specified
        return self.team_config.default_role
    
    def get_specialized_query(self, agent: Agent, base_query: str) -> str:
        """
        Transform query based on agent's role.
        
        Args:
            agent: The agent to transform query for
            base_query: The original query
            
        Returns:
            The transformed query
        """
        role = self.get_agent_role(agent)
        if not role:
            self.logger.warn(f"No role defined for agent {agent.name}, using base query")
            return base_query
        
        try:
            transformed_query = role.query_transform(base_query)
            self.logger.log(f"Query transformed for {agent.name} ({role.name})", {
                "original": base_query,
                "transformed": transformed_query
            })
            return transformed_query
        except Exception as error:
            self.logger.error(f"Error transforming query for {agent.name}", {"error": str(error)})
            return base_query
    
    async def track_contribution(
        self,
        agent: Agent,
        content: str,
        has_converged: bool
    ) -> None:
        """
        Improved contribution tracking with metadata.
        
        Args:
            agent: The agent that contributed
            content: The content of the contribution
            has_converged: Whether the content has converged
        """
        role = self.get_agent_role(agent)
        self.logger.log(f"Tracking contribution from {agent.name}", {
            "role": role.name if role else 'Unspecified',
            "content_length": len(content),
            "has_converged": has_converged
        })
        
        # Add to shared memory with metadata
        if self.shared_memory:
            await self.shared_memory.add_message({
                "role": "assistant",
                "content": content,
                "metadata": {
                    "agent_name": agent.name,
                    "role_name": role.name if role else None,
                    "timestamp": int(time.time() * 1000),
                    "has_converged": has_converged
                }
            })
    
    def build_team_system_prompt(self) -> str:
        """
        Build team-level system prompt.
        
        Returns:
            A system prompt for the team
        """
        role_descriptions = ""
        if self.team_config:
            role_descriptions = "\n".join([
                f"{name}: {role.description}"
                for name, role in self.team_config.roles.items()
            ])
        
        return f"""
This is a collaborative analysis by multiple expert agents.
Each agent has a specific role and expertise:
{role_descriptions}

Agents will build on each other's insights while maintaining their specialized focus.
Final responses should be marked with "FINAL ANSWER:".
"""
    
    async def initialize_shared_context(self, query: str) -> None:
        """
        Initialize shared memory with system and user context.
        
        Args:
            query: The user query to initialize with
        """
        if not self.shared_memory:
            return
        
        # Clear any previous conversation
        await self.shared_memory.clear()
        
        # Add system context
        await self.shared_memory.add_message({
            "role": "system",
            "content": self.build_team_system_prompt()
        })
        
        # Add user query
        await self.shared_memory.add_message({
            "role": "user",
            "content": query
        })
        
        self.logger.log("Shared context initialized", {"query": query})
    
    def have_all_agents_contributed(
        self,
        contributions: Dict[str, AgentContribution]
    ) -> bool:
        """
        Check if all agents have contributed.
        
        Args:
            contributions: Map of agent name to contribution
            
        Returns:
            True if all agents have contributed, False otherwise
        """
        return len(contributions) == len(self.agents)
    
    def have_all_agents_converged(
        self,
        contributions: Dict[str, AgentContribution]
    ) -> bool:
        """
        Check if all agents have contributed AND converged.
        
        Args:
            contributions: Map of agent name to contribution
            
        Returns:
            True if all agents have contributed and converged, False otherwise
        """
        if not self.have_all_agents_contributed(contributions):
            return False
        
        # Check if all contributions have converged
        all_converged = all(
            contribution.has_final_answer
            for contribution in contributions.values()
        )
        
        self.logger.log("Checking convergence status", {
            "total_agents": len(self.agents),
            "contributing_agents": len(contributions),
            "all_converged": all_converged,
            "convergence_status": [
                {
                    "agent": name,
                    "has_converged": c.has_final_answer
                }
                for name, c in contributions.items()
            ]
        })
        
        return all_converged
    
    async def run_interleaved(
        self,
        user_query: str,
        max_rounds: int,
        is_converged: Callable[[str], Union[bool, Awaitable[bool]]],
        require_all_agents: bool = False
    ) -> str:
        """
        Interleaved/Chat-like approach where agents build on each other's contributions.
        
        Args:
            user_query: The user query to run
            max_rounds: Maximum number of rounds to run
            is_converged: Function to check if content has converged
            require_all_agents: Whether to require all agents to contribute before stopping
            
        Returns:
            The final result
        """
        if require_all_agents:
            self.logger.log("requireAllAgents is true. Waiting for all agents to contribute.")
            self.logger.log(f"Total agents: {len(self.agents)}\n")
        
        self.logger.log("Starting interleaved team workflow", {
            "query": user_query,
            "max_rounds": max_rounds,
            "require_all_agents": require_all_agents,
            "team_size": len(self.agents)
        })
        
        # Track contributions per round
        contributions: Dict[str, AgentContribution] = {}
        current_round = 0
        final_answer = None
        
        # Initialize shared memory if enabled
        await self.initialize_shared_context(user_query)
        
        # Main interaction loop
        while current_round < max_rounds:
            current_round += 1
            self.logger.log(f"Starting round {current_round}/{max_rounds}")
            
            if self.hooks and self.hooks.on_round_start:
                self.hooks.on_round_start(current_round, max_rounds)
            
            # Each agent takes a turn in the current round
            for agent in self.agents:
                self.logger.log(f"Round {current_round}: {agent.name}'s turn")
                
                if self.hooks and self.hooks.on_agent_start:
                    self.hooks.on_agent_start(agent.name, user_query)
                
                # Get agent's specialized query based on their role
                agent_query = self.get_specialized_query(agent, user_query)
                
                try:
                    agent_output = await agent.run(agent_query)
                    self.logger.log(f"{agent.name} response received", {"agent_output": agent_output})
                    
                    # Check if this output meets convergence criteria
                    has_converged_result = is_converged(agent_output)
                    if isinstance(has_converged_result, bool):
                        has_converged = has_converged_result
                    else:
                        has_converged = await has_converged_result
                    
                    # Track agent contribution with metadata
                    contributions[agent.name] = AgentContribution(
                        agent=agent,
                        content=agent_output,
                        has_final_answer=has_converged,
                        timestamp=int(time.time() * 1000)
                    )
                    
                    await self.track_contribution(agent, agent_output, has_converged)
                    
                    if self.hooks and self.hooks.on_agent_end:
                        self.hooks.on_agent_end(agent.name, agent_output)
                    
                    # Check convergence conditions
                    if has_converged:
                        if self.hooks and self.hooks.on_convergence:
                            self.hooks.on_convergence(agent, agent_output)
                        
                        if not require_all_agents:
                            # Stop at first convergence if not requiring all agents
                            final_answer = agent_output
                            self.logger.log(f"{agent.name} met convergence criteria, stopping early")
                            break
                        elif self.have_all_agents_converged(contributions):
                            # Stop only if all agents have contributed AND converged
                            final_answer = self.combine_contributions(contributions)
                            self.logger.log("All agents have contributed and converged")
                            break
                
                except Exception as error:
                    self.logger.error(f"Error during {agent.name}'s turn", {"error": str(error)})
                    if self.hooks and self.hooks.on_error:
                        self.hooks.on_error(agent.name, error)
                    
                    contributions[agent.name] = AgentContribution(
                        agent=agent,
                        content=f"Error during execution: {str(error)}",
                        has_final_answer=False,
                        timestamp=int(time.time() * 1000)
                    )
            
            if self.hooks and self.hooks.on_round_end:
                self.hooks.on_round_end(current_round, contributions)
            
            # Break if we found a final answer
            if final_answer:
                self.logger.log("Convergence achieved", {"final_answer": final_answer})
                break
            
            # If all agents have contributed but not all converged, log and continue
            if (
                self.have_all_agents_contributed(contributions) and
                not self.have_all_agents_converged(contributions)
            ):
                self.logger.log("All agents contributed but not all converged, continuing to next round")
                continue
            
            # Check if we should continue
            if current_round == max_rounds:
                self.logger.warn(f"Maximum rounds ({max_rounds}) reached without convergence")
        
        # If no final answer was reached, combine all contributions
        if not final_answer:
            self.logger.warn("No convergence reached, combining all contributions")
            final_answer = self.combine_contributions(contributions)
        
        formatted_output = self.format_final_output(final_answer, contributions)
        
        if self.hooks and self.hooks.on_aggregation:
            self.hooks.on_aggregation(formatted_output)
        
        return formatted_output
    
    def combine_contributions(self, contributions: Dict[str, AgentContribution]) -> str:
        """
        Combine all agent contributions into a final response.
        
        Args:
            contributions: Map of agent name to contribution
            
        Returns:
            Combined contributions as a string
        """
        return "\n---\n".join([
            f"[{c.agent.name}{' (' + self.get_agent_role(c.agent).name + ')' if self.get_agent_role(c.agent) else ''}]\n{c.content}"
            for c in sorted(
                contributions.values(),
                key=lambda x: x.timestamp or 0  # Sort by timestamp if available
            )
        ])
    
    def format_final_output(
        self,
        final_answer: str,
        contributions: Dict[str, AgentContribution]
    ) -> str:
        """
        Format the final output with additional context if needed.
        
        Args:
            final_answer: The final answer content
            contributions: Map of agent name to contribution
            
        Returns:
            Formatted final output
        """
        contributing_agents = ", ".join([
            f"{name}{' ✓' if c.has_final_answer else ''}"
            for name, c in contributions.items()
        ])
        
        header = f"Team Response (Contributors: {contributing_agents})\n{'=' * 40}\n"
        return f"{header}{final_answer}"
    
    async def run_sequential(self, query: str, hooks: Optional[TeamHooks] = None) -> str:
        """
        Runs agents sequentially with specialized query transformations.
        This override properly transforms queries based on agent roles and ensures
        proper workflow between agents.
        
        Args:
            query: The user input or initial query
            hooks: Optional TeamHooks for debugging/logging steps and errors
            
        Returns:
            The final output string after all agents have processed it
        """
        self.logger.log(f"[AdvancedAgentTeam:{self.name}] Starting sequential execution", {
            "query": query,
            "agents": [agent.name for agent in self.agents]
        })
        
        # Initialize shared memory if available
        await self.initialize_shared_context(query)
        
        current_input = query
        all_contributions = []
        
        for idx, agent in enumerate(self.agents):
            agent_role = self.get_agent_role(agent)
            role_name = agent_role.name if agent_role else "Unspecified"
            
            self.logger.log(f"[AdvancedAgentTeam:{self.name}] Running agent {idx+1}/{len(self.agents)}: {agent.name} ({role_name})")
            
            # Get specialized query for this agent based on its role
            specialized_query = self.get_specialized_query(agent, current_input)
            
            # Call hooks if available
            if hooks and hooks.on_agent_start:
                hooks.on_agent_start(agent.name, specialized_query)
            
            try:
                # Run the agent with its specialized query
                self.logger.log(f"[AdvancedAgentTeam:{self.name}] Running {agent.name} with specialized query", {
                    "original": current_input,
                    "transformed": specialized_query
                })
                
                output = await agent.run(specialized_query)
                
                self.logger.log(f"[AdvancedAgentTeam:{self.name}] {agent.name} completed", {
                    "output_length": len(output)
                })
                
                # Store contribution for reporting
                all_contributions.append({
                    "agent": agent.name,
                    "role": role_name,
                    "output": output
                })
                
                # Call hooks if available
                if hooks and hooks.on_agent_end:
                    hooks.on_agent_end(agent.name, output)
                
                # Pass to the next agent
                current_input = output
                
                # Add to shared memory as a contribution
                await self.track_contribution(agent, output, False)
                
            except Exception as err:
                error_msg = f"Error from agent {agent.name}: {str(err)}"
                self.logger.error(f"[AdvancedAgentTeam:{self.name}] Error in sequential execution", {
                    "agent": agent.name,
                    "error": str(err)
                })
                
                if hooks and hooks.on_error:
                    hooks.on_error(agent.name, err)
                
                # Add error to contributions
                all_contributions.append({
                    "agent": agent.name,
                    "role": role_name,
                    "error": str(err)
                })
                
                # Continue with a default message
                current_input = error_msg
                
                # Add error to shared memory
                if self.shared_memory:
                    await self.shared_memory.add_message({
                        "role": "system",
                        "content": error_msg
                    })
        
        # Handle final result with hooks
        if hooks and hooks.on_final:
            hooks.on_final([current_input])
        
        self.logger.log(f"[AdvancedAgentTeam:{self.name}] Sequential execution completed", {
            "contributions": len(all_contributions)
        })
        
        return current_input 
    
    async def run_in_parallel(self, query: str, hooks: Optional[TeamHooks] = None) -> List[str]:
        """
        Runs all agents in parallel with specialized query transformations.
        This override properly transforms queries based on agent roles and ensures
        proper tracking of contributions.
        
        Args:
            query: The user input or initial query
            hooks: Optional TeamHooks for debugging/logging steps and errors
            
        Returns:
            An array of output strings from each agent
        """
        import asyncio
        
        self.logger.log(f"[AdvancedAgentTeam:{self.name}] Starting parallel execution", {
            "query": query,
            "agents": [agent.name for agent in self.agents]
        })
        
        # Initialize shared memory if available
        await self.initialize_shared_context(query)
        
        async def run_agent_with_role(agent: Agent) -> str:
            agent_role = self.get_agent_role(agent)
            role_name = agent_role.name if agent_role else "Unspecified"
            
            self.logger.log(f"[AdvancedAgentTeam:{self.name}] Running agent: {agent.name} ({role_name})")
            
            # Get specialized query for this agent based on its role
            specialized_query = self.get_specialized_query(agent, query)
            
            # Call hooks if available
            if hooks and hooks.on_agent_start:
                hooks.on_agent_start(agent.name, specialized_query)
            
            try:
                # Run the agent with its specialized query
                self.logger.log(f"[AdvancedAgentTeam:{self.name}] Running {agent.name} with specialized query", {
                    "original": query,
                    "transformed": specialized_query
                })
                
                output = await agent.run(specialized_query)
                
                self.logger.log(f"[AdvancedAgentTeam:{self.name}] {agent.name} completed", {
                    "output_length": len(output)
                })
                
                # Call hooks if available
                if hooks and hooks.on_agent_end:
                    hooks.on_agent_end(agent.name, output)
                
                # Add to shared memory as a contribution
                await self.track_contribution(agent, output, False)
                
                return output
                
            except Exception as err:
                error_msg = f"Error from agent {agent.name}: {str(err)}"
                self.logger.error(f"[AdvancedAgentTeam:{self.name}] Error in parallel execution", {
                    "agent": agent.name,
                    "error": str(err)
                })
                
                if hooks and hooks.on_error:
                    hooks.on_error(agent.name, err)
                
                # Add error to shared memory
                if self.shared_memory:
                    await self.shared_memory.add_message({
                        "role": "system",
                        "content": error_msg
                    })
                
                return error_msg
        
        # Create a task for each agent
        tasks = [run_agent_with_role(agent) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        # Handle final results with hooks
        if hooks and hooks.on_final:
            hooks.on_final(results)
        
        self.logger.log(f"[AdvancedAgentTeam:{self.name}] Parallel execution completed", {
            "results": len(results)
        })
        
        return results