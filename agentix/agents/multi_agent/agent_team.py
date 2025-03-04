from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass

from ..agent import Agent


@dataclass
class TeamHooks:
    """Lifecycle hooks for agent team execution."""
    on_agent_start: Optional[Callable[[str, str], None]] = None
    on_agent_end: Optional[Callable[[str, str], None]] = None
    on_error: Optional[Callable[[str, Exception], None]] = None
    on_final: Optional[Callable[[List[str]], None]] = None


class AgentTeam:
    """
    Orchestrates multiple agents.
    Could be parallel calls or passing context from one to the next.
    """
    
    def __init__(self, name: str, agents: List[Agent]):
        """
        Initialize an agent team.
        
        Args:
            name: The name of the team
            agents: List of agents in the team
        """
        self.name = name
        self.agents = agents
    
    async def run_in_parallel(self, query: str, hooks: Optional[TeamHooks] = None) -> List[str]:
        """
        Runs all agents in parallel on the same input query.
        Each agent processes the query independently and returns its result.
        
        Args:
            query: The user input or initial query
            hooks: Optional TeamHooks for debugging/logging steps and errors
            
        Returns:
            An array of output strings from each agent
        """
        import asyncio
        
        async def run_agent(agent: Agent) -> str:
            if hooks and hooks.on_agent_start:
                hooks.on_agent_start(agent.name, query)
            
            try:
                output = await agent.run(query)
                if hooks and hooks.on_agent_end:
                    hooks.on_agent_end(agent.name, output)
                
                return output
            
            except Exception as err:
                if hooks and hooks.on_error:
                    hooks.on_error(agent.name, err)
                
                raise  # re-raise the exception
        
        # Create a task for each agent
        tasks = [run_agent(agent) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        if hooks and hooks.on_final:
            hooks.on_final(results)
        
        return results
    
    async def run_sequential(self, query: str, hooks: Optional[TeamHooks] = None) -> str:
        """
        Runs agents sequentially, passing each agent's output as input to the next agent.
        Forms a processing pipeline where agents transform the data in sequence.
        
        Args:
            query: The user input or initial query
            hooks: Optional TeamHooks for debugging/logging steps and errors
            
        Returns:
            The final output string after all agents have processed it
        """
        current_input = query
        
        for agent in self.agents:
            if hooks and hooks.on_agent_start:
                hooks.on_agent_start(agent.name, current_input)
            
            try:
                output = await agent.run(current_input)
                if hooks and hooks.on_agent_end:
                    hooks.on_agent_end(agent.name, output)
                
                current_input = output
            
            except Exception as err:
                if hooks and hooks.on_error:
                    hooks.on_error(agent.name, err)
                
                raise  # re-raise the exception
        
        if hooks and hooks.on_final:
            hooks.on_final([current_input])
        
        return current_input
    
    async def run_in_parallel_safe(self, query: str, hooks: Optional[TeamHooks] = None) -> List[Dict[str, Any]]:
        """
        A "safe" version of run_in_parallel that catches errors from individual agents.
        
        Args:
            query: The user input or initial query
            hooks: Optional TeamHooks for debugging/logging steps and errors
            
        Returns:
            An array of results, each containing success status and output.
            For successful agents, {'success': True, 'output': string}.
            For failed agents, {'success': False, 'output': error message}.
        """
        import asyncio
        
        async def run_agent_safe(agent: Agent) -> Dict[str, Any]:
            if hooks and hooks.on_agent_start:
                hooks.on_agent_start(agent.name, query)
            
            try:
                out = await agent.run(query)
                if hooks and hooks.on_agent_end:
                    hooks.on_agent_end(agent.name, out)
                
                return {"success": True, "output": out}
            
            except Exception as err:
                if hooks and hooks.on_error:
                    hooks.on_error(agent.name, err)
                
                return {"success": False, "output": str(err)}
        
        # Create a task for each agent
        tasks = [run_agent_safe(agent) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        if hooks and hooks.on_final:
            hooks.on_final([r["output"] for r in results])
        
        return results
    
    async def run_sequential_safe(self, query: str, stop_on_error: bool, hooks: Optional[TeamHooks] = None) -> List[str]:
        """
        A "safe" version of run_sequential that catches errors from individual agents.
        
        Args:
            query: The user input or initial query
            stop_on_error: If true, stop executing further agents after the first error.
                          If false, record the error and keep going with the next agent.
            hooks: Optional TeamHooks for debugging/logging steps and errors
            
        Returns:
            An array of output strings from each agent in sequence
        """
        outputs: List[str] = []
        current_input = query
        
        for agent in self.agents:
            # onAgentStart hook
            if hooks and hooks.on_agent_start:
                hooks.on_agent_start(agent.name, current_input)
            
            try:
                out = await agent.run(current_input)
                
                # onAgentEnd hook
                if hooks and hooks.on_agent_end:
                    hooks.on_agent_end(agent.name, out)
                
                # record output, pass to next agent
                outputs.append(out)
                current_input = out
            
            except Exception as err:
                # onError hook
                if hooks and hooks.on_error:
                    hooks.on_error(agent.name, err)
                
                # record the error as an output
                error_msg = f"Error from agent {agent.name}: {str(err)}"
                outputs.append(error_msg)
                
                # break or continue based on stop_on_error
                if stop_on_error:
                    break
            
        # onFinal hook after the sequence completes
        if hooks and hooks.on_final:
            hooks.on_final(outputs)
        
        return outputs
    
    async def aggregate_results(self, query: str) -> str:
        """
        Run all agents in parallel and aggregate their results.
        
        Args:
            query: The query to run
            
        Returns:
            Aggregated results from all agents
        """
        results = await self.run_in_parallel(query)
        return "\n---\n".join(results)  # Combine all results 