from typing import List, Callable, Union, Awaitable, TypeVar, overload, cast

from ..agent import Agent

# Define a type for the routing function
RoutingFunction = Callable[[str], Union[int, Awaitable[int]]]  # returns index of the agent to call

class AgentRouter:
    """
    A simple router that directs queries to specific agents based on a routing function.
    """
    
    def __init__(self, agents: List[Agent], routing_fn: RoutingFunction):
        """
        Initialize the agent router.
        
        Args:
            agents: List of agents to route between
            routing_fn: Function that determines which agent to use for a query
        """
        self.agents = agents
        self.routing_fn = routing_fn
    
    async def run(self, query: str) -> str:
        """
        Route a query to the appropriate agent and run it.
        
        Args:
            query: The user query to route
            
        Returns:
            The result from the selected agent
        """
        # Handle both synchronous and asynchronous routing functions
        result = self.routing_fn(query)
        if isinstance(result, int):
            idx = result
        else:
            idx = await result
        
        agent = self.agents[idx]
        return await agent.run(query) 