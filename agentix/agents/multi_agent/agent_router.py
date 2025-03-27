import json
import time
from typing import List, Dict, Any, Optional, Set, Callable, Union, Awaitable
from dataclasses import dataclass

from ..agent import Agent
from ...llms import OpenAIChat, LLM
from ...utils.debug_logger import DebugLogger


# Define a type for the routing function
RoutingFunction = Callable[[str], Union[int, Awaitable[int]]]  # returns index of the agent to call


@dataclass
class AgentCapability:
    """Agent capability metadata."""
    name: str
    description: str
    keywords: List[str]
    examples: List[str]


@dataclass
class RouterOptions:
    """Options for AgentRouter."""
    use_llm: bool = False
    debug: bool = False
    fallback_index: Optional[int] = None
    confidence_threshold: float = 0.7
    router_llm: Optional[LLM] = None  # Optional custom LLM for routing


@dataclass
class RoutingMetadata:
    """
    Metadata for routing decisions.
    Used for logging and analysis.
    """
    timestamp: int
    query: str
    selected_agent: str
    confidence: float
    reasoning: Optional[str] = None


class AgentRouter:
    """
    Routes queries to specialized agents based on capabilities.
    Features:
    1) Rule-based or LLM-powered routing
    2) Capability-based agent selection
    3) Confidence thresholds and fallbacks
    4) Routing history tracking
    """
    
    def __init__(
        self,
        agents: List[Agent],
        capabilities: Optional[Dict[int, AgentCapability]] = None,
        routing_fn: Optional[RoutingFunction] = None,
        options: Optional[RouterOptions] = None
    ):
        """
        Initialize an agent router.
        
        Args:
            agents: List of agents to route between
            capabilities: Map of agent index to capability metadata (required for capability-based routing)
            routing_fn: Optional custom routing function (if not using capability-based routing)
            options: Configuration options for the router
        """
        self.agents = agents
        self.capabilities = capabilities or {}
        self.routing_fn = routing_fn
        options = options or RouterOptions()
        
        self.logger = DebugLogger(options.debug)
        self.fallback_index = options.fallback_index if options.fallback_index is not None else len(agents) - 1
        self.confidence_threshold = options.confidence_threshold
        
        self.router_llm = None
        if options.use_llm:
            # Use provided LLM if available, otherwise create default
            if options.router_llm:
                self.router_llm = options.router_llm
            else:
                self.router_llm = OpenAIChat(
                    model="gpt-4o-mini",
                    temperature=0.2  # Lower temperature for more consistent routing
                )
        
        self.routing_history: List[RoutingMetadata] = []
    
    async def run(self, query: str) -> str:
        """
        Route a query to the appropriate agent and run it.
        
        Args:
            query: The user query to route
            
        Returns:
            The result from the selected agent
        """
        if self.routing_fn:
            # Use custom routing function if provided
            result = self.routing_fn(query)
            if isinstance(result, int):
                idx = result
            else:
                idx = await result
            return await self.agents[idx].run(query)
        
        # Use capability-based routing
        routing_result = await self.route_query(query)
        
        self.logger.log("Routing decision", {
            "query": query,
            "selected_agent": self.agents[routing_result["agent_index"]].name,
            "confidence": routing_result["confidence"],
            "reasoning": routing_result.get("reasoning")
        })
        
        # Track routing metadata
        self.routing_history.append(RoutingMetadata(
            timestamp=int(time.time() * 1000),
            query=query,
            selected_agent=self.agents[routing_result["agent_index"]].name,
            confidence=routing_result["confidence"],
            reasoning=routing_result.get("reasoning")
        ))
        
        # Use fallback if confidence is too low
        if routing_result["confidence"] < self.confidence_threshold:
            self.logger.warn(
                f"Low confidence routing ({routing_result['confidence']}), using fallback agent"
            )
            return await self.agents[self.fallback_index].run(query)
        
        return await self.agents[routing_result["agent_index"]].run(query)
    
    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        Main routing logic that can use either rule-based or LLM-based routing.
        
        Args:
            query: The query to route
            
        Returns:
            Dictionary with agent_index, confidence, and reasoning
        """
        if self.router_llm:
            try:
                llm_result = await self.route_with_llm(query)
                self.logger.log('LLM routing result', llm_result)
                
                if llm_result["confidence"] >= self.confidence_threshold:
                    return llm_result
                
                self.logger.log('LLM routing confidence too low, trying rule-based', {
                    "confidence": llm_result["confidence"],
                    "threshold": self.confidence_threshold
                })
            except Exception as error:
                self.logger.error('LLM routing failed, falling back to rule-based', {"error": str(error)})
        
        # Fall back to rule-based routing
        return await self.route_with_rules(query)
    
    async def route_with_rules(self, query: str) -> Dict[str, Any]:
        """
        Rule-based routing using capabilities and keywords.
        
        Args:
            query: The query to route
            
        Returns:
            Dictionary with agent_index, confidence, and reasoning
        """
        lower_query = query.lower()
        best_match = {
            "index": self.fallback_index,
            "confidence": 0,
            "matches": 0
        }
        
        # Check each agent's capabilities
        for index, capability in self.capabilities.items():
            matches = 0
            total_keywords = len(capability.keywords)
            
            # Check keywords
            for keyword in capability.keywords:
                if keyword.lower() in lower_query:
                    matches += 1
            
            # Check examples for similar patterns
            for example in capability.examples:
                if self.calculate_similarity(query, example) > 0.7:
                    matches += 1
            
            confidence = matches / (total_keywords + len(capability.examples))
            
            if confidence > best_match["confidence"]:
                best_match = {"index": index, "confidence": confidence, "matches": matches}
        
        return {
            "agent_index": best_match["index"],
            "confidence": best_match["confidence"],
            "reasoning": f"Matched {best_match['matches']} keywords/patterns"
        }
    
    async def route_with_llm(self, query: str) -> Dict[str, Any]:
        """
        LLM-based intelligent routing.
        
        Args:
            query: The query to route
            
        Returns:
            Dictionary with agent_index, confidence, and reasoning
        """
        try:
            prompt = self.build_routing_prompt(query)
            response = await self.router_llm.call([{
                "role": "user",
                "content": prompt
            }])
            
            # Clean the response - remove any markdown formatting or extra text
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            
            try:
                parsed = json.loads(cleaned_response)
                
                # Validate the parsed response
                if (
                    not isinstance(parsed.get("selectedAgent"), int) or
                    not isinstance(parsed.get("confidence"), (int, float)) or
                    not isinstance(parsed.get("reasoning"), str)
                ):
                    raise ValueError("Invalid response format")
                
                # Ensure values are within expected ranges
                return {
                    "agent_index": min(max(0, parsed["selectedAgent"]), len(self.agents) - 1),
                    "confidence": min(max(0, float(parsed["confidence"])), 1),
                    "reasoning": parsed["reasoning"]
                }
            except Exception as parse_error:
                self.logger.error('Failed to parse LLM response', {
                    "response": cleaned_response,
                    "error": str(parse_error)
                })
                raise ValueError(f"Failed to parse LLM response: {str(parse_error)}")
        
        except Exception as error:
            self.logger.error('LLM routing error', {"error": str(error)})
            raise
    
    def build_routing_prompt(self, query: str) -> str:
        """
        Build prompt for LLM-based routing.
        
        Args:
            query: The query to route
            
        Returns:
            Prompt for LLM-based routing
        """
        capabilities_desc = []
        for idx, cap in self.capabilities.items():
            capabilities_desc.append(
                f"Agent {idx}: {cap.name}\n"
                f"Description: {cap.description}\n"
                f"Example queries: {', '.join(cap.examples)}"
            )
        
        return f"""You are a routing system that determines which specialized agent should handle a user query.

Available agents and their capabilities:
{"\n\n".join(capabilities_desc)}

Additionally, there is a general-purpose agent (index: {self.fallback_index}) for queries that don't clearly match any specialist.

Analyze this query and determine the best agent to handle it:
"{query}"

Respond in this exact JSON format:
{{
  "selectedAgent": <agent index number>,
  "confidence": <number between 0 and 1>,
  "reasoning": "<brief explanation of your choice>"
}}

Your response should ONLY contain the JSON object, nothing else.
Choose the general agent ({self.fallback_index}) if no specialist is clearly suitable.
Set confidence above 0.8 only if you're very sure about the routing."""
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Simple similarity calculation for example matching.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(str1.lower().split(' '))
        words2 = set(str2.lower().split(' '))
        intersection = words1.intersection(words2)
        return len(intersection) / max(len(words1), len(words2))
    
    def get_routing_history(self) -> List[RoutingMetadata]:
        """
        Get routing history for analysis.
        
        Returns:
            List of routing metadata entries
        """
        return self.routing_history
    
    def set_capability(self, agent_index: int, capability: AgentCapability) -> None:
        """
        Add or update agent capabilities.
        
        Args:
            agent_index: Index of the agent to update
            capability: New capability metadata
        """
        self.capabilities[agent_index] = capability
        self.logger.log(f"Updated capabilities for agent {agent_index}", vars(capability)) 