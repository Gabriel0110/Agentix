from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .tools.tools import Tool
from .memory.memory import Memory
from .llms import LLM


class Planner(ABC):
    """
    Abstract base class for a Planner that can produce a plan string or structured plan
    from a user query, the known tools, and conversation memory.
    """
    
    @abstractmethod
    async def generate_plan(
        self,
        user_query: str,
        tools: List[Tool],
        memory: Memory
    ) -> str:
        """
        Generate a plan based on the user query, available tools, and memory.
        
        Args:
            user_query: The user's query or request
            tools: List of available tools
            memory: Memory containing conversation history
            
        Returns:
            A string representation of the plan
        """
        pass


class SimpleLLMPlanner(Planner):
    """
    A naive LLM-based planner that uses prompt instructions for plan generation.
    """
    
    def __init__(self, planner_model: LLM):
        """
        Initialize the simple LLM planner.
        
        Args:
            planner_model: The LLM instance to use for planning (OpenAIChat or TogetherChat)
        """
        self.planner_model = planner_model
    
    async def generate_plan(
        self,
        user_query: str,
        tools: List[Tool],
        memory: Memory
    ) -> str:
        """
        Generate a plan using the LLM based on the user query, available tools, and memory.
        
        Args:
            user_query: The user's query or request
            tools: List of available tools
            memory: Memory containing conversation history
            
        Returns:
            A string representation of the plan in JSON format
        """
        context = await memory.get_context()
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])
        
        context_str = "\n".join([
            f"{m['role'] if isinstance(m, dict) else m.role}: {m['content'] if isinstance(m, dict) else m.content}"
            for m in context
        ])
        
        plan_prompt = [
            {"role": "system", "content": "You are a task planning assistant."},
            {
                "role": "user",
                "content": f"""
User query: "{user_query}"

Tools available:
{tool_descriptions}

Context:
{context_str}

Plan the steps required to solve the user's query. You may also refine plans based on intermediate results. Use JSON format like this:
[
  {{ "action": "tool", "details": "ToolName" }},
  {{ "action": "message", "details": "Message to user or model" }},
  {{ "action": "complete", "details": "FINAL ANSWER" }}
]
"""
            }
        ]
        
        return await self.planner_model.call(plan_prompt) 