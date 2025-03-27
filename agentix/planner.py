"""
Planner module for generating execution plans.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .tools.tools import Tool
from .memory.memory import Memory
from .llms import LLM
from .utils.debug_logger import DebugLogger

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
    An enhanced LLM-based planner that generates tool-aware execution plans.
    """
    
    def __init__(self, planner_model: LLM, debug: bool = False):
        """
        Initialize the simple LLM planner.
        
        Args:
            planner_model: The LLM instance to use for planning (OpenAIChat or TogetherChat)
            debug: Whether to enable debug logging
        """
        self.planner_model = planner_model
        self.debug = debug
        self.logger = DebugLogger(debug)
    
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
        self.logger.log("[Planner] Generating plan", {
            "query": user_query,
            "num_tools": len(tools) if tools else 0
        })
        
        context = await memory.get_context()
        
        # Build detailed tool descriptions including parameters
        tool_descriptions = []
        for t in tools:
            desc = [f"Tool: {t.name}", f"Description: {t.description}"]
            if t.parameters:
                desc.append("Parameters:")
                for p in t.parameters:
                    required = "(required)" if p.required else "(optional)"
                    desc.append(f"  - {p.name}: {p.type} {required}")
                    if p.description:
                        desc.append(f"    {p.description}")
            if t.docs and t.docs.usage_example:
                desc.append(f"Example: {t.docs.usage_example}")
            tool_descriptions.append("\n".join(desc))
        
        tools_str = "\n\n".join(tool_descriptions) if tools else "No tools available."
        
        context_str = "\n".join([
            f"{m['role'] if isinstance(m, dict) else m.role}: {m['content'] if isinstance(m, dict) else m.content}"
            for m in context
        ])
        
        plan_prompt = [
            {"role": "system", "content": """You are a task planning assistant that creates structured, tool-aware execution plans.
Your plans MUST follow these strict rules:

1. RESPONSE TYPES:
For queries that don't require tools or can be answered directly:
[
  {
    "action": "complete",
    "details": "Your comprehensive answer to the query"
  }
]

For queries requiring tool usage:
[
  {
    "action": "tool",
    "details": "ToolName",
    "args": {
      "param1": "value1"
    }
  },
  {
    "action": "message",
    "details": "Analysis of the tool result"
  },
  {
    "action": "complete",
    "details": "Final comprehensive answer"
  }
]

2. TOOL USAGE (when tools are available):
- Before using any tool, verify it exists in the available tools list
- Include ALL required parameters for each tool
- After each tool call, add a message step to process the result
- If a tool fails, include fallback steps

3. PLAN FLOW:
- For simple queries or when no tools are needed, use a single "complete" action
- For complex queries with tools:
  * Start with information gathering
  * Process each tool's output
  * End with a complete action summarizing findings

4. VALIDATION:
- Ensure the plan is valid JSON
- All steps must have "action" and "details" fields
- Tool steps must include "args" with required parameters
- Final step must always be a "complete" action

IMPORTANT: If no tools are needed or available, return a simple plan with just a complete action."""},
            {
                "role": "user",
                "content": f"""
User query: "{user_query}"

Available Tools:
{tools_str}

Context:
{context_str}

Create a plan to answer this query. If no tools are needed or none are available, provide a direct answer plan.
Return ONLY a valid JSON array of steps."""
            }
        ]
        
        try:
            # Get the plan from the model
            self.logger.log("[Planner] Requesting plan from model")
            plan = await self.planner_model.call(plan_prompt)
            self.logger.log("[Planner] LLM Output:", {"llm_output": plan})
            
            # Try to parse the plan as JSON
            import json
            try:
                parsed_plan = json.loads(plan)
                self.logger.log("[Planner] Successfully parsed plan as JSON")
            except json.JSONDecodeError as e:
                self.logger.log("[Planner] Failed to parse as JSON, wrapping as simple plan", {"error": str(e)})
                # If the model didn't return valid JSON, wrap the response in a simple plan
                return json.dumps([{
                    "action": "complete",
                    "details": plan.strip()
                }])
            
            # Validate plan structure
            if not isinstance(parsed_plan, list):
                self.logger.log("[Planner] Plan is not a list, converting to simple plan")
                return json.dumps([{
                    "action": "complete",
                    "details": "Based on the information provided, here is the answer: " + str(parsed_plan)
                }])
            
            # Ensure plan ends with complete action
            if not parsed_plan or parsed_plan[-1].get("action") != "complete":
                self.logger.log("[Planner] Adding missing complete action")
                parsed_plan.append({
                    "action": "complete",
                    "details": "Based on the gathered information, here is the final answer..."
                })
            
            final_plan = json.dumps(parsed_plan)
            self.logger.log("[Planner] Generated final plan", {"plan": final_plan})
            return final_plan
            
        except Exception as e:
            self.logger.error("[Planner] Error during plan generation", {"error": str(e)})
            # If anything goes wrong, return a simple plan with an error message
            return json.dumps([{
                "action": "complete",
                "details": f"I encountered an error while planning ({str(e)}), but I'll provide a direct answer: {user_query}"
            }]) 