"""
Planner module for generating execution plans.
"""

from abc import ABC, abstractmethod
from typing import List

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
    An enhanced LLM-based planner that generates tool-aware execution plans.
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
        
        tools_str = "\n\n".join(tool_descriptions)
        
        context_str = "\n".join([
            f"{m['role'] if isinstance(m, dict) else m.role}: {m['content'] if isinstance(m, dict) else m.content}"
            for m in context
        ])
        
        plan_prompt = [
            {"role": "system", "content": """You are a task planning assistant that creates structured, tool-aware execution plans.
Your plans MUST follow these strict rules:

1. TOOL USAGE:
- Before using any tool, verify it exists in the available tools list
- Include ALL required parameters for each tool
- After each tool call, add a message step to process and analyze the result
- If a tool fails, include fallback steps in the plan

2. STEP STRUCTURE:
For tool steps, use EXACTLY this format:
{
  "action": "tool",
  "details": "ToolName",
  "args": {
    "param1": "value1",
    "param2": "value2"
  }
}

For message/analysis steps, use:
{
  "action": "message",
  "details": "Your analysis of the previous step's result"
}

For final answers, use:
{
  "action": "complete",
  "details": "Your comprehensive final answer that includes all gathered information"
}

3. PLAN FLOW:
- Start with information gathering using appropriate tools
- Process and analyze each tool's output before proceeding
- Include error handling and alternative paths
- ALWAYS end with a complete action that summarizes all findings

4. VALIDATION:
- Verify all required tool parameters are included
- Ensure tool names match exactly with available tools
- Check that each tool result is properly analyzed
- Validate that the final answer incorporates all gathered information

Example plan structure:
[
  {
    "action": "tool",
    "details": "SearchTool",
    "args": {"query": "specific search terms"}
  },
  {
    "action": "message",
    "details": "Analyzing search results to determine next steps..."
  },
  {
    "action": "tool",
    "details": "ProcessData",
    "args": {"input": "data from search"}
  },
  {
    "action": "message",
    "details": "Processing complete. Key findings: [...]"
  },
  {
    "action": "complete",
    "details": "Based on the search and processing results, here is the comprehensive answer: [...]"
  }
]"""},
            {
                "role": "user",
                "content": f"""
User query: "{user_query}"

Available Tools:
{tools_str}

Context:
{context_str}

Create a detailed, tool-aware plan to solve this query.
Remember:
1. Use ONLY tools from the available tools list
2. Include ALL required parameters
3. Analyze each tool's output
4. End with a comprehensive final answer

Return ONLY a valid JSON array of steps."""
            }
        ]
        
        plan = await self.planner_model.call(plan_prompt)
        
        # Validate plan structure
        try:
            import json
            parsed_plan = json.loads(plan)
            if not isinstance(parsed_plan, list):
                raise ValueError("Plan must be a JSON array")
            
            # Ensure plan ends with complete action
            if not parsed_plan or parsed_plan[-1].get("action") != "complete":
                parsed_plan.append({
                    "action": "complete",
                    "details": "Based on the gathered information, here is the final answer..."
                })
                plan = json.dumps(parsed_plan)
                
        except Exception as e:
            # If plan parsing fails, return a simple one-step plan
            plan = json.dumps([{
                "action": "message",
                "details": f"Error parsing plan: {str(e)}. Proceeding with direct response."
            }])
            
        return plan 