from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from .tool_metadata import ToolParameter, ToolDocumentation


class Tool(ABC):
    """
    Defines the interface for Tools that an Agent can call.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        A short name or identifier for the tool (e.g., "DuckDuckGo").
        """
        pass
    
    @property
    def description(self) -> str:
        """
        A human-readable description of what the tool does.
        This should be detailed enough for an LLM to understand when to use the tool.
        """
        return "No description provided"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        """
        A list of parameter definitions that document and validate the tool's input.
        Each parameter should include name, type, description, and whether it's required.
        """
        return []
    
    @property
    def parameter_schema(self) -> Optional[Dict[str, Any]]:
        """
        Optional JSON schema that defines the tool's parameter structure.
        This is useful for tools that accept complex structured input.
        Example:
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"}
            },
            "required": ["query"]
        }
        """
        return None
    
    @property
    def docs(self) -> Optional[ToolDocumentation]:
        """
        Additional documentation including usage examples and advanced patterns.
        This helps the agent understand how to use the tool effectively.
        """
        return None
    
    @property
    def usage_examples(self) -> List[str]:
        """
        List of example tool calls showing how to use the tool.
        These examples help the agent understand proper usage patterns.
        """
        examples = []
        if self.parameters:
            # Generate example with parameters
            param_example = {
                p.name: f"example_{p.name}" 
                for p in self.parameters if p.required
            }
            examples.append(f'TOOL REQUEST: {self.name} {param_example}')
        else:
            # Simple example
            examples.append(f'TOOL REQUEST: {self.name} "example query"')
        return examples
    
    @abstractmethod
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Executes the tool with the given input string and optionally a structured args object.
        The 'input_str' will usually be the extracted string from "TOOL REQUEST: <ToolName> \"<Query>\""
        
        Args:
            input_str: The input string for the tool
            args: Optional structured arguments for the tool
            
        Returns:
            The result of running the tool as a string
        """
        pass 