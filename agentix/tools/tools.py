from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

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
    def description(self) -> Optional[str]:
        """
        A human-readable description of what the tool does.
        """
        return None
    
    @property
    def parameters(self) -> Optional[List[ToolParameter]]:
        """
        (Optional) A list of parameter definitions for validating or documenting input.
        """
        return None
    
    @property
    def docs(self) -> Optional[ToolDocumentation]:
        """
        Additional documentation for advanced usage or function-calling patterns.
        """
        return None
    
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