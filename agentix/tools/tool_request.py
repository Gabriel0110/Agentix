import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .tools import Tool
from .tool_error import ToolError


@dataclass
class ParsedToolRequest:
    """
    Represents the parsed request from an LLM's output.
    Example: TOOL REQUEST: MyTool "some query"
    """
    tool_name: str
    query: str
    args: Optional[Dict[str, Any]] = None


class ToolRequestParser:
    """Parser for tool requests from LLM outputs."""
    
    # We match either:
    # 1. Simple pattern: TOOL REQUEST: <ToolName> "<Query>"
    # 2. JSON pattern: TOOL REQUEST: <ToolName> {"key": "value"}
    SIMPLE_PATTERN = re.compile(r'TOOL REQUEST:\s*(\w+)\s+"([^"]+)"', re.IGNORECASE)
    JSON_PATTERN = re.compile(r'TOOL REQUEST:\s*(\w+)\s+(\{.*\})', re.IGNORECASE | re.DOTALL)
    
    @classmethod
    def parse(cls, input_str: str) -> Optional[ParsedToolRequest]:
        """
        Attempt to parse a string into a ParsedToolRequest.
        If we detect JSON, we parse the `args` object.
        
        Args:
            input_str: The string to parse
            
        Returns:
            A ParsedToolRequest object or None if parsing fails
        """
        # First try to find JSON pattern anywhere in the input
        json_match = cls.JSON_PATTERN.search(input_str)
        if json_match:
            try:
                tool_name = json_match.group(1)
                json_string = json_match.group(2)
                # Clean up any potential trailing quotes or characters
                json_string = json_string.strip()
                if json_string.endswith('"'):
                    json_string = json_string[:-1]
                parsed_args = json.loads(json_string)
                return ParsedToolRequest(
                    tool_name=tool_name,
                    query=json.dumps(parsed_args),  # Store full JSON as query
                    args=parsed_args
                )
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e} in string: {json_string}")
                # If JSON parsing fails, continue to try simple pattern
                pass
        
        # Try simple pattern
        match = cls.SIMPLE_PATTERN.search(input_str)
        if not match:
            return None
        
        return ParsedToolRequest(
            tool_name=match.group(1),
            query=match.group(2)
        )
    
    @classmethod
    def validate_basic(cls, request: ParsedToolRequest, tools: List[Tool]) -> Optional[str]:
        """
        Validates that the request references an available tool and has minimal required fields.
        
        Args:
            request: The parsed tool request
            tools: List of available tools
            
        Returns:
            Error message if validation fails, None otherwise
        """
        if not request.tool_name or not request.query:
            return "Invalid tool request format."
        
        if not tools:
            return f"No tools are available, but a tool request was made: \"{request.tool_name}\"."
        
        tool = next((t for t in tools if t.name.lower() == request.tool_name.lower()), None)
        if not tool:
            return f"Tool \"{request.tool_name}\" is not available."
        
        return None  # Valid request
    
    @classmethod
    def validate_parameters(cls, tool: Tool, request: ParsedToolRequest) -> None:
        """
        Enhanced parameter validation if the tool supports parameters.
        We can handle either (a) a single string query or (b) a JSON 'args' object.
        
        Args:
            tool: The tool to validate against
            request: The parsed tool request
            
        Raises:
            ToolError: If parameter validation fails
        """
        # If no parameters are declared, no further validation needed
        if not tool.parameters:
            return
        
        # If the tool expects structured parameters, we should parse from request.args
        if request.args:
            for param in tool.parameters:
                if param.required and param.name not in request.args:
                    raise ToolError(tool.name, f"Missing required parameter: \"{param.name}\"")
            # Optional: check param types, etc.
        else:
            # The tool might expect a single string if it only has one param
            # If multiple parameters are expected but we only have a single query,
            # that might be insufficient or invalid.
            if len(tool.parameters) > 1:
                raise ToolError(
                    tool.name,
                    f"Tool \"{tool.name}\" requires multiple parameters, but only a single query string was provided."
                ) 