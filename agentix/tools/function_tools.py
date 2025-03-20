"""
Function-based tools for Agentix.

This module provides decorators and utilities to convert Python functions into Agentix tools.
Functions can be synchronous or asynchronous, and parameters are automatically extracted
from function signatures using type hints and docstrings.
"""

import inspect
import functools
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from .tools import Tool
from .tool_metadata import ToolParameter, ToolDocumentation


def function_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    usage_example: Optional[str] = None
) -> Callable:
    """
    Decorator to convert a Python function into an Agentix Tool.
    
    The decorator automatically extracts:
    - Parameter names and types from type hints
    - Parameter descriptions from docstring
    - Function description from docstring
    - Required/optional status from default values
    
    Args:
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description (defaults to function docstring)
        usage_example: Optional example of how to use the tool
        
    Example:
        @function_tool(
            name="GetWeather",
            description="Get current weather for a city",
            usage_example='TOOL REQUEST: GetWeather {"city": "San Francisco"}'
        )
        def get_weather(city: str) -> str:
            '''Get current weather for the specified city.'''
            return f"The weather in {city} is sunny"
    """
    # Handle case when decorator is used without parentheses
    if callable(name):
        func = name
        name = None
        return function_tool()(func)
        
    def decorator(func: Callable) -> Tool:
        # Get function signature info
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        type_hints = get_type_hints(func)
        
        def parse_docstring(docstring: str) -> Dict[str, str]:
            """Parse docstring to extract parameter descriptions."""
            param_docs = {}
            if not docstring:
                return param_docs
                
            lines = docstring.split("\n")
            in_args_section = False
            current_param = None
            current_desc = []
            
            for line in lines:
                line = line.strip()
                
                # Check for Args/Parameters section
                if line.lower().startswith(("args:", "parameters:")):
                    in_args_section = True
                    continue
                    
                # Check for end of Args section
                if in_args_section and (not line or line.lower().startswith(("returns:", "raises:"))):
                    if current_param and current_desc:
                        param_docs[current_param] = " ".join(current_desc)
                    in_args_section = False
                    continue
                
                if in_args_section and line:
                    # New parameter definition
                    if line[0].isalnum() and ":" in line:
                        # Save previous parameter if exists
                        if current_param and current_desc:
                            param_docs[current_param] = " ".join(current_desc)
                            
                        # Parse new parameter
                        param_parts = line.split(":", 1)
                        current_param = param_parts[0].strip()
                        current_desc = [param_parts[1].strip()] if len(param_parts) > 1 else []
                    # Continuation of current parameter description
                    elif current_param:
                        current_desc.append(line)
            
            # Add last parameter if exists
            if current_param and current_desc:
                param_docs[current_param] = " ".join(current_desc)
                
            return param_docs
        
        # Create tool class dynamically
        class FunctionTool(Tool):
            def __init__(self):
                self._func = func
                self._is_async = inspect.iscoroutinefunction(func)
                self._param_docs = parse_docstring(doc)
                self._name = name or func.__name__
            
            @property
            def name(self) -> str:
                return self._name
            
            @property
            def description(self) -> str:
                # Use provided description or first line of docstring
                if description:
                    return description
                if doc:
                    return doc.split("\n")[0]
                return f"Call the {self.name} function"
            
            @property
            def parameters(self) -> List[ToolParameter]:
                params = []
                for param_name, param in sig.parameters.items():
                    # Skip self for methods
                    if param_name == "self":
                        continue
                        
                    param_type = type_hints.get(param_name, Any).__name__
                    param_required = param.default == param.empty
                    param_desc = self._param_docs.get(param_name, f"Parameter {param_name}")
                    
                    params.append(ToolParameter(
                        name=param_name,
                        type=param_type,
                        description=param_desc,
                        required=param_required
                    ))
                return params
            
            @property
            def docs(self) -> Optional[ToolDocumentation]:
                return ToolDocumentation(
                    name=self.name,
                    description=self.description,
                    usage_example=usage_example or f'TOOL REQUEST: {self.name} "example input"',
                    parameters=self.parameters
                )
            
            async def run(self, input_str: str = "", args: Optional[Dict[str, Any]] = None) -> str:
                """Run the function with provided arguments."""
                try:
                    # Initialize args if not provided
                    if args is None:
                        args = {}
                    
                    # If no args provided but input_str exists and we have exactly one required parameter,
                    # use input_str as the value for that parameter
                    if not args and input_str and len(self.parameters) == 1:
                        args = {self.parameters[0].name: input_str}
                    
                    # Validate required parameters
                    missing_params = [
                        p.name for p in self.parameters 
                        if p.required and p.name not in args
                    ]
                    if missing_params:
                        return f"Error: Missing required parameters: {', '.join(missing_params)}"
                    
                    # Call the function with the provided arguments
                    if self._is_async:
                        result = await self._func(**args)
                    else:
                        result = self._func(**args)
                    
                    # Convert result to string if needed
                    if not isinstance(result, str):
                        result = str(result)
                    
                    return result
                    
                except Exception as e:
                    return f"Error calling {self.name}: {str(e)}"
            
            def __str__(self) -> str:
                return f"Tool({self.name})"
                
            def __repr__(self) -> str:
                return f"Tool({self.name})"
        
        # Create an instance of the tool
        tool_instance = FunctionTool()
        
        # Add tool attributes to the original function for compatibility
        functools.update_wrapper(tool_instance, func)
        
        # Return the tool instance
        return tool_instance
    
    return decorator


# Example usage:
"""
@function_tool(
    name="GetWeather",
    description="Get current weather for a city",
    usage_example='TOOL REQUEST: GetWeather {"city": "San Francisco"}'
)
async def get_weather(city: str) -> str:
    '''
    Get the current weather for the specified city.
    
    Args:
        city: The name of the city to get weather for
    
    Returns:
        A string describing the current weather
    '''
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny"
""" 