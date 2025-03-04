from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from .tool_metadata import ToolParameter
from .tools import Tool

@dataclass
class ToolAnalysis:
    """Complete analysis of a tool's capabilities and usage."""
    name: str
    description: str
    parameters: List[ToolParameter]
    usage_examples: List[str]
    calling_patterns: List[str]
    parameter_schema: Optional[Dict[str, Any]] = None
    valid_parameter_values: Optional[Dict[str, List[str]]] = None

class ToolAnalyzer:
    """
    Analyzes tools to extract complete documentation and usage patterns.
    This helps agents understand how to use tools without explicit instructions.
    """
    
    @classmethod
    def analyze_tool(cls, tool: Tool) -> ToolAnalysis:
        """
        Analyze a single tool to extract its complete documentation.
        
        Args:
            tool: The tool to analyze
            
        Returns:
            ToolAnalysis containing complete tool documentation
        """
        # Get basic info
        name = tool.name
        description = tool.description or "No description provided"
        
        # Get parameters
        parameters = tool.parameters or []
        
        # Extract parameter schema if available
        parameter_schema = getattr(tool, 'parameter_schema', None)
        
        # Extract valid parameter values from descriptions
        valid_parameter_values = cls._extract_valid_parameter_values(parameters)
        
        # Generate calling patterns
        calling_patterns = cls._generate_calling_patterns(tool)
        
        # Get usage examples
        usage_examples = cls._extract_usage_examples(tool)
        
        return ToolAnalysis(
            name=name,
            description=description,
            parameters=parameters,
            usage_examples=usage_examples,
            calling_patterns=calling_patterns,
            parameter_schema=parameter_schema,
            valid_parameter_values=valid_parameter_values
        )
    
    @classmethod
    def analyze_tools(cls, tools: List[Tool]) -> List[ToolAnalysis]:
        """
        Analyze multiple tools to extract their documentation.
        
        Args:
            tools: List of tools to analyze
            
        Returns:
            List of ToolAnalysis objects, one for each tool
        """
        return [cls.analyze_tool(tool) for tool in tools]
    
    @classmethod
    def _extract_valid_parameter_values(cls, parameters: List[ToolParameter]) -> Dict[str, List[str]]:
        """Extract valid values from parameter descriptions."""
        valid_values = {}
        for param in parameters:
            # Look for patterns like "(value1, value2, value3)" in description
            if param.description and "(" in param.description and ")" in param.description:
                start = param.description.find("(")
                end = param.description.find(")")
                if start != -1 and end != -1:
                    values_str = param.description[start + 1:end]
                    values = [v.strip() for v in values_str.split(",")]
                    if values:
                        valid_values[param.name] = values
        return valid_values
    
    @classmethod
    def _generate_calling_patterns(cls, tool: Tool) -> List[str]:
        """Generate example calling patterns for the tool."""
        patterns = []
        
        # If tool has parameters, prioritize JSON pattern
        if tool.parameters:
            # Create example with all parameters
            param_example = {}
            for param in tool.parameters:
                # Try to use an example value from the description
                example_value = cls._get_example_value(param)
                param_example[param.name] = example_value
            
            json_pattern = f'TOOL REQUEST: {tool.name} {json.dumps(param_example)}'
            patterns.append(json_pattern)
            
            # Add pattern with only required parameters
            required_params = {
                p.name: cls._get_example_value(p)
                for p in tool.parameters if p.required
            }
            if required_params:
                required_pattern = f'TOOL REQUEST: {tool.name} {json.dumps(required_params)}'
                if required_pattern != patterns[0]:
                    patterns.append(required_pattern)
        
        # Add simple pattern for tools that can work with just a string
        if not tool.parameters or len(tool.parameters) == 1:
            patterns.append(f'TOOL REQUEST: {tool.name} "your query here"')
            
        return patterns
    
    @classmethod
    def _get_example_value(cls, param: ToolParameter) -> Any:
        """Get an example value for a parameter based on its type and description."""
        if param.description:
            # Extract example from "e.g., X" or "(e.g. X)" patterns
            if "e.g.," in param.description:
                example = param.description.split("e.g.,")[1].strip()
                if " " in example:
                    example = example.split(" ")[0]
                return example.strip("().")
            elif "(e.g." in param.description:
                example = param.description.split("(e.g.")[1].strip()
                if ")" in example:
                    example = example.split(")")[0]
                return example.strip()
        
        # Default examples based on type
        type_examples = {
            "string": f"example_{param.name}",
            "integer": 42,
            "number": 3.14,
            "boolean": True,
            "array": [],
            "object": {}
        }
        return type_examples.get(param.type, f"<{param.type}>")
    
    @classmethod
    def _extract_usage_examples(cls, tool: Tool) -> List[str]:
        """Extract usage examples from tool documentation."""
        examples = []
        
        # Get examples from docs if available
        if tool.docs and tool.docs.usage_example:
            examples.append(tool.docs.usage_example)
        
        # Get examples from the tool's usage_examples property
        if hasattr(tool, 'usage_examples'):
            tool_examples = tool.usage_examples
            if isinstance(tool_examples, list):
                examples.extend(tool_examples)
        
        # Generate examples if none provided
        if not examples:
            if tool.parameters:
                # Example with all parameters
                param_values = {
                    p.name: cls._get_example_value(p)
                    for p in tool.parameters
                }
                examples.append(f'TOOL REQUEST: {tool.name} {json.dumps(param_values)}')
                
                # Example with only required parameters
                required_values = {
                    p.name: cls._get_example_value(p)
                    for p in tool.parameters if p.required
                }
                if required_values != param_values:
                    examples.append(f'TOOL REQUEST: {tool.name} {json.dumps(required_values)}')
            else:
                # Simple example
                examples.append(f'TOOL REQUEST: {tool.name} "example query"')
                
        return examples 