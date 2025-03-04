from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False


@dataclass
class ToolDocumentation:
    """Documentation for a tool."""
    name: str
    description: str
    usage_example: Optional[str] = None
    parameters: List[ToolParameter] = field(default_factory=list) 