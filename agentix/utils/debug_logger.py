import json
from typing import Any, Dict, Optional

class DebugLogger:
    """Enhanced debug logger with pretty formatting."""
    
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'white': '\033[97m',
        'grey': '\033[90m'
    }
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
    
    def _format_json(self, obj: Any, indent: int = 2) -> str:
        """Format JSON with consistent indentation."""
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, indent=indent)
        return str(obj)
    
    def _format_section(self, title: str, content: str) -> str:
        """Format a section with a title and content."""
        width = 80
        padding = "=" * ((width - len(title) - 2) // 2)
        header = f"{padding} {title} {padding}"
        return f"\n{self.COLORS['bold']}{header}{self.COLORS['reset']}\n{content}\n"
    
    def _format_key_value(self, key: str, value: Any) -> str:
        """Format a key-value pair with color."""
        return f"{self.COLORS['cyan']}{key}{self.COLORS['reset']}: {value}"
    
    def log(self, message: str, data: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
        """Log a message with optional data."""
        if not self.enabled:
            return
            
        # Color based on level
        level_colors = {
            "info": self.COLORS['blue'],
            "warn": self.COLORS['yellow'],
            "error": self.COLORS['red'],
            "success": self.COLORS['green']
        }
        color = level_colors.get(level, self.COLORS['white'])
        
        # Format the message
        formatted_msg = f"{color}{message}{self.COLORS['reset']}"
        
        # Format any data
        if data:
            # Special handling for LLM output
            if "llm_output" in data:
                print(self._format_section("LLM Output", self._format_json(data["llm_output"])))
                return
                
            # Format other data nicely
            data_str = "\n".join(
                f"  {self._format_key_value(k, self._format_json(v))}"
                for k, v in data.items()
            )
            print(f"{formatted_msg}\n{data_str}\n")
        else:
            print(f"{formatted_msg}\n")
    
    def stats(self, stats: Dict[str, Any]) -> None:
        """Log agent statistics in a formatted table."""
        if not self.enabled:
            return
            
        stats_str = "\n".join(
            f"  {self._format_key_value(k, v)}"
            for k, v in stats.items()
        )
        
        print(self._format_section("Agent Stats", stats_str))
    
    def warn(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message."""
        self.log(message, data, level="warn")
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log an error message."""
        self.log(message, data, level="error")
    
    def success(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a success message."""
        self.log(message, data, level="success") 