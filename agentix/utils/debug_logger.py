import json
from typing import Any, Dict, Optional, Union

class DebugLogger:
    """
    A utility class for debug logging with different levels and JSON formatting.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the debug logger.
        
        Args:
            debug: Whether to print debug messages or not
        """
        self.debug = debug
    
    def log(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message at the debug level.
        
        Args:
            message: The message to log
            data: Optional data to include in the log
        """
        if self.debug:
            if data:
                formatted_data = json.dumps(data, default=str, indent=2)
                print(f"{message}\n{formatted_data}")
            else:
                print(message)
    
    def warn(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning message.
        
        Args:
            message: The warning message
            data: Optional data to include in the log
        """
        if self.debug:
            prefix = "WARNING: "
            if data:
                formatted_data = json.dumps(data, default=str, indent=2)
                print(f"{prefix}{message}\n{formatted_data}")
            else:
                print(f"{prefix}{message}")
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error message.
        
        Args:
            message: The error message
            data: Optional data to include in the log
        """
        if self.debug:
            prefix = "ERROR: "
            if data:
                formatted_data = json.dumps(data, default=str, indent=2)
                print(f"{prefix}{message}\n{formatted_data}")
            else:
                print(f"{prefix}{message}")
    
    def stats(self, stats_data: Dict[str, Union[int, str, float]]) -> None:
        """
        Log statistics about the agent's execution.
        
        Args:
            stats_data: Statistics data to log
        """
        if self.debug:
            print("--- Agent Stats ---")
            for key, value in stats_data.items():
                print(f"{key}: {value}")
            print("------------------") 