class ToolError(Exception):
    """
    A dedicated error class for tool-related issues.
    This centralizes error handling logic so that the Agent or other parts
    can catch and process errors consistently.
    """
    
    def __init__(self, tool_name: str, message: str):
        """
        Initialize a tool error.
        
        Args:
            tool_name: The name of the tool that caused the error
            message: The error message
        """
        super().__init__(f"ToolError [{tool_name}]: {message}")
        self.tool_name = tool_name 