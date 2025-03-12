from typing import Protocol, List, Dict, runtime_checkable

@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol defining the interface that all LLM classes should implement."""
    
    async def call(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM with the given messages.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            
        Returns:
            The model's response as a string
        """
        ...

# Base class for LLM implementations
class LLM:
    """Base class for all LLM implementations."""
    async def call(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM with the given messages.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            
        Returns:
            The model's response as a string
        """
        raise NotImplementedError("Subclasses must implement this method")

# For strict typing with Protocol
LLMInterface = LLMProtocol 