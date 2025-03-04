from typing import Union, Protocol, List, Dict, Any, Awaitable

from .openai_chat import OpenAIChat
from .together_chat import TogetherChat


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


# Type alias for all supported LLM classes
LLM = Union[OpenAIChat, TogetherChat]

# For strict typing with Protocol
LLMInterface = LLMProtocol 