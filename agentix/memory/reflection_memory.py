from typing import List, Dict, Any, Union, Optional

from .memory import Memory, ConversationMessage, MemoryRole


class ReflectionMemory(Memory):
    """
    Memory that specifically stores and manages reflection messages.
    """
    
    def __init__(self, include_reflections: bool = False):
        """
        Initialize reflection memory.
        
        Args:
            include_reflections: Whether to include reflections in context by default
        """
        self.reflections: List[Union[ConversationMessage, Dict[str, Any]]] = []
        self.include_reflections = include_reflections
    
    async def add_message(self, message: Union[ConversationMessage, Dict[str, Any]]) -> None:
        """
        Add a message to reflection memory, but only if it's a reflection.
        
        Args:
            message: The message to add
        """
        # Only store if role is 'reflection'
        if isinstance(message, dict):
            if message.get("role") == "reflection":
                self.reflections.append(message)
        else:
            if message.role == "reflection":
                self.reflections.append(message)
    
    async def get_context(self) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get all reflection messages if include_reflections is True,
        otherwise returns an empty list.
        
        Returns:
            List of stored reflection messages or empty list
        """
        return self.reflections if self.include_reflections else []
    
    async def get_context_for_prompt(self, _query: str) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get reflection messages for prompts.
        
        Args:
            _query: The query (unused in this implementation)
            
        Returns:
            List of stored reflection messages or empty list
        """
        return await self.get_context()
    
    async def clear(self) -> None:
        """Clear all reflection messages from memory."""
        self.reflections = [] 