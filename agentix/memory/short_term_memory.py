from typing import List, Dict, Any, Union, Optional

from .memory import Memory, ConversationMessage


class ShortTermMemory(Memory):
    """
    A simple memory implementation that keeps a fixed number of recent messages.
    """
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize short term memory with a maximum message limit.
        
        Args:
            max_messages: Maximum number of messages to store (default: 20)
        """
        self.messages: List[Union[ConversationMessage, Dict[str, Any]]] = []
        self.max_messages = max_messages
    
    async def add_message(self, message: Union[ConversationMessage, Dict[str, Any]]) -> None:
        """
        Add a message to short-term memory, removing oldest messages if limit is reached.
        
        Args:
            message: The message to add
        """
        # Normalize the message format if it's a dict
        if isinstance(message, dict):
            if 'metadata' not in message:
                message['metadata'] = {}
                
        self.messages.append(message)
        
        # Remove oldest messages if we exceed the maximum
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    async def get_context(self) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get all messages in the memory.
        
        Returns:
            List of all stored messages
        """
        return self.messages
    
    async def get_context_for_prompt(self, _query: str) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        By default, we just return all short-term messages regardless of the query.
        
        Args:
            _query: The query (unused in this implementation)
            
        Returns:
            List of all stored messages
        """
        return await self.get_context()
    
    async def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = [] 