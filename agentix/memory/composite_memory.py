from typing import List, Dict, Any, Union

from .memory import Memory, ConversationMessage


class CompositeMemory(Memory):
    """
    Combines multiple memory implementations into a single memory interface.
    """
    
    def __init__(self, *memories: Memory):
        """
        Initialize composite memory with a list of memory instances.
        
        Args:
            *memories: Variable number of Memory instances to combine
        """
        self.memories = list(memories)
    
    async def add_message(self, message: Union[ConversationMessage, Dict[str, Any]]) -> None:
        """
        Add a message to all contained memories.
        
        Args:
            message: The message to add
        """
        for mem in self.memories:
            await mem.add_message(message)
    
    async def get_context(self) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get combined context from all memories.
        
        Returns:
            Combined list of messages from all memories, sorted by timestamp
        """
        all_messages = []
        for mem in self.memories:
            ctx = await mem.get_context()
            all_messages.extend(ctx)
        
        return self.sort_by_timestamp(all_messages)
    
    async def get_context_for_prompt(self, query: str) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get combined context relevant to the query from all memories.
        
        Args:
            query: The query to get relevant context for
            
        Returns:
            Combined list of relevant messages from all memories, sorted by timestamp
        """
        all_messages = []
        for mem in self.memories:
            partial = await mem.get_context_for_prompt(query)
            all_messages.extend(partial)
        
        return self.sort_by_timestamp(all_messages)
    
    async def clear(self) -> None:
        """Clear all messages from all memories."""
        for mem in self.memories:
            await mem.clear()
    
    def sort_by_timestamp(self, messages: List[Union[ConversationMessage, Dict[str, Any]]]) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Sort messages by timestamp if available.
        
        Args:
            messages: List of messages to sort
            
        Returns:
            Sorted list of messages
        """
        def get_timestamp(msg):
            if isinstance(msg, dict):
                return msg.get("metadata", {}).get("timestamp", 0)
            else:
                return msg.metadata.get("timestamp", 0) if msg.metadata else 0
        
        return sorted(messages, key=get_timestamp) 