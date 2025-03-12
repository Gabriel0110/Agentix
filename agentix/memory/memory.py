from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass

# Define memory role types
MemoryRole = Literal["system", "user", "assistant", "reflection"]

@dataclass
class ConversationMessage:
    """A single message in the conversation or agent's reasoning process."""
    role: MemoryRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


class Memory(ABC):
    """
    The abstract base class that every Memory implementation should inherit from.
    """
    
    @abstractmethod
    async def add_message(self, message: Union[ConversationMessage, Dict[str, Any]]) -> None:
        """
        Add a new message to the memory.
        
        Args:
            message: The message to add, either as a ConversationMessage object
                    or a dict with 'role' and 'content' keys.
        """
        pass
    
    @abstractmethod
    async def get_context(self) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Returns the entire stored context (used in older design).
        In new design, consider using `get_context_for_prompt()` instead.
        
        Returns:
            A list of conversation messages
        """
        pass
    
    @abstractmethod
    async def get_context_for_prompt(self, query: str) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        A more advanced method that returns only the relevant or necessary context
        for a given query or scenario.
        
        Args:
            query: The query to get relevant context for
            
        Returns:
            A list of conversation messages relevant to the query
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all messages from memory."""
        pass 