from typing import List, Dict, Any, Union, Optional

from .memory import Memory, ConversationMessage
from ..llms import LLM


class SummarizingMemory(Memory):
    """
    SummarizingMemory with optional hierarchical chunk-level approach.
    Automatically summarizes old messages when they exceed a threshold.
    """
    
    def __init__(
        self,
        summarizer_model: LLM,
        threshold: int = 10,
        summary_prompt: Optional[str] = None,
        max_summary_tokens: int = 150,
        hierarchical: bool = False
    ):
        """
        Initialize summarizing memory.
        
        Args:
            summarizer_model: LLM instance used for summarization (OpenAIChat or TogetherChat)
            threshold: Number of messages before summarization is triggered
            summary_prompt: Custom prompt for summarization
            max_summary_tokens: Maximum tokens in summaries (approximate)
            hierarchical: If true, store multiple chunk-level summaries
        """
        self.messages: List[Union[ConversationMessage, Dict[str, Any]]] = []
        self.threshold = threshold
        self.summarizer_model = summarizer_model
        self.summary_prompt = summary_prompt or "Summarize the following conversation:"
        self.max_summary_tokens = max_summary_tokens
        self.hierarchical = hierarchical
        self.chunk_summaries: List[str] = []  # Store summaries for sub-chunks
    
    async def add_message(self, message: Union[ConversationMessage, Dict[str, Any]]) -> None:
        """
        Add a message and trigger summarization if threshold is exceeded.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        
        if len(self.messages) > self.threshold:
            # If hierarchical is enabled, we keep chunk-level summary
            if self.hierarchical:
                await self.summarize_and_store_chunk()
            else:
                await self.summarize_older_messages()
    
    async def get_context(self) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get all messages in the memory.
        
        Returns:
            List of all stored messages
        """
        return self.messages
    
    async def get_context_for_prompt(self, _query: str) -> List[Union[ConversationMessage, Dict[str, Any]]]:
        """
        Get all messages, including any summaries.
        
        Args:
            _query: The query (unused in this implementation)
            
        Returns:
            List of all stored messages
        """
        return self.messages
    
    async def clear(self) -> None:
        """Clear all messages and summaries from memory."""
        self.messages = []
        self.chunk_summaries = []
    
    async def summarize_older_messages(self) -> None:
        """Summarize older messages, keeping only the most recent ones."""
        # We'll keep the most recent 3 messages unsummarized
        keep_count = 3
        if len(self.messages) <= keep_count:
            return
        
        older_messages = self.messages[:-keep_count]
        recent_messages = self.messages[-keep_count:]
        
        # Extract content for summarization
        conversation_text = ""
        for msg in older_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
            else:
                role = msg.role.upper()
                content = msg.content
            
            conversation_text += f"{role}: {content}\n"
        
        # Generate summary with LLM
        summary = await self.summarizer_model.call([
            {"role": "system", "content": self.summary_prompt},
            {"role": "user", "content": conversation_text},
        ])
        
        # Create summary message
        summary_message = {
            "role": "assistant", 
            "content": f"Summary of earlier discussion:\n{summary}",
            "metadata": {
                "is_summary": True,
                "summarized_messages_count": len(older_messages)
            }
        }
        
        # Replace older messages with summary
        self.messages = [summary_message] + recent_messages
    
    async def summarize_and_store_chunk(self) -> None:
        """Hierarchical: summarize and store chunks, then clear older messages."""
        # Get the chunk to summarize
        chunk_messages = self.messages[:self.threshold]
        remainder = self.messages[self.threshold:]
        
        # Extract content for summarization
        conversation_text = ""
        for msg in chunk_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
            else:
                role = msg.role.upper()
                content = msg.content
            
            conversation_text += f"{role}: {content}\n"
        
        # Generate summary with LLM
        summary = await self.summarizer_model.call([
            {"role": "system", "content": self.summary_prompt},
            {"role": "user", "content": conversation_text},
        ])
        
        # Store chunk summary
        self.chunk_summaries.append(summary)
        
        # Create summary message
        summary_message = {
            "role": "assistant",
            "content": f"Chunk summary: {summary}",
            "metadata": {
                "is_chunk_summary": True,
                "chunk_index": len(self.chunk_summaries) - 1
            }
        }
        
        # Replace older messages with a single summary message
        self.messages = [summary_message] + remainder 