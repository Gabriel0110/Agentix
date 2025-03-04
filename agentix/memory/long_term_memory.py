import time
import uuid
from typing import List, Dict, Any, Union, Optional

from .memory import Memory, ConversationMessage
from .vector_store import InMemoryVectorStore, VectorStoreItem
from ..llms.openai_embeddings import OpenAIEmbeddings


class LongTermMemory(Memory):
    """
    Long-term memory with vector store retrieval.
    """
    
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        max_messages: int = 1000,
        top_k: int = 3
    ):
        """
        Initialize long-term memory with vector storage.
        
        Args:
            embeddings: OpenAIEmbeddings instance for encoding text
            max_messages: Maximum number of messages to store (default: 1000)
            top_k: Number of most similar messages to retrieve (default: 3)
        """
        self.embeddings = embeddings
        self.vector_store = InMemoryVectorStore()
        self.max_messages = max_messages
        self.top_k = top_k
    
    async def add_message(self, message: Union[ConversationMessage, Dict[str, Any]]) -> None:
        """
        Add a message to long-term memory with embedding.
        
        Args:
            message: The message to add
        """
        # Generate a unique ID
        item_id = f"msg-{int(time.time() * 1000)}-{uuid.uuid4().hex[:5]}"
        
        # Extract content and metadata
        if isinstance(message, dict):
            content = message.get("content", "")
            role = message.get("role", "user")
            metadata = message.get("metadata", {}) or {}
            metadata["role"] = role
        else:
            content = message.content
            metadata = message.metadata or {}
            metadata["role"] = message.role
        
        # Skip empty messages
        if not content.strip():
            return
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = int(time.time() * 1000)
        
        # Generate embedding asynchronously
        embedding = await self.embeddings.aembed(content)
        
        # Add to vector store
        self.vector_store.add_item(VectorStoreItem(
            id=item_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        ))
        
        # Manage size if needed
        if len(self.vector_store.get_all_items()) > self.max_messages:
            # Remove oldest items (assuming they're added in chronological order)
            items = self.vector_store.get_all_items()
            self.vector_store = InMemoryVectorStore()
            for item in items[1:]:  # Skip the oldest
                self.vector_store.add_item(item)
    
    async def get_context(self) -> List[Dict[str, Any]]:
        """
        By default, no immediate context unless you do retrieval.
        
        Returns:
            Empty list by default
        """
        return []
    
    async def get_context_for_prompt(self, query: str) -> List[Dict[str, Any]]:
        """
        Return top-K relevant messages to the query.
        
        Args:
            query: The query to get relevant context for
            
        Returns:
            List of relevant messages
        """
        if not query.strip():
            return []
        
        # Generate query embedding
        query_embedding = await self.embeddings.aembed(query)
        
        # Find similar items
        results = self.vector_store.similarity_search(query_embedding, self.top_k)
        
        # Convert to message format
        messages = []
        for result in results:
            messages.append({
                "role": result.metadata.get("role", "assistant"),
                "content": result.content,
                "metadata": {
                    "source": "long_term_memory",
                    "timestamp": result.metadata.get("timestamp", 0),
                    "similarity_context": True
                }
            })
        
        return messages
    
    async def clear(self) -> None:
        """Clear all messages from memory."""
        self.vector_store = InMemoryVectorStore() 