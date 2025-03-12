from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class VectorStoreItem:
    """Represents a single item in a vector store (content + embedding)."""
    id: str
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


class InMemoryVectorStore:
    """
    Basic in-memory vector store with cosine similarity search.
    For larger-scale usage, we'd want to have a real vector database.
    """
    
    def __init__(self):
        """Initialize an empty in-memory vector store."""
        self.items: List[VectorStoreItem] = []
    
    def add_item(self, item: VectorStoreItem) -> None:
        """
        Add an item to the vector store.
        
        Args:
            item: The VectorStoreItem to add
        """
        self.items.append(item)
    
    def get_all_items(self) -> List[VectorStoreItem]:
        """
        Get all items in the vector store.
        
        Returns:
            List of all stored items
        """
        return self.items
    
    def similarity_search(self, query_embedding: List[float], k: int = 3) -> List[VectorStoreItem]:
        """
        Find the k most similar items to the query embedding.
        
        Args:
            query_embedding: The embedding vector to compare against
            k: Number of results to return (default: 3)
            
        Returns:
            List of the k most similar VectorStoreItems
        """
        if not self.items:
            return []
        
        # Calculate similarity scores
        scores = [(item, cosine_similarity(item.embedding, query_embedding)) 
                  for item in self.items]
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k items
        return [item for item, _ in scores[:k]]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    
    Raises:
        ValueError: If vectors are not the same length
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length for cosine similarity.")
    
    # Convert to numpy arrays for more efficient calculation
    a_array = np.array(a)
    b_array = np.array(b)
    
    # Calculate dot product
    dot_product = np.dot(a_array, b_array)
    
    # Calculate magnitudes
    norm_a = np.linalg.norm(a_array)
    norm_b = np.linalg.norm(b_array)
    
    # Handle zero division
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b) 