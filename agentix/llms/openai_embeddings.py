import os
from typing import List, Optional

from openai import OpenAI, AsyncOpenAI


class OpenAIEmbeddings:
    """
    A wrapper for OpenAI's embedding API using the official Python client.
    
    Args:
        model: The embedding model to use (default: "text-embedding-3-small")
        api_key: Your OpenAI API key (defaults to OPENAI_API_KEY environment variable)
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """Initialize the OpenAI embeddings client."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key. Either pass it as api_key parameter "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        # Initialize the OpenAI clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the provided text using the synchronous client.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of float values representing the embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    async def aembed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the provided text using the asynchronous client.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of float values representing the embedding vector
        """
        response = await self.async_client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding 