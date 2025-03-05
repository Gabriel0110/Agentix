import os
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
import json
import asyncio

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .types import LLM

from dotenv import load_dotenv
load_dotenv()

class OpenAIChat(LLM):
    """
    A wrapper for OpenAI's chat completion API using the official Python client.
    
    Args:
        model: The model to use (e.g., "gpt-4o-mini")
        api_key: Your OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        temperature: Controls randomness (0-1)
        stream: Whether to use streaming mode
        on_token: Callback function for streaming tokens
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None
    ):
        """Initialize the OpenAI chat client."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.on_token = on_token
        
        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key. Either pass it as api_key parameter "
                "or set the OPENAI_API_KEY environment variable."
            )
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    async def call(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the OpenAI chat completion API with the given messages.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
        
        Returns:
            The model's response as a string
        """
        if self.stream:
            return await self._stream_call(messages)
        else:
            return await self._standard_call(messages)
    
    async def _standard_call(self, messages: List[Dict[str, str]]) -> str:
        """Make a standard (non-streaming) API call."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            error_message = str(e)
            try:
                # Try to extract more detailed error information if available
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    if 'error' in error_data and 'message' in error_data['error']:
                        error_message = f"OpenAI API error: {error_data['error']['message']}"
            except:
                pass
            
            print(f"Error calling OpenAI API: {error_message}")
            raise
    
    async def _stream_call(self, messages: List[Dict[str, str]]) -> str:
        """Make a streaming API call and process chunks as they arrive."""
        final_text = ""
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    final_text += content
                    
                    # Call the token callback if provided
                    if self.on_token:
                        self.on_token(content)
            
            return final_text
        except Exception as e:
            error_message = str(e)
            try:
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    if 'error' in error_data and 'message' in error_data['error']:
                        error_message = f"OpenAI API error: {error_data['error']['message']}"
            except:
                pass
            
            print(f"Error in streaming call to OpenAI API: {error_message}")
            raise 