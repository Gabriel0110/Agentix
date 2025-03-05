import os
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
import json
import asyncio

from together import Together, AsyncTogether
from .types import LLM

from dotenv import load_dotenv
load_dotenv()

class TogetherChat(LLM):
    """
    A wrapper for Together.ai's chat completion API using their official Python client.
    
    Args:
        model: The model to use (e.g., "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
        api_key: Your Together API key (defaults to TOGETHER_API_KEY environment variable)
        temperature: Controls randomness (0-1)
        stream: Whether to use streaming mode
        on_token: Callback function for streaming tokens
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None
    ):
        """Initialize the Together chat client."""
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.on_token = on_token
        
        if not self.api_key:
            raise ValueError(
                "Missing Together API key. Either pass it as api_key parameter "
                "or set the TOGETHER_API_KEY environment variable."
            )
        
        # Initialize the Together clients
        self.client = Together(api_key=self.api_key)
        self.async_client = AsyncTogether(api_key=self.api_key)
    
    async def call(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the Together chat completion API with the given messages.
        
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
                        error_message = f"Together API error: {error_data['error']['message']}"
            except:
                pass
            
            print(f"Error calling Together API: {error_message}")
            raise
    
    async def _stream_call(self, messages: List[Dict[str, str]]) -> str:
        """Make a streaming API call and process chunks as they arrive."""
        final_text = ""
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    if self.on_token:
                        self.on_token(token)
                    final_text += token
            
            return final_text
        except Exception as e:
            error_message = str(e)
            try:
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    if 'error' in error_data and 'message' in error_data['error']:
                        error_message = f"Together API error: {error_data['error']['message']}"
            except:
                pass
            
            print(f"Error in streaming call to Together API: {error_message}")
            raise
    
    async def call_with_vision(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None
    ) -> str:
        """
        Call the Together chat completion API with vision capabilities.
        
        Args:
            prompt: The text prompt to send
            image_url: URL of the image to analyze (optional)
            image_base64: Base64-encoded image data (optional)
            
        Returns:
            The model's response as a string
        """
        if not image_url and not image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")
        
        # Prepare the message content with image
        content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add the image data
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        elif image_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        messages = [{"role": "user", "content": content}]
        
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
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    if 'error' in error_data and 'message' in error_data['error']:
                        error_message = f"Together API error: {error_data['error']['message']}"
            except:
                pass
            
            print(f"Error calling Together Vision API: {error_message}")
            raise 