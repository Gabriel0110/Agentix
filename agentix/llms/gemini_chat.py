"""
Google Gemini models for Agentix agents.

This module provides integration with Google's Gemini models via their official Python SDK.
It supports text generation, multimodal input (text + images), streaming, and advanced
configuration options.
"""

import base64
import os
import json
from typing import Any, Callable, Dict, List, Optional, Union, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .types import LLM
from ..utils.debug_logger import DebugLogger


class GeminiChat(LLM):
    """
    Interface for Google's Gemini language models.
    
    This class provides methods to interact with Google's Gemini models,
    supporting both text-only and multimodal (text + images) inputs.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        system_instruction: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        debug: bool = False
    ):
        """
        Initialize a Google Gemini model.
        
        Args:
            api_key: Google API key, defaults to GOOGLE_API_KEY env var if not provided
            model: Gemini model to use (default: "gemini-2.0-flash")
            temperature: Controls randomness (0.0 to 1.0)
            max_output_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Number of highest probability tokens to consider
            stream: Whether to stream the response tokens
            system_instruction: System instruction for the model
            on_token: Callback function for streaming tokens
            debug: Whether to enable debug logging
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        self.system_instruction = system_instruction
        self.on_token = on_token
        self.debug = debug
        self.logger = DebugLogger(debug=debug)
        
        if not GENAI_AVAILABLE:
            self.logger.warn("Google AI package not installed. Please install with: pip install google-generativeai")
        
        if not self.api_key and GENAI_AVAILABLE:
            self.logger.warn("No Google API key provided. Please provide an API key or set the GOOGLE_API_KEY environment variable.")

        # Initialize the client if the API key and package are available
        if GENAI_AVAILABLE and self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    async def call(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the Gemini model with the given messages.
        
        This method complies with the LLM interface by delegating to the chat method.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            
        Returns:
            The model's response as a string
        """
        return await self.chat(messages)
        
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Send a chat message to the Gemini model and get a response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Model's response as a string
        """
        if not GENAI_AVAILABLE:
            raise ImportError("Google AI package not installed. Please install with: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("No Google API key provided. Please provide an API key or set the GOOGLE_API_KEY environment variable.")
        
        # Override instance parameters with any provided in kwargs
        model = kwargs.get("model", self.model_name)
        temperature = kwargs.get("temperature", self.temperature)
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        stream = kwargs.get("stream", self.stream)
        system_instruction = kwargs.get("system_instruction", self.system_instruction)
        
        # Set up the generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        
        if max_output_tokens is not None:
            config.max_output_tokens = max_output_tokens
        
        if top_p is not None:
            config.top_p = top_p
            
        if top_k is not None:
            config.top_k = top_k
            
        if system_instruction is not None:
            config.system_instruction = system_instruction
        
        # Format messages for Gemini
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Handle special roles
            if role == "system":
                # Use system instruction if it's a system message
                config.system_instruction = content
                continue
            
            # For user and assistant messages, add them directly
            gemini_messages.append(content)
        
        try:
            self.logger.log("Sending request to Gemini API", {
                "model": model,
                "message_count": len(gemini_messages),
                "system_instruction": system_instruction is not None
            })
            
            if stream:
                return await self._stream_chat(model, gemini_messages, config)
            else:
                # Use the synchronous API and await the result
                response = self.client.models.generate_content(
                    model=model,
                    contents=gemini_messages,
                    config=config
                )
                return response.text
                
        except Exception as e:
            self.logger.error(f"Error in Gemini API request: {str(e)}")
            raise
    
    async def _stream_chat(
        self, 
        model: str,
        messages: List[str],
        config: types.GenerateContentConfig
    ) -> str:
        """
        Stream chat responses from the Gemini model.
        
        Args:
            model: Model name to use
            messages: List of message strings
            config: Generation configuration
            
        Returns:
            The complete response as a string
        """
        full_response = ""
        
        # Use the streaming API
        response_stream = self.client.models.generate_content_stream(
            model=model,
            contents=messages,
            config=config
        )
        
        # Process the streaming response
        for chunk in response_stream:
            if not chunk.text:
                continue
                
            full_response += chunk.text
            
            # Call the on_token callback if provided
            if self.on_token:
                self.on_token(chunk.text)
        
        return full_response
    
    async def chat_with_images(
        self,
        text_prompt: str,
        image_paths: List[str],
        **kwargs
    ) -> str:
        """
        Send a multimodal request with text and images to the Gemini model.
        
        Args:
            text_prompt: Text prompt to send
            image_paths: List of paths to image files
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Model's response as a string
        """
        if not GENAI_AVAILABLE:
            raise ImportError("Google AI package not installed. Please install with: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("No Google API key provided. Please provide an API key or set the GOOGLE_API_KEY environment variable.")
        
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL package not installed. Please install with: pip install pillow")
        
        # Override instance parameters with any provided in kwargs
        model = kwargs.get("model", self.model_name)
        temperature = kwargs.get("temperature", self.temperature)
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        system_instruction = kwargs.get("system_instruction", self.system_instruction)
        
        # Set up the generation config
        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        
        if max_output_tokens is not None:
            config.max_output_tokens = max_output_tokens
        
        if top_p is not None:
            config.top_p = top_p
            
        if top_k is not None:
            config.top_k = top_k
            
        if system_instruction is not None:
            config.system_instruction = system_instruction
        
        # Load the images
        loaded_images = []
        for img_path in image_paths:
            loaded_images.append(Image.open(img_path))
        
        # Create the content array with images and text
        contents = loaded_images + [text_prompt]
        
        try:
            self.logger.log("Sending multimodal request to Gemini API", {
                "model": model,
                "image_count": len(image_paths),
                "system_instruction": system_instruction is not None
            })
            
            # Use the API for multimodal content
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
            return response.text
                
        except Exception as e:
            self.logger.error(f"Error in Gemini API multimodal request: {str(e)}")
            raise

    async def stream(
        self, 
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Stream responses from the Gemini model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Yields:
            Response tokens as they are generated
        """
        if not GENAI_AVAILABLE:
            raise ImportError("Google AI package not installed. Please install with: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("No Google API key provided. Please provide an API key or set the GOOGLE_API_KEY environment variable.")
        
        # Set up the generation config
        config = types.GenerateContentConfig(
            temperature=self.temperature,
        )
        
        if self.max_output_tokens is not None:
            config.max_output_tokens = self.max_output_tokens
        
        if self.top_p is not None:
            config.top_p = self.top_p
            
        if self.top_k is not None:
            config.top_k = self.top_k
            
        if self.system_instruction is not None:
            config.system_instruction = self.system_instruction
        
        # Format messages for Gemini
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Handle special roles
            if role == "system":
                # Use system instruction if it's a system message
                config.system_instruction = content
                continue
            
            # For user and assistant messages, add them directly
            gemini_messages.append(content)
        
        try:
            self.logger.log("Streaming request to Gemini API", {
                "model": self.model_name,
                "message_count": len(gemini_messages),
                "system_instruction": self.system_instruction is not None
            })
            
            # Use the streaming API
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=gemini_messages,
                config=config
            )
            
            # Process the streaming response
            for chunk in response_stream:
                if not chunk.text:
                    continue
                
                # Call the on_token callback if provided
                if self.on_token:
                    self.on_token(chunk.text)
                    
                yield chunk.text
                
        except Exception as e:
            self.logger.error(f"Error in Gemini API streaming request: {str(e)}")
            raise 