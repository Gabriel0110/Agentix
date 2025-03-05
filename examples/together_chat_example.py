import asyncio
import os
from agentix.llms.together_chat import TogetherChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    # Initialize the Together chat client
    chat = TogetherChat(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.7
    )
    
    # Example 1: Basic chat completion
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = await chat.call(messages)
    print("\nBasic Chat Response:")
    print(response)
    
    # Example 2: Streaming chat completion
    streaming_chat = TogetherChat(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.7,
        stream=True,
        on_token=lambda token: print(token, end="", flush=True)
    )
    
    print("\n\nStreaming Chat Response:")
    await streaming_chat.call(messages)
    
    # Example 3: Vision capabilities (if using a vision-enabled model)
    vision_chat = TogetherChat(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        temperature=0.7
    )
    
    image_url = "https://example.com/image.jpg"  # Replace with actual image URL
    
    print("\n\nVision Model Response:")
    vision_response = await vision_chat.call_with_vision(
        prompt="What do you see in this image?",
        image_url=image_url
    )
    print(vision_response)

if __name__ == "__main__":
    asyncio.run(main()) 