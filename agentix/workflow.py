from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union, Awaitable

from .llms import LLM
from .memory.memory import Memory, ConversationMessage, MemoryRole


class WorkflowStep(ABC):
    """
    A workflow step: calls the model or performs some transformation,
    returning a single conversation message (role + content).
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a workflow step.
        
        Args:
            name: Optional name for the step
        """
        self.name = name
    
    @abstractmethod
    async def run(self, messages: List[Union[ConversationMessage, Dict[str, Any]]]) -> Union[ConversationMessage, Dict[str, Any]]:
        """
        Execute the workflow step.
        
        Args:
            messages: List of conversation messages to process
            
        Returns:
            A single conversation message (role + content)
        """
        pass


class Workflow:
    """
    A simple orchestrator that runs a sequence of steps.
    """
    
    def __init__(self, steps: List[WorkflowStep], memory: Memory):
        """
        Initialize a workflow with steps and memory.
        
        Args:
            steps: List of workflow steps to execute
            memory: Memory to store conversation history
        """
        self.steps = steps
        self.memory = memory
    
    async def run_sequential(self, input_str: str) -> str:
        """
        Run steps sequentially.
        
        Args:
            input_str: The user input string
            
        Returns:
            The final output from the workflow
        """
        # 1) Add user message
        await self.memory.add_message({"role": "user", "content": input_str})
        
        final_output = ""
        for step in self.steps:
            # 2) Gather context
            context = await self.memory.get_context()
            
            # 3) Step returns a conversation message (role + content)
            step_result = await step.run(context)
            await self.memory.add_message(step_result)
            
            if isinstance(step_result, dict):
                final_output = step_result.get("content", "")
            else:
                final_output = step_result.content
        
        return final_output
    
    async def run_parallel(self, input_str: str) -> List[str]:
        """
        Run steps in parallel.
        
        Args:
            input_str: The user input string
            
        Returns:
            List of outputs from each step
        """
        # 1) Add user message
        await self.memory.add_message({"role": "user", "content": input_str})
        context = await self.memory.get_context()
        
        # 2) Run all steps concurrently
        import asyncio
        results = await asyncio.gather(*[step.run(context) for step in self.steps])
        
        # 3) Add each result to memory
        for result in results:
            await self.memory.add_message(result)
        
        # 4) Return an array of content
        return [
            result.get("content", "") if isinstance(result, dict) else result.content
            for result in results
        ]
    
    async def run_conditional(
        self,
        input_str: str,
        condition_fn: Callable[[str], Union[bool, Awaitable[bool]]]
    ) -> str:
        """
        Run steps conditionally based on the output of the previous step.
        
        Args:
            input_str: The user input string
            condition_fn: Function that evaluates whether to continue to the next step
            
        Returns:
            The final output from the workflow
        """
        await self.memory.add_message({"role": "user", "content": input_str})
        
        final_output = ""
        for step in self.steps:
            context = await self.memory.get_context()
            step_result = await step.run(context)
            await self.memory.add_message(step_result)
            
            content = step_result.get("content", "") if isinstance(step_result, dict) else step_result.content
            
            # Check if the condition function is a coroutine
            condition_result = condition_fn(content)
            if isinstance(condition_result, Awaitable):
                condition_passed = await condition_result
            else:
                condition_passed = condition_result
                
            if not condition_passed:
                # Exit if condition fails
                break
            
            final_output = content
        
        return final_output


class LLMCallStep(WorkflowStep):
    """
    Example step that just calls an LLM with the entire conversation as input.
    """
    
    def __init__(
        self,
        model: LLM,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None
    ):
        """
        Initialize an LLM call step.
        
        Args:
            model: The LLM model to use (OpenAIChat or TogetherChat)
            system_prompt: Optional system prompt to use
            name: Optional name for the step
        """
        super().__init__(name)
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    async def run(self, messages: List[Union[ConversationMessage, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run the LLM call step.
        
        Args:
            messages: List of conversation messages to process
            
        Returns:
            A conversation message with the LLM's response
        """
        # We treat messages as the conversation so far
        enhanced_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Convert any ConversationMessage objects to dictionaries
        for msg in messages:
            if isinstance(msg, dict):
                enhanced_messages.append(msg)
            else:
                enhanced_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        response = await self.model.call(enhanced_messages)
        
        # Return a conversation message
        return {"role": "assistant", "content": response} 