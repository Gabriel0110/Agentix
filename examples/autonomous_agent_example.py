"""
Autonomous Agent Example

This example demonstrates how to create and use autonomous agents with self-healing
capabilities, planning, and optional human intervention points.
"""

import os
import asyncio
import time

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import (
    AutonomousAgent, 
    AutoAgentOptions, 
    AutoAgentHooks
)
from agentix.tools.function_tools import function_tool
from agentix.metrics.workflow_metrics import BaseWorkflowMetrics


# Create a simple example tool that might sometimes fail
@function_tool(
    name="CalculateResult",
    description="Calculate a result using the specified operation (add, subtract, multiply, divide).",
    usage_example='TOOL REQUEST: CalculateResult {"a": 25, "b": 17, "operation": "add"}'
)
def calculate_result(a: int, b: int, operation: str) -> str:
    """
    Calculate a result using the specified operation (add, subtract, multiply, divide).
    May fail for division by zero or unsupported operations.
    
    Args:
        a: First number
        b: Second number
        operation: The operation to perform (add, subtract, multiply, divide)
        
    Returns:
        The result as a string
    """
    if operation == "add":
        return f"Result: {a + b}"
    elif operation == "subtract":
        return f"Result: {a - b}"
    elif operation == "multiply":
        return f"Result: {a * b}"
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return f"Result: {a / b}"
    else:
        raise ValueError(f"Unsupported operation: {operation}")


# Create a more complex tool to demonstrate real-world use
@function_tool(
    name="FetchData",
    description="Fetch data from a URL with optional retries.",
    usage_example='TOOL REQUEST: FetchData {"url": "https://example.com", "max_retries": 3}'
)
async def fetch_data(url: str, max_retries: int = 3) -> str:
    """
    Simulate fetching data from a URL with potential failures.
    
    Args:
        url: The URL to fetch data from
        max_retries: Maximum number of retries before giving up
        
    Returns:
        The fetched data as a string
    """
    # Simulate network delays
    await asyncio.sleep(1)
    
    # Simulate random failures based on URL
    if "fail" in url:
        if max_retries <= 1:
            raise ConnectionError(f"Failed to connect to {url}")
        
        return f"Successfully fetched data after retry: {url}"
    
    if "error" in url:
        raise ValueError(f"Invalid URL format: {url}")
    
    if "timeout" in url:
        await asyncio.sleep(2)
        return f"Data fetched after delay: {url}"
    
    return f"Data fetched successfully from {url}"


# Handler for approval requests
async def handle_approval(action_type: str, details: str) -> bool:
    """
    Handle approval requests from the autonomous agent.
    
    Args:
        action_type: Type of action requiring approval
        details: Details about the action
    
    Returns:
        True if approved, False otherwise
    """
    print(f"\n--- Approval Required for {action_type} ---")
    print(details)
    response = input("\nApprove? (yes/no): ").strip().lower()
    return response in ("yes", "y", "")  # Empty string defaults to yes


# Progress callback
def report_progress(task_id: str, progress: float) -> None:
    """Report progress of a task."""
    print(f"Task {task_id} progress: {progress:.1%}")


async def run_autonomous_agent_example() -> None:
    """Run the autonomous agent example."""
    print("\n=== Autonomous Agent Example ===\n")
    
    # Initialize the LLM
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Set up metrics
    metrics = BaseWorkflowMetrics()
    
    # Define hooks for lifecycle events
    hooks = AutoAgentHooks(
        on_recovery_attempt=lambda task_id, error, attempt: print(f"\n🔄 Recovery attempt {attempt} for error: {str(error)}"),
        on_task_start=lambda task: print(f"\n🚀 Starting task: {task}"),
        on_task_complete=lambda task, result: print(f"\n✅ Task completed: {task}\nResult: {result}"),
        on_task_failed=lambda task, error: print(f"\n❌ Task failed: {task}\nError: {str(error)}"),
        on_thinking=lambda task_id, thinking: print(f"\n🤔 Agent thinking: {thinking[:100]}..."),
        on_approval_request=handle_approval
    )
    
    # Create the autonomous agent
    agent = AutonomousAgent.create(
        name="AutonomousAgent",
        model=model,
        memory=ShortTermMemory(max_messages=10),
        tools=[calculate_result, fetch_data],
        instructions=[
            "You are an autonomous problem-solving agent.",
            "You can solve tasks on your own, recovering from errors if needed.",
            "Be efficient and direct in your approach."
        ],
        options=AutoAgentOptions(
            max_recovery_attempts=3,
            enable_planning=True,
            auto_recovery=True,
            require_approval_for={},
            max_thinking_depth=5,
            approval_callback=handle_approval,
            progress_callback=report_progress,
            debug=True
        ),
        hooks=hooks,
        metrics=metrics
    )
    
    # Example tasks to demonstrate capabilities
    tasks = [
        "Calculate the result of adding 25 and 17, then multiply it by 2.",
        "Try to divide 10 by 0 and handle the error appropriately.",
        "Fetch data from https://example.com and summarize it.",
        "Fetch data from https://example.com/fail and handle any errors."
    ]
    
    # Execute each task
    for i, task in enumerate(tasks):
        task_id = f"example_task_{i}"
        print(f"\n\n{'='*50}")
        print(f"Task {i+1}: {task}")
        print(f"{'='*50}")
        
        try:
            result = await agent.execute_task(task, task_id)
            print(f"\n🎯 Final result: {result}")
        except Exception as e:
            print(f"\n💥 Unrecoverable error: {str(e)}")
        
        # Show current status
        status = agent.get_task_status(task_id)
        print(f"\nTask status: {status['status']}")
        print(f"Steps: {status['current_step']}/{status['total_steps']}")
        print(f"Time: {time.time() - status['start_time']:.2f}s")
    
    # Report metrics
    print("\n\n--- Metrics Report ---")
    print(metrics.report())


async def main() -> None:
    """Main function running the examples."""
    # Run the single autonomous agent example
    await run_autonomous_agent_example()

if __name__ == "__main__":
    asyncio.run(main()) 