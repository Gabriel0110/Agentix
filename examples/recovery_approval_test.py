#!/usr/bin/env python
import asyncio
import os
from typing import Optional, Dict

from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions
from agentix.tools import function_tool

from dotenv import load_dotenv
load_dotenv()

# A tool that sometimes fails to trigger recovery
@function_tool(
    name="UnreliableTool",
    description="A tool that sometimes fails to demonstrate recovery",
    usage_example='TOOL REQUEST: UnreliableTool {"input": "test"}'
)
def unreliable_tool(input: str, should_fail: bool = False) -> str:
    """
    A tool that fails when should_fail is True to test recovery.
    
    Args:
        input: Input string to process
        should_fail: If True, the tool will raise an exception
    """
    if should_fail:
        raise Exception("Tool failed as requested")
    return f"Successfully processed: {input}"

# A tool that requires approval
@function_tool(
    name="SensitiveTool",
    description="A tool that requires approval before execution",
    usage_example='TOOL REQUEST: SensitiveTool {"data": "sensitive_info"}'
)
def sensitive_tool(data: str) -> str:
    """
    A tool that processes sensitive data and requires approval.
    
    Args:
        data: Sensitive data to process
    """
    return f"Processed sensitive data: {data}"

async def approval_callback(action_type: str, details: str) -> bool:
    """Mock approval callback that simulates user interaction."""
    print(f"\n🔒 Approval Required for {action_type}")
    print(f"Details: {details}")
    print("Auto-approving for test... ✅")
    return True

async def main():
    """
    Example demonstrating recovery and approval capabilities.
    """
    print("\n🔧 Recovery and Approval Test")
    print("=" * 50)

    # Create the agent with our test configuration
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
    )

    # Configure agent with recovery and approval settings
    agent = Agent.create(
        name="RecoveryApprovalAgent",
        model=chat_model,
        memory=ShortTermMemory(max_messages=5),
        tools=[unreliable_tool, sensitive_tool],
        instructions=[
            "You are a test agent demonstrating recovery and approval capabilities.",
            "When using UnreliableTool, try with should_fail=false first.",
            "If it fails, try recovery by setting should_fail=false explicitly.",
            "For SensitiveTool, always explain what sensitive data you're processing."
        ],
        options=AgentOptions(
            enable_planning=True,
            debug=True,
            auto_recovery=True,
            max_recovery_attempts=3,
            require_approval_for={
                "tool.SensitiveTool",  # Tool-specific approval
                "Tool.UnreliableTool",  # Note the capital T to test case-insensitivity
                "Planning",
                "Recovery"
            },
            approval_callback=approval_callback
        )
    )

    # Test 1: Recovery Capability
    print("\n📝 Test 1: Recovery Capability")
    print("-" * 30)
    recovery_query = "Use the UnreliableTool with should_fail=true and handle the failure"
    print("User Query:", recovery_query)
    answer = await agent.run(recovery_query)
    print("\nAgent's Answer:", answer)

    # Test 2: Approval Workflow for SensitiveTool
    print("\n📝 Test 2: Approval Workflow - SensitiveTool")
    print("-" * 30)
    approval_query = "Process this sensitive data: 'test_data' using the SensitiveTool"
    print("User Query:", approval_query)
    answer = await agent.run(approval_query)
    print("\nAgent's Answer:", answer)

    # Test 3: Case-Insensitive Tool Approval
    print("\n📝 Test 3: Case-Insensitive Tool Approval")
    print("-" * 30)
    case_test_query = "Use the UnreliableTool with should_fail=false"
    print("User Query:", case_test_query)
    answer = await agent.run(case_test_query)
    print("\nAgent's Answer:", answer)

if __name__ == "__main__":
    asyncio.run(main()) 