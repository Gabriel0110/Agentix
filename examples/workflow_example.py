#!/usr/bin/env python
"""
workflow_example.py

This example demonstrates using the Workflow system to chain together multiple steps.
The example shows multiple types of workflows:
1. Sequential workflow - steps that run one after another
2. Parallel workflow - steps that run at the same time
3. Conditional workflow - steps that only run if a condition is met
"""
import os
import asyncio

from agentix.workflow import Workflow, WorkflowStep, LLMCallStep
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory

from dotenv import load_dotenv
load_dotenv()

async def main():
  # Create a model
  model = OpenAIChat(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7,
  )

  memory = ShortTermMemory(10)

  # Define steps
  step1 = LLMCallStep(model, "Step 1: Greet the user politely.")
  step2 = LLMCallStep(model, "Step 2: Provide a brief motivational quote.")

  workflow = Workflow([step1, step2], memory)

  userInput = "I need some positivity today!"
  print("User says:", userInput)

  finalOutput = await workflow.runSequential(userInput)
  print("Workflow Final Output:", finalOutput)

if __name__ == "__main__":
    asyncio.run(main()) 