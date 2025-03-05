#!/usr/bin/env python
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.evaluators import SimpleEvaluator
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Example demonstrating how to evaluate an agent's responses.
    """
    # 1) Create a model for the agent
    agent_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 2) Create memory
    memory = ShortTermMemory(max_messages=10)

    # 3) Create the agent
    agent = Agent.create(
        name="EvaluatedAgent",
        model=agent_model,
        memory=memory,
        instructions=["Provide a concise but detailed explanation. Always include 'FINAL ANSWER:' before your final response."],
        options=AgentOptions(
            max_steps=3, 
            usage_limit=5, 
            use_reflection=True, 
            debug=True
        )
    )

    # 4) Create a model for the evaluator
    eval_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    evaluator = SimpleEvaluator(model=eval_model)

    # 5) Run the agent
    user_question = "Explain the difference between supervised and unsupervised learning algorithms."
    print(f"User Question: {user_question}")
    
    answer = await agent.run(user_question)
    print("\nAgent's Final Answer:\n", answer)

    # 6) Evaluate the final answer
    messages = await memory.get_context()
    result = await evaluator.evaluate(messages)

    print("\nEvaluation Result:")
    print(f"Score: {result.score}")
    print(f"Feedback: {result.feedback}")
    if result.improvements:
        print(f"Improvements: {result.improvements}")
    else:
        print("No improvements suggested.")


if __name__ == "__main__":
    asyncio.run(main()) 