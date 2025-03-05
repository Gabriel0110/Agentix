import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def runValidatedAgent():
  # Optionally use different models or the same model for both the agent and validation
  main_model = OpenAIChat(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o-mini"
  )

  validator_model = OpenAIChat(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o-mini"
  )

  memory = ShortTermMemory(20)

  agent_options = AgentOptions(
    validate_output=True, # We want to validate agent responses (requires the validator model)
    debug=True
  )

  agent = Agent.create(
    name="ValidatorAgent",
    model=main_model,
    validation_model=validator_model,   # <--- Provide the validator model
    memory=memory,
    instructions=["You are an agent that does simple math."],
    task="User wants the sum of two numbers",  # <--- Short example task specification
    options=agent_options,
  )

  user_query = "Add 2 and 2 for me, thanks!"
  final_ans = await agent.run(user_query)
  print("Final Answer from Agent:", final_ans)

if __name__ == "__main__":
    asyncio.run(runValidatedAgent())