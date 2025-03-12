
[![GitHub stars](https://img.shields.io/github/stars/gabriel0110/agentix?style=flat-square)](https://github.com/gabriel0110/agentix/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/gabriel0110/agentix?style=flat-square)](https://github.com/gabriel0110/agentix/issues)
[![GitHub license](https://img.shields.io/github/license/gabriel0110/agentix?style=flat-square)](./LICENSE)  

# Agentix Framework

A **simple and extensible** Python framework for building AI-powered agents based on the original [Webby-Agents](https://github.com/gabriel0110/webby-agents) web-native TypeScript framework. This framework provides the core building blocks needed to integrate Large Language Models (LLMs) into applications and empower them with “agentic” capabilities. The goal is to provide simplicity but also customization and extensibility for more advanced use cases ***if you so desire***.

The core of this framework is built upon the writings of Chip Nguyen's [Agents](https://huyenchip.com/2025/01/07/agents.html) blog post and Anthropic's [Building effective agents](https://www.anthropic.com/research/building-effective-agents) blog post.

> **Note**: The framework is **experimental** and still under **active** development and tuning. ***Use at your own risk***. Please report any issues you encounter, and feel free to contribute!

## Key Features

- **OpenAI Integration**  
  Wrappers for OpenAI LLMs 
- **Together.AI Integration**  
  Wrapper for Together.AI LLMs 
- **Google Gemini Integration**  
  Wrapper for Google Gemini LLMs  

- **Flexible Memory**  
  - **ShortTermMemory** – stores recent messages for immediate context.  
  - **SummarizingMemory** – automatically summarizes older messages to keep the context manageable (supports optional hierarchical chunk-based summarization).  
  - **LongTermMemory** – an in-memory vector store for semantically relevant retrieval of older context.  
  - **CompositeMemory** – combine multiple memory classes into a single interface (e.g., short-term + summarizing + vector).  

- **Multi-Agent Orchestration**  
  Classes like `AgentTeam` and `AgentRouter` let you run multiple agents in parallel, sequentially, or with routing logic.

- **Pluggable Planning & Workflows**  
  - **Planner** interface for generating structured task plans.  
  - **Workflow** for fixed step-by-step or parallel tasks.

- **Tool Usage**  
  Agents can call custom external “Tools” in a multi-step loop, retrieving data and incorporating it into final answers. You can extend the `Tool` interface for your own use cases.
    - **Parameterized Tools** – tools that take input parameters for more dynamic behavior. See the `tool_parameter_demo.ts` example on how to call tools with required and optional parameters.
    - **Function-based Tools** – tools that are defined as Python functions.
    - **Example Tools**
      - **Firecrawl** – scrape and crawl websites (https://firecrawl.dev/)
      - **YFinance** – get stock market data.
      - **DuckDuckGoSearch** – search the web using DuckDuckGo.
      - **Tavily** - search the web using Tavily (https://tavily.com/)

- **Safety Controls**  
  Configure max reflection steps, usage limits, time-to-live, plus hooks for user approval on tool calls and task validation.

- **Debug Logging & Hooks**  
  Add custom hooks for logging, debugging, or user approval on tool calls. You can enable agent debugging for more detailed logs via the `debug` option, setting it to `true`.

- **Lightweight & Modular**  
  Use only the parts you need, or extend them for advanced use cases (e.g., reflection memory, external vector DBs).

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage & Examples](#usage--examples)  
   - [Basic Agent (Single-Pass)](#1-basic-agent-single-pass)  
   - [Workflow Example (Fixed Steps)](#2-workflow-example-fixed-steps)  
   - [Multi-Tool Agent](#3-multi-tool-agent)  
   - [Agent Team (Parallel/Sequential)](#4-agent-team-parallelsequential)  
   - [RAG Demo (Long-Term Memory Retrieval)](#5-rag-demo-long-term-memory-retrieval)  
   - [Planner Example](#6-planner-example)  
   - [Evaluator Example](#7-evaluator-example)  
   - [Agent with Logging Hooks](#8-agent-with-logging-hooks) 
   - [Agent Task Specification and Output Validation](#9-agent-task-specification-and-output-validation) 
   - [Advanced Team Orchestration](#10-advanced-team-orchestration)
   - [Agent Team with Summarizer](#11-agent-team-with-summarizer)
   - [Production Agentic Workflow Example](#12-production-agentic-workflow-example)
   - [Function-based Tools Example](#13-function-based-tools-example)
3. [Agent Options & Settings](#agent-options--settings)  
4. [Memory](#memory)  
5. [Models](#models)  
6. [Multi-Agent Orchestration](#multi-agent-orchestration)  
7. [Planner & Workflow](#planner--workflow)  
8. [Evaluators](#evaluators)  
9. [Advanced Patterns and Best Practices](#advanced-patterns-and-best-practices)  
   - [Reflection Memory](#reflection-memory)  
   - [Safe Run Methods](#safe-run-methods)  
   - [Advanced Multi-Agent Synergy](#advanced-multi-agent-synergy)
10. [Building & Running](#building--running)  
11. [FAQ](#faq)  
12. [Roadmap](#roadmap)  
13. [License](#license)

---

## Installation

```bash
pip install agentix
```

---

## Usage & Examples

Below are demos demonstrating different ways to build and orchestrate agents.

### 1) **Basic Agent (Single-Pass)**

**Goal**: A minimal agent that performs a single LLM call. No reflection, no tools, just a direct “Question -> Answer.”

```python
import asyncio
import sys
import os

from agentix.llms import OpenAIChat, TogetherChat
from agentix.memory import ShortTermMemory
from agentix.agents import Agent, AgentOptions

# Callback function for token streaming
def on_token(token):
    sys.stdout.write(token)
    sys.stdout.flush()


async def main():
    """
    Example demonstrating a minimal agent setup.
    """
    # 1) Create a minimal LLM
    chat_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7,
        #stream=True,  # Stream output to console
        #on_token=on_token  # Hook to process tokens
    )

    # chat_model = TogetherChat(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    #     temperature=0.7,
    #     #stream=True,  # Stream output to console
    #     #on_token=on_token  # Hook to process tokens
    # )

    # 2) Create a simple short-term memory
    short_term_memory = ShortTermMemory(max_messages=5)

    # 2.1) Create agent options
    agent_options = AgentOptions(
        use_reflection=False,
        max_steps=5,
        usage_limit=5,
        time_to_live=5000,
    )

    # 3) Instantiate an Agent with NO reflection or tools
    agent = Agent.create(
        model=chat_model,
        memory=short_term_memory,
        instructions=[
            "You are a simple agent. Answer only in one short sentence."
        ],
        options=agent_options,
    )

    # 4) Run the agent with a simple question
    user_question = "What's a quick tip for staying productive at work?"
    print("User Question:", user_question)

    answer = await agent.run(user_question)
    print("\n\nAgent's Answer:", answer)


if __name__ == "__main__":
    asyncio.run(main()) 
```

**Key Observations**:
- Agent is **single-pass** by setting `useReflection: false`.  
- Only **ShortTermMemory** is appended automatically, storing the last few messages.  
- Good for trivial or low-cost tasks with no tool usage.

---

### 2) **Workflow Example (Fixed Steps)**

**Goal**: Demonstrate a fixed-step approach using the `Workflow` class, which is simpler than an agent for known tasks.

```python
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

async def main():
  # Create a model
  model = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
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
```

**Key Observations**:
- Each `LLMCallStep` has its own “system prompt.”  
- The user input is added to memory, each step sees the updated context.  
- Great for “scripted” or “predefined” pipelines.

---

### 3) **Multi-Tool Agent**

**Goal**: Show how an agent can have multiple tools (fake or real) and call them autonomously.

```python
import os
import asyncio
from typing import Dict, Any, Optional

from agentix.agents import Agent, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import Tool

# Dummy tool #1
class FakeSearchTool(Tool):
    """A dummy search tool that returns fake results."""
    
    @property
    def name(self) -> str:
        return "FakeSearch"
    
    @property
    def description(self) -> str:
        return "Simulates a search engine lookup (dummy)."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f'FAKE SEARCH RESULTS for "{input_str}" (no real search done).'


# Dummy tool #2
class FakeTranslatorTool(Tool):
    """A dummy translator tool that returns fake French translations."""
    
    @property
    def name(self) -> str:
        return "FakeTranslator"
    
    @property
    def description(self) -> str:
        return "Pretends to translate input text into French."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f'FAKE TRANSLATION to French of: "{input_str}" => [Ceci est une traduction factice]'


async def main():
    """
    Example demonstrating an agent with multiple tools.
    """
    # 1) Create LLM
    chat_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )

    # 2) Memory
    mem = ShortTermMemory(max_messages=10)

    # 3) Tools
    search_tool = FakeSearchTool()
    translator_tool = FakeTranslatorTool()

    # 4) Agent Options
    options = AgentOptions(
        max_steps=5,
        usage_limit=5,
        time_to_live=60000,
        use_reflection=True,
        debug=True,
    )

    # 5) Create Agent with multiple tools
    agent = Agent.create(
        name="MultiToolAgent",
        model=chat_model,
        memory=mem,
        tools=[search_tool, translator_tool],
        instructions=[
            "You can use FakeSearch to look up information.",
            "You can use FakeTranslator to convert text to French.",
            "Use tools by responding EXACTLY in the format: TOOL REQUEST: <ToolName> \"<Query>\"",
            "Integrate tool results before proceeding to the next step.",
        ],
        options=options,
    )

    # 6) User question
    user_question = "Search for today's top news and then translate the summary into French."
    print("\nUser Question:", user_question)

    # 7) Run agent
    answer = await agent.run(user_question)
    print("\nAgent's Final Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 
```

**Key Observations**:
- Multiple tools allow more complex tasks.  
- The agent can choose to use one or both tools in its reflection loop.  
- Tools are minimal “run(input: string) => string” classes.

---

### 4) **Agent Team (Parallel/Sequential)**

**Goal**: Show how multiple agents can coordinate using `AgentTeam`.

```python
import os
import asyncio

from agentix.agents import Agent, AgentTeam, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory

async def main():
    """
    Example demonstrating how to use AgentTeam with parallel and sequential execution.
    """
    # 1) Create base LLMs
    agent1_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    agent2_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 2) Memory
    mem1 = ShortTermMemory(max_messages=5)
    mem2 = ShortTermMemory(max_messages=5)

    # 3) Agent #1: "GreetingAgent"
    greeting_agent = Agent.create(
        name="GreetingAgent",
        model=agent1_model,
        memory=mem1,
        instructions=["Greet the user in a friendly way."],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )

    # 4) Agent #2: "MotivationAgent"
    motivation_agent = Agent.create(
        name="MotivationAgent",
        model=agent2_model,
        memory=mem2,
        instructions=["Provide a short motivational statement or advice to the user."],
        options=AgentOptions(max_steps=1, use_reflection=False)
    )

    # 5) Create an AgentTeam
    team = AgentTeam("Greeting+MotivationTeam", [greeting_agent, motivation_agent])

    # 6) Use run_in_parallel
    user_prompt = "I could use some positivity today!"
    print("User Prompt:", user_prompt)

    parallel_results = await team.run_in_parallel(user_prompt)
    print("\nParallel Results:\n", parallel_results)

    # 7) Use run_sequential
    sequential_result = await team.run_sequential(user_prompt)
    print("\nSequential Result:\n", sequential_result)


if __name__ == "__main__":
    asyncio.run(main()) 
```

**Key Observations**:
- `runInParallel` returns an array of answers.  
- `runSequential` passes the previous agent’s output as the next agent’s input.  
- Each agent can have its own memory, instructions, or tools.

---

### 5) **RAG Demo (Long-Term Memory Retrieval)**

**Goal**: Show how to store older context in a semantic vector store and retrieve it later.

```python
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.memory import (
    CompositeMemory,
    ShortTermMemory,
    SummarizingMemory,
    LongTermMemory,
)
from agentix.llms import OpenAIChat, OpenAIEmbeddings

async def main():
    """
    Example demonstrating a RAG (Retrieval-Augmented Generation) agent with
    composite memory including short-term, summarizing, and long-term memory.
    """
    # 1) Chat model
    chat_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 2) Summarizer model
    summarizer_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    # 3) Embeddings for long-term
    embeddings_model = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 4) Memory instances
    short_mem = ShortTermMemory(max_messages=10)
    summarizing_mem = SummarizingMemory(
        summarizer_model=summarizer_model,
        threshold=5,
        summary_prompt="Summarize earlier conversation:",
        max_summary_tokens=200
    )
    long_term_mem = LongTermMemory(
        embeddings=embeddings_model,
        max_messages=100,
        top_k=3
    )

    # 5) Composite memory
    composite_mem = CompositeMemory(short_mem, summarizing_mem, long_term_mem)

    # 6) Agent
    agent_options = AgentOptions(
        max_steps=5,
        usage_limit=10,
        time_to_live=60000,
        use_reflection=True,
        debug=True
    )

    agent = Agent.create(
        name="RAGAgent",
        model=chat_model,
        memory=composite_mem,
        instructions=[
            "If the user asks about older content, recall from memory. If uncertain, say so politely."
        ],
        options=agent_options
    )

    # 7) Simulate a user adding data, then later asking about it
    # First: user provides some info
    print("User: I'm planning a road trip from LA to Vegas next month, maybe around the 15th.")
    await agent.run("I'm planning a road trip from LA to Vegas next month, maybe around the 15th.")
    
    print("\nUser: I want to remember that I'll have a budget of $500 total.")
    await agent.run("I want to remember that I'll have a budget of $500 total.")

    # Later: user asks
    question = "Hey, do you recall how much money I budgeted for my LA to Vegas trip?"
    print(f"\nUser: {question}")
    answer = await agent.run(question)
    print("\nFinal Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 
```

**Key Observations**:
- ShortTermMemory captures immediate recency, SummarizingMemory condenses older conversation, LongTermMemory performs semantic retrieval.  
- CompositeMemory merges them all, so the agent has a holistic memory.  
- By default, the agent tries to append everything, but can be adapted for more advanced usage.

---

### 6) **Planner Example**

**Goal**: Show how a `Planner` can generate a structured plan (JSON or bullet list) that the agent may follow before final reasoning.

```python
import os
import asyncio
from typing import Dict, Any, Optional

from agentix.agents import Agent, AgentOptions
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory, Tool
from agentix.planner import SimpleLLMPlanner

# Dummy tool
class DummyCalendarTool(Tool):
    """A dummy calendar tool that simulates scheduling events."""
    
    @property
    def name(self) -> str:
        return "Calendar"
    
    @property
    def description(self) -> str:
        return "Manages event scheduling and date lookups (dummy)."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f"FAKE CALENDAR ACTION: {input_str}"


async def main():
    """
    Example demonstrating how to use a planner with an agent.
    """
    # 1) Create an LLM for both agent & planner
    main_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    planner_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.3
    )

    # 2) Planner
    planner = SimpleLLMPlanner(planner_model=planner_model)

    # 3) Memory
    memory = ShortTermMemory(max_messages=5)

    # 4) Tool
    calendar = DummyCalendarTool()

    # 5) Create Agent with Planner
    def on_plan_generated(plan):
        print("[PLAN GENERATED]\n", plan)
    
    agent = Agent.create(
        name="PlannerAgent",
        model=main_model,
        memory=memory,
        tools=[calendar],
        planner=planner,
        instructions=[
            "You can plan tasks first, then execute them. If a plan step references 'Calendar', call the Calendar tool."
        ],
        options=AgentOptions(
            max_steps=5,
            usage_limit=10,
            time_to_live=30000,
            use_reflection=True,
            debug=True,
        ),
        hooks={
            "on_plan_generated": on_plan_generated
        }
    )

    # 6) User request
    user_query = "Schedule a meeting next Friday to discuss project updates."
    print("User Query:", user_query)

    # 7) Run agent
    answer = await agent.run(user_query)
    print("\nFinal Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 
```

**Key Observations**:
- `SimpleLLMPlanner` can produce a plan describing steps or tools to call.  
- The agent can parse or interpret that plan in a multi-step loop.  
- `onPlanGenerated` hook logs the plan for debugging.

---

### 7) **Evaluator Example**

**Goal**: Show how an additional LLM call can critique or score the agent’s final output using `SimpleEvaluator`.

```python
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.evaluators import SimpleEvaluator
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

async def main():
    """
    Example demonstrating how to evaluate an agent's responses.
    """
    # 1) Create a model for the agent
    agent_model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
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
        api_key=os.getenv("OPENAI_API_KEY"),
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
```

**Key Observations**:
- A separate LLM pass can generate a `score` and `feedback`.  
- In production, you might automate a re-try loop if score < threshold.  
- Evaluation is an optional feature to refine or grade agent outputs.

---

### 8) **Agent with Logging Hooks**

**Goal**: Demonstrate using `hooks` (`onStep`, `onToolCall`, `onFinalAnswer`) for debugging and user approvals.

```python
import os
import asyncio
from typing import Dict, Any, List, Optional, Union, Awaitable

from agentix.agents import Agent, AgentOptions, AgentHooks
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat
from agentix.tools import Tool

# Dummy tool
class DummyMathTool(Tool):
    """A dummy math tool that always returns 42."""
    
    @property
    def name(self) -> str:
        return "DummyMath"
    
    @property
    def description(self) -> Optional[str]:
        return "Performs fake math calculations (dummy)."
    
    async def run(self, input_str: str, args: Optional[Dict[str, Any]] = None) -> str:
        return f'FAKE MATH RESULT for "{input_str}": 42 (always).'


async def main():
    """
    Example demonstrating how to use hooks with an Agent.
    """
    # 1) Create LLM
    model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.6
    )

    # 2) Memory
    memory = ShortTermMemory(max_messages=5)

    # 3) Hooks
    async def on_tool_call(tool_name: str, query: str) -> bool:
        print(f'[Hook: on_tool_call] About to call "{tool_name}" with query="{query}"')
        # Could confirm or deny usage. If we return False, the call is canceled
        return True
    
    def on_step(messages: List[Dict[str, Any]]):
        print("[Hook: on_step] Current conversation so far:", messages)
    
    def on_final_answer(answer: str):
        print("[Hook: on_final_answer] The final answer is:", answer)
    
    hooks = AgentHooks(
        on_tool_call=on_tool_call,
        on_step=on_step,
        on_final_answer=on_final_answer
    )

    # 4) Create Agent
    math_tool = DummyMathTool()
    agent = Agent.create(
        name="HookedAgent",
        model=model,
        memory=memory,
        tools=[math_tool],
        instructions=['Use DummyMath if the user needs a calculation. Request it in the format "TOOL REQUEST: DummyMath {"num1": "123", "num2": "456"}"'],
        hooks=hooks,
        options=AgentOptions(
            use_reflection=True,
            max_steps=5,
            usage_limit=5,
            time_to_live=60000,
            debug=True,
        )
    )

    # 5) Run agent
    question = "What is 123 + 456, approximately?"
    print("User asks:", question)

    answer = await agent.run(question)
    print("\nFinal Answer from Agent:\n", answer)


if __name__ == "__main__":
    asyncio.run(main()) 
```

**Key Observations**:
- `onToolCall` can be used to require user confirmation or log usage.  
- `onStep` shows the conversation state after each reflection step.  
- `debug: true` provides more console logs for diagnosing agent flow.

---

### 9) **Agent Task Specification and Output Validation**

**Goal**: Demonstrate how to specify a task for the agent and have the output validated by a validation model.

```python
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

async def runValidatedAgent():
  # Optionally use different models or the same model for both the agent and validation
  main_model = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
  )
  validator_model = OpenAIChat(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
  )

  memory = ShortTermMemory(20)

  agent_options = AgentOptions(
    validate_output=True, # We want to validate agent responses
    debug=True
  )

  agent = Agent.create(
    name="ValidatorAgent",
    model=main_model,
    validation_model=validator_model,   # <--- Provide the validator
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
```

**Key Observations**:
- `validateOutput: true` tells the agent to validate its output.
- The `task` field is a short description of the task the agent is expected to perform.
- The `validationModel` is used to validate the agent's output.

---

### 10) **Advanced Team Orchestration**

**Goal**: Show how to orchestrate multiple agents in parallel or sequentially.

```python
#!/usr/bin/env python
"""
advanced_team_collaboration_example.py

This example demonstrates the use of AdvancedAgentTeam for collaborative problem-solving where:
1. Agents have specialized roles with custom query transformations
2. Agents communicate with each other through shared memory
3. The team runs in an interleaved manner, where each agent builds on others' insights
4. The process continues until convergence criteria are met
5. Advanced hooks track the progress of the collaboration

This pattern is particularly useful for complex problem-solving that requires
iterative refinement from different perspectives working together.
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import (
    AdvancedAgentTeam, 
    AdvancedTeamOptions, 
    AdvancedTeamHooks,
    AgentRole, 
    TeamConfiguration, 
)
from agentix.memory import ShortTermMemory, CompositeMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Main function demonstrating an advanced agent team collaboration.
    """
    # Create shared model for all agents
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create a shared memory for the team
    shared_memory = ShortTermMemory(max_messages=20)
    
    # Create agent roles with specialized query transformations
    
    # Analyst role focuses on breaking down problems and identifying key components
    def analyst_transform(query: str) -> str:
        return f"As a strategic analyst, break down this problem into key components: {query}\n\nConsider what information we already have and what we still need to determine."
    
    # Critic role focuses on identifying potential issues or weaknesses
    def critic_transform(query: str) -> str:
        return f"As a critical thinker, evaluate the current approach to this problem: {query}\n\nIdentify any logical flaws, missing information, or alternative perspectives that should be considered."
    
    # Innovator role focuses on creative solutions and novel approaches
    def innovator_transform(query: str) -> str:
        return f"As an innovative thinker, suggest creative approaches to this problem: {query}\n\nBuild upon the team's current insights and propose solutions that might not be immediately obvious."
    
    # Synthesizer role focuses on combining insights and creating a cohesive solution
    def synthesizer_transform(query: str) -> str:
        return f"As a synthesizing expert, combine our collective insights on this problem: {query}\n\nCreate a cohesive solution that addresses the key points raised by the team."
    
    # Define team configuration with specialized roles
    team_config = TeamConfiguration(
        roles={
            "Analyst": AgentRole(
                name="Analyst",
                description="Breaks down problems and identifies key components",
                query_transform=analyst_transform
            ),
            "Critic": AgentRole(
                name="Critic", 
                description="Identifies potential issues or weaknesses",
                query_transform=critic_transform
            ),
            "Innovator": AgentRole(
                name="Innovator",
                description="Proposes creative solutions and novel approaches",
                query_transform=innovator_transform
            ),
            "Synthesizer": AgentRole(
                name="Synthesizer",
                description="Combines insights and creates cohesive solutions",
                query_transform=synthesizer_transform
            )
        },
        default_role=None  # No default role; each agent must have a specific role
    )
    
    # Create the specialized agents
    analyst_agent = Agent.create(
        name="Analyst",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are a strategic analyst who excels at breaking down complex problems.",
            "Identify key components, available information, and knowledge gaps.",
            "Create structured analyses that help the team understand the problem space."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    critic_agent = Agent.create(
        name="Critic",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are a critical thinker who evaluates proposed approaches and solutions.",
            "Identify logical flaws, missing information, and unexplored alternatives.",
            "Your role is not to be negative, but to strengthen the team's thinking."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    innovator_agent = Agent.create(
        name="Innovator",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are an innovative thinker who generates creative solutions.",
            "Build upon the team's analysis to propose novel approaches.",
            "Don't hesitate to suggest unconventional ideas that might lead to breakthroughs."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    synthesizer_agent = Agent.create(
        name="Synthesizer",
        model=model,
        memory=CompositeMemory(ShortTermMemory(max_messages=5)),  # Will be replaced with shared memory
        instructions=[
            "You are a synthesis expert who combines diverse perspectives into cohesive solutions.",
            "Integrate the team's insights, addressing conflicts and finding common ground.",
            "Create comprehensive solutions that reflect the collective intelligence of the team."
        ],
        options=AgentOptions(use_reflection=True, max_steps=1)
    )
    
    # Create advanced team hooks for monitoring the collaboration
    hooks = AdvancedTeamHooks(
        # Basic team hooks
        on_agent_start=lambda agent_name, query: print(f"\n🚀 {agent_name} starting work..."),
        on_agent_end=lambda agent_name, result: print(f"✅ {agent_name} contributed"),
        on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
        on_final=lambda results: print(f"🏁 Team process completed with {len(results)} contributions"),
        
        # Advanced hooks for round-based collaboration
        on_round_start=lambda round_num, max_rounds: print(f"\n📊 Starting collaboration round {round_num}/{max_rounds}"),
        on_round_end=lambda round_num, contributions: print(f"📝 Round {round_num} complete with {len(contributions)} contributions"),
        on_convergence=lambda agent, content: print(f"🎯 {agent.name} proposed a solution that meets convergence criteria"),
        on_aggregation=lambda final_result: print(f"🧩 Final solution synthesized from {len(final_result.split())} words")
    )
    
    # Configure the advanced team
    team_options = AdvancedTeamOptions(
        shared_memory=shared_memory,
        team_config=team_config,
        hooks=hooks,
        debug=True
    )
    
    # Create the advanced agent team
    team = AdvancedAgentTeam(
        name="ProblemSolvingTeam",
        agents=[analyst_agent, critic_agent, innovator_agent, synthesizer_agent],
        options=team_options
    )
    
    # Enable shared memory so all agents can see each other's contributions
    team.enable_shared_memory()
    
    # Define a convergence check function to determine when the team has reached a solution
    def check_convergence(content: str) -> bool:
        """
        Check if the content represents a converged solution.
        
        A solution is converged when:
        1. It contains "FINAL SOLUTION:" indicating the team believes they've solved it
        
        Args:
            content: The content to check
            
        Returns:
            True if convergence criteria are met, False otherwise
        """
        # Check for explicit final solution marker
        has_final_marker = "FINAL SOLUTION:" in content.upper()
        
        return has_final_marker
    
    async def solve_problem_collaboratively(query: str, max_rounds: int = 5) -> str:
        """
        Use the advanced agent team to solve a complex problem collaboratively.
        
        Args:
            query: The problem to solve
            max_rounds: Maximum number of collaboration rounds
            
        Returns:
            The team's final solution
        """
        print(f"\n🔍 Team tackling problem: '{query}'")
        
        # Run the team in interleaved mode until convergence or max rounds
        # Each agent will see others' contributions through shared memory
        final_solution = await team.run_interleaved(
            user_query=query,
            max_rounds=max_rounds,
            is_converged=check_convergence
        )
        
        return final_solution
    
    # Example complex problems that benefit from collaborative problem-solving
    problems = [
        "Design a sustainable urban transportation system that reduces carbon emissions while improving accessibility for all residents.",
        
        #"Develop a strategy for a community to prepare for and adapt to increasing climate-related disasters with limited resources.",
        
        #"Create an education system that better prepares students for the rapidly changing job market of the future."
    ]
    
    for problem in problems:
        print("\n" + "=" * 100)
        print(f"COMPLEX PROBLEM: {problem}")
        print("=" * 100)
        
        solution = await solve_problem_collaboratively(problem)
        
        print("\n" + "=" * 40 + " COLLABORATIVE SOLUTION " + "=" * 40)
        print(solution)
        print("=" * 100)
        
        # Add a pause between problems
        if problem != problems[-1]:
            print("\nMoving to next problem in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main()) 
```

---

### 11) **Agent Team with Summarizer**

See `examples/advanced/agent_team_summarizer_example.py`

---

### 12) **Production Agentic Workflow Example**

See `examples/advanced/startup_research_workflow.py`

---

### 13) **Function-based Tools Example**

See `examples/function_tool_example.py`

---

## Agent Options & Settings

**`AgentOptions`** let you shape agent behavior:

| Option         | Default   | Description                                                                     |
|----------------|-----------|---------------------------------------------------------------------------------|
| **`maxSteps`** | `15`       | Max reflection steps in the reasoning loop (`-1` = unlimited).                 |
| **`usageLimit`** | `15`     | Maximum total LLM calls (cost control) (`-1` = unlimited)                      |
| **`useReflection`** | `true` | If `false`, a single pass only. Tools require reflection to see their results.|
| **`timeToLive`** | `60000` | (ms) Halts the agent if it runs too long. (`-1` = unlimited).                   |
| **`debug`** | `false`     | More logs about each step and the final plan.                                    |
| **`validateOutput`** | `false` | If `true`, the agent validates its output with a second LLM.                |

---

## Memory

### Memory Philosophy

- **ShortTermMemory** is best for immediate context (most recent messages).  
- **SummarizingMemory** prevents bloat by condensing older conversation; optionally can store multiple chunk-level summaries if `hierarchical` is set.  
- **LongTermMemory** uses semantic embeddings for retrieving older messages by similarity (mini RAG).  
- **CompositeMemory** merges multiple memory strategies into one.  

### ReflectionMemory (Optional)

You can create a specialized memory just for the agent’s chain-of-thought or self-critique (“reflection”) that is never shown to the user. This can be helpful for debugging or advanced self-correction patterns.

---

## Models

### `OpenAIChat`

- **`model`**: e.g., `"gpt-4o-mini"` 
- **`temperature`**: Controls creativity.  
- **`stream`** + **`onToken`**: For partial token streaming.  

### `OpenAIEmbeddings`

- **`model`**: e.g., `"text-embedding-3-small"`.  
- Used for semantic similarity in `LongTermMemory`.  

### `TogetherAIChat`

- **`model`**: e.g., `"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"` 
- **`temperature`**: Controls creativity.  
- **`stream`** + **`onToken`**: For partial token streaming. 

---

## Multi-Agent Orchestration

### `AgentTeam`

Runs multiple Agents in **parallel** (`runInParallel`) or **sequential** (`runSequential`). Good for combining domain-specific agents (e.g. finance + web search + summarizer).

### `AgentRouter`

Uses a custom routing function to pick which agent handles a query.

### `AdvancedAgentTeam`

A more advanced version of `AgentTeam` that allows for more complex routing logic, hooks, and interleaved round-robin-style execution.

### `AdvancedAgentRouter`

A more advanced version of `AgentRouter` that allows for more complex routing logic, including LLM-based routing, agent capability specifications, and more.

### `LLMConvergenceChecker`

A custom convergence check for multi-agent orchestration that uses an LLM to decide if convergence has been reached. This can be useful for more complex multi-agent orchestration scenarios.

---

## Planner & Workflow

- **`Planner`** interface + **`SimpleLLMPlanner`** let you do a “plan-then-execute” approach, where the LLM can propose a structured plan (for example in JSON) and the system executes each step. This is typically for more open-ended tasks where you want some autonomy, but still want to parse or validate a plan.

- **`Workflow`** provides a **simpler**, more **prescriptive** pattern for tasks that follow a **known sequence** of steps. Instead of letting the LLM dynamically decide how to solve the problem (like an Agent would), the developer defines a series of steps in code. Each step receives the current conversation context (from memory) and returns a new message. The `Workflow` then appends that message to memory and continues to the next step.

### Workflows in Detail

A **Workflow** is composed of multiple **workflow steps**. Each step implements the interface:

```python
class WorkflowStep(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    @abstractmethod
    async def run(self, messages: List[Union[ConversationMessage, Dict[str, Any]]]) -> Union[ConversationMessage, Dict[str, Any]]:
        pass
```

The **`Workflow`** class orchestrates how these steps are invoked:

1. **`runSequential`**:
   - Calls each step **in order**, passing in the updated conversation context (from memory).
   - Each step returns a new message, which is appended to memory.
   - The final output is the `content` of the last step’s message.

2. **`runParallel`**:
   - Runs **all steps at once** on the same conversation context, gathering all results and appending them to memory.
   - Returns an array of the messages `content` values.

3. **`runConditional`**:
   - Similar to `runSequential`, but you provide a `conditionFn` that checks the last step’s output. If the condition fails, it stops immediately.

### When to Use a Workflow vs. an Agent?

- **Workflow**:  
  - You have a **predefined** or **fixed** series of steps you want to run each time (e.g., “collect user input, summarize, translate, finalize”).
  - You need **predictability** or a **scripted** approach.  
  - Each step is a known function or LLM call; the model does not “choose” how to proceed.

- **Agent**:  
  - The LLM is **autonomous** and decides which tool(s) to call, in which order, and when to produce a final answer.
  - You want **dynamic** multi-step reasoning or “tool usage” in a ReAct-like loop.
  - The agent uses reflection, tool requests, memory, and potentially self-correction or planning.

### Combining an Agent with a Workflow

You can place an **Agent** call inside a **WorkflowStep** if you want a hybrid approach:

```python
class AgentCallStep(WorkflowStep):
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    async def run(self, messages: List[Union[ConversationMessage, Dict[str, Any]]]) -> Union[ConversationMessage, Dict[str, Any]]:
        # Possibly parse 'messages' to get user input or context
        user_input = next((m.content for m in messages if m.role == "user"), "")
        agent_answer = await self.agent.run(user_input)
        return { role: "assistant", content: agent_answer }
```

Then, include `AgentCallStep` in your workflow steps array if you want “one step” to let the LLM operate in a more autonomous, tool-using manner, but still in a bigger scripted flow.

**In short**, **Workflows** are a simpler, **prescriptive** approach to orchestrating multiple LLM calls or transformations, while **Agents** handle open-ended tasks where the LLM can reason about which steps (Tools) to use to get to the final answer.


---

## Evaluators

- **`SimpleEvaluator`** uses a second LLM to critique or rate the final output.  
- Could be extended for “chain-of-thought” improvement loops, auto-correction, or advanced QA.

---

## Advanced Patterns and Best Practices

Beyond the standard usage patterns (single-pass Agents, Workflows, multi-tool or multi-agent orchestration), **Agentix** supports more advanced scenarios that can significantly expand agent capabilities. Below are additional patterns and tips for **self-reflection**, **multi-agent synergy**, **error-safe runs**, and more.

### Reflection Memory

**What is it?**  
A specialized `ReflectionMemory` allows the agent to store an internal “chain-of-thought” or self-critique messages (role: `"reflection"`) that aren’t shown to the user. This can be useful for:
- **Self-correction**: The agent can note mistakes, then fix them in subsequent steps (if `includeReflections` is `true`).
- **Debugging**: Developers can review the chain-of-thought to see where the agent might have gone wrong without exposing it to end users.
- **Audit / Logging**: Keep an internal record of the agent’s reasoning steps for advanced QA.

**Example**  
```python
from agentix.agents import Agent
from agentix.memory import ShortTermMemory, CompositeMemory, ReflectionMemory
from agentix.llms import OpenAIChat

async def main():
  chat_model = OpenAIChat(api_key="YOUR_API_KEY", model="gpt-4o-mini")

  # Public conversation memory
  public_mem = ShortTermMemory(5)

  # Reflection memory (not shown to user)
  reflection_mem = ReflectionMemory(include_reflections=False) # False => do NOT append reflection to prompt

  # Combine them so the agent has a single memory object
  composite = CompositeMemory(public_mem, reflection_mem)

  agent = Agent.create(
    name="ReflectiveAgent",
    model=chat_model,
    memory=composite,
    instructions=[
      "You are a reflective agent; keep your chain-of-thought hidden from the user."
    ]
  )

  # Add logic to store reflection after final answer or each step
  original_hooks = agent["hooks"] or {}
  agent["hooks"] = {
    ...original_hooks,
    onFinalAnswer: (answer: str) => {
      # Save a reflection message
      reflection_mem.add_message({
        role: "reflection",
        content: f"I produced answer=\"{answer}\". Next time, double-check for accuracy."
      })
    }
  }

  user_question = "How tall is Mount Everest in meters?"
  final_answer = await agent.run(user_question)
  print("Agent's Final Answer =>", final_answer)

  # Inspect reflection memory for debugging
  reflections = await reflection_mem.get_context()
  print("ReflectionMemory =>", reflections)
```

> **Chain-of-Thought Disclaimer**: If you choose to feed the reflection messages back into the prompt (`includeReflections=true`), be aware of token usage and the potential to leak chain-of-thought if not handled carefully in final user outputs.

---

### Safe Run Methods

When orchestrating multiple agents, you may want more robust error handling. For example:

- **Stop On Error**: Immediately stop if any agent fails.  
- **Continue On Error**: Log the error but proceed with subsequent agents.

**Example**  
```python
from agentix.agents import Agent, AgentTeam, AgentOptions
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

# Extend AgentTeam for the sake of having a custom class
class SafeAgentTeam(AgentTeam):
    # You can change the constructor or add more methods if you want
    pass

async def main():
  # 1) Create LLM(s)
  model1 = OpenAIChat(
    api_key="YOUR-API-KEY",
    model="gpt-4o-mini",
    temperature=0.7,
  )
  model2 = OpenAIChat(
    api_key="YOUR-API-KEY",
    model="gpt-4o-mini",
    temperature=0.7,
  )
  model3 = OpenAIChat(
    api_key="YOUR-API-KEY",
    model="gpt-4o-mini",
    temperature=0.7,
  )

  # 2) Create memory for each agent
  memA = ShortTermMemory(5)
  memB = ShortTermMemory(5)
  memC = ShortTermMemory(5)

  # 3) Create agents
  agentA = Agent.create(
    name="AgentA",
    model=model1,
    memory=memA,
    instructions=["Respond politely. (No error here)"],
    options=AgentOptions(maxSteps=1, useReflection=False)
  )

  # AgentB intentionally might throw an error or produce unexpected output
  agentB = Agent.create(
    name="AgentB",
    model=model2,
    memory=memB,
    instructions=["Pretend to attempt the user query but throw an error for demonstration."],
    options=AgentOptions(maxSteps=1, useReflection=False)
  )

  # Force an error for agentB to demonstrate safe run
  agentB.run = async (input: str) => {
    raise Exception("Intentional error from AgentB for demonstration!")
  }

  agentC = Agent.create(
    name="AgentC",
    model=model3,
    memory=memC,
    instructions=["Provide a short helpful answer. (No error)"],
    options=AgentOptions(maxSteps=1, useReflection=False)
  )

  # 4) Create our SafeAgentTeam (again, extends AgentTeam - see AgentTeam.ts)
  team = SafeAgentTeam("DemoTeam", [agentA, agentB, agentC])

  # 5) Define some hooks to see what happens behind the scenes
  hooks = {
    onAgentStart: (agentName, input) => {
      print(f"[START] {agentName} with input: \"{input}\"")
    },
    onAgentEnd: (agentName, output) => {
      print(f"[END] {agentName}: output => \"{output}\"")
    },
    onError: (agentName, error) => {
      print(f"[ERROR] in {agentName}: {error.message}")
    },
    onFinal: (outputs) => {
      print("Final outputs from the entire sequential run =>", outputs)
    },
  }

  # 6a) Demonstrate runSequentialSafe with stopOnError=true
  #         - With stopOnError=true, the loop breaks immediately after AgentB throws an error,
  #           so AgentC never runs.
  print("\n--- runSequentialSafe (stopOnError = true) ---")
  userPrompt = "Hello from the user!"
  resultsStopOnError = await team.runSequentialSafe(userPrompt, true, hooks)
  print("\nResults (stopOnError=true):", resultsStopOnError)

  # 6b) Demonstrate runSequentialSafe with stopOnError=false
  #         - With stopOnError=false, AgentB's error is logged, but AgentC still gets a chance to run,
  #           producing its output as the final step.
  print("\n--- runSequentialSafe (stopOnError = false) ---")
  userPrompt2 = "Another user query - let's see if we continue after errors."
  resultsContinue = await team.runSequentialSafe(userPrompt2, false, hooks)
  print("\nResults (stopOnError=false):", resultsContinue)

```

---

### Advanced Multi-Agent Synergy

Your **AgentTeam** and **AgentRouter** can be extended for more collaborative or specialized interactions:

1. **Shared Memory**: Give each agent the **same** memory instance so they see the entire conversation as it evolves.
2. **Interleaved/Chat-Like**: Round-robin the agents in a while loop until a convergence condition (like `"FINAL ANSWER"`) is met.
3. **Sub-Teams**: Combine `AgentRouter` (for domain routing) with an `AgentTeam` (for parallel or sequential synergy among a subset).

**Example**: Interleaved approach with a shared memory

```python
from agentix.agents import AdvancedAgentTeam

async def main():
  # Build 2 specialized agents
  # Enable shared memory so they see each other's messages
  advancedTeam = AdvancedAgentTeam("RoundRobinTeam", [agent1, agent2], sharedMem)
  advancedTeam.enableSharedMemory()

  # They "talk" to each other until "FINAL ANSWER" or max 10 rounds
  def checkConverged(msg: str) -> bool:
    return "FINAL ANSWER" in msg


  final = await advancedTeam.runInterleaved("Collaborate on a solution, finalize with 'FINAL ANSWER:'", 10, checkConverged)
  print("Final synergy output =>", final)

```

---

### Aggregator & Consensus

You might want a final “aggregator” agent that merges the outputs of multiple sub-agents into a single consensus answer.

```python
class AggregatorAgentTeam(AgentTeam):
  aggregator: Agent

  def __init__(self, name: str, agents: List[Agent], aggregator: Agent):
    super().__init__(name, agents)
    self.aggregator = aggregator

  # For instance, gather parallel results, pass them to aggregator
  async def runWithAggregator(self, query: str) -> str:
    results = await self.runInParallel(query)
    combined = "\n---\n".join(results)
    return self.aggregator.run(f"Sub-agent answers:\n{combined}\nPlease unify them:")
```

---

## Additional Recommendations & Thoughts

- **Agent Performance & Prompting**: Agentic systems are all about the prompts. They will only work as well as the prompts you provide. Ensure they are clear, concise, and tailored to the task at hand for every use case. There are many guides on prompting LLMs effectively, and I would advise reading them.
- **Security / Tools**: If you use “write actions” or potentially destructive tools, ensure you have human approval hooks or environment isolation (sandboxing).  
- **Chain-of-thought Safety**: If reflection memory is fed back into the final prompt or user response, carefully ensure it does not leak internal reasoning to the user if that is not desired.  
- **External Vector DB**: For production scale retrieval, integrate with an actual vector database instead of in-memory stores.  
- **Local LLM**: For on-prem or offline scenarios, adapt the code to use local inference with something like Transformers.js or custom endpoints.  

---

## Building & Running

1. **Install** dependencies:
```bash
pip install -r requirements.txt
```

2. **Run** a specific demo:
```bash
python src/examples/basic_agent.py
```

---

## FAQ

1. **Why multi-step reflection?**  
   Because tool usage, memory retrieval, or planning steps require the agent to see the result of each action before finalizing an answer.

2. **Can I swap SummarizingMemory for another approach?**  
   Absolutely. Any class implementing `Memory` works. You can also create a chunk-based or hierarchical summarizing approach.

3. **Is everything stored in memory ephemeral?**  
   By default, yes. For a persistent store, integrate an external vector DB or a database for your conversation logs.

4. **How do I see partial streaming tokens?**  
   Set `stream = true` in `OpenAIChat`, and provide an `onToken` callback to process partial output in real time.  

5. **Do I need to use an agent framework?**
   Absolutely not. Frameworks are just tools to assist in building more complex agents. You can use the LLMs directly with loops if you prefer.

6. **Do I have to use everything in the library?**  
   Nope. You can pick and choose the components you need. The library is designed to be modular and flexible. You can use the most basic agent implementation for basic agentic tasks, or you can use the more advanced features for more complex scenarios. You can even extend the library with your own custom components and features. The goal is to provide you with the ***options*** to build the agent you need for your desired use case. A lot of the advanced features are there to help you build more robust, more capable agents, but you don't have to use them if you don't need them.

---

## Roadmap

- **External Vector DB Integrations** (FAISS, Pinecone, Weaviate, etc.)  
- **Local LLMs** via Transformers.js and WebGPU-based inference if available  
- ~~**More LLM API integrations** (e.g., Together.ai, Anthropic, Google, etc.)~~  
- ~~**More External Tools**  (e.g., Firecrawl, SerpAPI, etc.)~~  
- **Browser Vision Tools** (image recognition, OCR, etc.)  
- **Multi-step self-correction** (auto re-try if evaluator score < threshold)  
- ~~**Improved Observability** (agent API metrics, logging, and tracing)~~

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

Feel free to submit PRs or open issues for new tools, memory ideas, or advanced agent patterns.
