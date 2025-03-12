
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

**Goal**: Show how to use an agent team to summarize content.

```python
#!/usr/bin/env python
"""
agent_team_summarizer_example.py

This example demonstrates the use of AgentTeam with a summarizer pattern, where:
1. Three specialized domain agents work on a problem in parallel
2. Their outputs are collected and sent to a summarizer agent
3. The summarizer agent creates a cohesive final response

This pattern is useful for complex queries that benefit from multiple perspectives,
ensuring the user gets a comprehensive, well-structured answer that incorporates
insights from all domain experts.
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import AgentTeam, TeamHooks
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Main function demonstrating an agent team with summarizer pattern.
    
    Args:
        debug: Whether to enable debug logging
    """
    # Create shared model for all agents
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create three specialized domain agents
    finance_agent = Agent.create(
        name="FinanceExpert",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a financial expert specialized in investments, economics, and business strategy.",
            "Provide detailed financial analysis and perspective on topics.",
            "Focus on economic implications, market trends, and financial considerations."
        ],
        options=AgentOptions(use_reflection=False, max_steps=1)
    )
    
    tech_agent = Agent.create(
        name="TechnologyExpert",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a technology expert with deep knowledge of computers, AI, and digital innovation.",
            "Analyze topics from a technological perspective.",
            "Consider current technology trends, technical feasibility, and future tech developments."
        ],
        options=AgentOptions(use_reflection=False, max_steps=1)
    )
    
    society_agent = Agent.create(
        name="SocialImpactExpert",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are an expert in sociology and social impact analysis.",
            "Focus on how topics affect different populations, social structures, and human behavior.",
            "Consider ethical implications, social trends, and cultural factors."
        ],
        options=AgentOptions(use_reflection=False, max_steps=1)
    )
    
    # Create a summarizer agent with special instructions for synthesis
    summarizer_agent = Agent.create(
        name="SynthesisExpert",
        model=model,
        memory=ShortTermMemory(max_messages=10),
        instructions=[
            "You are a synthesis expert who combines multiple perspectives into cohesive summaries.",
            "When presented with different expert opinions on a topic, your job is to:",
            "1. Identify key insights from each expert's contribution",
            "2. Find areas of agreement and disagreement",
            "3. Synthesize a comprehensive, balanced viewpoint that incorporates all perspectives",
            "4. Structure your response clearly with sections for each major point",
            "5. Conclude with the most important takeaways",
            "Note when experts disagree and explain the different viewpoints fairly."
        ],
        options=AgentOptions(use_reflection=True, max_steps=2)
    )
    
    # Create an agent team with the domain experts
    team = AgentTeam("DomainExpertTeam", [finance_agent, tech_agent, society_agent])
    
    # Create hooks for monitoring team execution
    hooks = TeamHooks(
        on_agent_start=lambda agent_name, query: print(f"\n🚀 Starting {agent_name} with query: '{query}'"),
        on_agent_end=lambda agent_name, result: print(f"✅ {agent_name} completed"),
        on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
        on_final=lambda results: print(f"🏁 All agents completed their analysis")
    )
    
    async def analyze_with_team_and_summarize(query: str) -> str:
        """
        Run the team of experts in parallel and then summarize their outputs.
        
        Args:
            query: The user query to analyze
            
        Returns:
            A synthesized response combining all expert perspectives
        """
        print(f"\n📝 Team analyzing: '{query}'")
        
        # Step 1: Run all domain experts in parallel
        expert_results = await team.run_in_parallel(query, hooks)
        
        # Step 2: Format results with expert names for the summarizer
        formatted_expert_input = ""
        for i, result in enumerate(expert_results):
            agent = team.agents[i]
            formatted_expert_input += f"--- {agent.name} Analysis ---\n{result}\n\n"
        
        # Step 3: Create a summarization request for the synthesis agent
        summarization_request = f"""
        The following are expert analyses on this topic: "{query}"
        
        {formatted_expert_input}
        
        Please synthesize these perspectives into a comprehensive response that 
        incorporates insights from all experts. Identify key points of agreement 
        and disagreement, and structure your response clearly.
        """
        
        print("\n🔄 Generating synthesis of expert perspectives...")
        
        # Step 4: Have the summarizer agent create the final synthesis
        final_summary = await summarizer_agent.run(summarization_request)
        
        return final_summary
    
    # Example queries that benefit from multiple perspectives
    example_queries = [
        "What are the implications of artificial intelligence on the job market?",
        #"How might blockchain technology change the financial industry?",
        #"What are the potential impacts of remote work becoming permanent for many companies?"
    ]
    
    for query in example_queries:
        print("\n" + "=" * 80)
        print(f"USER QUERY: {query}")
        print("=" * 80)
        
        final_answer = await analyze_with_team_and_summarize(query)
        
        print("\n" + "=" * 40 + " FINAL SYNTHESIZED RESPONSE " + "=" * 40)
        print(final_answer)
        print("=" * 100)
        
        # Add a pause between queries for easier reading of output
        if query != example_queries[-1]:
            print("\nMoving to next query in 3 seconds...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(main()) 
```

---

### 12) **Production Agentic Workflow Example**

**Goal**: Show how to use an agent team to summarize content.

```python
#!/usr/bin/env python
"""
startup_research_workflow.py

An advanced production-grade workflow that demonstrates how to build a complex
multi-agent system for startup research and analysis. This example integrates:

1. News Collection Agent: Gathers recent startup news using DuckDuckGo
2. Summarization Agent: Creates concise summaries of the gathered information
3. Trend Analysis Agent: Identifies patterns in startup funding and technology
4. Validation and quality control using task validators
5. Multi-agent coordination via advanced agent team and router

The workflow demonstrates these Agentix features:
- Advanced Agent Team for collaboration
- LLM Convergence Checking for quality assurance
- Task Validation for output verification
- Reflection Memory for agent chain-of-thought
- Composite Memory with Long Term and Summarizing Memory
- Advanced Agent Router for dynamic task assignment
- Planner for workflow orchestration
- Hooks for monitoring and debugging
- Evaluator for output quality assessment

This production-grade workflow handles real-world startup research tasks
and can be extended for additional research domains.
"""
import os
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Agentix Core Components
from agentix.agents import Agent, AgentOptions, AgentHooks
from agentix.planner import SimpleLLMPlanner
from agentix.evaluators import SimpleEvaluator, EvaluationResult

# Agentix Multi-agent Components
from agentix.agents.multi_agent import (
    AdvancedAgentTeam,
    AdvancedTeamOptions,
    AdvancedTeamHooks,
    AgentRole,
    TeamConfiguration,
    LLMConvergenceChecker,
    AdvancedAgentRouter,
    RouterOptions,
    AgentCapability
)

from agentix.agents.multi_agent.llm_convergence_checker import ConvergenceCriteria

# Agentix Memory Components
from agentix.memory import (
    ShortTermMemory,
    LongTermMemory,
    ReflectionMemory,
    SummarizingMemory,
    CompositeMemory
)

# Agentix LLM and Tools
from agentix.llms import OpenAIChat, OpenAIEmbeddings
from agentix.tools import (
    DuckDuckGoToolkit,
    DuckDuckGoNewsSearchTool,
    DuckDuckGoTextSearchTool,
)

from dotenv import load_dotenv
load_dotenv()


@dataclass
class WorkflowMetrics:
    """Tracks performance metrics for the workflow."""
    query_count: int = 0
    agent_calls: int = 0
    search_queries: int = 0
    validation_runs: int = 0
    total_processing_time: float = 0.0
    convergence_attempts: int = 0
    successful_convergence: int = 0
    evaluation_scores: List[float] = None

    def __post_init__(self):
        if self.evaluation_scores is None:
            self.evaluation_scores = []

    def add_evaluation_score(self, score: float):
        self.evaluation_scores.append(score)
        
    def get_average_evaluation_score(self) -> float:
        return sum(self.evaluation_scores) / len(self.evaluation_scores) if self.evaluation_scores else 0.0
    
    def report(self) -> str:
        """Generate a summary report of metrics."""
        return f"""
Workflow Performance Metrics:
----------------------------
Queries Processed: {self.query_count}
Agent Calls: {self.agent_calls}
Search Queries: {self.search_queries}
Validation Runs: {self.validation_runs}
Total Processing Time: {self.total_processing_time:.2f}s
Convergence Rate: {self.successful_convergence}/{self.convergence_attempts} ({(self.successful_convergence/self.convergence_attempts)*100 if self.convergence_attempts else 0:.1f}%)
Average Evaluation Score: {self.get_average_evaluation_score():.2f}/1.0
"""


class StartupResearchWorkflow:
    """
    A comprehensive workflow for startup research and analysis.
    Demonstrates a production-grade setup using multiple integrated agents.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the startup research workflow with all required components.
        
        Args:
            debug: Whether to enable debug logging
        """
        self.debug = debug
        self.metrics = WorkflowMetrics()
        
        # Initialize LLM models with different temperature settings for specialized tasks
        self.main_model = OpenAIChat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.7  # Higher temperature for creative analysis
        )
        
        self.validator_model = OpenAIChat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.1  # Lower temperature for validation
        )
        
        self.router_model = OpenAIChat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.2  # Balanced temperature for routing decisions
        )
        
        # Initialize embeddings for long-term memory
        self.embeddings = OpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        
        # Initialize search tools
        self.search_tools = self._create_search_tools()
        
        # Initialize memories
        self.memories = self._create_memories()
        
        # Initialize specialized agents
        self.agents = self._create_agents()
        
        # Initialize router
        self.router = self._create_router()
        
        # Initialize team
        self.team = self._create_team()
        
        # Initialize planner
        self.planner = SimpleLLMPlanner(self.main_model)
        
        # Initialize evaluator
        self.evaluator = SimpleEvaluator(self.validator_model)
        
        # Initialize convergence checker
        self.convergence_checker = self._create_convergence_checker()
        
        print("✅ Startup Research Workflow initialized successfully")
    
    def _create_search_tools(self) -> Dict[str, Any]:
        """Initialize search tools for the agents."""
        print("🔧 Initializing search tools...")
        
        try:
            # Create DuckDuckGo toolkit with specific tools enabled
            ddg_toolkit = DuckDuckGoToolkit(
                enable_text_search=True,
                enable_news_search=True,
                enable_image_search=False,
                enable_video_search=False,
                enable_chat=False,
                timeout=20  # Increased timeout for more reliable results
            )
            
            # Extract individual tools from the toolkit
            news_search = next((t for t in ddg_toolkit.get_tools() 
                              if isinstance(t, DuckDuckGoNewsSearchTool)), None)
            
            text_search = next((t for t in ddg_toolkit.get_tools() 
                              if isinstance(t, DuckDuckGoTextSearchTool)), None)
            
            # Check if we have the tools we need
            if not news_search or not text_search:
                print("⚠️ Warning: Could not find required DuckDuckGo search tools. Using empty toolkit.")
                return {
                    "toolkit": None,
                    "news_search": None,
                    "text_search": None,
                    "all_tools": []
                }
            
            # Print success message
            print("✅ DuckDuckGo search tools initialized successfully")
            
            return {
                "toolkit": ddg_toolkit,
                "news_search": news_search,
                "text_search": text_search,
                "all_tools": ddg_toolkit.get_tools()
            }
        except Exception as e:
            print(f"⚠️ Error initializing DuckDuckGo tools: {e}")
            print("⚠️ Continuing without search tools. Please ensure duckduckgo-search is installed.")
            print("⚠️ Run: pip install duckduckgo-search")
            return {
                "toolkit": None,
                "news_search": None,
                "text_search": None,
                "all_tools": []
            }
    
    def _create_memories(self) -> Dict[str, Any]:
        """Initialize memory systems for the agents."""
        print("🧠 Initializing memory systems...")
        
        # Create short-term memories for each agent
        news_mem = ShortTermMemory(max_messages=10)
        summary_mem = ShortTermMemory(max_messages=15)
        trends_mem = ShortTermMemory(max_messages=15)
        
        # Create reflection memories for each agent
        news_reflection = ReflectionMemory(include_reflections=False)
        summary_reflection = ReflectionMemory(include_reflections=False)
        trends_reflection = ReflectionMemory(include_reflections=False)
        
        # Create long-term memory for persistent storage
        long_term = LongTermMemory(self.embeddings, max_messages=1000, top_k=5)
        
        # Create summarizing memory for past research
        summarizing = SummarizingMemory(
            summarizer_model=self.validator_model,
            threshold=15,
            max_summary_tokens=200,
            hierarchical=True
        )
        
        # Create composite memories combining different memory types
        news_composite = CompositeMemory(news_mem, news_reflection, long_term)
        summary_composite = CompositeMemory(summary_mem, summary_reflection, summarizing)
        trends_composite = CompositeMemory(trends_mem, trends_reflection, long_term, summarizing)
        
        # Create shared memory for team collaboration
        shared_memory = CompositeMemory(ShortTermMemory(max_messages=30), long_term)
        
        return {
            "news": news_composite,
            "summary": summary_composite,
            "trends": trends_composite,
            "shared": shared_memory,
            "long_term": long_term,
            "summary_memory": summarizing,
            "reflections": {
                "news": news_reflection,
                "summary": summary_reflection,
                "trends": trends_reflection
            }
        }
    
    async def _track_search_query_async(self, name: str, query: str) -> bool:
        """Asynchronous version of search query tracking that returns awaitable bool."""
        self.metrics.search_queries += 1
        return True
    
    async def _log_reflection_async(self, reflection_memory: ReflectionMemory, content: str) -> bool:
        """
        Asynchronous version of reflection logging that returns awaitable bool.
        
        Args:
            reflection_memory: The reflection memory to log to
            content: The reflection content
            
        Returns:
            True to indicate successful logging
        """
        await reflection_memory.add_message({
            "role": "reflection",
            "content": content,
            "metadata": {"timestamp": int(time.time() * 1000)}
        })
        return True
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Initialize specialized agents for the workflow."""
        print("🤖 Creating specialized agents...")
        
        # Check if we have search tools available
        has_tools = len(self.search_tools["all_tools"]) > 0
        if not has_tools:
            print("⚠️ No search tools available. Agents will continue with reasoning only.")
        
        # 1. Create the News Collection Agent
        news_agent = Agent.create(
            name="NewsCollector",
            model=self.main_model,
            memory=self.memories["news"],
            tools=self.search_tools["all_tools"] if has_tools else [],
            instructions=[
                "You are a startup news research specialist focused on gathering accurate information.",
                "Use DuckDuckGo tools to find recent news, funding announcements, and market analysis." if has_tools else "Analyze startup trends and news based on your knowledge.",
                "Focus on factual information like funding amounts, investors, technologies, and market data.",
                "Include timestamps and sources for each piece of information.",
                "Organize findings into clearly structured categories.",
                "Always verify information across multiple sources when possible.",
            ],
            validation_model=self.validator_model,
            task="Gather comprehensive, verified news about startups in specified sectors",
            options=AgentOptions(
                validate_output=True,
                use_reflection=True,
                max_steps=10,
                debug=self.debug
            ),
            hooks=AgentHooks(
                on_tool_call=self._track_search_query_async if has_tools else None,
                on_final_answer=lambda ans: self._log_reflection_async(
                    self.memories["reflections"]["news"],
                    "Collected startup news data. Need to verify: sources cited? recent enough? comprehensive?"
                )
            )
        )
        
        # 2. Create the Summary Generation Agent
        summary_agent = Agent.create(
            name="Summarizer",
            model=self.main_model,
            memory=self.memories["summary"],
            instructions=[
                "You are a startup data summarization specialist.",
                "Create concise, factual summaries of startup information.",
                "Highlight key metrics like funding amounts, growth rates, team size, and technologies.",
                "Structure summaries with clear sections for financials, technology, team, and market.",
                "Remove redundant information and focus on verifiable facts.",
                "Include confidence ratings for information based on source reliability.",
            ],
            validation_model=self.validator_model,
            task="Create accurate, concise summaries of startup information",
            options=AgentOptions(
                validate_output=True,
                use_reflection=True,
                max_steps=5,
                debug=self.debug
            ),
            hooks=AgentHooks(
                on_final_answer=lambda ans: self._log_reflection_async(
                    self.memories["reflections"]["summary"],
                    "Created summary of startup data. Focus on clarity, accuracy, and insight density."
                )
            )
        )
        
        # 3. Create the Trend Analysis Agent
        trends_agent = Agent.create(
            name="TrendAnalyst",
            model=self.main_model,
            memory=self.memories["trends"],
            tools=[self.search_tools["text_search"]] if has_tools and self.search_tools["text_search"] else [],
            instructions=[
                "You are a startup trend analysis expert.",
                "Identify emerging patterns in startup funding, technology adoption, and market opportunities.",
                "Look for connections across different startups and sectors.",
                "Analyze trends quantitatively when possible (growth rates, funding increases, etc.).",
                "Compare trends against established market contexts.",
                "Distinguish between temporary hype and substantive movements.",
                "Provide actionable insights based on identified trends.",
            ],
            validation_model=self.validator_model,
            task="Identify meaningful patterns and trends in startup data",
            options=AgentOptions(
                validate_output=True,
                use_reflection=True,
                max_steps=7,
                debug=self.debug
            ),
            hooks=AgentHooks(
                on_tool_call=self._track_search_query_async if has_tools else None,
                on_final_answer=lambda ans: self._log_reflection_async(
                    self.memories["reflections"]["trends"],
                    "Analyzed startup trends. Consider: economic context, technological limitations, market adoption curves."
                )
            )
        )
        
        return {
            "news": news_agent,
            "summary": summary_agent, 
            "trends": trends_agent
        }
    
    def _create_router(self) -> AdvancedAgentRouter:
        """Initialize the advanced agent router."""
        print("🔀 Setting up agent router...")
        
        # Define agent capabilities for the router
        capabilities = {
            0: AgentCapability(
                name="Startup News Collection",
                description="Gathers recent news, funding rounds, and market data about startups",
                keywords=["news", "latest", "funding", "investment", "announced", "launched", "raised"],
                examples=[
                    "Find recent news about AI startups in healthcare",
                    "What are the latest funding rounds for fintech startups?",
                    "Research recent developments in clean energy startups"
                ]
            ),
            1: AgentCapability(
                name="Startup Information Summarization",
                description="Creates concise summaries of startup information and market data",
                keywords=["summarize", "summary", "digest", "brief", "overview", "condensed"],
                examples=[
                    "Summarize the funding landscape for AI startups",
                    "Give me a brief overview of the top climate tech startups",
                    "Provide a condensed summary of recent biotech startup innovations"
                ]
            ),
            2: AgentCapability(
                name="Startup Trend Analysis",
                description="Identifies patterns in startup funding, technology, and market opportunities",
                keywords=["trends", "patterns", "analysis", "emerging", "growing", "opportunities", "future"],
                examples=[
                    "What are the emerging trends in AI startup funding?",
                    "Analyze patterns in startup growth across fintech",
                    "Identify future opportunities in the clean energy startup space"
                ]
            )
        }
        
        # Create router options
        router_options = RouterOptions(
            use_llm=True,
            debug=self.debug,
            fallback_index=2,  # Trend analysis as fallback (most general)
            confidence_threshold=0.65,
            router_llm=self.router_model
        )
        
        # Create the router with the agent list
        return AdvancedAgentRouter(
            agents=[self.agents["news"], self.agents["summary"], self.agents["trends"]],
            capabilities=capabilities,
            options=router_options
        )
    
    def _create_team(self) -> AdvancedAgentTeam:
        """Initialize the advanced agent team for data gathering."""
        print("👥 Building agent team...")
        
        # Define specialized query transformations for each role
        def news_collector_transform(query: str) -> str:
            return f"As a startup news specialist, gather comprehensive information about: {query}\n\nFocus on recent funding announcements, technology developments, market entries, and verified facts. Provide your findings in a clear, structured format."
        
        def trend_analyst_transform(query: str) -> str:
            return f"As a startup trend analyst, identify emerging patterns related to: {query}\n\nFocus on funding trends, technology adoption curves, market opportunities, and future projections. Structure your analysis clearly with sections for different trend categories."
        
        # Create team configuration with specialized roles for data gathering
        team_config = TeamConfiguration(
            roles={
                "NewsCollector": AgentRole(
                    name="News Collector",
                    description="Gathers comprehensive startup news and research",
                    query_transform=news_collector_transform
                ),
                "TrendAnalyst": AgentRole(
                    name="Trend Analyst",
                    description="Identifies patterns and emerging trends in startup data",
                    query_transform=trend_analyst_transform
                )
            },
            default_role=None  # No default role; each agent must have a specific role
        )
        
        # Helper function for on_agent_end to avoid issues with boolean logic in lambda
        async def agent_end_handler(agent_name: str, result: str) -> None:
            print(f"✅ {agent_name} completed research")
            self._increment_agent_calls()
        
        # Define team hooks for monitoring and debugging
        team_hooks = AdvancedTeamHooks(
            on_agent_start=lambda agent_name, query: print(f"\n🚀 {agent_name} starting research..."),
            on_agent_end=agent_end_handler,
            on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
            on_final=lambda results: print(f"🏁 Data gathering completed with {len(results)} contributions"),
            on_round_start=lambda round_num, max_rounds: print(f"\n📊 Starting research round {round_num}/{max_rounds}"),
            on_round_end=lambda round_num, contributions: print(f"📝 Round {round_num} complete with {len(contributions)} contributions")
        )
        
        # Configure the team options
        team_options = AdvancedTeamOptions(
            shared_memory=self.memories["shared"],
            team_config=team_config,
            hooks=team_hooks,
            debug=self.debug
        )
        
        # Create and configure the team with just the data gathering agents
        team = AdvancedAgentTeam(
            name="StartupResearchTeam",
            agents=[self.agents["news"], self.agents["trends"]],  # Only include data gathering agents
            options=team_options
        )
        
        # Enable shared memory
        team.enable_shared_memory()
        
        return team
    
    def _create_convergence_checker(self) -> LLMConvergenceChecker:
        """Initialize the LLM convergence checker for validating individual agent outputs."""
        print("👁️ Setting up convergence checker...")
        
        # Define convergence criteria for individual agent outputs
        criteria = ConvergenceCriteria(
            required_elements=[
                "factual information",
                "data sources",
                "startup details"
            ],
            required_structure=[
                "clear sections",
                "organized data"
            ],
            minimum_length=300,  # Minimum length in words for a complete response
            custom_instructions=[
                "Ensure the response contains specific, verifiable information",
                "Check that all claims are supported by sources",
                "Verify the response is properly structured and organized"
            ]
        )
        
        # Create the convergence checker
        checker = LLMConvergenceChecker(
            model=self.validator_model,
            criteria=criteria,
            debug=self.debug
        )
        
        return checker
    
    async def process_single_query(self, query: str) -> Dict[str, Any]:
        """
        Process a single startup research query using the appropriate agent.
        
        Args:
            query: User's research query
            
        Returns:
            Dictionary with results and performance metrics
        """
        self.metrics.query_count += 1
        print(f"\n🔍 Processing query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Use the router to select the most appropriate agent
            result = await self.router.run(query)
            
            # Validate the output
            self.metrics.validation_runs += 1
            evaluation = await self.evaluate_result(result)
            
            # Store in long-term memory
            await self.memories["long_term"].add_message({
                "role": "assistant",
                "content": result,
                "metadata": {
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "evaluation_score": evaluation.score
                }
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            return {
                "result": result,
                "evaluation": evaluation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            print(f"❌ Error processing query: {str(e)}")
            return {
                "result": f"Error: {str(e)}",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def process_complex_query(self, query: str, max_rounds: int = 3) -> Dict[str, Any]:
        """
        Process a complex startup research query using a two-phase approach:
        1. Parallel data gathering from news and trend agents
        2. Summarization of gathered data
        
        Args:
            query: User's research query
            max_rounds: Maximum number of rounds (used for convergence in data gathering)
            
        Returns:
            Dictionary with results and performance metrics
        """
        self.metrics.query_count += 1
        print(f"\n🔍 Processing complex query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Phase 1: Parallel Data Gathering
            print("\n📊 Phase 1: Gathering Data (News and Trends)")
            
            # Create specialized query transformations for data gathering
            def news_collector_transform(query: str) -> str:
                return f"As a startup news specialist, gather comprehensive information about: {query}\n\nFocus on recent funding announcements, technology developments, market entries, and verified facts. Provide your findings in a clear, structured format."
            
            def trend_analyst_transform(query: str) -> str:
                return f"As a startup trend analyst, identify emerging patterns related to: {query}\n\nFocus on funding trends, technology adoption curves, market opportunities, and future projections. Structure your analysis clearly with sections for different trend categories."
            
            # Create team configuration for data gathering
            team_config = TeamConfiguration(
                roles={
                    "NewsCollector": AgentRole(
                        name="News Collector",
                        description="Gathers comprehensive startup news and research",
                        query_transform=news_collector_transform
                    ),
                    "TrendAnalyst": AgentRole(
                        name="Trend Analyst",
                        description="Identifies patterns and emerging trends in startup data",
                        query_transform=trend_analyst_transform
                    )
                },
                default_role=None
            )
            
            # Define team hooks for monitoring
            team_hooks = AdvancedTeamHooks(
                on_agent_start=lambda agent_name, query: print(f"\n🚀 {agent_name} starting research..."),
                on_agent_end=lambda agent_name, result: print(f"✅ {agent_name} completed research"),
                on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
                on_final=lambda results: print(f"🏁 Data gathering completed with {len(results)} contributions")
            )
            
            # Create a data gathering team with just news and trend agents
            data_team = AdvancedAgentTeam(
                name="DataGatheringTeam",
                agents=[self.agents["news"], self.agents["trends"]],
                options=AdvancedTeamOptions(
                    shared_memory=self.memories["shared"],
                    team_config=team_config,
                    hooks=team_hooks,
                    debug=self.debug
                )
            )
            
            # Enable shared memory for the data team
            data_team.enable_shared_memory()
            
            # Run news and trend analysis in parallel
            gathering_results = await data_team.run_in_parallel(query)
            news_data = gathering_results[0]
            trend_data = gathering_results[1]
            
            print("\n✅ Data gathering complete")
            print(f"News data length: {len(news_data)} chars")
            print(f"Trend data length: {len(trend_data)} chars")
            
            # Phase 2: Summarization
            print("\n📝 Phase 2: Synthesizing Results")
            
            # Prepare a structured input for the summarizer
            summary_prompt = f"""
Please synthesize and analyze the following startup research data.

GATHERED NEWS:
{news_data}

TREND ANALYSIS:
{trend_data}

Create a comprehensive analysis that:
1. Synthesizes key findings from both news and trend research
2. Identifies the most significant market developments
3. Highlights emerging patterns and opportunities
4. Provides actionable insights and recommendations
5. Notes any potential risks or challenges

Structure your response with clear sections for:
- Executive Summary
- Key Market Developments
- Emerging Trends
- Opportunities and Risks
- Actionable Recommendations
"""
            
            # Get the final synthesis from the summarizer
            final_research = await self.agents["summary"].run(summary_prompt)
            
            print("\n✅ Synthesis complete")
            
            # Validate the output
            self.metrics.validation_runs += 1
            evaluation = await self.evaluate_result(final_research)
            
            # Store in long-term memory
            await self.memories["long_term"].add_message({
                "role": "assistant",
                "content": final_research,
                "metadata": {
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "evaluation_score": evaluation.score,
                    "team_research": True,
                    "raw_data": {
                        "news": news_data,
                        "trends": trend_data
                    }
                }
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            return {
                "result": final_research,
                "evaluation": evaluation,
                "processing_time": processing_time,
                "raw_data": {
                    "news": news_data,
                    "trends": trend_data
                }
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            print(f"❌ Error processing complex query: {str(e)}")
            return {
                "result": f"Error: {str(e)}",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def generate_research_plan(self, query: str) -> str:
        """
        Generate a research plan using the SimpleLLMPlanner.
        
        Args:
            query: User's research query
            
        Returns:
            JSON string with research plan steps
        """
        print(f"\n📋 Generating research plan for: '{query}'")
        
        plan = await self.planner.generate_plan(
            query, 
            self.search_tools["all_tools"], 
            self.memories["shared"]
        )
        
        print(f"Plan generated: {len(plan)} characters")
        return plan
    
    async def evaluate_result(self, result: str) -> EvaluationResult:
        """
        Evaluate the quality of a research result.
        
        Args:
            result: The research result to evaluate
            
        Returns:
            Evaluation result with score, feedback, and improvements
        """
        print("\n⭐ Evaluating research quality...")
        
        # Create a message list for the evaluator
        messages = [
            {"role": "system", "content": "Evaluating startup research output quality."},
            {"role": "user", "content": "Provide comprehensive startup research."},
            {"role": "assistant", "content": f"FINAL ANSWER: {result}"}
        ]
        
        # Evaluate the result
        evaluation = await self.evaluator.evaluate(messages)
        
        # Track the evaluation score
        self.metrics.add_evaluation_score(evaluation.score)
        
        print(f"Evaluation complete: Score {evaluation.score:.2f}/1.0")
        return evaluation
    
    def _increment_agent_calls(self) -> None:
        """Increment agent calls counter."""
        self.metrics.agent_calls += 1
    
    def get_metrics_report(self) -> str:
        """Generate a report of performance metrics."""
        return self.metrics.report()


async def main():
    """
    Main function to demonstrate the startup research workflow.
    """
    print("\n" + "=" * 80)
    print("🚀 STARTUP RESEARCH AND ANALYSIS WORKFLOW DEMO")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = StartupResearchWorkflow(debug=True)
    
    # Define example queries for different complexity levels
    simple_queries = [
        "What are the latest funding rounds for AI startups in healthcare?",
        #"Summarize recent developments in climate tech startups"
    ]
    
    complex_queries = [
        "Analyze emerging trends in AI startups focusing on enterprise applications, including recent funding patterns, technology adoption trends, and potential market opportunities",
        #"Research the intersection of blockchain and sustainability startups, including funding rounds, technological innovations, and market challenges"
    ]
    
    # Process simple queries using the router to select individual agents
    # print("\n" + "=" * 80)
    # print("📱 PROCESSING SINGLE-AGENT QUERIES")
    # print("=" * 80)
    
    # for query in simple_queries:
    #     result = await workflow.process_single_query(query)
        
    #     print("\n" + "-" * 60)
    #     print(f"Query: {query}")
    #     print(f"Processing time: {result['processing_time']:.2f} seconds")
        
    #     if "evaluation" in result:
    #         print(f"Evaluation score: {result['evaluation'].score:.2f}/1.0")
    #         print(f"Feedback: {result['evaluation'].feedback}")
        
    #     print("\nResult:")
    #     print("-" * 40)
    #     print(result["result"])
    #     print("-" * 60)
    
    # Generate a research plan for a complex query
    plan_query = complex_queries[0]
    research_plan = await workflow.generate_research_plan(plan_query)
    
    print("\n" + "=" * 80)
    print("📋 RESEARCH PLAN")
    print("=" * 80)
    print(f"Query: {plan_query}")
    print("\nPlan:")
    print("-" * 40)
    print(research_plan)
    print("-" * 40)
    
    # Process complex queries using the agent team
    print("\n" + "=" * 80)
    print("👥 PROCESSING MULTI-AGENT TEAM QUERIES")
    print("=" * 80)
    
    for query in complex_queries:
        result = await workflow.process_complex_query(query)
        
        print("\n" + "-" * 60)
        print(f"Query: {query}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        if "evaluation" in result:
            print(f"Evaluation score: {result['evaluation'].score:.2f}/1.0")
            print(f"Feedback: {result['evaluation'].feedback}")
        
        print("\nResult:")
        print("-" * 40)
        print(result["result"])
        print("-" * 60)
    
    # Display metrics report
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE METRICS")
    print("=" * 80)
    print(workflow.get_metrics_report())
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main()) 
```

---

### 13) **Function-based Tools Example**

**Goal**: Show how to use function-based tools.

```python
"""
Example demonstrating how to use function-based tools in Agentix.

This example shows:
1. Converting sync and async functions to tools
2. Automatic parameter extraction from type hints and docstrings
3. Using function tools with an agent
"""

import os
import asyncio
from typing import Dict, List, Optional

from agentix.agents import Agent
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory
from agentix.tools import function_tool

# Example 1: Simple synchronous function tool
@function_tool(
    name="GetWeather",
    description="Get current weather for a city",
    usage_example='TOOL REQUEST: GetWeather {"city": "San Francisco"}'
)
def get_weather(city: str) -> str:
    """
    Get the current weather for the specified city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the current weather
    """
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny"

# Example 2: Async function with multiple parameters
@function_tool(
    name="SearchDatabase",
    description="Search a database for records"
)
async def search_database(
    query: str,
    limit: Optional[int] = 5,
    category: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Search the database for matching records.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        category: Optional category to filter by
    
    Returns:
        List of matching records
    """
    # Simulate async database query
    await asyncio.sleep(0.5)
    
    # Mock results
    results = [
        {"id": "1", "title": f"Result {i} for {query}", "category": category or "general"}
        for i in range(limit)
    ]
    
    return results

# Example 3: Function with complex return type
@function_tool(
    name="ProcessData",
    description="Process data and return statistics"
)
def process_data(data: List[float], include_advanced: bool = False) -> Dict[str, float]:
    """
    Calculate statistics for a list of numbers.
    
    Args:
        data: List of numbers to process
        include_advanced: Whether to include advanced statistics
        
    Returns:
        Dictionary of calculated statistics
    """
    import statistics
    
    stats = {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "std_dev": statistics.stdev(data) if len(data) > 1 else 0
    }
    
    if include_advanced:
        stats.update({
            "variance": statistics.variance(data) if len(data) > 1 else 0,
            "mode": statistics.mode(data) if data else None
        })
    
    return stats

async def main():
    print("🔧 Function Tools Demo")
    print("=" * 50)
    
    # Initialize the agent
    chat_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.8
    )
    
    memory = ShortTermMemory(max_messages=10)
    
    # Create agent with our function tools
    agent = Agent.create(
        name="FunctionToolAgent",
        model=chat_model,
        memory=memory,
        tools=[
            get_weather,  # Note: decorator returns a Tool instance
            search_database,
            process_data
        ],
        instructions=[
            "You are a helpful assistant with access to various function-based tools.",
            "Use the tools to help answer user queries.",
            "Always analyze tool results before making additional tool calls."
        ]
    )
    
    # Example 1: Weather query
    print("\n🌤️ Weather Query Example")
    print("-" * 30)
    response = await agent.run("What's the weather like in Tokyo?")
    print(f"Response: {response}")
    
    # Example 2: Database search
    print("\n🔍 Database Search Example")
    print("-" * 30)
    response = await agent.run(
        "Search the database for 'machine learning' articles, limit to 3 results in the 'tech' category"
    )
    print(f"Response: {response}")
    
    # Example 3: Data processing
    print("\n📊 Data Processing Example")
    print("-" * 30)
    response = await agent.run(
        "Process this data with advanced statistics: [1.5, 2.5, 3.5, 4.5, 5.5]"
    )
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main()) 
```

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
