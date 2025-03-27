"""
Agentix - A Python framework for building AI agents
"""

__version__ = "0.3.0"

from .agents.agent import Agent, AgentOptions
from .agents.prompt_builder import AgentPromptBuilder
from .llms.openai_chat import OpenAIChat
from .llms.openai_embeddings import OpenAIEmbeddings
from .llms.together_chat import TogetherChat
from .llms.types import LLM
from .memory.memory import Memory, ConversationMessage, MemoryRole
from .memory.short_term_memory import ShortTermMemory
from .memory.long_term_memory import LongTermMemory
from .memory.summarizing_memory import SummarizingMemory
from .memory.reflection_memory import ReflectionMemory
from .memory.composite_memory import CompositeMemory
from .agents.multi_agent import (
    AgentRouter,
    AgentTeam,
    TeamHooks,
    AgentRole,
    TeamConfiguration,
    AgentContribution, 
    TeamOptions,
    LLMConvergenceChecker
)
from .planner import Planner, SimpleLLMPlanner
from .workflow import Workflow, WorkflowStep, LLMCallStep
from .evaluators.simple_evaluator import SimpleEvaluator, EvaluationResult

__all__ = [
    "Agent",
    "AgentOptions",
    "AgentPromptBuilder",
    "OpenAIChat",
    "OpenAIEmbeddings",
    "TogetherChat",
    "LLM",
    "Memory",
    "ConversationMessage",
    "MemoryRole",
    "ShortTermMemory",
    "LongTermMemory",
    "SummarizingMemory",
    "ReflectionMemory",
    "CompositeMemory",
    "AgentRouter",
    "AgentTeam",
    "TeamHooks",
    "AgentRole",
    "TeamConfiguration",
    "AgentContribution",
    "TeamOptions",
    "LLMConvergenceChecker",
    "Planner",
    "SimpleLLMPlanner",
    "Workflow",
    "WorkflowStep",
    "LLMCallStep",
    "SimpleEvaluator",
    "EvaluationResult"
] 