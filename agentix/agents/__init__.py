"""
Agentix Agent Module

This module provides the core Agent class and various specialized agents.
"""

from .agent import Agent, AgentOptions, AgentHooks
from .prompt_builder import AgentPromptBuilder
from .multi_agent import (
    AgentRouter,
    AgentTeam,
    TeamHooks,
    AgentRole,
    TeamConfiguration,
    AgentContribution,
    TeamOptions,
    LLMConvergenceChecker
)
from .autonomous_agent import (
    AutonomousAgent, 
    AutoAgentOptions, 
    AutoAgentHooks
)

__all__ = [
    "Agent",
    "AgentRouter",
    "AgentTeam",
    "TeamHooks",
    "AgentRole",
    "TeamConfiguration",
    "AgentContribution",
    "TeamOptions",
    "LLMConvergenceChecker",
    "AgentOptions",
    "AgentHooks",
    "AgentPromptBuilder",
    "AutonomousAgent",
    "AutoAgentOptions",
    "AutoAgentHooks"
] 