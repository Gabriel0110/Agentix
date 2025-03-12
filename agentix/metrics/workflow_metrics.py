"""
Workflow metrics tracking system for agent-based workflows.

This module provides:
1. Base workflow metrics tracking
2. Extensible custom metrics
3. Automatic metrics collection via decorators
4. Formatted reporting
"""
from typing import List, Dict, Any, Optional, Type, TypeVar, Generic
from dataclasses import dataclass, field, fields
import time
from abc import ABC, abstractmethod
import functools
import json

T = TypeVar('T', bound='BaseWorkflowMetrics')

class MetricsLogger:
    """
    Logger for workflow metrics that handles automatic collection and aggregation.
    """
    def __init__(self, metrics_instance: 'BaseWorkflowMetrics'):
        self.metrics = metrics_instance
        self.start_time = None
    
    def __enter__(self):
        """Start timing when entering a context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log timing when exiting a context."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.add_timing(duration)

def track_metrics(metric_name: str):
    """
    Decorator to track metrics for a function.
    
    Args:
        metric_name: Name of the metric to increment
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            with MetricsLogger(self.metrics):
                self.metrics.increment(metric_name)
                result = await func(self, *args, **kwargs)
                return result
                
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            with MetricsLogger(self.metrics):
                self.metrics.increment(metric_name)
                result = func(self, *args, **kwargs)
                return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@dataclass
class BaseWorkflowMetrics:
    """Base class for workflow metrics tracking."""
    
    # Core metrics that all workflows should track
    total_queries: int = 0
    total_agent_calls: int = 0
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_processing_time: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    evaluation_scores: List[float] = field(default_factory=list)
    
    # Timing metrics
    step_timings: List[float] = field(default_factory=list)
    average_step_time: float = 0.0
    
    # Custom metrics storage
    _custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def increment(self, metric_name: str, amount: int = 1) -> None:
        """
        Increment a metric by a given amount.
        
        Args:
            metric_name: Name of the metric to increment
            amount: Amount to increment by (default: 1)
        """
        if hasattr(self, metric_name):
            current_value = getattr(self, metric_name)
            if isinstance(current_value, (int, float)):
                setattr(self, metric_name, current_value + amount)
        elif metric_name in self._custom_metrics:
            if isinstance(self._custom_metrics[metric_name], (int, float)):
                self._custom_metrics[metric_name] += amount
    
    def add_timing(self, duration: float) -> None:
        """
        Add a timing measurement.
        
        Args:
            duration: Time duration to add
        """
        self.step_timings.append(duration)
        self.total_processing_time += duration
        self.average_step_time = sum(self.step_timings) / len(self.step_timings)
    
    def add_evaluation_score(self, score: float) -> None:
        """
        Add an evaluation score.
        
        Args:
            score: Evaluation score to add
        """
        self.evaluation_scores.append(score)
        if self.total_queries > 0:
            self.success_rate = len([s for s in self.evaluation_scores if s >= 0.7]) / self.total_queries
    
    def add_custom_metric(self, name: str, value: Any) -> None:
        """
        Add a custom metric.
        
        Args:
            name: Name of the custom metric
            value: Initial value for the metric
        """
        self._custom_metrics[name] = value
    
    def get_custom_metric(self, name: str) -> Any:
        """
        Get a custom metric value.
        
        Args:
            name: Name of the custom metric
            
        Returns:
            Value of the custom metric
        """
        return self._custom_metrics.get(name)
    
    def get_average_evaluation_score(self) -> float:
        """Get the average evaluation score."""
        return sum(self.evaluation_scores) / len(self.evaluation_scores) if self.evaluation_scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        base_metrics = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith('_')
        }
        return {
            **base_metrics,
            "custom_metrics": self._custom_metrics,
            "average_evaluation_score": self.get_average_evaluation_score()
        }
    
    def report(self, include_custom: bool = True) -> str:
        """
        Generate a formatted metrics report.
        
        Args:
            include_custom: Whether to include custom metrics
            
        Returns:
            Formatted report string
        """
        metrics_dict = self.to_dict()
        
        # Format core metrics
        lines = [
            "Workflow Performance Metrics:",
            "----------------------------",
            f"Queries Processed: {self.total_queries}",
            f"Agent Calls: {self.total_agent_calls}",
            f"Tool Calls: {self.total_tool_calls}",
            f"LLM Calls: {self.total_llm_calls}",
            f"Total Processing Time: {self.total_processing_time:.2f}s",
            f"Average Step Time: {self.average_step_time:.2f}s",
            f"Success Rate: {self.success_rate:.1%}",
            f"Error Count: {self.error_count}",
            f"Average Evaluation Score: {self.get_average_evaluation_score():.2f}/1.0"
        ]
        
        # Add custom metrics if requested
        if include_custom and self._custom_metrics:
            lines.extend([
                "",
                "Custom Metrics:",
                "--------------"
            ])
            for name, value in self._custom_metrics.items():
                if isinstance(value, float):
                    lines.append(f"{name}: {value:.2f}")
                else:
                    lines.append(f"{name}: {value}")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class AgentTeamMetrics(BaseWorkflowMetrics):
    """Specialized metrics for agent team workflows."""
    
    # Team-specific metrics
    team_size: int = 0
    parallel_executions: int = 0
    sequential_executions: int = 0
    interleaved_executions: int = 0
    convergence_attempts: int = 0
    successful_convergence: int = 0
    agent_contributions: Dict[str, int] = field(default_factory=dict)
    
    def record_agent_contribution(self, agent_name: str) -> None:
        """Record a contribution from an agent."""
        if agent_name not in self.agent_contributions:
            self.agent_contributions[agent_name] = 0
        self.agent_contributions[agent_name] += 1
    
    def get_contribution_distribution(self) -> Dict[str, float]:
        """Get the distribution of agent contributions."""
        total = sum(self.agent_contributions.values())
        return {
            agent: (count / total) if total > 0 else 0
            for agent, count in self.agent_contributions.items()
        }

@dataclass
class ResearchWorkflowMetrics(BaseWorkflowMetrics):
    """Specialized metrics for research workflows."""
    
    # Research-specific metrics
    search_queries: int = 0
    sources_found: int = 0
    validation_runs: int = 0
    facts_extracted: int = 0
    synthesis_rounds: int = 0
    
    def calculate_source_efficiency(self) -> float:
        """Calculate how efficiently sources are being found."""
        return self.sources_found / self.search_queries if self.search_queries > 0 else 0

# Example usage:
"""
# Basic usage
metrics = BaseWorkflowMetrics()

# Using with a workflow
class MyWorkflow:
    def __init__(self):
        self.metrics = BaseWorkflowMetrics()
    
    @track_metrics("total_queries")
    async def process_query(self, query: str):
        # Processing logic here
        pass

# Using team metrics
team_metrics = AgentTeamMetrics(team_size=3)
team_metrics.record_agent_contribution("Agent1")

# Using research metrics
research_metrics = ResearchWorkflowMetrics()
research_metrics.add_custom_metric("domain_coverage", 0.85)

# Generate reports
print(metrics.report())
print(team_metrics.report())
print(research_metrics.report())
""" 