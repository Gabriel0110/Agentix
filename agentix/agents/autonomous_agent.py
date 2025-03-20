"""
Autonomous Agent Module

This module extends the basic Agent and AgentTeam classes with autonomous capabilities
that enable agents to work independently with minimal user intervention, including:
- Advanced planning and execution
- Self-healing and error recovery
- Task monitoring and reporting
- Configurable human intervention points
"""

from typing import List, Dict, Any, Optional, Callable, Union, Set, TypedDict, Awaitable
from dataclasses import dataclass, field
import asyncio
import time

from .agent import Agent, AgentOptions, AgentHooks
from ..planner import Planner, SimpleLLMPlanner
from ..llms import LLM
from ..memory.memory import Memory
from ..tools.tools import Tool
from ..metrics.workflow_metrics import BaseWorkflowMetrics
from ..utils.debug_logger import DebugLogger


@dataclass
class AutoAgentOptions(AgentOptions):
    """Configuration options for autonomous agents."""
    # Maximum number of self-healing attempts before giving up
    max_recovery_attempts: int = 3
    
    # Whether to enable self-planning capabilities
    enable_planning: bool = True
    
    # Whether to automatically attempt recovery from errors
    auto_recovery: bool = True
    
    # Which operations require human approval
    require_approval_for: Set[str] = field(default_factory=lambda: set())
    
    # Maximum thinking iterations before requiring intervention
    max_thinking_depth: int = 5
    
    # User approval callback, should return True if approved
    approval_callback: Optional[Callable[[str, str], Awaitable[bool]]] = None
    
    # Progress reporting callback
    progress_callback: Optional[Callable[[str, float], None]] = None


@dataclass
class AutoAgentHooks(AgentHooks):
    """Lifecycle hooks for autonomous agent execution."""
    # Called when agent needs to make a recovery attempt
    on_recovery_attempt: Optional[Callable[[str, Exception, int], None]] = None
    
    # Called when agent is starting a new task
    on_task_start: Optional[Callable[[str], None]] = None
    
    # Called when agent completes a task
    on_task_complete: Optional[Callable[[str, Any], None]] = None
    
    # Called when agent's task fails permanently
    on_task_failed: Optional[Callable[[str, Exception], None]] = None
    
    # Called when agent is thinking/planning
    on_thinking: Optional[Callable[[str, str], None]] = None
    
    # Called when agent requires user approval
    on_approval_request: Optional[Callable[[str, str], Awaitable[bool]]] = None


class ExecutionState(TypedDict, total=False):
    """Tracks the state of task execution."""
    task_id: str
    status: str  # 'running', 'failed', 'completed', 'waiting_approval'
    current_step: int
    total_steps: int
    error: Optional[str]
    recovery_attempts: int
    result: Any
    start_time: float
    last_updated: float


class AutonomousAgent(Agent):
    """
    Agent with autonomous capabilities including self-healing, planning,
    and configurable human intervention points.
    """
    
    def __init__(self,
                 name: Optional[str] = None,
                 model: LLM = None,
                 memory: Memory = None,
                 tools: List[Tool] = None,
                 instructions: List[str] = None,
                 planner: Optional[Planner] = None,
                 options: Optional[Union[AutoAgentOptions, Dict[str, Any]]] = None,
                 hooks: Optional[Union[AutoAgentHooks, Dict[str, Any]]] = None,
                 task: Optional[str] = None,
                 validation_model: Optional[LLM] = None,
                 metrics: Optional[BaseWorkflowMetrics] = None):
        """
        Initialize an autonomous agent with extended capabilities.
        
        Args:
            name: The name of the agent
            model: LLM instance for the agent
            memory: Memory instance for the agent
            tools: List of tools available to the agent
            instructions: List of system instructions
            planner: Planner for task planning (if None, will create one)
            options: Configuration options for the agent
            hooks: Lifecycle hooks for the agent
            task: Current task description
            validation_model: LLM for output validation
            metrics: Metrics tracker for the agent
        """
        # Convert options dict to AutoAgentOptions if needed
        if options is None:
            auto_options = AutoAgentOptions()
        elif isinstance(options, dict):
            auto_options = AutoAgentOptions()
            for key, value in options.items():
                if hasattr(auto_options, key):
                    setattr(auto_options, key, value)
        else:
            auto_options = options
            
        # Convert hooks dict to AutoAgentHooks if needed
        if hooks is None:
            auto_hooks = AutoAgentHooks()
        elif isinstance(hooks, dict):
            auto_hooks = AutoAgentHooks()
            for key, value in hooks.items():
                if hasattr(auto_hooks, key):
                    setattr(auto_hooks, key, value)
        else:
            auto_hooks = hooks
            
        # Initialize the basic Agent with debug enabled
        super().__init__(
            name=name,
            model=model,
            memory=memory,
            tools=tools,
            instructions=instructions,
            planner=planner,
            options=auto_options,  # This passes debug=True to parent
            hooks=auto_hooks,
            task=task,
            validation_model=validation_model
        )
        
        # Store autonomous-specific attributes
        self.auto_options = auto_options
        self.auto_hooks = auto_hooks
        self.metrics = metrics or BaseWorkflowMetrics()
        
        # Log initialization using parent's logger
        self.logger.log(f"[AutonomousAgent:{self.name}] Initialized with options", {
            "enable_planning": auto_options.enable_planning,
            "auto_recovery": auto_options.auto_recovery,
            "max_recovery_attempts": auto_options.max_recovery_attempts,
            "require_approval_for": list(auto_options.require_approval_for),
            "max_thinking_depth": auto_options.max_thinking_depth
        })
        
        # Default planner if none is provided and planning is enabled
        if self.planner is None and auto_options.enable_planning:
            self.planner = SimpleLLMPlanner(model)
            self.logger.log(f"[AutonomousAgent:{self.name}] Created default planner")
            
        # State management for autonomous execution
        self.execution_state: Dict[str, ExecutionState] = {}
        self.current_task_id: Optional[str] = None
            
    async def execute_task(self, task: str, task_id: Optional[str] = None) -> Any:
        """
        Execute a task autonomously with self-healing and progress tracking.
        """
        task_id = task_id or f"task_{int(time.time())}"
        self.current_task_id = task_id
        task_start_time = time.time()
        
        self.logger.log(f"[AutonomousAgent:{self.name}] Starting task execution", {
            "task_id": task_id,
            "task": task
        })
        
        # Reset tracking counters for this task
        self.start_time = int(time.time() * 1000)
        self.step_count = 0
        self._consecutive_tool_calls = 0
        self._last_n_tool_calls = []
        self._last_n_assistant_messages = []
        
        # Initialize execution state
        self.execution_state[task_id] = {
            "task_id": task_id,
            "status": "running",
            "current_step": 0,
            "total_steps": 0,
            "error": None,
            "recovery_attempts": 0,
            "result": None,
            "start_time": time.time(),
            "last_updated": time.time()
        }
        
        self.logger.log(f"[AutonomousAgent:{self.name}] Initialized execution state", self.execution_state[task_id])
        
        # Track metrics
        self.metrics.increment("total_queries")
        self.metrics.increment("total_agent_calls")
        
        # Initialize conversation for this task
        await self.memory.add_message({
            "role": "system",
            "content": self.build_system_prompt()
        })
        await self.memory.add_message({"role": "user", "content": task})
        
        # Notify task start
        if self.auto_hooks.on_task_start:
            self.auto_hooks.on_task_start(task)
            
        try:
            # If planning is enabled, plan first then execute
            if self.auto_options.enable_planning and self.planner:
                self.logger.log(f"[AutonomousAgent:{self.name}] Executing with planning")
                result = await self._execute_with_planning(task, task_id)
            else:
                # Execute without explicit planning
                self.logger.log(f"[AutonomousAgent:{self.name}] Executing without planning")
                result = await self._execute_with_recovery(
                    lambda: self.run(task),
                    task_id
                )
                
            # Update state on completion
            self.execution_state[task_id].update({
                "status": "completed",
                "result": result,
                "last_updated": time.time()
            })
            
            self.logger.log(f"[AutonomousAgent:{self.name}] Task completed successfully", {
                "task_id": task_id,
                "execution_time": time.time() - task_start_time
            })
            
            # Update metrics
            execution_time = time.time() - task_start_time
            self.metrics.add_custom_metric("total_processing_time", execution_time)
            if self.step_count > 0:
                self.metrics.add_custom_metric("average_step_time", execution_time / self.step_count)
            self.metrics.add_custom_metric("success_rate", 1.0)
            
            # Notify task completion
            if self.auto_hooks.on_task_complete:
                self.auto_hooks.on_task_complete(task, result)
                
            return result
            
        except Exception as e:
            # Update state on failure
            self.execution_state[task_id].update({
                "status": "failed",
                "error": str(e),
                "last_updated": time.time()
            })
            
            self.logger.error(f"[AutonomousAgent:{self.name}] Task failed", {
                "task_id": task_id,
                "error": str(e),
                "execution_time": time.time() - task_start_time
            })
            
            # Update metrics
            self.metrics.increment("error_count")
            self.metrics.add_custom_metric("success_rate", 0.0)
            
            # Notify task failure
            if self.auto_hooks.on_task_failed:
                self.auto_hooks.on_task_failed(task, e)
                
            raise e
            
    async def _execute_with_planning(self, task: str, task_id: str) -> Any:
        """
        Plan and execute a task with recovery capabilities.
        """
        # First generate a plan
        self.logger.log(f"[AutonomousAgent:{self.name}] Generating plan", {
            "task_id": task_id,
            "task": task
        })
        
        plan = await self._generate_plan_with_recovery(task, task_id)
        self.metrics.increment("total_llm_calls")
        
        self.logger.log(f"[AutonomousAgent:{self.name}] Plan generated", {
            "task_id": task_id,
            "plan": plan
        })
        
        # Call the base hooks for plan generation
        if self.hooks.on_plan_generated:
            self.hooks.on_plan_generated(plan)
        
        # Check if planning requires approval
        if "planning" in self.auto_options.require_approval_for:
            self.logger.log(f"[AutonomousAgent:{self.name}] Requesting plan approval")
            approved = await self._request_approval(
                "Planning",
                f"Generated plan for task: {task}\n\n{plan}"
            )
            if not approved:
                self.logger.warn(f"[AutonomousAgent:{self.name}] Plan was not approved")
                raise Exception("Plan was not approved by user")
                
        # Parse plan into steps
        steps = self.parse_plan(plan)
        total_steps = len(steps)
        
        self.logger.log(f"[AutonomousAgent:{self.name}] Parsed plan into steps", {
            "task_id": task_id,
            "total_steps": total_steps
        })
        
        # Update execution state
        self.execution_state[task_id].update({
            "total_steps": total_steps,
            "current_step": 0
        })
        
        # Update progress
        if self.auto_options.progress_callback:
            self.auto_options.progress_callback(task_id, 0.0)
            
        # Execute each step with recovery
        results = []
        final_answer = None
        
        for i, step in enumerate(steps):
            # Update current step
            self.execution_state[task_id]["current_step"] = i + 1
            self.step_count = i + 1
            
            self.logger.log(f"[AutonomousAgent:{self.name}] Executing step {i+1}/{total_steps}", {
                "task_id": task_id,
                "step": step
            })
            
            # Execute step with recovery
            step_result = await self._execute_with_recovery(
                lambda: self.execute_plan_step(step, task),
                task_id
            )
            
            self.logger.log(f"[AutonomousAgent:{self.name}] Step {i+1} completed", {
                "task_id": task_id,
                "result": step_result
            })
            
            # Update metrics based on step execution
            if step.get("action") == "tool":
                self.metrics.increment("total_tool_calls")
                self.metrics.increment("total_agent_calls")
            elif step.get("action") == "message":
                self.metrics.increment("total_llm_calls")
            
            # Store step result in memory
            if step.get("action") == "tool":
                await self.memory.add_message({
                    "role": "assistant",
                    "content": f"Tool '{step['details']}' returned: {step_result}",
                    "metadata": {
                        "type": "tool_result",
                        "tool_name": step["details"],
                        "result": step_result
                    }
                })
            else:
                await self.memory.add_message({
                    "role": "assistant",
                    "content": step_result
                })
            
            # Collect result
            results.append(step_result)
            
            # Check if this is the final answer
            if step.get("action") == "complete":
                final_answer = step_result.replace("FINAL ANSWER:", "").strip()
                
                self.logger.log(f"[AutonomousAgent:{self.name}] Final answer found", {
                    "task_id": task_id,
                    "final_answer": final_answer
                })
                
                # Call base hook for final answer
                if self.hooks.on_final_answer:
                    await self.hooks.on_final_answer(final_answer)
                
                # Update progress to complete
                if self.auto_options.progress_callback:
                    self.auto_options.progress_callback(task_id, 1.0)
                    
                return final_answer
                
            # Update progress
            if self.auto_options.progress_callback:
                progress = (i + 1) / total_steps
                self.auto_options.progress_callback(task_id, progress)
                
        # If we get here and have results but no final answer, construct one from the results
        if results:
            final_result = "\n".join(str(r) for r in results if r)
            self.logger.log(f"[AutonomousAgent:{self.name}] Constructed final result from steps", {
                "task_id": task_id,
                "final_result": final_result
            })
            return f"FINAL ANSWER: {final_result}"
            
        self.logger.warn(f"[AutonomousAgent:{self.name}] No results produced from plan execution")
        return "Plan executed but no results were produced."
        
    async def _generate_plan_with_recovery(self, task: str, task_id: str) -> str:
        """
        Generate a plan with recovery capabilities.
        """
        if not self.planner:
            self.logger.error(f"[AutonomousAgent:{self.name}] No planner configured")
            raise Exception("No planner is configured")
            
        self.logger.log(f"[AutonomousAgent:{self.name}] Generating plan with recovery", {
            "task_id": task_id,
            "task": task
        })
            
        return await self._execute_with_recovery(
            lambda: self.planner.generate_plan(task, self.tools, self.memory),
            task_id
        )
        
    async def _execute_with_recovery(self, fn: Callable[[], Awaitable[Any]], task_id: str) -> Any:
        """
        Execute a function with automatic recovery attempts on failure.
        """
        recovery_attempts = 0
        max_attempts = self.auto_options.max_recovery_attempts
        
        while True:
            try:
                # Execute the function
                return await fn()
                
            except Exception as e:
                # Don't attempt recovery if auto-recovery is disabled
                if not self.auto_options.auto_recovery:
                    self.logger.error(f"[AutonomousAgent:{self.name}] Auto-recovery disabled, failing", {
                        "task_id": task_id,
                        "error": str(e)
                    })
                    raise e
                    
                # Track the error
                self.logger.error(f"[AutonomousAgent:{self.name}] Error during execution", {
                    "task_id": task_id,
                    "error": str(e),
                    "recovery_attempt": recovery_attempts + 1,
                    "max_attempts": max_attempts
                })
                self.metrics.increment("error_count")
                
                # Update execution state
                self.execution_state[task_id].update({
                    "error": str(e),
                    "recovery_attempts": recovery_attempts + 1,
                    "last_updated": time.time()
                })
                
                recovery_attempts += 1
                
                # Check if we've exceeded max attempts
                if recovery_attempts > max_attempts:
                    self.logger.error(f"[AutonomousAgent:{self.name}] Max recovery attempts exceeded", {
                        "task_id": task_id,
                        "max_attempts": max_attempts
                    })
                    raise Exception(f"Failed after {max_attempts} recovery attempts: {str(e)}")
                    
                # Notify recovery attempt
                if self.auto_hooks.on_recovery_attempt:
                    self.auto_hooks.on_recovery_attempt(task_id, e, recovery_attempts)
                    
                # Attempt recovery through reflection
                recovery_prompt = self._build_recovery_prompt(str(e), recovery_attempts)
                
                # Check if recovery requires approval
                if "recovery" in self.auto_options.require_approval_for:
                    self.logger.log(f"[AutonomousAgent:{self.name}] Requesting recovery approval", {
                        "task_id": task_id,
                        "attempt": recovery_attempts
                    })
                    approved = await self._request_approval(
                        "Recovery",
                        f"Recovery attempt {recovery_attempts}/{max_attempts} for error: {str(e)}"
                    )
                    if not approved:
                        self.logger.warn(f"[AutonomousAgent:{self.name}] Recovery not approved")
                        raise Exception("Recovery was not approved by user")
                        
                # Execute recovery thinking
                self.logger.log(f"[AutonomousAgent:{self.name}] Executing recovery thinking", {
                    "task_id": task_id,
                    "attempt": recovery_attempts
                })
                recovery_thinking = await self.model.call([
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": recovery_prompt}
                ])
                
                # Notify thinking
                if self.auto_hooks.on_thinking:
                    self.auto_hooks.on_thinking(task_id, recovery_thinking)
                    
                # Add recovery thinking to memory
                await self.memory.add_message({"role": "assistant", "content": f"[RECOVERY THINKING] {recovery_thinking}"})
                
                self.logger.log(f"[AutonomousAgent:{self.name}] Recovery thinking complete, retrying", {
                    "task_id": task_id,
                    "attempt": recovery_attempts
                })
                
                # Wait briefly before retrying
                await asyncio.sleep(1)
                
    async def _request_approval(self, action_type: str, details: str) -> bool:
        """
        Request user approval for an action.
        """
        # Update execution state
        if self.current_task_id:
            self.execution_state[self.current_task_id].update({
                "status": "waiting_approval",
                "last_updated": time.time()
            })
            
        self.logger.log(f"[AutonomousAgent:{self.name}] Requesting approval", {
            "action_type": action_type,
            "details": details
        })
            
        # Use custom approval callback if provided
        if self.auto_options.approval_callback:
            approved = await self.auto_options.approval_callback(action_type, details)
            self.logger.log(f"[AutonomousAgent:{self.name}] Approval callback result: {approved}")
            return approved
            
        # Use hook if provided
        if self.auto_hooks.on_approval_request:
            approved = await self.auto_hooks.on_approval_request(action_type, details)
            self.logger.log(f"[AutonomousAgent:{self.name}] Approval hook result: {approved}")
            return approved
            
        # Default to True if no callback is provided
        self.logger.log(f"[AutonomousAgent:{self.name}] No approval callback/hook, defaulting to True")
        return True
        
    def _build_recovery_prompt(self, error: str, attempt: int) -> str:
        """
        Build a prompt for error recovery.
        
        Args:
            error: The error message
            attempt: The current recovery attempt number
            
        Returns:
            A prompt for recovery
        """
        return f"""
I encountered an error during execution: {error}

This is recovery attempt {attempt}. Please help me recover from this error by:

1. Analyzing what went wrong
2. Suggesting an alternative approach 
3. Providing a specific plan to overcome this issue

Be specific and precise in your recovery plan. Consider:
- Were there incorrect assumptions?
- Was there missing information?
- Was the approach incorrect?
- Is there a simpler way to achieve the goal?

Please think step by step and provide a clear recovery plan.
"""
    
    def get_task_status(self, task_id: str) -> Optional[ExecutionState]:
        """
        Get the current status of a task.
        
        Args:
            task_id: The task identifier
            
        Returns:
            The execution state or None if not found
        """
        return self.execution_state.get(task_id)
        
    def get_all_tasks(self) -> Dict[str, ExecutionState]:
        """
        Get all tracked tasks and their states.
        
        Returns:
            Dictionary of task IDs to execution states
        """
        return self.execution_state
        
    @classmethod
    def create(cls, **kwargs) -> 'AutonomousAgent':
        """
        Factory method to create an autonomous agent with default settings.
        
        Args:
            **kwargs: Arguments to pass to the constructor
            
        Returns:
            An initialized AutonomousAgent
        """
        # Ensure we have hooks and options of the right type
        if 'hooks' in kwargs and not isinstance(kwargs['hooks'], (dict, AutoAgentHooks)):
            raise ValueError("hooks must be an AutoAgentHooks instance or a dict")
            
        if 'options' in kwargs and not isinstance(kwargs['options'], (dict, AutoAgentOptions)):
            raise ValueError("options must be an AutoAgentOptions instance or a dict")
            
        return cls(**kwargs)

    async def execute_with_planning(self, task: str) -> str:
        """
        Execute a task using the planner for structured execution
        """
        self.logger.log(f"[Agent:{self.name}] Starting planned execution for task: {task}")
        
        try:
            # Generate plan using available tools
            self.logger.log(f"[Agent:{self.name}] Generating plan...")
            plan = await self.planner.generate_plan(task, self.tools, self.memory)
            
            # Parse and validate plan
            try:
                steps = self.parse_plan(plan)
                if not steps:
                    raise ValueError("Empty plan generated")
                    
                self.logger.log(f"[Agent:{self.name}] Plan generated with {len(steps)} steps")
                
                # Store plan in memory for context
                await self.memory.add_message({
                    "role": "system",
                    "content": f"Generated execution plan:\n{plan}"
                })
                
            except Exception as e:
                self.logger.error(f"[Agent:{self.name}] Error parsing plan", {"error": str(e)})
                # Fall back to direct execution
                return await self.execute_without_planning(task)
            
            # Execute each step with proper error handling
            results = []
            for i, step in enumerate(steps):
                try:
                    self.logger.log(f"[Agent:{self.name}] Executing step {i+1}/{len(steps)}")
                    
                    # Check stopping conditions
                    elapsed = int((time.time() - self.start_time) * 1000)
                    if self.should_stop(elapsed):
                        reason = self.get_stopping_reason(elapsed)
                        self.logger.warn(f"[Agent:{self.name}] Stopping execution", {"reason": reason})
                        await self.memory.add_message({
                            "role": "system",
                            "content": f"Execution stopped: {reason}"
                        })
                        break
                    
                    # Execute step with retry logic for transient failures
                    max_retries = 2
                    retry_count = 0
                    while retry_count <= max_retries:
                        try:
                            step_result = await self.execute_plan_step(step, task)
                            results.append(step_result)
                            break
                        except Exception as e:
                            retry_count += 1
                            if retry_count > max_retries:
                                self.logger.error(f"[Agent:{self.name}] Step failed after retries", {
                                    "step": step,
                                    "error": str(e)
                                })
                                results.append(f"Error in step {i+1}: {str(e)}")
                                break
                            else:
                                self.logger.warn(f"[Agent:{self.name}] Retrying step after error", {
                                    "retry": retry_count,
                                    "error": str(e)
                                })
                                await asyncio.sleep(1)  # Brief delay before retry
                    
                    # Update progress
                    self.step_count += 1
                    
                except Exception as e:
                    self.logger.error(f"[Agent:{self.name}] Unexpected error in step execution", {"error": str(e)})
                    results.append(f"Unexpected error in step {i+1}: {str(e)}")
                    continue
            
            # Process final results
            final_result = results[-1] if results else "No results generated"
            
            # Validate final answer if needed
            if self.validate_output and final_result.startswith("FINAL ANSWER:"):
                is_valid = await self.validate_final_answer(final_result)
                if not is_valid:
                    self.logger.warn(f"[Agent:{self.name}] Final answer validation failed")
                    # Attempt to improve the answer
                    final_result = await self.improve_final_answer(final_result, results)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"[Agent:{self.name}] Error in planned execution", {"error": str(e)})
            # Fall back to direct execution
            return await self.execute_without_planning(task)
    
    async def improve_final_answer(self, current_answer: str, all_results: List[str]) -> str:
        """
        Attempt to improve a final answer that failed validation
        """
        try:
            # Get context from memory
            context = await self.memory.get_context()
            
            # Create improvement prompt
            improvement_prompt = [
                {"role": "system", "content": "You are improving a final answer that failed validation. Use all available information to create a more complete and accurate response."},
                {"role": "user", "content": f"""
Current answer: {current_answer}

All execution results:
{all_results}

Recent context:
{context[-5:] if context else 'No context available'}

Create an improved final answer that:
1. Addresses the original query completely
2. Incorporates all relevant information
3. Provides clear and accurate conclusions
4. Maintains a professional tone
"""}
            ]
            
            # Get improved answer
            self.metrics.increment("total_llm_calls")
            improved_answer = await self.model.call(improvement_prompt)
            
            return f"FINAL ANSWER: {improved_answer}"
            
        except Exception as e:
            self.logger.error(f"[Agent:{self.name}] Error improving answer", {"error": str(e)})
            return current_answer  # Fall back to original answer