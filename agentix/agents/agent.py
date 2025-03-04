import time
from typing import List, Dict, Any, Optional, Callable, Union, TypedDict, Awaitable, cast
from dataclasses import dataclass

from ..llms import LLM
from ..memory.memory import Memory, MemoryRole, ConversationMessage
from ..memory.reflection_memory import ReflectionMemory
from ..tools.tools import Tool
from ..planner import Planner
from ..tools.tool_request import ToolRequestParser, ParsedToolRequest
from ..utils.debug_logger import DebugLogger

@dataclass
class AgentOptions:
    """Options to configure agent behavior and safety checks."""
    max_steps: int = 15
    usage_limit: int = 15
    use_reflection: bool = True
    time_to_live: int = 60000  # ms
    debug: bool = False
    validate_output: bool = False


@dataclass
class AgentHooks:
    """Lifecycle hooks for debugging or advanced usage."""
    on_plan_generated: Optional[Callable[[str], None]] = None
    on_tool_call: Optional[Callable[[str, str], Union[Awaitable[bool], bool]]] = None
    on_tool_validation_error: Optional[Callable[[str, str], None]] = None
    on_tool_result: Optional[Callable[[str, str], None]] = None
    on_final_answer: Optional[Callable[[str], None]] = None
    on_step: Optional[Callable[[List[ConversationMessage]], None]] = None


class Agent:
    """
    The main Agent class that can do multi-step reasoning, tool usage, etc.
    """
    
    def __init__(self, 
                 name: Optional[str] = None,
                 model: LLM = None,
                 memory: Memory = None,
                 tools: List[Tool] = None,
                 instructions: List[str] = None,
                 planner: Optional[Planner] = None,
                 options: Optional[Union[AgentOptions, Dict[str, Any]]] = None,
                 hooks: Optional[AgentHooks] = None,
                 task: Optional[str] = None,
                 validation_model: Optional[LLM] = None):
        """
        Initialize an agent with the given parameters.
        """
        self.name = name or "UnnamedAgent"
        self.model = model
        self.memory = memory
        self.tools = tools or []
        self.instructions = instructions or []
        self.planner = planner
        self.hooks = hooks or AgentHooks()
        
        self.task = task
        self.validation_model = validation_model
        
        # Options - handle both dict and AgentOptions
        if options is None:
            options = AgentOptions()
        elif isinstance(options, dict):
            # Convert dict to AgentOptions
            agent_options = AgentOptions()
            for key, value in options.items():
                if hasattr(agent_options, key):
                    setattr(agent_options, key, value)
                else:
                    print(f"Warning: Ignoring unknown option '{key}'")
            options = agent_options
        
        self.max_steps = options.max_steps
        self.usage_limit = options.usage_limit
        self.time_to_live = options.time_to_live
        self.debug = options.debug
        self.logger = DebugLogger(self.debug)
        self.validate_output = options.validate_output
        
        # Reflection toggling
        if len(self.tools) > 0:
            self.use_reflection = True
            if not options.use_reflection:
                self.logger.warn(f"[Agent] Tools were provided, forcing use_reflection to True.")
        else:
            self.use_reflection = options.use_reflection
            
        # Internal counters
        self.llm_calls_used = 0
        self.start_time = 0
        self.step_count = 0
    
    @classmethod
    def create(cls, 
               name: Optional[str] = None,
               model: LLM = None,
               memory: Memory = None,
               tools: Optional[List[Tool]] = None,
               instructions: Optional[List[str]] = None,
               planner: Optional[Planner] = None,
               options: Optional[Union[AgentOptions, Dict[str, Any]]] = None,
               hooks: Optional[AgentHooks] = None,
               task: Optional[str] = None,
               validation_model: Optional[LLM] = None) -> 'Agent':
        """
        Factory method to create an Agent instance.
        
        Args:
            name: Agent name
            model: The LLM to use (OpenAIChat or TogetherChat)
            memory: Memory implementation to use
            tools: Available tools
            instructions: Custom instructions to add to the prompt
            planner: Optional planner for more complex reasoning
            options: Agent configuration options
            hooks: Lifecycle hooks for debugging or advanced usage
            task: The user's stated task (for validation context)
            validation_model: Optional separate model for validation
            
        Returns:
            An initialized Agent
        """
        return cls(
            name=name,
            model=model,
            memory=memory,
            tools=tools,
            instructions=instructions,
            planner=planner,
            options=options,
            hooks=hooks,
            task=task,
            validation_model=validation_model
        )
    
    @classmethod
    async def create_with_generated_prompt(
        cls,
        name: str,
        model: LLM,
        memory: Memory,
        tools: List[Tool],
        task_description: str,
        additional_instructions: Optional[List[str]] = None,
        prompt_builder_model: Optional[LLM] = None,
        use_json_format: bool = False,
        options: Optional[Union[AgentOptions, Dict[str, Any]]] = None,
        hooks: Optional[AgentHooks] = None,
        validation_model: Optional[LLM] = None
    ) -> 'Agent':
        """
        Factory method to create an Agent with an auto-generated system prompt.
        
        This method uses the AgentPromptBuilder to automatically generate tailored
        instructions based on the agent's purpose and available tools.
        
        Args:
            name: Agent name
            model: The LLM to use for the agent
            memory: Memory implementation to use
            tools: Available tools
            task_description: Description of the agent's purpose/task
            additional_instructions: Optional custom instruction points to include
            prompt_builder_model: Optional custom LLM for prompt generation
            use_json_format: If True, generates prompt in JSON format
            options: Agent configuration options
            hooks: Lifecycle hooks for debugging or advanced usage
            validation_model: Optional separate model for validation
            
        Returns:
            An initialized Agent with auto-generated instructions
        """
        from .prompt_builder import AgentPromptBuilder
        
        # Create the prompt builder
        # If no model is provided, it will initialize its own default model
        builder = AgentPromptBuilder(model=prompt_builder_model)
        
        # Generate the prompt
        prompt_result = await builder.generate_prompt(
            agent_name=name,
            tools=tools,
            task_description=task_description,
            additional_instructions=additional_instructions,
            return_json=use_json_format
        )
        
        # Extract the prompt text from JSON if necessary
        prompt_text = prompt_result["prompt"] if use_json_format else prompt_result
        
        # Convert the prompt into instruction lines
        instruction_lines = [line.strip() for line in prompt_text.split('\n') if line.strip()]
        
        # Create the agent with the generated instructions
        return cls.create(
            name=name,
            model=model,
            memory=memory,
            tools=tools,
            instructions=instruction_lines,
            options=options,
            hooks=hooks,
            task=task_description,
            validation_model=validation_model
        )
    
    async def run(self, query: str) -> str:
        """
        The main entry point for the agent.
        """
        self.start_time = int(time.time() * 1000)  # current time in ms
        self.step_count = 0
        self._tool_usage_counter = {}  # Initialize tool usage counter
        self._consecutive_tool_calls = 0  # Track consecutive tool calls
        
        self.logger.log(f"[Agent:{self.name}] Starting run", {"query": query})
        
        # Initialize conversation
        await self.memory.add_message({
            "role": "system",
            "content": self.build_system_prompt()
        })
        await self.memory.add_message({"role": "user", "content": query})
        
        # Single-pass if reflection is off
        if not self.use_reflection:
            return await self.single_pass()
        
        # If planner is specified
        if self.planner:
            return await self.execute_planner_flow(query)
        
        # Default reflection loop
        while True:
            # Check usage/time
            elapsed = int(time.time() * 1000) - self.start_time
            self.logger.stats({
                "llm_calls_used": self.llm_calls_used,
                "llm_calls_limit": self.usage_limit,
                "steps_used": self.step_count,
                "max_steps": self.max_steps,
                "elapsed_ms": elapsed,
                "time_to_live": self.time_to_live,
            })
            
            if self.should_stop(elapsed):
                return self.get_stopping_reason(elapsed)
            
            self.llm_calls_used += 1
            self.step_count += 1
            
            context = await self.memory.get_context_for_prompt(query)
            llm_output = await self.model.call(context)
            self.logger.log(f"[Agent:{self.name}] LLM Output:", {"llm_output": llm_output})
            
            # Tool usage?
            tool_request = ToolRequestParser.parse(llm_output)
            if tool_request:
                self._consecutive_tool_calls += 1
                
                # Check for too many consecutive tool calls without a final answer
                if self._consecutive_tool_calls > 3:
                    # Add a reminder to produce a final answer
                    await self.memory.add_message({
                        "role": "system", 
                        "content": "You have made several tool calls in a row. If you have enough information to answer the user's question, please provide a FINAL ANSWER: now instead of making more tool calls."
                    })
                
                # Store the tool request in memory to help the model see its own requests
                await self.memory.add_message({
                    "role": "assistant", 
                    "content": llm_output,
                    "metadata": {"type": "tool_request"}
                })
                
                result = await self.handle_tool_request(tool_request)
                
                # Store tool results with a more distinctive format
                await self.memory.add_message({
                    "role": "function", 
                    "name": tool_request.tool_name,
                    "content": f"TOOL RESULT: {result}",
                    "metadata": {"type": "tool_result"}
                })
                
                continue  # Next iteration
            
            # Reset consecutive tool calls counter if we're not making a tool call
            self._consecutive_tool_calls = 0
            
            # Final answer check
            if llm_output.startswith("FINAL ANSWER:"):
                final_ans = llm_output.replace("FINAL ANSWER:", "").strip()
                self.logger.log(f"[Agent:{self.name}] Final answer found", {"final_ans": final_ans})
                
                # Add final answer to memory
                await self.memory.add_message({"role": "assistant", "content": llm_output})
                
                # If validate_output is true, attempt validation
                if self.validate_output and self.validation_model:
                    validated = await self.validate_final_answer(final_ans)
                    if not validated:
                        self.logger.log(f"[Agent:{self.name}] Validation failed. Continuing loop to refine...")
                        # We can either continue the loop or forcibly revise the answer
                        # We'll continue the loop here:
                        continue
                
                if self.hooks.on_final_answer:
                    await self.hooks.on_final_answer(final_ans)
                return final_ans
            
            # Otherwise, treat as intermediate output
            await self.memory.add_message({"role": "assistant", "content": llm_output})
    
    async def single_pass(self) -> str:
        """
        Single pass execution without reflection
        """
        if self.llm_calls_used >= self.usage_limit and self.usage_limit != -1:
            return "Usage limit reached. No more LLM calls allowed."
            
        self.llm_calls_used += 1
        single_response = await self.model.call(await self.memory.get_context())
        await self.memory.add_message({"role": "assistant", "content": single_response})
        
        # If final answer, optionally validate
        if self.validate_output and self.validation_model:
            validated = await self.validate_final_answer(single_response)
            if not validated:
                # If single-pass and fails validation, we just return it anyway, or we can override
                self.logger.log(f"[Agent:{self.name}] Single pass validation failed. Returning anyway.")
        
        if self.hooks.on_final_answer:
            await self.hooks.on_final_answer(single_response)
        return single_response
    
    async def execute_planner_flow(self, query: str) -> str:
        """
        Plan-then-execute approach if a planner is provided
        """
        if not self.planner:
            return "No planner specified."
            
        plan = await self.planner.generate_plan(query, self.tools, self.memory)
        if self.hooks.on_plan_generated:
            self.hooks.on_plan_generated(plan)
        
        steps = self.parse_plan(plan)
        for step in steps:
            step_response = await self.execute_plan_step(step, query)
            await self.memory.add_message({"role": "assistant", "content": step_response})
            
            if "FINAL ANSWER" in step_response:
                # Extract the final answer string
                final_answer = step_response.replace("FINAL ANSWER:", "").strip()
                
                # Validate if required
                if self.validate_output and self.validation_model:
                    pass_validation = await self.validate_final_answer(final_answer)
                    if not pass_validation:
                        self.logger.log(
                            f"[Agent:{self.name}] Validation failed in planner flow. Possibly continue or refine?"
                        )
                        # Could do more refinement or just return
                
                return final_answer
        
        return "Plan executed but no final answer was found."
    
    def build_system_prompt(self) -> str:
        """
        Build the system prompt, enumerating tools etc.
        """
        tool_lines = "\n".join([
            f"- {t.name}: {t.description or '(no description)'}"
            for t in self.tools
        ])
        
        lines = []
        lines.append(f'You are an intelligent AI agent named "{self.name}".')
        
        if tool_lines:
            lines.append(
                f"You have access to these tools:\n{tool_lines}\n"
                "Use them by responding with EXACT format:\n"
                'TOOL REQUEST: <ToolName> "<Query>"'
            )
        else:
            lines.append("You do not have any tools available.")
        
        lines.append(
            "When you have the final answer, format EXACTLY:\n"
            "FINAL ANSWER: <Your answer>\n"
            "\n"
            "Important guidelines:\n"
            "1. After using tools to gather information, you MUST provide a final answer to the user.\n"
            "2. Do not repeat the same tool calls unnecessarily.\n"
            "3. Once you have the information needed, use 'FINAL ANSWER:' to respond directly to the user.\n"
            "4. When you see 'TOOL RESULT:' in the conversation, this means you've already received a response from that tool.\n"
            "5. PAY CLOSE ATTENTION to previous tool results in the conversation before making new tool calls.\n"
            "6. If you already have the information needed from a previous tool result, DO NOT call the same tool again.\n"
            "7. Always analyze provided tool results before making additional tool calls."
        )
        lines.extend(self.instructions)
        
        return "\n\n".join(lines)
    
    def parse_plan(self, plan: str) -> List[Dict[str, str]]:
        """
        Parse the plan into steps
        """
        try:
            import json
            return json.loads(plan)
        except Exception:
            return [{"action": "message", "details": plan}]
    
    async def execute_plan_step(self, step: Dict[str, str], query: str) -> str:
        """
        Execute a single step from a plan
        """
        action = step.get("action", "")
        details = step.get("details", "")
        
        if action == "tool":
            tool = next((t for t in self.tools if t.name == details), None)
            if not tool:
                return f'Error: Tool "{details}" not found.'
            return await tool.run(query)
        elif action == "message":
            return await self.model.call([{"role": "user", "content": details}])
        elif action == "complete":
            return f"FINAL ANSWER: {details}"
        else:
            return f"Unknown action: {action}"
    
    async def handle_tool_request(self, request: ParsedToolRequest) -> str:
        """
        Handle a tool request from the agent
        """
        self.logger.log("Processing tool request", request)
        
        # Track tool usage to detect loops
        tool_key = f"{request.tool_name}:{request.query}"
        self._tool_usage_counter[tool_key] = self._tool_usage_counter.get(tool_key, 0) + 1
        
        # If the same tool with the same parameters is called more than twice, suggest moving to a final answer
        if self._tool_usage_counter.get(tool_key, 0) > 2:
            self.logger.log(f"Tool loop detected: {tool_key} used {self._tool_usage_counter[tool_key]} times")
            return (
                f"I've noticed you've called '{request.tool_name}' with the same parameters multiple times. "
                f"You already have this information from previous calls. "
                f"Please analyze the previous tool results and provide a FINAL ANSWER: to respond to the user's query."
            )
        
        try:
            validation_error = ToolRequestParser.validate_basic(request, self.tools)
            if validation_error:
                raise Exception(validation_error)
            
            tool = next(
                (t for t in self.tools if t.name.lower() == request.tool_name.lower()),
                None
            )
            if not tool:
                raise Exception(f'Tool "{request.tool_name}" is not available.')
            
            ToolRequestParser.validate_parameters(tool, request)
            
            if self.hooks.on_tool_call:
                proceed = await self.hooks.on_tool_call(tool.name, request.query)
                if not proceed:
                    self.logger.log("Tool call cancelled by hook", {"tool_name": tool.name})
                    return f'Tool call to "{tool.name}" cancelled by user approval.'
            
            result = await tool.run("", request.args) if request.args else await tool.run(request.query)
            
            self.logger.log("Tool execution result", {"tool_name": tool.name, "result": result})
            
            if self.hooks.on_tool_result:
                await self.hooks.on_tool_result(tool.name, result)
            
            return result
        except Exception as err:
            error_msg = str(err)
            self.logger.error("Tool request failed", {"error": error_msg})
            return f"Error processing tool request: {error_msg}"
    
    def should_stop(self, elapsed: int) -> bool:
        """
        Called each iteration to see if we should stop for usage/time reasons
        """
        if self.max_steps != -1 and self.step_count >= self.max_steps:
            return True
        if self.usage_limit != -1 and self.llm_calls_used >= self.usage_limit:
            return True
        if self.time_to_live != -1 and elapsed >= self.time_to_live:
            return True
        return False
    
    def get_stopping_reason(self, elapsed: int) -> str:
        """
        Get a message explaining why execution stopped
        """
        if self.step_count >= self.max_steps:
            return f"Max steps ({self.max_steps}) reached without final answer."
        if self.usage_limit != -1 and self.llm_calls_used >= self.usage_limit:
            return f"Usage limit ({self.usage_limit} calls) reached."
        if elapsed >= self.time_to_live:
            return f"Time limit ({self.time_to_live}ms) reached after {elapsed}ms."
        return "Unknown stopping condition reached."
    
    async def handle_reflection(self, reflection_content: str) -> None:
        """
        Handle reflection content
        """
        reflection_message = {
            "role": "reflection",
            "content": reflection_content,
        }
        
        if isinstance(self.memory, ReflectionMemory):
            await self.memory.add_message(reflection_message)
        
        self.logger.log(f"Reflection stored: {reflection_content}")
    
    async def validate_final_answer(self, final_answer: str) -> bool:
        """
        Validate final answer if validate_output is True and we have a validation_model.
        If passes validation, returns True. If fails, returns False.
        """
        if not self.validation_model:
            # No separate model, skip
            return True
        
        system_prompt = f"""
You are a validator that checks if an agent's final answer meets the user's task requirements.
If the final answer is correct and satisfies the task, respond with a JSON:
{{"is_valid":true,"reason":"some short reason"}}
If it fails or is incomplete, respond with:
{{"is_valid":false,"reason":"some short reason"}}

User's task (if any): {self.task or "(none provided)"}

Agent's final answer to validate:
{final_answer}
        """
        
        validator_output = await self.validation_model.call([
            {"role": "system", "content": system_prompt},
        ])
        
        self.logger.log(f"[Agent:{self.name}] Validator output:", {"validator_output": validator_output})
        
        # Attempt to parse
        try:
            import json
            parsed = json.loads(validator_output)
            if parsed.get("is_valid") is True:
                self.logger.log(f"[Agent:{self.name}] Validation PASSED: {parsed.get('reason')}")
                return True
            else:
                self.logger.log(f"[Agent:{self.name}] Validation FAILED: {parsed.get('reason')}")
                return False
        except Exception:
            self.logger.warn(f"[Agent:{self.name}] Could not parse validator output. Assuming fail.",
                            {"validator_output": validator_output})
            return False 