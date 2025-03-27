import os
import time
from typing import List, Dict, Any, Optional, Union
import json

from ..llms import LLM, OpenAIChat
from ..tools.tools import Tool


class AgentPromptBuilder:
    """
    A utility class that uses LLMs to generate effective system prompts for agents.
    
    This class helps developers create comprehensive system prompts without having to
    manually write detailed instructions for each agent. It leverages an LLM to generate
    tailored prompts based on the tools, task description, and other parameters.
    """
    
    def __init__(
        self,
        model: Optional[LLM] = None,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
    ):
        """
        Initialize the prompt builder with an LLM for generating prompts.
        
        Args:
            model: Optional pre-configured OpenAIChat instance
            model_name: Model to use if no model is provided (default: gpt-4o)
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            temperature: Temperature setting for prompt generation (lower is more precise)
        """
        if model:
            self.model = model
        else:
            self.model = OpenAIChat(
                model=model_name,
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                temperature=temperature
            )
        
        # Default template patterns for different sections of the prompt
        self.templates = {
            "tool_usage": (
                "Use tools by responding with EXACT format:\n"
                'TOOL REQUEST: <ToolName> "<Query>"\n'
                "or for tools with parameters:\n"
                'TOOL REQUEST: <ToolName> {"param1": "value1", "param2": "value2"}'
            ),
            "final_answer": (
                "When you have the final answer, format EXACTLY:\n"
                "FINAL ANSWER: <Your answer>"
            ),
            "guidelines": [
                "After using tools to gather information, you MUST provide a final answer to the user.",
                "Do not repeat the same tool calls unnecessarily.",
                "Once you have the information needed, use 'FINAL ANSWER:' to respond directly to the user.",
                "When you see 'TOOL RESULT:' in the conversation, this means you've already received a response from that tool.",
                "PAY CLOSE ATTENTION to previous tool results in the conversation before making new tool calls.",
                "If you already have the information needed from a previous tool result, DO NOT call the same tool again.",
                "Always analyze provided tool results before making additional tool calls."
            ]
        }
        
        # Example high-quality prompts to help the LLM understand good prompt structure
        self.example_prompts = {
            "stock_analysis": """
You are a stock market analysis assistant that helps users analyze stocks and financial data.
You have access to the following YFinance tools that you must use correctly:
- StockPrice: Get current price with 'TOOL REQUEST: StockPrice "SYMBOL"'
- CompanyInfo: Get company details with 'TOOL REQUEST: CompanyInfo "SYMBOL"'
- StockHistoricalPrices: Get price history with 'TOOL REQUEST: StockHistoricalPrices {"symbol": "SYMBOL", "period": "1mo", "interval": "1d"}'
- StockFundamentals: Get key metrics with 'TOOL REQUEST: StockFundamentals "SYMBOL"'
- FinancialStatements: Get statements with 'TOOL REQUEST: FinancialStatements {"symbol": "SYMBOL", "statement_type": "income"}'

When analyzing stocks:
1. Always start by getting the current stock price using StockPrice
2. Follow up with CompanyInfo to understand the business
3. Use StockFundamentals to analyze key financial metrics
4. For historical analysis, use StockHistoricalPrices with appropriate period/interval
5. For detailed financials, use FinancialStatements with the right statement type

After retrieving the necessary information with tools, ALWAYS provide your final answer to the user with the format:
FINAL ANSWER: <Your comprehensive answer>

Remember: You must use FINAL ANSWER: to conclude your response after using tools.
Do not keep using tools repeatedly if you already have the information needed to answer the question.

Provide clear explanations of all data retrieved and what it means for investors.
Format tool requests exactly as shown in the examples above.
Always verify the stock symbol exists before making multiple requests.
"""
        }
    
    def _extract_tool_info(self, tools: List[Tool]) -> Dict[str, Any]:
        """Extract comprehensive information about tools for prompt generation."""
        tool_info = []
        
        for tool in tools:
            info = {
                "name": tool.name,
                "description": tool.description or "(No description provided)",
                "parameters": [],
                "example_usage": None
            }
            
            # Add parameter information
            if tool.parameters:
                for param in tool.parameters:
                    info["parameters"].append({
                        "name": param.name,
                        "type": param.type,
                        "description": param.description,
                        "required": param.required
                    })
            
            # Add example usage if available
            if tool.docs and tool.docs.usage_example:
                info["example_usage"] = tool.docs.usage_example
            
            tool_info.append(info)
        
        return {"tools": tool_info}
    
    def _build_tool_descriptions(self, tools: List[Tool]) -> str:
        """Build formatted tool descriptions."""
        descriptions = []
        
        for tool in tools:
            # Basic tool info
            desc = f"- {tool.name}: {tool.description or '(No description provided)'}"
            
            # Add parameter info if available
            if tool.parameters:
                param_desc = []
                for param in tool.parameters:
                    required_str = " (required)" if param.required else " (optional)"
                    param_desc.append(f"  - {param.name}: {param.description}{required_str}")
                
                if param_desc:
                    desc += "\n" + "\n".join(param_desc)
            
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def _build_tool_usage_examples(self, tools: List[Tool]) -> str:
        """Build examples of how to use each tool properly."""
        examples = []
        
        for tool in tools:
            example = f"- {tool.name}:"
            
            # Simple example with no parameters
            example += f'\n  TOOL REQUEST: {tool.name} "example query"'
            
            # Example with parameters if available
            if tool.parameters:
                required_params = {p.name: f"example_{p.name}" for p in tool.parameters if p.required}
                if required_params:
                    json_params = json.dumps(required_params, ensure_ascii=False)
                    example += f'\n  TOOL REQUEST: {tool.name} {json_params}'
            
            # Add doc examples if available
            if tool.docs and tool.docs.usage_example:
                example += f"\n  Example: {tool.docs.usage_example}"
                
            examples.append(example)
        
        return "\n".join(examples)
    
    async def generate_prompt(
        self,
        agent_name: str,
        tools: List[Tool],
        task_description: str,
        additional_instructions: Optional[List[str]] = None,
        include_examples: bool = True,
        custom_template: Optional[Dict[str, Any]] = None,
        return_json: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a comprehensive system prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            tools: List of Tool objects available to the agent
            task_description: Description of the agent's primary task/purpose
            additional_instructions: Additional custom instructions to include
            include_examples: Whether to include example usage patterns
            custom_template: Custom template sections to override defaults
            return_json: If True, returns a JSON structure instead of just the prompt text
            
        Returns:
            A comprehensive system prompt string or a JSON dictionary if return_json is True
        """
        # Combine default and custom templates if provided
        template = self.templates.copy()
        if custom_template:
            for key, value in custom_template.items():
                template[key] = value
        
        # Extract structured information about tools
        tool_info = self._extract_tool_info(tools)
        
        # Build tool descriptions and examples
        tool_descriptions = self._build_tool_descriptions(tools)
        tool_usage_examples = self._build_tool_usage_examples(tools)
        
        # Prepare the LLM prompting context
        llm_context = [
            {
                "role": "system", 
                "content": (
                    "You are an expert system prompt engineer for AI agents. "
                    "Your task is to create an effective, clear, and comprehensive system prompt "
                    "for an AI agent that will be using tools to complete tasks."
                )
            },
            {
                "role": "user",
                "content": f"""
Create a comprehensive system prompt for an agent named "{agent_name}" with the following task description:

{task_description}

The agent has access to these tools:
{tool_descriptions}

These tools should be used with the following syntax:
{template["tool_usage"]}

The agent should provide final answers with:
{template["final_answer"]}

Important guidelines for the agent:
- {"\n- ".join(template["guidelines"])}

Additional instructions to include:
{additional_instructions or []}

Here's an example of a good prompt for a different use case:
{self.example_prompts["stock_analysis"] if include_examples else ""}

Now, please create a comprehensive system prompt that:
1. Clearly introduces the agent's purpose and capabilities
2. Explains the available tools accurately
3. Provides clear usage instructions for each tool
4. Includes helpful examples of when to use specific tools
5. Emphasizes providing a final answer in the correct format
6. Includes the important guidelines listed above
7. Incorporates any additional instructions
8. Optimizes for clarity, comprehensiveness, and effectiveness

Structure the prompt with clear sections and appropriate formatting.
"""
            }
        ]
        
        # Generate the prompt using the LLM
        generated_prompt = await self.model.call(llm_context)
        
        if return_json:
            # Create a structured JSON response
            return {
                "agent_name": agent_name,
                "prompt": generated_prompt,
                "metadata": {
                    "task_description": task_description,
                    "additional_instructions": additional_instructions or [],
                    "tools": tool_info["tools"],
                    "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        
        return generated_prompt
    
    async def preview_prompt(
        self,
        agent_name: str,
        tools: List[Tool],
        task_description: str,
        additional_instructions: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        output_format: str = "txt"
    ) -> str:
        """
        Generate and optionally save a prompt to a file for review.
        
        Args:
            agent_name: Name of the agent
            tools: List of Tool objects available to the agent
            task_description: Description of the agent's primary task/purpose
            additional_instructions: Additional custom instructions to include
            output_file: Optional file path to save the generated prompt
            output_format: Format to save the output file in ("txt" or "json")
            
        Returns:
            The generated prompt
        """
        use_json = output_format.lower() == "json"
        
        result = await self.generate_prompt(
            agent_name=agent_name,
            tools=tools,
            task_description=task_description,
            additional_instructions=additional_instructions,
            return_json=use_json
        )
        
        # Extract the prompt text
        prompt = result["prompt"] if use_json else result
        
        # Output to console with formatting
        print("\n" + "="*80)
        print(f"GENERATED PROMPT FOR: {agent_name}")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        # Save to file if requested
        if output_file:
            if use_json:
                # Ensure file has .json extension
                if not output_file.lower().endswith('.json'):
                    output_file = f"{output_file}.json"
                    
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                # Default to text format
                if not output_file.lower().endswith('.txt'):
                    output_file = f"{output_file}.txt"
                    
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(prompt)
                    
            print(f"Prompt saved to: {output_file}")
        
        return prompt


# Example usage:
# async def main():
#     # Initialize the prompt builder
#     builder = AgentPromptBuilder()
#     
#     # Get your tools
#     from agentix.tools import YFinanceToolkit
#     toolkit = YFinanceToolkit()
#     tools = toolkit.get_tools()
#     
#     # Generate a prompt
#     prompt = await builder.preview_prompt(
#         agent_name="FinanceGPT",
#         tools=tools,
#         task_description="Provide detailed financial analysis and stock information to users.",
#         additional_instructions=[
#             "Always use the most recent data available.",
#             "Explain financial terms in plain language if the user seems unfamiliar with them.",
#         ],
#         output_file="finance_agent_prompt.txt"
#     ) 