from dataclasses import dataclass
import re
from typing import List, Optional

from ..llms import LLM
from ..memory.memory import ConversationMessage


@dataclass
class EvaluationResult:
    """
    Result of an evaluation containing score, feedback, and optional improvements.
    """
    score: float  # e.g., 0-1
    feedback: str  # Detailed feedback
    improvements: Optional[str] = None  # Suggested improvements


class SimpleEvaluator:
    """
    A simple evaluator that assesses the quality of assistant responses
    using an LLM.
    """
    
    def __init__(self, model: LLM):
        """
        Initialize the SimpleEvaluator with an LLM model.
        
        Args:
            model: An instance of an LLM (OpenAIChat or TogetherChat) to use for evaluation
        """
        self.model = model
        
    async def evaluate(self, messages: List[ConversationMessage]) -> EvaluationResult:
        """
        Evaluates the last assistant response in the conversation memory.
        
        Args:
            messages: Conversation history
            
        Returns:
            EvaluationResult with score, feedback, and improvements
        """
        print(f"[SimpleEvaluator] Retrieved messages for evaluation: {messages}")
        
        # Identify the last assistant message with "FINAL ANSWER:"
        last_assistant_msg = None
        for msg in reversed(messages):
            if (isinstance(msg, dict) and 
                msg.get("role") == "assistant" and 
                msg.get("content", "").startswith("FINAL ANSWER:")):
                last_assistant_msg = msg
                break
            elif (hasattr(msg, "role") and 
                  msg.role == "assistant" and 
                  msg.content.startswith("FINAL ANSWER:")):
                last_assistant_msg = msg
                break
                
        if not last_assistant_msg:
            print("[SimpleEvaluator] No valid assistant message found to evaluate.")
            return EvaluationResult(
                score=0,
                feedback="No valid assistant response found to evaluate.",
                improvements="Ensure the assistant provides a response to the user query."
            )
            
        # Extract content based on message type
        if isinstance(last_assistant_msg, dict):
            assistant_response = last_assistant_msg.get("content", "").replace("FINAL ANSWER:", "").strip()
        else:
            assistant_response = last_assistant_msg.content.replace("FINAL ANSWER:", "").strip()
            
        prompt = [
            {"role": "system", "content": "You are an AI evaluator that critiques assistant responses."},
            {
                "role": "user",
                "content": f"""Evaluate the following assistant response:

"{assistant_response}"

Please provide:
1. A numeric score (0-1) assessing the quality and relevance of the response.
2. Detailed feedback about what the response did well or poorly.
3. Suggestions for improvements, if any. If no improvements are needed, leave the field blank.

Structure your response as follows:
Score: <numeric value>
Feedback: <detailed feedback>
Improvements: <suggested improvements>"""
            }
        ]
        
        eval_response = await self.model.call(prompt)
        
        print(f"[SimpleEvaluator] Raw evaluation response: {eval_response}")
        
        # Parse evaluation results
        score_match = re.search(r"Score:\s*([\d.]+)", eval_response, re.IGNORECASE)
        feedback_match = re.search(r"Feedback:\s*([\s\S]+?)(?=Improvements:)", eval_response, re.IGNORECASE)
        improvements_match = re.search(r"Improvements:\s*([\s\S]+)", eval_response, re.IGNORECASE)
        
        return EvaluationResult(
            score=float(score_match.group(1)) if score_match else 0,
            feedback=feedback_match.group(1).strip() if feedback_match else "No feedback provided.",
            improvements=improvements_match.group(1).strip() if improvements_match else None
        ) 