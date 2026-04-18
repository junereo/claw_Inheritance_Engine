import json
import logging
from typing import Any, Dict, Optional
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Ollama Client
# Using the local Ollama server which provides an OpenAI-compatible /v1 endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real API key but the SDK requires one
)

def ask_agentic_llm_json(system_prompt: str, user_prompt: str, model: str = "gemma4:31b") -> Dict[str, Any]:
    """
    Sends a request to the local Ollama LLM expecting a JSON response.
    Enforces JSON structure and provides a fallback mechanism for parsing errors.
    """
    try:
        logger.info(f"Calling Ollama LLM ({model}) with system prompt length: {len(system_prompt)}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,          # 0.1 이하라면 0.3~0.4 정도로 살짝 올려 반복을 깹니다.
            max_tokens=4096,          # 출력이 잘리지 않도록 충분히 큰 값을 줍니다.
            presence_penalty=0.5,     # 무한 반복 방지 옵션 (지원되는 경우)
            frequency_penalty=0.5     # 무한 반복 방지 옵션 (지원되는 경우)
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
            
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}. Raw content: {content}")
            # Fallback for hallucinated JSON
            return _extract_json_fallback(content)
            
    except Exception as e:
        logger.error(f"LLM Call Error: {e}")
        return {
            "thought": f"Error occurred during LLM call: {str(e)}",
            "tool_to_use": "error",
            "tool_payload": {"error": str(e)}
        }

def _extract_json_fallback(content: str) -> Dict[str, Any]:
    """
    Attempts to extract JSON from a string if the LLM wrapped it in text or markdown blocks.
    """
    try:
        # Try to find the first '{' and last '}'
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
    except:
        pass
        
    return {
        "thought": "Failed to parse JSON even with fallback.",
        "tool_to_use": "none",
        "tool_payload": {}
    }
