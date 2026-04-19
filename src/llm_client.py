import json
import re
import logging
from typing import Any, Dict

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Ollama Client ──────────────────────────────────────────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

DEFAULT_MODEL = "gemma4:31b"
NUM_CTX = 16384
STORY_NUM_CTX = 32768
STORY_MAX_TOKENS = 7000
REPAIR_MAX_TOKENS = 8000

# ── Repetition Detection ──────────────────────────────────────────────

def _detect_repetition(
    text: str,
    *,
    min_pat_len: int = 8,
    min_repeats: int = 6,
    max_field_len: int = 4000,
    check_field_lengths: bool = True,
) -> bool:
    """
    Returns True if pathological repetition is found.
    Checks:
      1. Any 8+ char substring repeated ≥6 times consecutively.
      2. Any JSON string-value field exceeding max_field_len chars when check_field_lengths=True.
    """
    # Check consecutive substring repetition  (e.g. "로로로로로", "////////")
    pattern = re.compile(r"(.{" + str(min_pat_len) + r",}?)\1{" + str(min_repeats - 1) + r",}")
    if pattern.search(text):
        return True

    # Check excessively long JSON string values
    if check_field_lengths:
        try:
            obj = json.loads(text)
            return _check_field_lengths(obj, max_field_len)
        except (json.JSONDecodeError, TypeError):
            pass

    return False


def _check_field_lengths(obj: Any, max_len: int) -> bool:
    """Recursively check if any string field exceeds max_len."""
    if isinstance(obj, str):
        return len(obj) > max_len
    if isinstance(obj, dict):
        return any(_check_field_lengths(v, max_len) for v in obj.values())
    if isinstance(obj, list):
        return any(_check_field_lengths(v, max_len) for v in obj)
    return False


# ── JSON Repair Pass ──────────────────────────────────────────────────

def _repair_json_pass(
    broken_content: str,
    model: str = DEFAULT_MODEL,
    *,
    max_tokens: int = REPAIR_MAX_TOKENS,
    num_ctx: int = STORY_NUM_CTX,
) -> Dict[str, Any]:
    """
    Sends the broken JSON back to the LLM with a repair-only prompt.
    Temperature 0, deterministic, short.
    """
    # Truncate to 6000 chars max so larger StoryJson payloads can still be repaired.
    truncated = broken_content[:6000]
    repair_prompt = (
        "The following JSON is broken (unterminated strings, missing brackets, or invalid syntax). "
        "Fix it to be valid JSON. Output ONLY the repaired JSON object, nothing else.\n\n"
        f"```\n{truncated}\n```"
    )
    try:
        logger.info("Attempting JSON repair pass...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON repair tool. Output ONLY valid JSON."},
                {"role": "user", "content": repair_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"options": {"num_ctx": num_ctx}},
        )
        content = response.choices[0].message.content or ""
        result = json.loads(content)
        if _detect_repetition(content):
            logger.warning("Repair pass output still contains repetition. Discarding.")
            return {}
        logger.info("JSON repair pass succeeded.")
        return result
    except Exception as e:
        logger.error(f"JSON repair pass failed: {e}")
        return {}


# ── Extract Fallback (bracket matching) ───────────────────────────────

def _extract_json_fallback(content: str) -> Dict[str, Any]:
    """Last-resort: find outermost { ... } and try to parse."""
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            return json.loads(content[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


# ── Core LLM Call (internal) ──────────────────────────────────────────

def _call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 512,
    num_ctx: int = NUM_CTX,
    top_p: float = 1.0,
    json_mode: bool = True,
) -> str:
    """Raw LLM call returning the content string."""
    extra_body = {"options": {"num_ctx": num_ctx}}
    
    response_format = None
    if json_mode:
        response_format = {"type": "json_object"}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        extra_body=extra_body,
    )
    raw = response.choices[0].message.content or ""
    logger.info(f"Raw LLM output length: {len(raw)} chars. Preview: {raw[:300]}")
    return raw


def _parse_with_retry(
    raw: str,
    *,
    model: str = DEFAULT_MODEL,
    repair_max_tokens: int = REPAIR_MAX_TOKENS,
    repair_num_ctx: int = STORY_NUM_CTX,
) -> Dict[str, Any]:
    """Parse JSON from raw string. On failure, try extract fallback, then repair pass."""
    # 1. Direct parse
    try:
        result = json.loads(raw)
        if not _detect_repetition(raw):
            return result
        logger.warning("Direct parse succeeded but repetition detected. Discarding.")
    except json.JSONDecodeError:
        pass

    # 2. Bracket-match fallback
    fallback = _extract_json_fallback(raw)
    if fallback:
        fallback_text = json.dumps(fallback, ensure_ascii=False)
        if not _detect_repetition(fallback_text):
            logger.info("Bracket-match fallback succeeded.")
            return fallback
        logger.warning("Bracket-match fallback produced repetitive payload. Discarding.")

    # 3. LLM repair pass
    repaired = _repair_json_pass(raw, model=model, max_tokens=repair_max_tokens, num_ctx=repair_num_ctx)
    if repaired:
        return repaired

    return {}


# ── Public API ────────────────────────────────────────────────────────

def ask_llm_decision(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Short tool-selection call.
    Returns: {"thought": "...", "tool_to_use": "..."}
    Temperature 0.1, max 512 tokens — virtually impossible to break.
    """
    try:
        logger.info(f"[Decision] Calling LLM ({model})")
        raw = _call_llm(system_prompt, user_prompt, model=model, temperature=0.1, max_tokens=512)
        result = _parse_with_retry(raw, model=model)
        if result:
            return result
        logger.error("[Decision] All parse attempts failed.")
        return {"thought": "Failed to parse decision JSON.", "tool_to_use": "none"}
    except Exception as e:
        logger.error(f"[Decision] LLM call error: {e}")
        return {"thought": f"LLM error: {e}", "tool_to_use": "error"}


def ask_llm_generate(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = STORY_MAX_TOKENS,
    num_ctx: int = STORY_NUM_CTX,
    json_mode: bool = False,
) -> Any:
    """
    Long content-generation call (e.g. story body).
    By default, now uses json_mode=False for structured text extraction.
    If json_mode=True, it returns a parsed JSON dict.
    Otherwise, returns the raw string.
    """
    try:
        logger.info(f"[Generate] Calling LLM ({model}) | JSON Mode: {json_mode}")
        raw = _call_llm(
            system_prompt, user_prompt,
            model=model,
            temperature=0.6,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            top_p=0.9,
            json_mode=json_mode,
        )
        
        if json_mode:
            result = _parse_with_retry(
                raw,
                model=model,
                repair_max_tokens=max_tokens,
                repair_num_ctx=num_ctx,
            )
            if result:
                return result
            logger.error("[Generate] All parse attempts failed.")
            return {}
        
        return raw
    except Exception as e:
        logger.error(f"[Generate] LLM call error: {e}")
        return {} if json_mode else ""


# ── Legacy compat shim (used by run_turn_loop) ────────────────────────

def ask_agentic_llm_json(
    system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Backward-compatible wrapper. Uses the decision path."""
    return ask_llm_decision(system_prompt, user_prompt, model=model)
