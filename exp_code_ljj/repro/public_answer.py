"""Final-answer boundaries for preference simulation, before any next prompt."""
import re
from pathlib import Path

THINK = re.compile(r"</?think\b", re.I)
POLICY = "final_answer_only"


def strip_thinking(value):
    filtered = bool(THINK.search(value))
    if filtered and not re.search(r"<think\b", value, re.I):
        value = re.split(r"</think\b[^>]*>", value, flags=re.I)[-1]
    value = re.sub(r"<think\b[^>]*>.*?</think\b[^>]*>", "", value, flags=re.I | re.S)
    value = re.sub(r"<think\b[^>]*>.*\Z", "", value, flags=re.I | re.S)
    value = re.sub(r"</think\b[^>]*>", "", value, flags=re.I)
    return value.strip(), filtered


def public_agent_answer(endpoint, *, system, user, decode):
    """Clear private text before a MA2P JSON answer reaches another agent."""
    filtered = False
    for attempt in range(4):
        retry = '' if attempt == 0 else (
            '\nThe previous completion had no public answer. Return only the '
            'requested final JSON object, without reasoning or <think> tags.'
        )
        raw = endpoint.complete(system=system, user=user + retry, **decode)
        visible, found = strip_thinking(raw)
        filtered = filtered or found
        if visible and not THINK.search(visible):
            return visible, filtered, attempt
    raise ValueError('MA2P agent returned no final-answer content after 3 retries')


def public_turn(raw, *, seed, role, endpoint, is_isr=None, force_rewrite=False):
    from .persuadee_simulation import clean_utterance, _direct_rewrite
    name = str(seed["scenario"][role])
    filtered = bool(THINK.search(raw))
    visible, _ = strip_thinking(raw)
    try:
        content = clean_utterance(visible, is_isr=role == "persuader" if is_isr is None else is_isr, speaker_name=name)
    except ValueError:
        content = ""
    retries = 0
    if not content or force_rewrite:
        content = ''
        for _ in range(3):
            retries += 1
            try:
                content, rewritten = _direct_rewrite(endpoint, raw, seed, name,
                                                     speaker_role=role.title(), allow_empty=True)
                visible, found = strip_thinking(rewritten)
                filtered = filtered or found
                content = clean_utterance(visible, is_isr=False, speaker_name=name)
            except ValueError:
                continue
            filtered = filtered or bool(THINK.search(rewritten))
            if content:
                break
    if not content or THINK.search(content):
        raise ValueError(f"No final-answer {role} reply for scenario {seed['scenario_id']}")
    return {"role": role, "content": content, "thinking_policy": POLICY,
            "thinking_filtered": filtered, "public_answer_retries": retries,
            "direct_answer_rewritten": retries > 0}


def model_preference_directory(model, data_dir):
    model = Path(model).expanduser().resolve()
    adapter_root = Path("/app/hzj/liujianjian/persuader_atapter_save").resolve()
    if (model.name == "baseline3_isr_sft" and model.parent.parent == adapter_root
            and model.parent.name in {"meta_llama31", "deepseek_r1_qwen14b"}):
        return Path(data_dir).parent / "artifacts/preference_simulations" / model.parent.name
    return Path(data_dir)
