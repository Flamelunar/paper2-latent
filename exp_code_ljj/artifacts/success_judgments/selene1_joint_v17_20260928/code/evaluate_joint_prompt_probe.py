#!/usr/bin/env python3
"""Judge five-round success and Win-Rate in one local-model call per sample."""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
import json
import re
from pathlib import Path
from typing import Any

from repro.profile import configure_profile_from_argv

configure_profile_from_argv()
from repro.constants import ROOT
from repro.simulation import LocalTransformersEndpoint, read_jsonl, transcript
from evaluate_baselines import ARCHIVED_METHODS, load_generation
from evaluate_success_prompt_probe import read_manifest


OUTPUT_DIR = ROOT / "artifacts/joint_success_win_probe"
PROMPT_REVISIONS = {"v1": "joint_v1_r1", "v2": "joint_v2_r2", "v3": "joint_v2_r3", "v4": "joint_v2_r4", "v5": "joint_v2_r5", "v6": "joint_v2_r6", "v7": "joint_v2_r7", "v8": "joint_v2_r8", "v9": "joint_v2_r9", "v10": "joint_v2_r10", "v11": "joint_v2_r11", "v12": "joint_selene_success_strict_v1", "v13": "joint_selene_success_action_v2", "v14": "joint_selene_success_format_v3", "v15": "joint_selene_success_compact_v4", "v16": "joint_selene_success_compact_pilot_v5", "v17": "joint_selene_success_literal_v6", "gemma_v11_format_v1": "joint_gemma_v11_format_v1"}


SELENE_SUCCESS_RULE_V12 = (
    "Use a strict threshold. For each R1-R5 separately, mark TRUE only when the "
    "Persuadee's own reply clearly accepts or unconditionally commits to doing the exact "
    "goal action. Use the stated goal to resolve a clear pronoun such as 'it', but do not "
    "infer acceptance from the Persuader's claims or from positive tone. Mark FALSE for "
    "interest, praise, thanks, questions, requests for details, 'I'll think about it', "
    "'maybe', 'probably', or a promise to decide later. A conditional promise is FALSE "
    "while its condition remains unresolved. Agreeing only to consider, research, view, "
    "discuss, attend an information session, or take another preliminary step is FALSE "
    "unless that exact step is the goal. If the goal requires multiple actions, every "
    "required action must be accepted. Treat hedged or ambiguous wording as FALSE. For "
    "TRUE, copy the shortest exact commitment from that same R line into round_evidence; "
    "otherwise use an empty string. Judge only the Persuadee replies and do not infer "
    "unspoken intent."
)

V11_SUCCESS_RULE = (
    "For each R1-R5, is there a clear agreement to perform or try the exact goal? "
    "A specific plan to perform the goal also counts. Ordinary scheduling details do not cancel agreement. "
    "Merely considering, asking, researching, viewing or attending an information meeting before deciding does not count as choosing, buying or supporting. "
    "If the goal includes two actions, both must be agreed to. Judge each R independently. "
    "If true, copy the shortest exact commitment from that same R line into round_evidence; "
    "if false use an empty string."
)

SUCCESS_RULES_V12 = {"selene1": SELENE_SUCCESS_RULE_V12, "gemma4": V11_SUCCESS_RULE}

SELENE_SUCCESS_RULE_V13 = (
    "Apply a literal acceptance test, not an optimism test. For each R1-R5, mark TRUE "
    "only when the Persuadee clearly decides to do or start the exact action requested by "
    "the GOAL. This measures acceptance, so the action need not already be completed. "
    "A reply that only considers, researches, learns about, reviews information, asks "
    "questions, discusses options, attends an information session, or agrees to a pilot "
    "for evaluation is FALSE for a later action goal such as investing, supporting, buying, "
    "applying, enrolling, or taking a trip. 'Open to it', 'maybe', 'might', 'probably', "
    "'I'll think about it', 'I'll get back to you', or a plan to decide later are FALSE. "
    "A conditional promise is FALSE while the condition is unresolved. A plan or promise "
    "to try counts only when it clearly refers to the exact goal action, not to research or "
    "a preliminary step. If the GOAL itself is to consider, explore, or learn, a clear "
    "commitment to that exact activity may count; do not infer a later decision beyond it. "
    "If the GOAL requires multiple actions, all must be accepted. If there is any doubt "
    "whether the reply says yes to the exact goal, mark FALSE. For TRUE, copy the shortest "
    "exact commitment from that same R line into round_evidence; for FALSE use an empty "
    "string. Examples: for 'apply for a scholarship', 'I'll submit an application' is TRUE "
    "and 'it might be worth a shot' is FALSE; for 'support a bill', 'I support it' is TRUE "
    "and 'I'll review the summary and consider my stance' is FALSE; for 'invest in a "
    "project', 'I'll invest' is TRUE and 'let's research projects' is FALSE. Judge only "
    "the Persuadee's own words."
)
SUCCESS_RULES_V13 = {"selene1": SELENE_SUCCESS_RULE_V13, "gemma4": V11_SUCCESS_RULE}


def joint_prompt(row: dict[str, Any], prompt_version: str, judge_name: str) -> str:
    scenario = row["scenario"]
    turns = row["trajectory"]
    if len(turns) != 10 or any(turns[i]["role"] != ("persuader" if i % 2 == 0 else "persuadee") for i in range(10)):
        raise ValueError(f"Expected five complete rounds at index {row.get('index')}")
    dialogue = "\n".join(
        f"Round {i + 1} Persuader: {turns[2 * i]['content']}\n"
        f"Round {i + 1} Persuadee: {turns[2 * i + 1]['content']}"
        for i in range(5)
    )
    if prompt_version == "v1":
        policy = (
            "For success, require the persuadee to clearly and unconditionally accept, "
            "commit to, or start the exact goal in their own words. Interest, questions, "
            "thinking about it, tentative or conditional promises, and intermediate steps "
            "do not count. Ambiguous evidence is failure."
        )
    elif prompt_version in {"v12", "v13", "v14", "v15", "v16", "v17"}:
        rule_set = SUCCESS_RULES_V12 if prompt_version == "v12" else SUCCESS_RULES_V13
        if judge_name not in rule_set:
            raise ValueError(f"No {prompt_version} success prompt configured for Judge: {judge_name}")
        policy = rule_set[judge_name]
    elif prompt_version in {"v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "gemma_v11_format_v1"}:
        policy = (
            "For success, first name the exact target action. Compare it with what the "
            "persuadee actually decides to do in each round. A clear agreement to perform "
            "or try the target action counts; uncertainty about enjoying the outcome or an "
            "ordinary scheduling detail does not undo that agreement. Earlier hesitation "
            "does not override a later clear decision. A later explicit retraction does. "
            "Mere interest, questions, 'I'll think about it', an unresolved condition, "
            "or agreeing only to research, view, meet, or discuss before deciding does "
            "not count unless that preliminary action is itself the goal. Viewing artwork "
            "does not mean buying it; researching insurance does not mean purchasing it. "
            "When the goal is merely to consider or learn, a clear commitment to consider "
            "or learn can count. Use only the persuadee's own words."
        )
    else:
        raise ValueError(f"Unknown prompt version: {prompt_version}")
    if prompt_version == "v5":
        replies = "\n".join(f"R{i + 1}: {turns[2 * i + 1]['content']}" for i in range(5))
        generated = "\n".join(f"G{i + 1}: {turns[2 * i]['content']}" for i in range(5))
    return f"""GOAL: {scenario['goal']}
PERSUADEE'S FIVE REPLIES:
{replies}

GENERATED PERSUADER'S FIVE REPLIES:
{generated}
GOLDEN PERSUADER REPLY: {row['reference']}

Make two judgments in one answer.
1. For each R1-R5, answer true ONLY if that reply explicitly commits to doing the GOAL. Considering, researching, asking, or planning to view something does not mean buying/donating. "I'll do it" counts only when its referent is clearly the GOAL. Use the five R lines, never the G lines.
2. Which persuasion is stronger: all five G replies together or the golden reply? For ab use A=G, B=golden. For ba use A=golden, B=G. Give one A/B winner per order.

Answer with one JSON object containing only: round_success (five JSON booleans), win_ab (A/B), win_ba (A/B)."""
    if prompt_version == "gemma_v11_format_v1":
        if judge_name != "gemma4":
            raise ValueError("gemma_v11_format_v1 is only for Gemma 4")
        persuadee_lines = "\n".join(f"R{i + 1}: {turns[2 * i + 1]['content']}" for i in range(5))
        persuader_lines = "\n".join(f"G{i + 1}: {turns[2 * i]['content']}" for i in range(5))
        return f"""Judge the goal: {scenario['goal']}
Background: {scenario['background']}

PERSUADEE replies for success judgments:
{persuadee_lines}

{V11_SUCCESS_RULE}

Persuasive quality comparison 1 (ignore persuadee replies):
A: {persuader_lines}
B: {row['reference']}
Choose A or B and write that one letter in win_ab.

Persuasive quality comparison 2 (ignore persuadee replies):
A: {row['reference']}
B: {persuader_lines}
Choose A or B and write that one letter in win_ba.
In both comparisons, judge which persuader text is more relevant, convincing and responsible. Do not choose based on length. Each winner must be a SINGLE letter.

Return exactly one valid JSON object, with no markdown, explanation, or extra keys:
{{"round_evidence":["","I'll attend the class.","","",""],"round_success":[false,true,false,false,false],"win_ab":"A","win_ba":"B"}}
The object above illustrates the FORMAT ONLY, not the answers. Judge this case independently. Both arrays must have exactly five items in R1-R5 order. Use JSON booleans, not quoted strings. For each true value, copy a short contiguous quote from that SAME numbered R line into the matching round_evidence item. For each false value, use an empty string. win_ab and win_ba must each be exactly one uppercase letter, A or B. Check array length, item alignment, quote source, and JSON syntax before responding."""
    if prompt_version in {"v15", "v16", "v17"}:
        persuadee_lines = "\n".join(f"R{i + 1}: {turns[2 * i + 1]['content']}" for i in range(5))
        persuader_lines = "\n".join(f"G{i + 1}: {turns[2 * i]['content']}" for i in range(5))
        pilot_rule = ("A small pilot, trial, or evaluation before deciding is FALSE when the goal is to invest in, purchase, adopt, or support the full proposal. The words 'give it a try' count only when the trial itself fulfills the exact goal.\n" if prompt_version in {"v16", "v17"} else "")
        pilot_example = ('Goal: invest in a workplace wellness program; R1: "Let us try a small pilot first and evaluate it before making a long-term commitment." -> {"first_success_round":0,"success_quote":"","win_ab":"B","win_ba":"A"}\n' if prompt_version in {"v16", "v17"} else "")
        literal_rule = ("Before marking a round positive, ask whether the quoted words themselves state a decision to DO the full goal. Merely appreciating a benefit, asking how to do it, promising to think more, looking into options, or asking someone else for permission is NOT acceptance. A clear promise to do only a preparatory action is NOT acceptance of a later goal. A conditional 'I'll try if possible' is NOT acceptance. If the quote can be read as either interest or commitment, choose 0 unless a different R line explicitly commits. Do not infer success from the Persuader's G line or a sympathetic tone.\n" if prompt_version == "v17" else "")
        literal_examples = ('Goal: support a policy; R1: "That is interesting. I will think more about it." -> {"first_success_round":0,"success_quote":"","win_ab":"B","win_ba":"A"}\nGoal: arrange long-term care; R1: "How would I go about arranging that?" -> {"first_success_round":0,"success_quote":"","win_ab":"B","win_ba":"A"}\nGoal: take a week off; R1: "I will ask my boss whether I could take leave." -> {"first_success_round":0,"success_quote":"","win_ab":"B","win_ba":"A"}\nGoal: apply for a grant; R1: "I will submit the application this week." -> {"first_success_round":1,"success_quote":"I will submit the application this week.","win_ab":"A","win_ba":"B"}\n' if prompt_version == "v17" else "")
        return f"""Judge the goal: {scenario['goal']}
Background: {scenario['background']}

PERSUADEE replies for success judgments:
{persuadee_lines}

{SUCCESS_RULES_V13[judge_name]}
{pilot_rule}
{literal_rule}

Output only the FIRST round whose R line clearly accepts the exact goal. If none does, use 0 and an empty quote. The quote must be a short, contiguous, verbatim substring of that numbered R line. Do not quote a G line or combine words from different lines.
Format examples (the A/B letters illustrate syntax only; decide the winners below independently):
Goal: apply for a scholarship; R1: "I'll read the eligibility rules."; R2: "I'll submit the application tomorrow." -> {{"first_success_round":2,"success_quote":"I'll submit the application tomorrow.","win_ab":"A","win_ba":"B"}}
Goal: support a bill; R1: "I'll review the summary and consider my stance."; R2: "Maybe later." -> {{"first_success_round":0,"success_quote":"","win_ab":"B","win_ba":"A"}}
{pilot_example}
{literal_examples}

Persuasive quality comparison 1 (ignore persuadee replies):
A: {persuader_lines}
B: {row['reference']}
Choose A or B and write that one letter in win_ab.

Persuasive quality comparison 2 (ignore persuadee replies):
A: {row['reference']}
B: {persuader_lines}
Choose A or B and write that one letter in win_ba.
In both comparisons, judge which persuader text is more relevant, convincing and responsible. Do not choose based on length. Each winner must be a SINGLE letter.

Return exactly one JSON object with exactly four keys: first_success_round (integer 0, 1, 2, 3, 4, or 5), success_quote (string), win_ab ("A" or "B"), win_ba ("A" or "B"). No arrays, markdown, or explanation. For a positive round, copy one exact quote from that same R line. For 0, use an empty quote."""
    if prompt_version == "v14":
        persuadee_lines = "\n".join(
            f"R{i + 1}: {turns[2 * i + 1]['content']}" for i in range(5)
        )
        persuader_lines = "\n".join(
            f"G{i + 1}: {turns[2 * i]['content']}" for i in range(5)
        )
        return f"""Judge the goal: {scenario['goal']}
Background: {scenario['background']}

PERSUADEE replies for success judgments:
{persuadee_lines}

{SUCCESS_RULES_V13[judge_name]}

Examples for the success rule and output format only; do not copy their answers:
- Goal: apply for a scholarship. Reply: "I'll read the eligibility rules." Decision: false, evidence: "".
- Goal: apply for a scholarship. Reply: "I'll submit the application tomorrow." Decision: true, evidence: "I'll submit the application tomorrow.".
- Goal: support a bill. Reply: "I'll review the summary and consider my stance." Decision: false, evidence: "".
- Goal: support a bill. Reply: "I support this bill." Decision: true, evidence: "I support this bill.".
If only R2 says "I'll submit the application tomorrow.", the JSON shape is:
{{"round_evidence":["","I'll submit the application tomorrow.","","",""],"round_success":[false,true,false,false,false],"win_ab":"A","win_ba":"B"}}
The example's A/B letters illustrate syntax only. Decide the winners for this case independently.

Persuasive quality comparison 1 (ignore persuadee replies):
A: {persuader_lines}
B: {row['reference']}
Choose A or B and write that one letter in win_ab.

Persuasive quality comparison 2 (ignore persuadee replies):
A: {row['reference']}
B: {persuader_lines}
Choose A or B and write that one letter in win_ba.
In both comparisons, judge which persuader text is more relevant, convincing and responsible. Do not choose based on length. Each winner must be a SINGLE letter.

Return exactly one JSON object with exactly four keys: round_evidence, round_success, win_ab, win_ba. Do not add markdown or explanation. Both arrays must contain exactly five items in R1-R5 order; array item 1 belongs to R1, item 2 to R2, and so on. Use JSON booleans true/false, not strings. For every true value, its evidence must be a contiguous verbatim quote from that SAME R line. For every false value, use an empty evidence string. If no quote in that R line proves acceptance of the exact goal, mark it false. Each win value must be one uppercase letter A or B. Check the R-line and array-position alignment before answering."""
    if prompt_version in {"v12", "v13"}:
        persuadee_lines = "\n".join(
            f"R{i + 1}: {turns[2 * i + 1]['content']}" for i in range(5)
        )
        persuader_lines = "\n".join(
            f"G{i + 1}: {turns[2 * i]['content']}" for i in range(5)
        )
        success_rule = (SUCCESS_RULES_V12 if prompt_version == "v12" else SUCCESS_RULES_V13)[judge_name]
        return f"""Judge the goal: {scenario['goal']}
Background: {scenario['background']}

PERSUADEE replies for success judgments:
{persuadee_lines}

{success_rule}

Persuasive quality comparison 1 (ignore persuadee replies):
A: {persuader_lines}
B: {row['reference']}
Choose A or B and write that one letter in win_ab.

Persuasive quality comparison 2 (ignore persuadee replies):
A: {row['reference']}
B: {persuader_lines}
Choose A or B and write that one letter in win_ba.
In both comparisons, judge which persuader text is more relevant, convincing and responsible. Do not choose based on length. Each winner must be a SINGLE letter.

Return only JSON with four keys: round_evidence (five strings), round_success (five true/false values), win_ab (A/B), win_ba (A/B)."""
    if prompt_version in {"v4", "v6", "v7", "v8", "v9", "v10", "v11"}:
        persuadee_lines = "\n".join(
            f"R{i + 1}: {turns[2 * i + 1]['content']}" for i in range(5)
        )
        persuader_lines = "\n".join(
            f"G{i + 1}: {turns[2 * i]['content']}" for i in range(5)
        )
        if prompt_version in {"v7", "v8", "v9", "v10", "v11"}:
            success_rule = (
                "For each R1-R5, did the Persuadee clearly commit to doing or trying the exact GOAL in that reply? "
                "Count 'I'll try it', 'Let's do it', or concrete planning of the goal when the referent is clear. "
                "Doubt about whether it will be enjoyable, an ordinary scheduling detail, or earlier hesitation does not cancel a clear commitment. "
                "Do not count considering, researching, asking, viewing, or scheduling a tour as choosing or buying. "
                "If the goal requires two actions, agreement to only one action is not enough. "
                "A promise to try stress techniques counts if trying them is the goal. "
                "For each true decision, quote the shortest exact commitment from that SAME R line in round_evidence; "
                "for false use an empty string. Judge each R independently."
                if prompt_version == "v8" else
                "For each R1-R5, is there a clear agreement to perform or try the exact goal? "
                "A specific plan to perform the goal also counts. Ordinary scheduling details do not cancel agreement. "
                "Merely considering, asking, researching, viewing or attending an information meeting before deciding does not count as choosing, buying or supporting. "
                "If the goal includes two actions, both must be agreed to. Judge each R independently. "
                "If true, copy the shortest exact commitment from that same R line into round_evidence; "
                "if false use an empty string."
                if prompt_version in {"v9", "v10", "v11"} else
                "For each R1-R5, is there an explicit commitment to perform the exact goal in that reply? "
                "Considering, asking, researching or viewing before deciding is not buying or donating. "
                "Mark each reply independently. If true, quote the shortest exact commitment from that same R line "
                "in round_evidence; otherwise give an empty string."
            )
            return f"""Judge the goal: {scenario['goal']}
Background: {scenario['background']}

PERSUADEE replies for success judgments:
{persuadee_lines}

{success_rule}

Persuasive quality comparison 1 (ignore persuadee replies):
A: {persuader_lines}
B: {row['reference']}
Choose A or B and write that one letter in win_ab.

Persuasive quality comparison 2 (ignore persuadee replies):
A: {row['reference']}
B: {persuader_lines}
Choose A or B and write that one letter in win_ba.
In both comparisons, judge which persuader text is more relevant, convincing and responsible. Do not choose based on length. Each winner must be a SINGLE letter.

Return only JSON with four keys: round_evidence (five strings), round_success (five true/false values), win_ab (A/B), win_ba (A/B)."""
        quality_rule = (
            "Compare all five G lines together against the golden reply. Answer twice. "
            "First comparison: A is generated G, B is golden; put the ONE LETTER winner "
            "A or B in win_ab. Second comparison: A is golden, B is generated G; put the "
            "ONE LETTER winner A or B in win_ba. Each field must contain exactly one "
            "letter. Never write AB or BA. Ignore the R lines for these comparisons."
            if prompt_version == "v6" else
            "Compare all five G lines as one candidate with the single golden reply. "
            "Ignore the R lines. Choose the stronger persuasion, not the longer text. "
            "AB: A=G, B=golden. BA: A=golden, B=G. Return one A/B choice for each order."
        )
        return f"""Judge the exact goal: {scenario['goal']}
Background: {scenario['background']}

PERSUADEE replies (only these can prove success):
{persuadee_lines}

Generated PERSUADER messages (only these enter the quality comparison):
{persuader_lines}
Golden Persuader reply: {row['reference']}

Success rule: In each R line, did the Persuadee explicitly agree to perform the exact goal? "I'll consider it", "I'll look into it", questions, and agreeing only to view something are FALSE when the goal is to buy or donate. "I'll buy it" or "I'll donate" are TRUE for those goals. Judge each R independently. For TRUE, copy the shortest verbatim commitment from that SAME R line into round_evidence. For FALSE, use an empty string. Never quote G lines or invent evidence.

Quality rule: {quality_rule}

Return ONLY one JSON object with these keys: round_evidence (five strings), round_success (five true/false values), win_ab (A or B), win_ba (A or B). Give no preset answer."""
    if prompt_version == "v3":
        return f"""You are judging one persuasion case. Read the GOAL literally.
GOAL: {scenario['goal']}
Background: {scenario['background']}
Earlier dialogue: {transcript(row.get('initial_history', []))}

{dialogue}

Golden reply: {row['reference']}

Task 1, success: For each round decide whether the Persuadee has explicitly agreed to the GOAL by that round. Use ONLY the Persuadee's words, never the Persuader's words. For each round_evidence item, copy a SHORT EXACT QUOTE from that round's Persuadee line. If the goal is buying, merely viewing or considering is false. If the goal is donating, merely considering or volunteering is false. "I'll think about it" is false; "I'll buy it" is true. Judge each round separately; a later retraction makes that later round false. {policy}

Task 2, quality: Compare ALL FIVE Persuader lines together against the ONE golden reply. Ignore the Persuadee lines for this task. In AB, A=generated and B=golden. In BA, A=golden and B=generated. Judge each order separately; choose A or B, no tie. Prefer relevant, convincing, responsible persuasion, not length.

Return ONLY compact JSON with exactly these keys: "round_evidence" (five short exact Persuadee quotes), "round_success" (five true/false values), "win_ab" ("A" or "B"), "win_ba" ("A" or "B")."""
    return f"""Evaluate two metrics for the SAME persuasion case in ONE response.

Background: {scenario['background']}
Goal: {scenario['goal']}
Persuader: {scenario['persuader']}
Persuadee: {scenario['persuadee']}
Dialogue before generation:
{transcript(row.get('initial_history', []))}

Generated five-round dialogue:
{dialogue}

Held-out golden Persuader reply:
{row['reference']}

SUCCESS: {policy} For each generated round, judge whether the goal has been accepted by the end of that round. Give five short persuadee quotes or reasons, then five Boolean judgments in round order.

WIN-RATE: Compare the persuasive quality of ALL FIVE generated Persuader messages above with the single held-out golden Persuader reply. Consider relevance, concrete reasoning, respectful tone, and factual responsibility. Ignore response length and the simulated Persuadee reactions. Choose a winner; no tie.
Comparison AB: Response A is the five generated Persuader messages; Response B is the golden reply.
Comparison BA: Response A is the golden reply; Response B is the five generated Persuader messages.
Judge both label orders and return the A/B winner for each.

Return only one compact JSON object with these keys in order: goal_action (short string), round_evidence (array of five short strings), round_success (array of five JSON Booleans), win_ab (A or B), win_ba (A or B). Do not copy a preset answer."""


def parse_joint(raw: str, *, require_evidence: bool = True, compact: bool = False) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S).strip()
    decoder = json.JSONDecoder()
    parsed: dict[str, Any] | None = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        raise ValueError("No parseable JSON object")
    if compact:
        first = parsed.get("first_success_round")
        if type(first) is not int or first not in range(6):
            raise ValueError("first_success_round must be an integer from 0 to 5")
        quote = parsed.get("success_quote")
        if not isinstance(quote, str) or (first == 0 and quote.strip()) or (first > 0 and not quote.strip()):
            raise ValueError("success_quote must be empty for 0 and nonempty for a positive round")
        win_ab = parsed.get("win_ab")
        win_ba = parsed.get("win_ba")
        if win_ab not in {"A", "B"} or win_ba not in {"A", "B"}:
            raise ValueError("Missing A/B winner for either order")
        rounds = [number == first for number in range(1, 6)]
        evidence = [quote if number == first else "" for number in range(1, 6)]
        return {
            "goal_action": "",
            "round_evidence": evidence,
            "round_success": rounds,
            "first_success_round": first or None,
            "success_within_5_rounds": first > 0,
            "win_ab": win_ab,
            "win_ba": win_ba,
            "win_credit": ((win_ab == "A") + (win_ba == "B")) / 2,
        }
    values = parsed.get("round_success")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("round_success is not a five-item array")
    rounds: list[bool] = []
    for value in values:
        if isinstance(value, bool):
            rounds.append(value)
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            rounds.append(value.strip().lower() == "true")
        else:
            raise ValueError("round_success contains a non-Boolean value")
    evidence = parsed.get("round_evidence")
    if require_evidence and (not isinstance(evidence, list) or len(evidence) != 5):
        raise ValueError("round_evidence is not a five-item array")
    if not require_evidence:
        evidence = [""] * 5
    win_ab = str(parsed.get("win_ab", "")).strip().upper()
    win_ba = str(parsed.get("win_ba", "")).strip().upper()
    if win_ab not in {"A", "B"} or win_ba not in {"A", "B"}:
        raise ValueError("Missing A/B winner for either order")
    first = next((number for number, value in enumerate(rounds, 1) if value), None)
    return {
        "goal_action": str(parsed.get("goal_action", "")),
        "round_evidence": [str(item) for item in evidence],
        "round_success": rounds,
        "first_success_round": first,
        "success_within_5_rounds": first is not None,
        "win_ab": win_ab,
        "win_ba": win_ba,
        "win_credit": ((win_ab == "A") + (win_ba == "B")) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/success_prompt_v2_diagnostic/sample_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--generation-dir", type=Path, default=ROOT / "artifacts/generation_5rounds",
                        help="Directory containing the five-round generations to judge")
    parser.add_argument("--full-evaluation", action="store_true",
                        help="Evaluate every row for the selected five-round generations")
    supported_methods = tuple(method.slug for method in ARCHIVED_METHODS) + ("baseline1_zero_shot", "baseline2_dialogue_sft", "baseline3_isr_simdpo")
    parser.add_argument("--methods", nargs="+", choices=supported_methods,
                        help="Methods to evaluate in full mode; defaults to both archived methods")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-name", choices=("llama31", "selene1", "qwen35", "persugpt", "gemma4"), required=True)
    parser.add_argument("--judge-device", required=True)
    parser.add_argument("--prompt-version", choices=tuple(PROMPT_REVISIONS), default="v2")
    parser.add_argument("--max-evaluations", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-parse-retries", type=int, default=3,
                        help="Additional model generations after a parsing or evidence error (default: 3)")
    args = parser.parse_args()
    if os.environ.get("ALLOW_JUDGE_EVAL") != "1":
        print("Judge evaluation is disabled for the active LatentRATT run; set ALLOW_JUDGE_EVAL=1 to resume.", flush=True)
        return
    if args.max_parse_retries < 0:
        parser.error("--max-parse-retries must be nonnegative")

    methods = set(supported_methods)
    if args.full_evaluation:
        selected_methods = args.methods or [method.slug for method in ARCHIVED_METHODS]
        samples = [
            {"trial": 1, "method": method, "index": index, "stratum": "full_test"}
            for method in selected_methods for index in range(1000)
        ]
    else:
        if args.methods:
            parser.error("--methods is only valid with --full-evaluation")
        samples = read_manifest(args.manifest)
    generation = {
        method: {int(row["index"]): row for row in load_generation(
            args.generation_dir / f"{method}.jsonl", expected_rows=1000
        )}
        for method in {sample["method"] for sample in samples}
        if method in methods
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = PROMPT_REVISIONS[args.prompt_version]
    if args.full_evaluation:
        output_paths = {
            method: args.output_dir / f"{method}.jsonl"
            for method in {sample["method"] for sample in samples}
        }
    else:
        output_paths = {"probe": args.output_dir / f"{args.judge_name}_{revision}.jsonl"}
    sample_keys = {
        (int(sample["trial"]), str(sample["method"]), int(sample["index"]))
        for sample in samples
    }
    existing: dict[tuple[int, str, int], dict[str, Any]] = {}
    for output in output_paths.values():
        if output.exists():
            for item in read_jsonl(output):
                key = (int(item["trial"]), str(item["method"]), int(item["index"]))
                if (key not in sample_keys or item.get("judge_model") != args.judge_model
                        or item.get("prompt_revision") != revision
                        or int(item.get("max_parse_retries", 0)) != args.max_parse_retries):
                    raise ValueError(f"Unexpected key, model mismatch, prompt mismatch, or retry-policy mismatch in {output}: {key}")
                source_row = generation[str(item["method"])].get(int(item["index"]))
                if source_row is None or item.get("scenario_id") != source_row.get("scenario_id"):
                    raise ValueError(f"Scenario mismatch in existing record {output}: {key}")
                # A shared output directory may contain earlier records from another GPU.
                # Re-evaluate those on the requested device, and tolerate duplicate rows
                # left by concurrent or interrupted runs so checkpoint resume can proceed.
                if item.get("judge_device") != args.judge_device:
                    continue
                previous = existing.get(key)
                if previous is None or (previous.get("parse_error") and not item.get("parse_error")):
                    existing[key] = item
    pending = [sample for sample in samples
               if (sample["trial"], sample["method"], sample["index"]) not in existing]
    if args.max_evaluations > 0:
        pending = pending[:args.max_evaluations]
    print(f"{args.judge_name} {revision}: {len(existing)} saved, {len(pending)} pending", flush=True)
    if not pending:
        return

    judge = LocalTransformersEndpoint(model=args.judge_model, device=args.judge_device, quantization="none")
    device_map = getattr(judge.model, "hf_device_map", {})
    offloaded = [(name, str(device)) for name, device in device_map.items()
                 if str(device).lower() in {"cpu", "disk"}]
    if offloaded:
        judge.close()
        raise RuntimeError(f"Judge has CPU/disk offload: {offloaded[:5]}")
    print(f"Loaded {args.judge_model} on {args.judge_device}; BF16, no quantization", flush=True)
    unresolved_count = sum(bool(item.get("parse_error")) for item in existing.values())
    try:
        with ExitStack() as stack:
            handles = {
                method: stack.enter_context(path.open("a", encoding="utf-8"))
                for method, path in output_paths.items()
            }
            for number, sample in enumerate(pending, 1):
                row = generation[sample["method"]][sample["index"]]
                prompt = joint_prompt(row, args.prompt_version, args.judge_name)
                outputs: list[str] = []
                parse_errors: list[str] = []
                verdict = None
                error = None
                for attempt in range(args.max_parse_retries + 1):
                    retry_note = ""
                    if attempt:
                        if args.prompt_version in {"v15", "v16", "v17"}:
                            retry_note = (
                                f"\n\nYour previous answer failed validation: {parse_errors[-1]}. "
                                f"Correction attempt {attempt}: return one complete JSON object with "
                                "only first_success_round (integer 0-5), success_quote (one exact "
                                "quote from that R line, or empty for 0), win_ab (A/B), and "
                                "win_ba (A/B). Answer both tasks again."
                            )
                        else:
                            retry_note = (
                                f"\n\nYour previous answer failed validation: {parse_errors[-1]}. "
                                f"Correction attempt {attempt}: return a new complete JSON answer. "
                                "Each win_ab and win_ba value must be exactly A or B. "
                                "If round_success is true, its round_evidence must be an exact quote "
                                "from that same numbered Persuadee reply. Answer both tasks again."
                            )
                        bad_quote = re.fullmatch(
                            r"Round ([1-5]) positive evidence is not a verbatim persuadee quote",
                            parse_errors[-1],
                        )
                        if args.prompt_version in {"v14", "v15", "v16", "v17"} and bad_quote:
                            bad_round = int(bad_quote.group(1))
                            reply = row["trajectory"][2 * (bad_round - 1) + 1]["content"]
                            retry_note += (
                                f"\nR{bad_round} is exactly: {reply}\n"
                                f"For array item {bad_round}, quote only from this line. "
                                "If this line does not clearly accept the goal, use false and an empty evidence string."
                            )
                    raw = judge.complete(
                        system="You are a careful evaluator of goal acceptance and persuasive quality.",
                        user=prompt + retry_note, temperature=0.0, max_tokens=args.max_tokens,
                        json_mode=True,
                    )
                    outputs.append(raw)
                    try:
                        verdict = parse_joint(raw, require_evidence=args.prompt_version != "v5", compact=args.prompt_version in {"v15", "v16", "v17"})
                        if args.prompt_version in {"v4", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "gemma_v11_format_v1"}:
                            for round_index, (is_success, quote) in enumerate(zip(
                                verdict["round_success"], verdict["round_evidence"]
                            )):
                                reply = row["trajectory"][2 * round_index + 1]["content"]
                                if is_success and (not quote.strip() or quote.strip().casefold() not in reply.casefold()):
                                    raise ValueError(f"Round {round_index + 1} positive evidence is not a verbatim persuadee quote")
                    except ValueError as exc:
                        verdict = None
                        error = str(exc)
                        parse_errors.append(error)
                    else:
                        error = None
                        break
                unresolved_count += error is not None
                result = {
                    **sample,
                    "scenario_id": row["scenario_id"],
                    "goal": row["scenario"]["goal"],
                    "model": row.get("persuader_model", row.get("model", "")),
                    "prompt_version": args.prompt_version,
                    "prompt_revision": revision,
                    "judge_model": args.judge_model,
                    "judge_device": args.judge_device,
                    "judge_compute_dtype": "bfloat16",
                    "judge_quantization": "none",
                    "model_calls": len(outputs),
                    "max_parse_retries": args.max_parse_retries,
                    "raw_output": outputs[-1],
                    "raw_outputs": outputs,
                    "retry_outputs": outputs[1:],
                    "parse_errors_by_attempt": parse_errors,
                    "parse_error": error,
                    "verdict": verdict,
                }
                handle_key = sample["method"] if args.full_evaluation else "probe"
                handles[handle_key].write(json.dumps(result, ensure_ascii=False) + "\n")
                handles[handle_key].flush()
                if number % 5 == 0 or number == len(pending):
                    print(f"{args.judge_name} {revision}: {number}/{len(pending)} new, {unresolved_count} total parse errors", flush=True)
    finally:
        judge.close()
    for output in output_paths.values():
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
