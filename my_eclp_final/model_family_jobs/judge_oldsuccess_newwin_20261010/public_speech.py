"""Public dialogue checks; the same rules apply to every persuader method."""
import re

SPEECH_RULE = (
    "Speak directly to the other person in your own voice. Give one natural spoken reply. "
    "Address their stated concern with relevant, concrete reasoning and a feasible next step "
    "toward the exact goal. If they have already clearly accepted, acknowledge that and "
    "discuss ordinary next steps; do not restart the argument. "
    "Use only established facts. Do not invent statistics, studies, guarantees or commitments. "
    "Output only the words you would actually say: no private reasoning, strategy labels, "
    "JSON, stage directions, or descriptions of what either person thinks or does."
)

SPEECH_SYSTEM = "You are a skilled persuader. Speak directly to the other person; output only your spoken reply."

def actor_prompt(point, history):
    from repro.persuadee_simulation import dialogue_prompt
    from protocol import PUBLIC_RULE
    # Keep the familiar training-compatible background/goal/history scaffold.
    return dialogue_prompt(point, history) + "\n\n" + PUBLIC_RULE + "\n" + SPEECH_RULE

def narration(text, point):
    names = [point['scenario']['persuader'], point['scenario']['persuadee'],
             'the persuader', 'the persuadee']
    subject = '(?:' + '|'.join(re.escape(n) for n in names) + ')'
    verbs = (r'acknowledges|recognizes|emphasizes|explains|suggests|proposes|addresses|'
             r'reassures|highlights|appeals|uses|offers|responds|points out|tries|'
             r'asks|tells|says|understands|encourages|reminds|presents|describes|'
             r'has agreed|seems|is concerned|should|needs to|will|can|could')
    # Match sentence-initial descriptions. Direct address ("Susan, ...") and
    # first-person self-introductions ("I'm Susan ...") do not match.
    if re.search(rf'(?im)(?:^|[.!?]\s+){subject}\s+(?:{verbs})\b', text):
        return True
    if re.search(r"(?im)^\s*(?:psychology|strategy|analysis|reasoning|plan)\s*:", text):
        return True
    return bool(re.search(
        r"(?i)^\s*(?:I(?:'ll| will) use the strategy|I need to (?:persuade|convince)|"
        r"the strategy is|the next (?:persuader|persuadee) (?:reply|response)|"
        r"let(?:'s| us) analyze|\[.*?\])", text))

def validate_public(raw, point):
    from protocol import speech_boundary
    value = speech_boundary(raw, point, 'persuader')
    if value['thinking_filtered'] or narration(value['content'], point):
        raise ValueError('Private reasoning or third-person narration in public answer')
    if raw.lstrip().startswith(('{', '<think>', 'Psychology:', 'Analysis:', 'Strategy:')):
        raise ValueError('Structured/private answer instead of public speech')
    return value

def retry_prompt(prompt, point):
    return prompt + ("\nFORMAT RETRY: Address " + point['scenario']['persuadee'] +
                     " directly. Start with a sentence you would say to them. "
                     "Give the spoken reply itself, not a description or a plan.")
