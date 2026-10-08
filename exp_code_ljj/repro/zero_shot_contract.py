"""Zero-shot-only input boundary; preserves the common public test context.

The allowlist deliberately ignores strategies, reference replies, psychological
annotations, and all other metadata. Public background may naturally describe
concerns; that is not an extra hidden-state annotation.
"""
from pathlib import Path


def zero_shot_prompt(point, history):
    scenario = point['scenario']
    background, goal = scenario['background'], scenario['goal']
    if not isinstance(background, str) or not isinstance(goal, str):
        raise ValueError('Zero-shot background and goal must be strings')
    turns = []
    for turn in history:
        role, content = turn['role'], turn['content']
        if role not in ('persuader', 'persuadee') or not isinstance(content, str):
            raise ValueError('Zero-shot history must contain public speaker utterances only')
        turns.append(f'{role.title()}: {content}')
    dialogue = '\n'.join(turns) if turns else '(empty)'
    return (f'The persuasion background: {background}\n'
            f'The persuasion goal: {goal}\n'
            'Give the next persuader utterance only.\n\n'
            f'Historical dialogue:\n{dialogue}\n\n'
            'Write only the next persuader utterance. Do not add a speaker label.')


def validate_zero_shot_model(model_path):
    """Reject explicit fine-tuning artifacts; merged weights need provenance too."""
    model = Path(model_path).expanduser().resolve()
    for filename in ('adapter_config.json', 'trainer_state.json', 'training_args.bin'):
        if (model / filename).exists():
            raise ValueError(f'Zero-shot requires an unmodified instruct model; found {model / filename}')
