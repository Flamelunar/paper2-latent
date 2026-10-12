"""Batch the four text actions while retaining independent seeded sampling."""
import copy


def sample_group(model, tokenizer, prompt_ids, seeds, config):
    import torch
    from transformers import LogitsProcessorList
    from batching import RowSampler
    if (config.temperature, config.top_p, config.top_k, config.max_new_tokens) != (0.95, 0.7, 50, 512):
        raise ValueError('Historical RowSampler decode contract changed')
    x = torch.tensor([prompt_ids] * len(seeds), device='cuda:0')
    generation = copy.deepcopy(model.generation_config)
    penalty = 1.0 if generation.repetition_penalty is None else float(generation.repetition_penalty)
    generation.repetition_penalty = 1.0
    generation.do_sample = False
    generation.num_beams = 1
    generation.temperature = generation.top_p = generation.top_k = None
    generation.pad_token_id = tokenizer.pad_token_id
    generation.use_cache = True
    if generation.eos_token_id is None:
        generation.eos_token_id = tokenizer.eos_token_id
    width = len(prompt_ids)
    sampler = RowSampler(torch, seeds, x.device, [x[i:i + 1] for i in range(len(seeds))], penalty, width)
    with torch.inference_mode():
        result = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=generation,
            max_new_tokens=config.max_new_tokens, logits_processor=LogitsProcessorList([sampler]))
    eos = generation.eos_token_id if isinstance(generation.eos_token_id, list) else [generation.eos_token_id]
    samples = []
    for row in result[:, width:]:
        ids = row.tolist()
        for index, token in enumerate(ids):
            if token in eos:
                ids = ids[:index + 1]
                break
        samples.append((ids, tokenizer.decode(ids, skip_special_tokens=True).strip()))
    return samples


def reward_group(point, utterances, seeds, user, judge, factory, config):
    """Infrastructure failures abort; actor-format penalties are handled upstream."""
    from batching import Request, complete_batch
    from protocol import GOAL_SCHEMA, speech_boundary, goal_prompt, validate_goal, json_batch
    from evaluate import excerpt_schema
    from repro.simulation import user_prompt
    pending = [i for i, utterance in enumerate(utterances) if utterance is not None]
    histories = {i: point['history'] + [{'role': 'persuader', 'content': utterances[i]}] for i in pending}
    simulator_traces = {i: [] for i in pending}
    replies = {}
    for attempt in range(3):
        requests = []
        for i in pending:
            system, prompt = user_prompt(point, histories[i])
            requests.append(Request(user, system, prompt, seeds[i] + 1 + 1000 * attempt, point))
        if not requests:
            break
        outputs = complete_batch(requests)
        retry = []
        for i, (raw, usage) in zip(pending, outputs):
            simulator_traces[i].append({'raw_output': raw, **usage})
            try:
                replies[i] = speech_boundary(raw, point, 'persuadee')
            except ValueError:
                retry.append(i)
        pending = retry
    if pending:
        raise ValueError(f'Simulator format unresolved for replicas {pending}; abort uncommitted block')
    prompts, traces = {}, {}
    for i, reply in replies.items():
        trajectory = histories[i] + [{'role': 'persuadee', 'content': reply['content']}]
        prompts[i] = goal_prompt({'scenario': point['scenario'], 'trajectory': trajectory}, len(trajectory) // 2)
        traces[i] = []
    pending = list(replies)
    values = {}
    for attempt in range(3):
        if not pending:
            break
        if attempt == 0:
            try:
                outputs = json_batch(judge, factory, [prompts[i] for i in pending], GOAL_SCHEMA,
                    'Judge exact-goal acceptance from the current persuadee reply.', max_tokens=256)
            except ValueError as error:
                outputs = [{'error': str(error)} for _ in pending]
        else:
            outputs = []
            for i in pending:
                try:
                    outputs.extend(json_batch(judge, factory, [prompts[i]], excerpt_schema(replies[i]['content']),
                        'Judge exact-goal acceptance from the current persuadee reply.', max_tokens=256))
                except ValueError as error:
                    outputs.append({'error': str(error)})
        retry = []
        for i, output in zip(pending, outputs):
            traces[i].append(output)
            try:
                if 'error' in output:
                    raise ValueError(output['error'])
                values[i] = validate_goal(output['value'], replies[i]['content'])
            except ValueError as error:
                traces[i].append({'validation_error': str(error)})
                retry.append(i)
        pending = retry
    if pending:
        raise ValueError(f'Unresolved training goal for replicas {pending}; abort uncommitted block')
    return {i: (['none', 'engagement', 'conditional', 'accepted'].index(v['progress']) / 3,
                {'simulator_reply': replies[i], 'simulator_attempts': simulator_traces[i],
                 'goal': v, 'judge_attempts': traces[i]}) for i, v in values.items()}
