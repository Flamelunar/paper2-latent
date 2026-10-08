"""Batch two independent scenario coroutines without duplicating model weights."""
import copy
from dataclasses import dataclass
import re

@dataclass
class Request:
    endpoint: object
    system: str
    prompt: str
    seed: int
    point: dict
    schema: object = None
    factory: object = None
    embeddings: object = None
    prompt_ids: object = None


class RowSampler:
    """Independent RNG per row; same temperature/top-k/top-p distribution as HF."""
    def __init__(self,torch,seeds,device,prompts,penalty,prompt_width=0):
        from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
        from eclp_elpo import PromptRepetitionPenalty
        self.torch=torch
        self.rng=[torch.Generator(device=device).manual_seed(s) for s in seeds]
        self.warpers=[TemperatureLogitsWarper(.95),TopKLogitsWarper(50),TopPLogitsWarper(.7)]
        self.penalties=[PromptRepetitionPenalty(p,penalty) if penalty!=1 else None for p in prompts]
        self.seen_steps=0
        self.prompt_width=prompt_width

    def __call__(self,input_ids,scores):
        torch=self.torch
        scores=scores.clone()
        for b,penalty in enumerate(self.penalties):
            if penalty is not None:scores[b:b+1]=penalty(input_ids[b:b+1,self.prompt_width:],scores[b:b+1])
        for warper in self.warpers:scores=warper(input_ids,scores)
        selected=[]
        for b,generator in enumerate(self.rng):
            probs=torch.softmax(scores[b],dim=-1)
            if not bool(torch.isfinite(probs).all()) or not bool(probs.sum()>0):
                raise ValueError('Invalid constrained sampling distribution')
            selected.append(torch.multinomial(probs,1,generator=generator))
        forced=torch.full_like(scores,-float('inf'))
        forced.scatter_(1,torch.stack(selected),0)
        self.seen_steps+=1
        return forced


class PublicStop:
    """Per-row public boundary; no cross-row names or vocabulary preprocessing."""
    def __init__(self,tokenizer,requests,prompt_width):
        self.tokenizer=tokenizer;self.width=prompt_width
        self.patterns=[]
        for request in requests:
            if request.schema is not None:self.patterns.append(None);continue
            labels=['user','assistant','system','persuader','persuadee']+[request.point['scenario'][r] for r in ['persuader','persuadee']]
            self.patterns.append(re.compile(r'(?im)\n[ \t]*(?:'+ '|'.join(re.escape(x) for x in labels)+r')[ \t]*(?::|\r?\n)|\n[ \t]*Write only the next natural (?:persuader|persuadee) utterance\.'))

    def __call__(self,input_ids,scores,**kwargs):
        import torch
        result=[]
        for b,pattern in enumerate(self.patterns):
            # Decode generated text only, never the prompt or a neighbour's tokens.
            tail=input_ids[b,max(self.width,input_ids.shape[1]-96):]
            text=self.tokenizer.decode(tail,skip_special_tokens=True)
            result.append(bool(pattern and pattern.search(text)))
        return torch.tensor(result,device=input_ids.device,dtype=torch.bool)


def complete_batch(requests):
    import torch
    from transformers import LogitsProcessorList,StoppingCriteriaList
    first=requests[0];endpoint=first.endpoint
    assert all(r.endpoint is endpoint for r in requests),'Never mix model endpoints'
    embedded=first.embeddings is not None
    assert all((r.embeddings is not None)==embedded for r in requests)
    tokenizer=endpoint.tokenizer
    model=endpoint.llm if embedded else endpoint.model
    device=endpoint.device if embedded else endpoint.input_device
    sequences=[];public_ids=[]
    for request in requests:
        if embedded:
            sequences.append(request.embeddings[0]);public_ids.append(request.prompt_ids)
        else:
            encoded=endpoint._encode_messages([{'role':'system','content':request.system},{'role':'user','content':request.prompt}])
            if hasattr(encoded,'input_ids'):encoded=encoded.input_ids
            sequences.append(encoded[0].to(device));public_ids.append(encoded.to(device))
    lengths=[len(x) for x in sequences];width=max(lengths)
    if not embedded and width+512>endpoint.context_limit:raise ValueError('Batch context overflow; do not truncate')
    if embedded and width+512>model.config.max_position_embeddings:raise ValueError('Embedded context overflow')
    mask=torch.zeros((len(requests),width),dtype=torch.long,device=device)
    if embedded:
        data=torch.zeros((len(requests),width,sequences[0].shape[-1]),dtype=sequences[0].dtype,device=device)
    else:data=torch.full((len(requests),width),tokenizer.pad_token_id,dtype=torch.long,device=device)
    for b,sequence in enumerate(sequences):data[b,width-len(sequence):]=sequence;mask[b,width-len(sequence):]=1
    cfg=copy.deepcopy(model.generation_config)
    penalty=1.0 if cfg.repetition_penalty is None else float(cfg.repetition_penalty)
    cfg.repetition_penalty=1.0;cfg.do_sample=False;cfg.num_beams=1
    cfg.temperature=None;cfg.top_p=None;cfg.top_k=None
    cfg.pad_token_id=tokenizer.pad_token_id;cfg.use_cache=True
    if cfg.eos_token_id is None:cfg.eos_token_id=tokenizer.eos_token_id
    sampler=RowSampler(torch,[r.seed for r in requests],device,public_ids,penalty,0 if embedded else width)
    kwargs=dict(attention_mask=mask,generation_config=cfg,max_new_tokens=512,
                logits_processor=LogitsProcessorList([sampler]),
                stopping_criteria=StoppingCriteriaList([PublicStop(tokenizer,requests,0 if embedded else width)]))
    constraints=[r.factory.constraint(r.schema,0 if embedded else width) if r.schema is not None else None for r in requests]
    if any(x is not None for x in constraints):
        assert all(x is not None for x in constraints),'Batch schema and free speech separately'
        kwargs['prefix_allowed_tokens_fn']=lambda b,ids:constraints[b](b,ids)
    kwargs['inputs_embeds' if embedded else 'input_ids']=data
    with torch.inference_mode():generated=model.generate(**kwargs)
    generated=generated if embedded else generated[:,width:]
    results=[]
    eos=cfg.eos_token_id if isinstance(cfg.eos_token_id,list) else [cfg.eos_token_id]
    for b,request in enumerate(requests):
        ids=generated[b].tolist()
        for j,token in enumerate(ids):
            if token in eos:ids=ids[:j+1];break
        raw=tokenizer.decode(ids,skip_special_tokens=True).strip()
        usage={'input_tokens_including_latents' if embedded else 'input_tokens':lengths[b],
               'output_tokens':len(ids),'batch_size_actual':len(requests),'sampling_seed':request.seed}
        results.append((raw,usage))
    del generated,data,mask
    return results


def drive(coroutines,on_complete):
    """Advance only matching stages; independent histories remain in each coroutine."""
    active={i:co for i,co in enumerate(coroutines)};waiting={}
    for lane,co in active.items():waiting[lane]=next(co)
    while active:
        head=next(iter(waiting.values()))
        # Retries can put lanes at different stages: split rather than reorder roles.
        group=[lane for lane,r in waiting.items() if r.endpoint is head.endpoint and
               (r.embeddings is not None)==(head.embeddings is not None) and r.schema==head.schema]
        requests=[waiting[lane] for lane in group]
        fallback=False
        try:responses=complete_batch(requests)
        except RuntimeError as exc:
            import torch
            if len(group)==1 or not isinstance(exc,torch.cuda.OutOfMemoryError):raise
            fallback=True
        if fallback:
            import gc
            gc.collect();torch.cuda.empty_cache()
            print('OOM batch fallback to singleton; same requests/seeds, no truncation',flush=True)
            responses=[]
            for request in requests:responses.extend(complete_batch([request]))
        for lane,response in zip(group,responses):
            try:waiting[lane]=active[lane].send(response)
            except StopIteration as done:
                del active[lane];del waiting[lane];on_complete(done.value)
