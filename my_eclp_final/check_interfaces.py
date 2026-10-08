from pathlib import Path
import sys,types
root=Path('/app/hzj/liujianjian/ljj-2-paper')
sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'))
from generate_our_ma2p192 import TrainedECLPEndpoint,public_point,selected_points
cfg=types.SimpleNamespace(actor_system_prompt='original')
bridge=types.SimpleNamespace(role_prompt=None)
class Runner:
 def __init__(self):
  self.llm=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=32768))
  self.tokenizer=object();self.device=types.SimpleNamespace(index=0);self.calls=[]
 def speak(self,point,live,seed):
  self.calls.append((point,live,seed,cfg.actor_system_prompt,bridge.role_prompt))
  return {'raw_output':'A trained ECLP public utterance.','content':'A trained ECLP public utterance.','input_tokens_including_latents':40,'public_tokens':12},None,None
r=Runner();e=TrainedECLPEndpoint(r,bridge,cfg);e.point={'scenario':{'persuader':'John','persuadee':'Sarah'}};e.live=object()
e.analysis_constraint=lambda system:(('preventive','generative') if 'Perception' in system else ('likely_reaction','strategy'),lambda batch,ids:None)
def fake_constrained(seed,prefix):
 r.calls.append((e.point,e.live,seed,cfg.actor_system_prompt,bridge.role_prompt))
 raw='{"preventive":"unknown","generative":"unknown"}' if cfg.actor_system_prompt.startswith("You are MA2P's Perception") else '{"likely_reaction":"unknown","strategy":"Ask respectfully"}'
 return {'raw_output':raw,'input_tokens_including_latents':40,'public_tokens':12}
e.constrained_speak=fake_constrained
decode={'temperature':0.95,'top_p':0.7,'top_k':50,'max_tokens':512}
for system in ["You are MA2P's Perception agent. Infer cautiously from observable evidence.","You are MA2P's World Model. Plan from public context and short-term memory only."]:
 assert e.complete(system=system,user='Original MA2P role prompt',**decode)
assert len(r.calls)==2
assert e.complete(system="You are a skilled persuader following the MA2P World Model's plan.",user='Original MA2P actor prompt',**decode)=='A trained ECLP public utterance.'
assert len(r.calls)==3 and cfg.actor_system_prompt=='original' and bridge.role_prompt is None
class PlanRunner(Runner):
 def speak(self,point,live,seed):
  turn,ctx,eff=super().speak(point,live,seed)
  turn.update(raw_output='Persuader: Sarah is considering it.',content='Sarah is considering it.')
  return turn,ctx,eff
pr=PlanRunner();pe=TrainedECLPEndpoint(pr,bridge,cfg)
pe.point={'scenario':{'persuader':'John','persuadee':'Sarah'}};pe.live=object()
assert pe.complete(system="You are a skilled persuader following the MA2P World Model's plan.",user='Original MA2P actor prompt',**decode)=='Persuader: Sarah is considering it.'
assert len(pr.calls)==3 and pe.calls[0]['retry_exhausted']=='plan_narration'
assert len(pe.calls[0]['rejected_plan_outputs'])==3
points,selection=selected_points(root);assert len(points)==192
p=public_point(dict(points[0][1],strategies='FORBIDDEN_CANARY',reference='FORBIDDEN_CANARY',psychological_state='FORBIDDEN_CANARY'))
assert 'FORBIDDEN_CANARY' not in str(p)
print('PASS: all three roles use full ECLP; role prompt restored; fixed 192 indexes; hidden/reference fields excluded.')
