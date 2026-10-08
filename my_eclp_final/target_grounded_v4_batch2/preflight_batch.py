"""Check batch RNG, padding and latent generation before real-model inference."""
from pathlib import Path
import sys
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'))
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from batching import RowSampler,Request,complete_batch,PublicStop

torch.set_num_threads(1)
prompts=[torch.tensor([[2,3]]),torch.tensor([[4,5,6]])]
batched=RowSampler(torch,[42,105],'cpu',prompts,1)
single=[RowSampler(torch,[seed],'cpu',[prompt],1) for seed,prompt in zip([42,105],prompts)]
for step in range(30):
    scores=torch.arange(64,dtype=torch.float32).repeat(2,1)/30
    scores[1]=scores[1].flip(0)
    ids=torch.ones((2,step),dtype=torch.long)
    result=batched(ids,scores).argmax(-1)
    for lane in range(2):assert result[lane]==single[lane](ids[lane:lane+1],scores[lane:lane+1]).argmax(-1)[0]

class Tokenizer:
    pad_token_id=0;eos_token_id=1
    def decode(self,ids,skip_special_tokens=True):
        if hasattr(ids,'tolist'):ids=ids.tolist()
        return ' '.join(str(i) for i in ids if i not in [0,1])
class Endpoint:
    tokenizer=Tokenizer();input_device='cpu';device=torch.device('cpu');context_limit=1024
    def _encode_messages(self,messages):return torch.tensor([[2,3,4,5]] if messages[-1]['content']=='long' else [[6,7]])

cfg=Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=1024,eos_token_id=1,pad_token_id=0)
torch.manual_seed(123);e=Endpoint();e.model=Qwen2ForCausalLM(cfg).eval();e.llm=e.model
point={'scenario':{'persuader':'Alice','persuadee':'Bob'}}
requests=[Request(e,'sys','long',42,point),Request(e,'sys','short',105,point)]
batch=complete_batch(requests)
individual=[complete_batch([r])[0] for r in requests]
assert [x[0] for x in batch]==[x[0] for x in individual],(batch,individual)
assert [x[1]['input_tokens'] for x in batch]==[4,2]
# Variable-length inputs_embeds (the complete ECLP interface).
for r in requests:
 r.prompt_ids=e._encode_messages([{'content':r.prompt}]);r.embeddings=e.model.get_input_embeddings()(r.prompt_ids).detach()
batch=complete_batch(requests);individual=[complete_batch([r])[0] for r in requests]
assert [x[0] for x in batch]==[x[0] for x in individual]
print('PASS: independent row RNG; different-length left padding; ordinary and inputs_embeds batch generation',flush=True)
