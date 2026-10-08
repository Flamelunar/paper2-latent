"""CPU checks for public boundary, strict target verdict and source contracts."""
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'))
from protocol import speech_boundary,validate_goal,strict_json
from experiment_io import read_json,read_jsonl,digest
p={'scenario':{'persuader':'Alice','persuadee':'Bob'}}
for raw in ['Hello.\nBob: Yes!','Alice: Hello.\nuser\nYes!','Hello.\nPersuadee: Yes!']:
    assert speech_boundary(raw,p,'persuader')['content']=='Hello.'
assert validate_goal({'evidence':'I will test it before deciding.','progress':'engagement','success':False},'I will test it before deciding.')['success'] is False
for value in [{'evidence':'yes','progress':'accepted','success':'true'}, {'evidence':'yes','progress':'engagement','success':True}, {'evidence':'invented','progress':'accepted','success':True}]:
    try:validate_goal(value,'yes')
    except ValueError:pass
    else:raise AssertionError('Invalid verdict accepted')
try:strict_json('{"success":true,"success":false}')
except ValueError:pass
else:raise AssertionError('Duplicate JSON accepted')
from evaluate import OLD_MAP
for method in OLD_MAP.values():
    c=read_json(root/'eclp_runs_gpu4/evaluation_v2/selene8b'/method/'run.json')
    assert digest(Path(c['generation_source']))==c['generation_sha256'],method
selection=read_json(root/'exp_code_ljj/data/test_selection_192.json')
gold={r['index']:r for r in read_jsonl(root/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl')}
for i in selection['original_indexes']:assert gold[i]['initial_history']==[]
print('PASS: public boundary; exact JSON booleans; evidence provenance; old-source hashes; 192 empty histories',flush=True)
