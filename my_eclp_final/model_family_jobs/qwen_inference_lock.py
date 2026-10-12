"""Verify original training provenance plus exactly reviewed inference revisions."""
from pathlib import Path
from eclp_elpo import read_json,object_hash,VERSION,sha256,model_fingerprint
CHANGES={
 'exp_code_ljj/repro/simulation.py':('a74d8206d64c39413fae76ce3b691cdb1e918b9dd5973b0accd552c4c922bf80','73b9f67e3fe9e9ee6682d3ef1859786cfc180283a7018508629903e1ef61d996'),
 'exp_code_ljj/repro/persuadee_simulation.py':('b8a88214519125e14358072d54ad193929a20ac66062e8b1cdba1a3c666a3fc8','3afcc8b3ccde2a9320813273d1fd02f82dfc960ba68ac0f726b7d97e02fb6077'),
}
def verify_inference_lock(cfg,lock_path,models=()):
    lock=read_json(lock_path)
    if object_hash({k:v for k,v in lock.items() if k!='lock_id'})!=lock['lock_id']:raise ValueError('Corrupt original training lock')
    if lock['version']!=VERSION or cfg.protocol()!=lock['protocol']:raise ValueError('Training implementation/protocol mismatch')
    for name,digest in lock['sources'].items():
        current=sha256(cfg.root/name)
        if current!=digest and CHANGES.get(name)!=(digest,current):raise ValueError('Unreviewed source change: '+name)
    for name,digest in lock['data'].items():
        if sha256(cfg.root/f'exp_code_ljj/data/{name}.jsonl')!=digest:raise ValueError('Training/test data changed: '+name)
    for name in models:
        if model_fingerprint(getattr(cfg,name))['digest']!=lock['models'][name]['digest']:raise ValueError('Model identity changed: '+name)
    print('Original training lock, data and base weights verified; reviewed inference-source revisions only',flush=True)
    return lock
