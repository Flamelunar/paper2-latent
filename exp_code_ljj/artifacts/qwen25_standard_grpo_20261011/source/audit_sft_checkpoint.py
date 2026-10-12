"""Inspect the saved real SFT checkpoint without loading a GPU model."""
import hashlib
import json
from pathlib import Path
import time
from safetensors import safe_open

RUN = Path('/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/qwen25_standard_grpo_20261011')
last = json.loads((RUN / 'training/sft/last.json').read_text())
checkpoint = Path(last['checkpoint'])
config = json.loads((checkpoint / 'adapter/adapter_config.json').read_text())
assert config['r'] == 16 and config['lora_alpha'] == 32 and config['lora_dropout'] == 0
assert set(config['target_modules']) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
path = checkpoint / 'adapter/adapter_model.safetensors'
shapes, dtypes = {}, set()
with safe_open(path, framework='pt', device='cpu') as tensors:
    for key in tensors.keys():
        assert '.lora_A.' in key or '.lora_B.' in key
        value = tensors.get_slice(key)
        shape = value.get_shape()
        dtypes.add(value.get_dtype())
        assert len(shape) == 2 and (shape[0] == 16 if '.lora_A.' in key else shape[1] == 16)
        shapes[key] = shape
assert dtypes == {'BF16'}
assert len(shapes) == 392
assert (checkpoint / 'trainer.pt').is_file()
h = hashlib.sha256()
with path.open('rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
result = {'observed_unix': time.time(), 'checkpoint': str(checkpoint), 'step': last['step'],
    'processed_sessions': last['position'], 'saved_tensor_count': len(shapes), 'dtypes': sorted(dtypes),
    'rank': config['r'], 'alpha': config['lora_alpha'], 'dropout': config['lora_dropout'],
    'adapter_sha256': h.hexdigest(), 'tensor_shapes': shapes, 'trainer_state_present': True}
(RUN / 'sft_checkpoint_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'tensor_shapes'}))
