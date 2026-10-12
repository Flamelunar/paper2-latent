"""Render only complete new judgments; archive all previous table scores once."""
import argparse,json,shutil
from pathlib import Path
LABELS={'zero_shot_repaired':'Zero-shot','dialogue_sft':'Dialogue-SFT','persugpt':'PersuGPT (ISR-SFT + SimDPO)','ma2p_repaired':'MA2P','latentratt':'LatentRATT','our_analysis':'Our: private analysis + persuasion','our_direct':'Our: persuasion only'}
def render(root,work):
 table=root/'exp_code_ljj/results/main_experiment_table.md';archive=work/'archive/main_experiment_table_before.md'
 if table.exists() and not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(table,archive)
 audit=json.loads((work/'historical_boundary_audit.json').read_text())
 latent_path=work/'historical_latentratt_provenance.json'
 latent=json.loads(latent_path.read_text()) if latent_path.exists() else {}
 out=['# Qwen2.5-7B — fixed 192 scenarios, new evaluation','', 'All prior scores are invalid for this run and archived. The fixed subset has target_turn = 1 and empty initial history. Maximum five rounds; success means exact goal acceptance in any round; failures count as five rounds.','', 'Dialogue-SFT, PersuGPT and LatentRATT reuse the corresponding 192 historical dialogues verbatim; Zero-shot, MA2P and the two Our variants generate new dialogues. Both Our variants expose only spoken persuasion to the simulator and judges. Historical simulator paths identify Qwen3.5-9B-persuadee; exact old/new weight equivalence is not established by the path name.','', f'Historical output boundary audit: {audit["n"]} flagged turns; preserved and disclosed, not silently repaired. See historical_boundary_audit.json. LatentRATT history is audited separately in historical_latentratt_provenance.json; flagged turn count: {latent.get("boundary_issue_count", "pending")}.','', 'Win@1 compares first replies; Win-Full compares all available conversations against golden. Each uses both candidate orders (win=1, tie=0.5, loss=0). These scores are preference credits including ties. Swapped calls are bias controls, not independent samples.','']
 for judge,title in [('selene8b','Selene-1-Mini-Llama-3.1-8B'),('qwen3_14b','Qwen3-14B')]:
  out+=['## '+title,'','| Method | N | Goal acceptance within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds to success (failures = 5) | Success format unresolved (rounds) |','|---|---:|---:|---:|---:|---:|---:|']
  for m,label in LABELS.items():
   p=work/'judges'/judge/'evaluation'/m/'summary.json';s=json.loads(p.read_text()) if p.exists() else {}
   def fmt(k,denominator=None):
    if not s.get('complete') or s.get(k) is None:return 'Pending'
    suffix=f' ({s[denominator]}/192)' if denominator else ''
    return f'{s[k]:.2f}{suffix}'
   out.append(f'| {label} | {s.get("n",0)}/192 | {fmt("success_percent","success_evaluable_n")} | {fmt("win1_percent","win1_evaluable_n")} | {fmt("win_full_percent","win_full_evaluable_n")} | {fmt("mean_rounds_fail5","success_evaluable_n")} | {s.get("format_unresolved","Pending")} |')
  out+=['','Order-conflict rates and tie rates are retained in each method summary. Partial runs do not populate final scores.','']
 temporary=table.with_suffix('.md.tmp');temporary.write_text('\n'.join(out),encoding='utf-8');temporary.replace(table)
 shutil.copy2(table,work/'main_experiment_table.md')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.repo_root,a.output)
