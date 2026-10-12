from repro.simulation import user_prompt as original_user_prompt,transcript

def user_prompt(point,history):
 system,_=original_user_prompt(point,history)
 prompt=f"Persuasion background: {point['scenario']['background']}\n\nDialogue so far:\n{transcript(history)}\n\nWrite only the next natural persuadee utterance. Do not add a speaker label."
 return system,prompt
