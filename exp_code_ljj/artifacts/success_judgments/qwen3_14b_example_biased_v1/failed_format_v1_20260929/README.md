# Failed Qwen3 V11 Output Format Attempt

These partial runs used the original Gemma V11 evidence-array output schema. Qwen3 repeatedly returned an empty evidence array, so the strict parser rejected the records after all allowed retries. The runs were stopped before completion; do not use these rows for scoring. The full evaluation uses a Qwen3-specific strict JSON schema with the same V11 decision rules.
