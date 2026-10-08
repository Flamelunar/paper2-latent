# Project instructions

## Fixed main test subset (2026-10-06)

The user requires all future main tests to use only the fixed 192 scenarios with target_turn == 1 and original history == []. Use exp_code_ljj/data/test_selection_192.json and evaluation_points_from_start192.jsonl; preserve original indexes. All methods and judges use the same subset. Keep training/dev data unchanged. Initial public dialogue is empty; subsequent generated history is retained. Success is any accepted round within five; use unified v2 Win@1 and Win-Full. New runs must use separate 192-scenario output contracts, never append to archived 1000-scenario runs. See TEST_SELECTION_192.md. Never stop other GPU4 tasks.
