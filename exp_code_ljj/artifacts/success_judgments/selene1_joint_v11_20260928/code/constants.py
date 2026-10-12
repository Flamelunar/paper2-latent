from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "paper10k"
SFT_SCENARIOS = 10000
PREFERENCE_SCENARIOS = 2000
TEST_SCENARIOS = 1000
MAX_NEW_TOKENS = 1024
SYSTEM_PERSUADER = "You are a skilled persuader."
DATA_DIR = ROOT / "data"
RUN_ROOT = ROOT
RUN_CONFIG_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "models"
CUTOFF_LEN = 2048
SAVE_STEPS = 100
DEFAULT_MODEL = "/data1/liujianjian/model/Qwen2.5-7B-Instruct"

def data_path(name: str) -> Path:
    if name not in {
        "dialogue_sft", "isr_sft", "user_model_sft", "preference_seeds",
        "simulation_preferences", "evaluation_points",
    }:
        raise ValueError(f"Unknown experiment dataset: {name}")
    return DATA_DIR / f"{name}.jsonl"
