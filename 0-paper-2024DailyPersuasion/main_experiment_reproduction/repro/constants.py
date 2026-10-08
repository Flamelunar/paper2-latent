from __future__ import annotations

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent
PROJECT_ROOT = PAPER_ROOT.parent
OFFICIAL_REPO = PAPER_ROOT / "PersuGPT"
OFFICIAL_SRC = OFFICIAL_REPO / "src"
PREPARED_ROOT = PROJECT_ROOT / "data-dailypersuasion/prepared"
ORIGINAL_ROOT = PREPARED_ROOT / "original_format"
SMALL_PROFILE_ROOT = PREPARED_ROOT / "small5000"
SMALL_ORIGINAL_ROOT = SMALL_PROFILE_ROOT / "original_format"

PROFILE = os.getenv("DAILYPERSUASION_PROFILE", "paper")
if PROFILE not in {"paper", "paper10k", "small5000"}:
    raise ValueError("DAILYPERSUASION_PROFILE must be paper, paper10k, or small5000")
SFT_SCENARIOS, PREFERENCE_SCENARIOS, TEST_SCENARIOS = (
    (5000, 1000, 500) if PROFILE == "small5000" else (10000, 2000, 1000)
)
RUN_ROOT = ROOT if PROFILE == "paper" else ROOT / "runs" / PROFILE
DATA_DIR = RUN_ROOT / "data"
OUTPUT_DIR = RUN_ROOT / "outputs"
RESULT_DIR = RUN_ROOT / "results"
RUN_CONFIG_DIR = RUN_ROOT / "run_configs"
SOURCE_ROOT = SMALL_ORIGINAL_ROOT if PROFILE == "small5000" else ORIGINAL_ROOT
SFT_SOURCE = SOURCE_ROOT / "sft.json"
PREFERENCE_SOURCE = SOURCE_ROOT / "preference.json"
TEST_SOURCE = SOURCE_ROOT / "test.json"

# Keep the internal dataset keys stable for the training code, while exposing
# short, role-oriented filenames in the reduced experiment profile. The paper
# profile keeps its historical names so existing full-scale files remain
# readable and untouched.
PAPER_DATA_FILENAMES = {
    "dialogue_sft": "dialogue_sft.jsonl",
    "isr_sft": "isr_sft.jsonl",
    "user_model_sft": "user_model_sft.jsonl",
    "preference_seeds": "preference_seeds.jsonl",
    "evaluation_points": "evaluation_points.jsonl",
    "simulation_preferences": "simulation_preferences.jsonl",
}
SMALL_DATA_FILENAMES = {
    "dialogue_sft": "sft_dialogue.jsonl",
    "isr_sft": "sft_add_isr.jsonl",
    "user_model_sft": "sft_persuadee.jsonl",
    "preference_seeds": "preference_seed.jsonl",
    "evaluation_points": "test.jsonl",
    "simulation_preferences": "dpo_preferences.jsonl",
}
DATA_FILENAMES = SMALL_DATA_FILENAMES if PROFILE == "small5000" else PAPER_DATA_FILENAMES


def data_path(name: str) -> Path:
    """Resolve an internal dataset key to the profile's public filename."""
    try:
        return DATA_DIR / DATA_FILENAMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown reproduction data file: {name}") from exc


# A five-thousand-session SFT run has about 313 optimizer updates with the
# default effective batch size, so retain useful resumable checkpoints.
SAVE_STEPS = 100 if PROFILE in {"small5000", "paper10k"} else 500
DEFAULT_MODEL = "/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct"
MODEL_LABEL = "Llama-3.1-8B-Instruct"
SYSTEM_PERSUADER = "You are a skilled persuader."
MAX_NEW_TOKENS = 1024
CUTOFF_LEN = 2048

PAPER_TARGETS = {
    "Llama-3.1 Zero-shot": {
        "paper_model": "LLaMA-2",
        "icl": False,
        "isr": False,
        "win_rate": 20.1,
        "rouge_l": 17.4,
    },
    "Llama-3.1 Dialogue SFT": {
        "paper_model": "LLaMA-2",
        "icl": False,
        "isr": False,
        "win_rate": 46.7,
        "rouge_l": 24.4,
    },
    "Llama-3.1 ISR SFT": {
        "paper_model": "LLaMA-2 + ISR",
        "icl": False,
        "isr": True,
        "win_rate": 47.8,
        "rouge_l": 25.6,
    },
    "Ours (Llama-3.1) ISR-SFT + DPO": {
        "paper_model": "Ours (LLaMA-2)",
        "icl": False,
        "isr": True,
        "win_rate": 60.4,
        "rouge_l": 25.2,
    },
}
