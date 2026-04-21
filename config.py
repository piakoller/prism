# config.py
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# --- LLM Configuration ---
LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0.0
MAX_TOKENS = 8192
NUM_CTX = 131072

MODEL_CONFIGS = {
    "google/gemini-2.5-pro": {
        "max_tokens": None,
        "temperature": LLM_TEMPERATURE,
    },
    "google/gemini-3-pro-preview": {
        "max_tokens": None,
        "temperature": LLM_TEMPERATURE,
    },
    "google/gemini-2.0-flash-exp:free": {
        "max_tokens": 8192,
        "temperature": LLM_TEMPERATURE,
    },
    "google/gemma-3-27b-it:free": {
        "max_tokens": 8192,
        "temperature": LLM_TEMPERATURE,
    },
    "qwen/qwen3-coder:free": {
        "max_tokens": 8192,
        "temperature": LLM_TEMPERATURE,
    },
}

MODEL_KWARGS = {
    "temperature": LLM_TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "num_ctx": NUM_CTX,
}

# --- OpenRouter API Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Set to True for production models (1M context), False for free test models
USE_PRODUCTION_MODEL = True

if USE_PRODUCTION_MODEL:
    OPENROUTER_MODEL = "google/gemini-3-pro-preview"
    DEFAULT_OPENROUTER_MODEL = "google/gemini-3-pro-preview"
else:
    OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"

MODEL_CONTEXT_LIMITS = {
    "google/gemini-2.5-pro": 1_000_000,
    "google/gemini-3-pro-preview": 1_000_000,
    "google/gemini-2.0-flash-exp:free": 128_000,
    "google/gemma-3-27b-it:free": 128_000,
    "qwen/qwen3-coder:free": 128_000,
}

# --- Cache Configuration ---
# Separate cache directories per model type to avoid mixing results
if USE_PRODUCTION_MODEL:
    CACHE_SUFFIX = "production"
    MODEL_CACHE_DIR = "cache_production"
else:
    CACHE_SUFFIX = "test"
    MODEL_CACHE_DIR = "cache_test"

GUIDELINES_CACHE_DIR = f"cache/{MODEL_CACHE_DIR}/guidelines"
TRIALS_CACHE_DIR = f"cache/{MODEL_CACHE_DIR}/trials"
ASSESSMENT_CACHE_DIR = f"cache/{MODEL_CACHE_DIR}/assessments"
TRIAL_ANALYSIS_CACHE_DIR = f"cache/{MODEL_CACHE_DIR}/trial_analysis"

# --- File Paths & Directories ---
DATA_ROOT_DIR = Path(__file__).resolve().parent.parent / "data"
TUBO_EXCEL_FILE_PATH = DATA_ROOT_DIR / "NET Tubo v2.xlsx"

REPORT_DIR = "generated_report"
REPORT_FILE_TYPE = "md"

# --- Clinical Trials API ---
CLINICAL_TRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_PAGE_SIZE = 20
REQUESTS_TIMEOUT = 30
MAX_LOCATIONS_TO_DISPLAY_PER_STUDY = 3

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(module)s.%(funcName)s - %(message)s"
