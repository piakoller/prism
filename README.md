# PRISM — Personalized Recommendation and Intelligent Study Matching

PRISM is a modular, LLM-powered pipeline for NET (Neuroendocrine Tumor) patient assessment, clinical trial matching, and therapy recommendation generation. It is designed for auditable and reproducible evidence-based decision support.

---

## Repository Structure

```
prism/
├── run_full_pipeline.py               # End-to-end entry point
├── config.py                          # Centralized configuration (model, paths, caching)
├── logging_setup.py                   # Shared logging configuration
├── shared_logic.py                    # Shared utility functions
├── .env                               # API key (not committed to version control)
├── .env_template                      # Template for .env setup
│
├── agentic_assessment/                # Core 7-step agentic workflow
│   ├── agentic_workflow.py            # Main orchestrator
│   ├── guidelines_matcher.py          # Step 1: Guidelines matching
│   ├── patient_assessor.py            # Step 2: Patient assessment
│   ├── trial_matcher.py               # Step 3: Trial matching
│   ├── trial_analyzer.py              # Step 4: Trial analysis
│   ├── therapy_recommender.py         # Step 5: Therapy recommendation
│   ├── recommendation_validator.py    # Step 6: Validation (3-layer)
│   ├── smart_regeneration.py          # Step 7: Smart regeneration
│   └── check_and_regenerate.py        # CLI tool for post-hoc validation/regeneration
│
├── clinical_trials/
│   └── core_utilities/
│       ├── study_collector.py         # Fetches trials from ClinicalTrials.gov API
│       ├── study_filter.py            # Filters and deduplicates collected trials
│       └── online_search.py           # Additional online evidence retrieval
│
├── prompts/                           # Prompt templates (one per pipeline step)
│   ├── 1_guideline_matching.txt
│   ├── 2_patient_assessment.txt
│   ├── 3_trial_matching.txt
│   ├── 4_trial_analysis.txt
│   ├── 5_therapy_recommendation.txt
│   ├── 6_recommendation_validator.txt
│   └── therapy_recommendation_baseline.txt
│
├── data/
│   ├── Combined_patientCases.xlsx     # Patient case data
│   └── guidelines/                    # Medical guideline files (.md)
│       ├── 1-0_data_singleprompt/
│       ├── 1-1_data_singleprompt/
│       └── 2-0/                       # ENET + ESMO guideline markdown files
│
└── results/                           # Generated outputs (gitignored)
    ├── collected_trials/              # Trials fetched from API
    └── agentic_outputs/               # Per-patient workflow results
```

---

## Setup

### 1. API Key

Copy `.env_template` to `.env` and fill in your OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

### 2. Dependencies

```bash
pip install requests pandas openpyxl python-dotenv
```

### 3. Patient Data

Patient cases are stored in:

```
data/Combined_patientCases.xlsx
```

Each row is one patient case. The column `ID` is used to identify patients throughout the pipeline.

---

## Running the Pipeline

### Full end-to-end run

```bash
cd prism/
python run_full_pipeline.py
```

This will:
1. Fetch NET clinical trials from ClinicalTrials.gov
2. Run the 7-step agentic assessment for all patients in `Combined_patientCases.xlsx`
3. Save all outputs to `results/agentic_outputs/`

### Single patient via CLI

```bash
cd prism/agentic_assessment/
python agentic_workflow.py \
  --patients "../data/Combined_patientCases.xlsx" \
  --trials "../results/collected_trials/<trials_file>.json" \
  --guidelines "../data/guidelines" \
  --single 2
```

**CLI arguments:**

| Argument | Description |
|---|---|
| `--patients` | Path to patient Excel file |
| `--trials` | Path to collected trials JSON |
| `--guidelines` | Path to guidelines directory |
| `--single <ID>` | Process only the patient with this ID |
| `--output <dir>` | Output directory (optional) |
| `--no-cache` | Disable result caching |
| `--force-regenerate` | Force regeneration, clear cache |

---

## The 7-Step Pipeline

All steps are orchestrated by `agentic_workflow.py` and run sequentially per patient.

### Step 1 — Guidelines Matching (`guidelines_matcher.py`)
- **Prompt:** `prompts/1_guideline_matching.txt`
- **Input:** Patient data + all guideline `.md` files in `data/guidelines/`
- **Process:** The LLM evaluates each guideline file for relevance to the specific patient (one API call per guideline file, typically ~25 calls)
- **Output:** `patient_<ID>_guidelines.json` — matched guidelines with relevance scores

### Step 2 — Patient Assessment (`patient_assessor.py`)
- **Prompt:** `prompts/2_patient_assessment.txt`
- **Input:** Patient data + matched guidelines from Step 1
- **Process:** Comprehensive clinical assessment — disease status, urgency, guideline adherence gaps, evidence priorities. **No trial data is included** at this stage.
- **Output:** `patient_<ID>_assessment.json` + prompt/response files

### Step 3 — Trial Matching (`trial_matcher.py`)
- **Prompt:** `prompts/3_trial_matching.txt`
- **Input:** Patient assessment from Step 2 + collected trial JSON
- **Process:** Each trial is individually evaluated with a YES/NO relevance decision and detailed explanation. **One API call per trial** (~86+ calls for a full trial set).
- **Output:** `patient_<ID>_trial_matching.json` — relevant and rejected trials with explanations

### Step 4 — Trial Analysis (`trial_analyzer.py`)
- **Prompt:** `prompts/4_trial_analysis.txt`
- **Input:** Patient assessment + all matched trials from Step 3
- **Process:** Synthesizes evidence across all relevant trials — eligibility, outcomes, efficacy, patient-specific interpretation
- **Output:** `patient_<ID>_trial_analysis.json` + raw response file

### Step 5 — Therapy Recommendation (`therapy_recommender.py`)
- **Prompt:** `prompts/5_therapy_recommendation.txt`
- **Input:** Patient assessment + trial analysis + full matched guideline text
- **Process:** Generates a structured therapy recommendation with primary recommendation, therapy options, rationale, and next steps
- **Output:** `patient_<ID>_therapy_recommendation.json` + prompt/response files

### Step 6 — Validation (`recommendation_validator.py`)
- **Prompt:** `prompts/6_recommendation_validator.txt` (for semantic layer)
- **Input:** Therapy recommendation + patient assessment + trial analysis + guidelines
- **Process:** Three-layer validation:
  1. **Structural** (rule-based): Required sections, content length, urgency alignment
  2. **Evidence consistency** (rule-based): Hallucinated NCT IDs, citation accuracy
  3. **Semantic** (LLM-based): Safety, accuracy, completeness, consistency
- **Decision:** `PASS` or `RECONSIDER`
- **Output:** `patient_<ID>_validation_results.json` + `patient_<ID>_validation_summary.txt`

### Step 7 — Smart Regeneration (`smart_regeneration.py`)
- **No new prompt file** — reuses existing prompts with feedback prepended
- **Triggers automatically** when validation returns `RECONSIDER`
- **Process:**
  1. Categorizes validation issues (safety, accuracy, completeness, consistency)
  2. Determines regeneration scope:
     - **Level 1:** Regenerate recommendation only
     - **Level 2:** Regenerate trial analysis + recommendation
  3. Prepends explicit feedback to the prompt and regenerates
  4. Re-validates the regenerated output
- **Output:** `patient_<ID>_therapy_recommendation_regenerated.json` + re-validation files

---

## Output Files per Patient

All outputs land in `results/agentic_outputs/batch_run_<timestamp>/patient_<ID>/`:

| File | Description |
|---|---|
| `patient_<ID>_guidelines.json` | Matched guidelines with relevance scores |
| `patient_<ID>_assessment.json` | Structured clinical assessment |
| `patient_<ID>_trial_matching.json` | Relevant + rejected trials |
| `patient_<ID>_trial_analysis.json` | Synthesized trial evidence |
| `patient_<ID>_therapy_recommendation.json` | Final therapy recommendation |
| `patient_<ID>_validation_results.json` | Detailed 3-layer validation results |
| `patient_<ID>_validation_summary.txt` | Human-readable validation report |
| `patient_<ID>_complete_workflow.json` | Summary of all steps and files |
| `patient_<ID>_workflow_progress.json` | Real-time progress tracking |

If regeneration occurred:

| File | Description |
|---|---|
| `patient_<ID>_therapy_recommendation_regenerated.json` | Fixed recommendation |
| `patient_<ID>_regeneration_context.txt` | Feedback provided to LLM |
| `patient_<ID>_validation_results_iter1.json` | Re-validation results |
| `patient_<ID>_validation_summary_iter1.txt` | Re-validation summary |

---

## Configuration (`config.py`)

Key settings:

| Variable | Description |
|---|---|
| `OPENROUTER_MODEL` | Active model (controlled by `USE_PRODUCTION_MODEL`) |
| `USE_PRODUCTION_MODEL` | `True` → `google/gemini-3-pro-preview`; `False` → free test model |
| `MODEL_CONTEXT_LIMITS` | Per-model token limits for dynamic truncation |
| `GUIDELINES_CACHE_DIR` / `TRIALS_CACHE_DIR` / etc. | Cache directories per model type |
| `DATA_ROOT_DIR` | Points to `prism/../data/` (relative to `config.py`) |
| `TUBO_EXCEL_FILE_PATH` | `data/NET Tubo v2.xlsx` (legacy reference) |

Cache is stored in `cache/cache_production/` or `cache/cache_test/` depending on `USE_PRODUCTION_MODEL`.

---

## Trial Collection (`clinical_trials/`)

Trials are fetched from the ClinicalTrials.gov v2 API using `study_collector.py`:

```bash
cd prism/clinical_trials/core_utilities/
python study_collector.py
```

This fetches Neuroendocrine Tumor studies (excluding Phase 1, with posted results) and saves them to `prism/results/collected_trials/`. The `run_full_pipeline.py` script does this automatically before running the agentic workflow.

---

## API Usage (per patient, no regeneration)

| Step | API calls |
|---|---|
| Guidelines matching | ~25 (one per guideline file) |
| Patient assessment | 1 |
| Trial matching | ~86 (one per trial) |
| Trial analysis | 1 |
| Therapy recommendation | 1 |
| Validation (semantic) | 1 |
| **Total** | **~115** |

If regeneration is needed: +2–3 additional calls. Smart regeneration reuses cached results for all other steps.

---

## Important Notes

- **Patient data file:** `data/Combined_patientCases.xlsx` — each row is one case, identified by an `ID` column.
- **Guidelines:** stored as `.md` files under `data/guidelines/2-0/` (ENET + ESMO guidelines).
- **Prompts are numbered** `1_` through `6_` to match pipeline steps exactly.
- **All recommendations** are decision support only and must be reviewed by a qualified clinician.
- **No absolute paths** are hardcoded anywhere — all paths are resolved relative to `config.py` or `__file__`.
