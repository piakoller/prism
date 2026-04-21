import os
import sys
import glob
from pathlib import Path
import logging

# Add current directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clinical_trials.core_utilities.study_collector import ClinicalTrialsAPI, StudyCollector
from agentic_assessment.agentic_workflow import AgenticWorkflow
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Starting End-to-End Pipeline in Prism")
    
    # 1. Trial API Extracting
    logger.info("--- Phase 1: Trial API Extraction ---")
    studies_dir = Path("results/collected_trials")
    studies_dir.mkdir(parents=True, exist_ok=True)
    
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    collector = StudyCollector(api, studies_dir)
    
    # We collect the NET studies from the API
    studies, query_used = collector.collect_net_studies_simple()
    if not studies:
        logger.error("No studies were collected. Pipeline cannot proceed.")
        return
        
    collector.save_studies(studies, "net_studies_all", query_used)
    
    # Find the recently saved JSON path (save_studies appends a timestamp)
    json_files = glob.glob(str(studies_dir / "*.json"))
    if not json_files:
        logger.error("Could not find the saved trials JSON.")
        return
        
    # Sort files by creation time
    json_files.sort(key=os.path.getmtime)
    trials_json_path = json_files[-1]
    
    logger.info(f"Trials extracted and saved to {trials_json_path}")
    
    # 2. Complete Agentic Assessment Pipeline
    logger.info("--- Phase 2: Agentic Workflow Execution ---")
    
    # Ensure OpenRouter API Key is set in environment (from .env or system)
    api_key = config.OPENROUTER_API_KEY
    if not api_key or api_key == "your_openrouter_api_key_here":
        logger.error("Please set a valid OPENROUTER_API_KEY in the .env file.")
        return
        
    workflow = AgenticWorkflow(
        api_key=api_key,
        output_dir="results/agentic_outputs"
    )
    
    patient_cases_path = "data/Combined_patientCases.xlsx"
    guidelines_dir = "data/guidelines"
    
    # We will run the batch on the patients.
    # Note: To test quickly, you can run a single patient using run_single_patient instead.
    # For full pipeline, we use run_batch_patients
    results = workflow.run_batch_patients(
        patients_file=patient_cases_path,
        trial_data_path=trials_json_path,
        guidelines_dir=guidelines_dir,
        output_dir="results/agentic_outputs",
        use_cache=True,
        force_regenerate=False
    )
    
    logger.info("--- Pipeline Completed ---")
    
if __name__ == "__main__":
    run_pipeline()
