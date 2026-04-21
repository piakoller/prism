# shared_logic.py

import json
import logging
import warnings
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Langchain dependencies temporarily disabled for clinical_trials_matcher.py
# from langchain.prompts import PromptTemplate
# from langchain.schema.language_model import BaseLanguageModel

warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Initialize logger
logger = logging.getLogger(__name__)

# --- Project-specific Imports (ensure they are in PYTHONPATH) ---
try:
    import config
    from data_loader import load_patient_data
    from logging_setup import setup_logging
except ImportError as e:
    print(f"Error importing project modules: {e}")
    exit(1)

# --- Shared Configuration ---
LLM_TEMPERATURE = 0.0

# Data Directories
BASE_PROJECT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo")
# BASE_PROJECT_DIR = Path("/home/pia/projects/netTubo")

EVAL_DATA_DIR = BASE_PROJECT_DIR / "netTubo/data_for_evaluation/single_prompt"

# Configuration modes - these will be overridden by command line arguments
# Mode 1.0: Only one ESMO and one ENET Guideline
GUIDELINE_SOURCE_DIR_1_0 = BASE_PROJECT_DIR / "netTubo/data/guidelines/1-0_data_singleprompt/mds"
PROMPT_FILE_1_0 = BASE_PROJECT_DIR / "prompts/prompt_v2_english.txt"

# Mode 1.1: Guidelines + additional context (NETpress + NETstudy)
GUIDELINE_SOURCE_DIR_1_1 = BASE_PROJECT_DIR / "netTubo/data/guidelines/1-0_data_singleprompt/mds"
ADDITIONAL_CONTEXT_1_1 = BASE_PROJECT_DIR / "netTubo/data/guidelines/1-1_data_singleprompt/mds"
PROMPT_FILE_1_1 = BASE_PROJECT_DIR / "prompts/prompt_v3_1-1_english.txt"

# Mode 1.2: Guidelines + NEW_NET_EVIDENCE
GUIDELINE_SOURCE_DIR_1_2 = BASE_PROJECT_DIR / "netTubo/data/guidelines/1-0_data_singleprompt/mds"
NEW_NET_EVIDENCE_1_2 = BASE_PROJECT_DIR / "New_NET_evidence/mds_docling"
PROMPT_FILE_1_2 = BASE_PROJECT_DIR / "prompts/prompt_v3_1-2_english.txt"

# Mode 2.0: Patient-specific ENET guidelines
GUIDELINE_SOURCE_DIR_2_0 = BASE_PROJECT_DIR / "netTubo/data/guidelines/1-0_data_singleprompt/mds"
PROMPT_FILE_2_0 = BASE_PROJECT_DIR / "prompts/prompt_v4_2-0_english.txt"

# Default values (will be overridden by set_configuration_mode)
GUIDELINE_SOURCE_DIR = GUIDELINE_SOURCE_DIR_1_0
ADDITIONAL_CONTEXT = False
NEW_NET_EVIDENCE = False
PROMPT_FILE_PATH = PROMPT_FILE_1_0
CONFIGURATION_MODE = "1-0"

def set_configuration_mode(mode: str):
    """Set the configuration mode for guidelines and prompt selection."""
    global GUIDELINE_SOURCE_DIR, ADDITIONAL_CONTEXT, NEW_NET_EVIDENCE, PROMPT_FILE_PATH, CONFIGURATION_MODE
    
    CONFIGURATION_MODE = mode
    
    if mode == "1-0":
        GUIDELINE_SOURCE_DIR = GUIDELINE_SOURCE_DIR_1_0
        ADDITIONAL_CONTEXT = False
        NEW_NET_EVIDENCE = False
        PROMPT_FILE_PATH = PROMPT_FILE_1_0
    elif mode == "1-1":
        GUIDELINE_SOURCE_DIR = GUIDELINE_SOURCE_DIR_1_1
        ADDITIONAL_CONTEXT = ADDITIONAL_CONTEXT_1_1
        NEW_NET_EVIDENCE = False
        PROMPT_FILE_PATH = PROMPT_FILE_1_1
    elif mode == "1-2":
        GUIDELINE_SOURCE_DIR = GUIDELINE_SOURCE_DIR_1_2
        ADDITIONAL_CONTEXT = False
        NEW_NET_EVIDENCE = NEW_NET_EVIDENCE_1_2
        PROMPT_FILE_PATH = PROMPT_FILE_1_2
    elif mode == "2-0":
        GUIDELINE_SOURCE_DIR = GUIDELINE_SOURCE_DIR_2_0
        ADDITIONAL_CONTEXT = False
        NEW_NET_EVIDENCE = False
        PROMPT_FILE_PATH = PROMPT_FILE_2_0
    else:
        raise ValueError(f"Unknown configuration mode: {mode}. Available modes: 1-0, 1-1, 1-2, 2-0")
    
    print(f'Configuration mode: {mode}')
    print(f'Guideline source: {GUIDELINE_SOURCE_DIR}')
    print(f'Additional context: {ADDITIONAL_CONTEXT}')
    print(f'NEW NET evidence: {NEW_NET_EVIDENCE}')
    print(f'Prompt version: {PROMPT_FILE_PATH}')

# Initialize with default mode 1-0
set_configuration_mode("1-0")

# Patient data fields to include in the prompt
PATIENT_FIELDS_FOR_PROMPT = [
    "id", "tumorboard_datum", "main_diagnosis_text", "Fragestellung"
]

# Constants for prompt tags
TAG_ASSESSMENT = "<beurteilung>"
TAG_RECOMMENDATION = "<therapieempfehlung>"
TAG_RATIONALE = "<begründung>"

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("shared_logic")

def get_prompt_version_from_path(prompt_path: Path) -> str:
    """Extract prompt version from prompt file path."""
    match = re.search(r'prompt_(v\d+(?:_[\d-]+)?)\.txt$', str(prompt_path))
    return match.group(1) if match else "unknown"

PROMPT_VERSION = get_prompt_version_from_path(PROMPT_FILE_PATH)

def _sanitize_tag_name(filename: str) -> str:
    """Converts a filename into a valid, clean XML-like tag name."""
    name = Path(filename).stem
    name = re.sub(r'[\s-]', '_', name)
    name = re.sub(r'[^\w_]', '', name)
    return name.lower()

def load_structured_guidelines(guideline_dir: Path, additional_dir: Optional[Path] = None, new_evidence_dir: Optional[Path] = None) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Recursively finds guideline files and organizes them by their source subdirectory.
    
    Args:
        guideline_dir: Primary directory containing guideline files
        additional_dir: Optional additional directory with more guidelines (1-1 mode)
        new_evidence_dir: Optional directory with new NET evidence (1-2 mode)
    """
    structured_docs: Dict[str, Dict[str, str]] = {}
    loaded_files: List[str] = []
    
    # Helper function to process a directory
    def process_directory(dir_path: Path, base_dir: Path, section_name: str = None) -> None:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Guidelines directory {dir_path} does not exist")
            return
            
        for item in sorted(dir_path.iterdir()):
            if section_name:
                source_name = section_name
            else:
                source_name = item.name.lower()
            
            files_to_load = []
            if item.is_dir():
                if source_name not in structured_docs:
                    structured_docs[source_name] = {}
                files_to_load = list(item.glob("*.md")) + list(item.glob("*.mds"))
            elif item.is_file() and item.suffix in ['.md', '.mds']:
                if section_name:
                    source_name = section_name
                else:
                    source_name = "root"
                if source_name not in structured_docs:
                    structured_docs[source_name] = {}
                files_to_load = [item]
            
            for file in sorted(files_to_load):
                try:
                    content = file.read_text(encoding='utf-8')
                    structured_docs[source_name][file.name] = content
                    loaded_files.append(str(file.relative_to(base_dir.parent if not section_name else base_dir)))
                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}")
    
    # Process primary directory (main guidelines)
    process_directory(guideline_dir, guideline_dir)
    
    # Process additional directory if provided (1-1 mode)
    if additional_dir:
        process_directory(additional_dir, additional_dir, "additional_context")
    
    # Process new evidence directory if provided (1-2 mode)
    if new_evidence_dir:
        process_directory(new_evidence_dir, new_evidence_dir, "new_net_evidence")
    
    return structured_docs, loaded_files

def format_patient_data_for_prompt(patient_row: Dict, fields: List[str]) -> str:
    """Formats patient data into a string for the LLM prompt."""
    lines = ["Patienteninformationen:"]
    field_map = {k.lower(): k for k in patient_row.keys()}  # Create case-insensitive field mapping
    
    for field in fields:
        actual_field = field_map.get(field.lower())  # Try to find the actual field name
        if actual_field:
            value = patient_row.get(actual_field)
            if value and str(value).strip():
                field_name_pretty = field.replace("_", " ").title()
                lines.append(f"- {field_name_pretty}: {str(value)}")
    return "\n".join(lines)

def load_clinical_trials_summary(clinical_trials_file: str = None) -> Dict:
    """Load the enhanced clinical trials summary file."""
    if not clinical_trials_file:
        # Look for the most recent enhanced summary file
        ct_dir = Path("clinical_trials_matches")
        if not ct_dir.exists():
            logger.warning("No clinical_trials_matches directory found")
            return {}
        
        enhanced_files = list(ct_dir.glob("clinical_trials_summary_enhanced_*.json"))
        if not enhanced_files:
            # Fallback to regular summary
            regular_file = ct_dir / "clinical_trials_summary_llm.json"
            if regular_file.exists():
                clinical_trials_file = str(regular_file)
                logger.info(f"Using regular clinical trials summary: {clinical_trials_file}")
            else:
                logger.warning("No clinical trials summary file found")
                return {}
        else:
            # Use the most recent enhanced file
            enhanced_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            clinical_trials_file = str(enhanced_files[0])
            logger.info(f"Using enhanced clinical trials summary: {clinical_trials_file}")
    
    try:
        with open(clinical_trials_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded clinical trials data from {clinical_trials_file}")
        return data
    except Exception as e:
        logger.error(f"Failed to load clinical trials summary from {clinical_trials_file}: {e}")
        return {}

def extract_study_evidence(study_data: Dict) -> str:
    """Extract the most relevant evidence from a clinical trial study."""
    evidence_parts = []
    
    # Get basic study info
    nct_id = study_data.get("nct_id", "")
    title = study_data.get("title", "")
    status = study_data.get("status", "")
    phase = study_data.get("phase", "")
    condition = study_data.get("condition", "")
    intervention = study_data.get("intervention", "")
    
    study_header = f"**{nct_id}: {title}**"
    study_info = f"Status: {status}, Phase: {phase}"
    if condition:
        study_info += f", Condition: {condition}"
    if intervention:
        study_info += f", Intervention: {intervention}"
    
    evidence_parts.append(study_header)
    evidence_parts.append(study_info)
    
    # Add brief summary if available
    brief_summary = study_data.get("brief_summary", "")
    if brief_summary and len(brief_summary) > 50:
        summary_text = brief_summary[:400] + "..." if len(brief_summary) > 400 else brief_summary
        evidence_parts.append(f"Summary: {summary_text}")
    
    # Add relevance information
    relevance_score = study_data.get("relevance_score", 0)
    relevance_reason = study_data.get("relevance_reason", "")
    if relevance_score > 0:
        evidence_parts.append(f"Relevance Score: {relevance_score}")
    if relevance_reason:
        reason_text = relevance_reason[:300] + "..." if len(relevance_reason) > 300 else relevance_reason
        evidence_parts.append(f"Relevance: {reason_text}")
    
    # Extract publication evidence (most important)
    enhanced_pubs = study_data.get("enhanced_publications", {})
    if enhanced_pubs:
        evidence_parts.append("**Publication Evidence:**")
        
        # Listed publications
        listed_pubs = enhanced_pubs.get("listed_publications", [])
        for pub in listed_pubs[:2]:  # Limit to first 2
            citation = pub.get("citation", "")
            if citation:
                evidence_parts.append(f"- {citation}")
        
        # Online search results with abstracts/content
        online_results = enhanced_pubs.get("online_search_results", {})
        
        # PubMed results (priority)
        pubmed_results = online_results.get("pubmed_results", [])
        for result in pubmed_results[:2]:  # Limit to first 2 most relevant
            title = result.get("title", "")
            abstract = result.get("abstract_text", "")
            full_content = result.get("full_content", "")
            
            if title:
                evidence_parts.append(f"- PubMed: {title}")
            
            # Use abstract if available, otherwise use content preview
            content_to_use = abstract or full_content
            if content_to_use:
                # Limit content length for prompt efficiency
                content_preview = content_to_use[:500] + "..." if len(content_to_use) > 500 else content_to_use
                evidence_parts.append(f"  Abstract/Results: {content_preview}")
        
        # Onclive results (clinical significance)
        onclive_results = online_results.get("onclive_results", [])
        for result in onclive_results[:1]:  # Limit to 1 most relevant
            title = result.get("title", "")
            full_content = result.get("full_content", "")
            
            if title:
                evidence_parts.append(f"- Onclive: {title}")
            
            if full_content:
                content_preview = full_content[:400] + "..." if len(full_content) > 400 else full_content
                evidence_parts.append(f"  Clinical Commentary: {content_preview}")
        
        # Google Scholar results (academic evidence)
        scholar_results = online_results.get("google_scholar_results", [])
        for result in scholar_results[:1]:  # Limit to 1 most relevant
            title = result.get("title", "")
            abstract = result.get("abstract_text", "")
            
            if title and abstract:
                evidence_parts.append(f"- Academic: {title}")
                abstract_preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                evidence_parts.append(f"  Abstract: {abstract_preview}")
    
    # Add eligibility criteria summary if relevant
    eligibility = study_data.get("eligibility_criteria", "")
    if eligibility:
        # Extract key inclusion/exclusion criteria
        eligibility_summary = eligibility[:200] + "..." if len(eligibility) > 200 else eligibility
        evidence_parts.append(f"Key Eligibility: {eligibility_summary}")
    
    return "\n".join(evidence_parts)

def format_clinical_trials_for_prompt(patient_id: str, clinical_trials_data: Dict) -> str:
    """Format clinical trials data for a specific patient into prompt text."""
    
    if not clinical_trials_data:
        return ""
    
    # Find patient data in clinical trials summary
    patient_matches = None
    
    # Check if it's an enhanced summary (with "patients" or "patient_results" key) or regular summary (direct list)
    if "patients" in clinical_trials_data:
        patients_data = clinical_trials_data["patients"]
    elif "patient_results" in clinical_trials_data:
        patients_data = clinical_trials_data["patient_results"]
    else:
        patients_data = clinical_trials_data if isinstance(clinical_trials_data, list) else []
    
    for patient_data in patients_data:
        if str(patient_data.get("patient_id", "")) == str(patient_id):
            patient_matches = patient_data
            break
    
    if not patient_matches:
        logger.warning(f"No clinical trials matches found for patient {patient_id}")
        return ""
    
    matches = patient_matches.get("matches", [])
    if not matches:
        return ""
    
    # Format the clinical trials section
    ct_sections = []
    ct_sections.append("**RELEVANTE KLINISCHE STUDIEN FÜR DIESEN PATIENTEN:**")
    ct_sections.append("")
    
    # Add summary statistics
    total_matches = len(matches)
    ct_sections.append(f"Anzahl relevanter Studien: {total_matches}")
    
    # Get evaluation info
    eval_method = patient_matches.get("llm_evaluation_method", "")
    min_score = patient_matches.get("min_relevance_score", "")
    if eval_method:
        ct_sections.append(f"Evaluiert mit: {eval_method} (Min. Relevanz-Score: {min_score})")
    
    ct_sections.append("")
    
    # Format each study (include ALL relevant studies)
    sorted_matches = sorted(matches, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    for i, study in enumerate(sorted_matches, 1):
        ct_sections.append(f"=== STUDIE {i}/{total_matches} ===")
        study_evidence = extract_study_evidence(study)
        ct_sections.append(study_evidence)
        ct_sections.append("")  # Add spacing between studies
    
    return "\n".join(ct_sections)

def build_prompt(patient_data_string: str, guidelines_context_string: str, patient_publications_string: str = "", clinical_trials_string: str = "") -> str:
    """Builds the complete prompt with patient data, guidelines, patient-specific publications, and clinical trials data."""

    # Read the prompt template from the file
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        logger.error(f"Failed to read the prompt file {PROMPT_FILE_PATH}: {e}")
        raise

    # Format the prompt with the provided variables first
    formatted_prompt = prompt_template.format(
        patient_data_string=patient_data_string,
        guidelines_context_string=guidelines_context_string
    )
    
    # Add patient publications section if available - insert after guidelines_context section
    if patient_publications_string:
        formatted_prompt = formatted_prompt.replace(
            "</guidelines_context>",
            "</guidelines_context>\n\n<patient_publications>\nHier sind die patientenspezifischen Publikationen und Studien:\n" + patient_publications_string + "\n</patient_publications>"
        )
    
    # Add clinical trials section if available - insert after patient publications or guidelines
    if clinical_trials_string:
        # Find the insertion point (after patient publications if they exist, otherwise after guidelines)
        if patient_publications_string:
            insertion_point = "</patient_publications>"
            replacement = "</patient_publications>\n\n<clinical_trials_evidence>\n" + clinical_trials_string + "\n</clinical_trials_evidence>"
        else:
            insertion_point = "</guidelines_context>"
            replacement = "</guidelines_context>\n\n<clinical_trials_evidence>\n" + clinical_trials_string + "\n</clinical_trials_evidence>"
        
        formatted_prompt = formatted_prompt.replace(insertion_point, replacement)

    return formatted_prompt

def format_guidelines_for_prompt(
    structured_docs: Dict[str, Dict[str, str]]
) -> str:
    """Formats structured guideline dictionaries into XML-like tagged text."""
    context_parts = ["<guidelines_context>"]
    
    # Process each section
    for source, files in structured_docs.items():
        if source == "additional_context":
            # Handle 1-1 mode: additional context (NETpress, NETstudy)
            context_parts.append("<additional_context>")
            for filename, content in files.items():
                file_tag = _sanitize_tag_name(filename)
                context_parts.append(f"<{file_tag}>\n{content}\n</{file_tag}>")
            context_parts.append("</additional_context>")
        elif source == "new_net_evidence":
            # Handle 1-2 mode: NEW_NET_EVIDENCE
            context_parts.append("<new_net_evidence>")
            for filename, content in files.items():
                # Use the original filename (without extension) as tag, preserving spaces and hyphens
                file_tag = Path(filename).stem
                context_parts.append(f"<{file_tag}>\n{content}\n</{file_tag}>")
            context_parts.append("</new_net_evidence>")
        else:
            # Handle main guidelines (ESMO, ENET)
            for filename, content in files.items():
                file_tag = _sanitize_tag_name(filename)
                context_parts.append(f"<{file_tag}>\n{content}\n</{file_tag}>")
    
    context_parts.append("</guidelines_context>")
    return "\n".join(context_parts)

def format_patient_data_for_prompt(patient_row: Dict, fields: List[str]) -> str:
    """Formats patient data into a string for the LLM prompt."""
    lines = ["Patienteninformationen:"]
    for field in fields:
        value = patient_row.get(field)
        if value and str(value).strip():
            field_name_pretty = field.replace("_", " ").title()
            lines.append(f"- {field_name_pretty}: {str(value)}")
    return "\n".join(lines)


def _parse_llm_response(response_text: str) -> Dict[str, Optional[str]]:
    """Extracts content from specified tags in the LLM's response."""
    def extract_tag_content(tag: str, text: str) -> Optional[str]:
        pattern = f"{tag}(.*?)</{tag[1:]}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    return {
        "assessment": extract_tag_content(TAG_ASSESSMENT, response_text),
        "recommendation": extract_tag_content(TAG_RECOMMENDATION, response_text),
        "rationale": extract_tag_content(TAG_RATIONALE, response_text),
    }

def load_patient_publications(patient_id: str, base_publications_dir: Optional[Path] = None) -> Tuple[Dict[str, str], List[str]]:
    """
    Load patient-specific publications from mds folder.
    
    Args:
        patient_id: The patient ID (e.g., "1", "2", etc.)
        base_publications_dir: Base directory containing patient folders. If None, uses default.
    
    Returns:
        Tuple of (publications_dict, loaded_files_list)
    """
    if base_publications_dir is None:
        base_publications_dir = BASE_PROJECT_DIR / "netTubo" / "clinical_trials_matches" / "publications"
    
    patient_folder = base_publications_dir / f"patient_{patient_id}" / "mds"
    publications = {}
    loaded_files = []
    
    if not patient_folder.exists():
        logger.debug(f"No patient-specific publications found for patient {patient_id} (folder: {patient_folder})")
        return publications, loaded_files
    
    logger.info(f"Loading patient-specific publications from: {patient_folder}")
    
    # Load all markdown files in the patient's mds folder
    for md_file in patient_folder.glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    publications[md_file.name] = content
                    loaded_files.append(str(md_file))
                    logger.debug(f"Loaded patient publication: {md_file.name}")
        except Exception as e:
            logger.error(f"Failed to load patient publication {md_file}: {e}")
    
    logger.info(f"Loaded {len(publications)} patient-specific publications for patient {patient_id}")
    return publications, loaded_files

def format_patient_publications_for_prompt(publications: Dict[str, str]) -> str:
    """Format patient-specific publications for inclusion in the prompt."""
    if not publications:
        return ""
    
    context_parts = ["<patient_publications>"]
    
    for filename, content in publications.items():
        file_tag = _sanitize_tag_name(filename)
        context_parts.append(f"<{file_tag}>\n{content}\n</{file_tag}>")
    
    context_parts.append("</patient_publications>")
    return "\n".join(context_parts)