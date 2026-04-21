#!/usr/bin/env python3
"""
Patient Assessor Module

This module handles the initial patient assessment using LLM with guidelines context.
It creates a structured assessment that identifies clinical needs and potential gaps.
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL, MODEL_CONTEXT_LIMITS, ASSESSMENT_CACHE_DIR

logger = logging.getLogger(__name__)

class PatientAssessor:
    """Handles initial patient assessment using LLM"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, prompt_file: Optional[str] = None):
        """
        Initialize the Patient Assessor
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: LLM model to use for assessment (defaults to config.OPENROUTER_MODEL)
            prompt_file: Path to the patient assessment prompt file (defaults to standard location)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        # Set default prompt file path
        if prompt_file is None:
            # Look for prompt file in the prompts directory
            base_dir = Path(__file__).parent.parent.parent
            # Use v2 prompt that excludes trials and focuses on patient data + guidelines
            prompt_file = str(base_dir / "prompts" / "2_patient_assessment.txt")
        
        self.prompt_file = Path(prompt_file)
        self.prompt_template = self._load_prompt_template()
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        logger.info(f"🧠 Patient Assessor initialized with model: {self.model}")
        logger.info(f"📄 Using prompt file: {self.prompt_file}")

        # Initialize cache directory
        self.cache_dir = Path(ASSESSMENT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, patient_data: Dict[str, Any], guidelines_context: Optional[str], matched_guidelines: Optional[Dict[str, Any]]) -> str:
        """Generate a cache key for assessment based on stable inputs."""
        import hashlib

        patient_id = str(patient_data.get('ID', patient_data.get('id', 'unknown')))
        # Use a concise hash of the question and key fields to avoid giant keys
        core_text = f"{patient_data.get('diagnosis','')}_{patient_data.get('Stage', patient_data.get('stage',''))}_{patient_data.get('question_for_tumorboard','')}"
        guidelines_count = 0
        if matched_guidelines and matched_guidelines.get('relevant_guidelines'):
            guidelines_count = len(matched_guidelines['relevant_guidelines'])
        gctx_len = len(guidelines_context) if isinstance(guidelines_context, str) else 0
        raw = f"{patient_id}|{core_text[:200]}|g{guidelines_count}|gl{gctx_len}|model:{self.model}"
        h = hashlib.md5(raw.encode('utf-8')).hexdigest()[:8]
        return f"patient_{patient_id}_assessment_{h}"

    def _get_cache_filepath(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load assessment from cache if available and compatible."""
        cache_file = self._get_cache_filepath(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                cached_model = data.get('metadata', {}).get('model_used') or data.get('model_used')
                if cached_model is None:
                    logger.info("🧩 Assessment cache missing model metadata; migrating and accepting")
                    data.setdefault('metadata', {})
                    data['metadata']['model_used'] = self.model
                    data['metadata'].setdefault('timestamp', datetime.now().isoformat())
                    data['metadata']['cache_reused'] = True
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as wf:
                            json.dump(data, wf, indent=2, ensure_ascii=False)
                    except Exception as se:
                        logger.warning(f"⚠️ Failed to write migrated assessment cache: {se}")
                    return data
                if cached_model == self.model:
                    logger.info(f"📂 Loaded assessment from cache: {cache_file}")
                    # mark reuse
                    if 'metadata' in data:
                        data['metadata']['cache_reused'] = True
                        data['metadata']['cached_at'] = datetime.now().isoformat()
                    return data
                else:
                    logger.info(f"🔄 Assessment cache model mismatch ({cached_model} vs {self.model}); regenerating")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load assessment cache {cache_file}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        try:
            cache_file = self._get_cache_filepath(cache_key)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved assessment to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save assessment cache: {e}")
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the text file."""
        if not self.prompt_file.exists():
            error_msg = f"❌ CRITICAL ERROR: Prompt file not found: {self.prompt_file}"
            logger.error(error_msg)
            print(error_msg)
            print("Please ensure the patient assessment prompt file exists before running the patient assessor.")
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            
            logger.info(f"✅ Loaded prompt template from: {self.prompt_file}")
            return prompt_content
            
        except Exception as e:
            error_msg = f"❌ CRITICAL ERROR: Failed to load prompt template: {e}"
            logger.error(error_msg)
            print(error_msg)
            print("Please check the prompt file permissions and content.")
            raise RuntimeError(f"Failed to load prompt template: {e}")
    
    def assess_patient(self, patient_data: Dict[str, Any], guidelines_context: Optional[str] = None, 
                      matched_guidelines: Optional[Dict[str, Any]] = None, matched_trials: Optional[Dict[str, Any]] = None, 
                      output_dir: str = ".", use_cache: bool = True) -> Dict[str, Any]:
        """
        Perform initial patient assessment
        
        Args:
            patient_data: Dictionary containing patient information
            guidelines_context: Optional general guidelines context
            matched_guidelines: Optional matched guidelines from guidelines_matcher
            matched_trials: Optional matched trials from trial_matcher
            output_dir: Directory to save assessment results
            use_cache: Whether to use cached assessment if available (default: True)
            
        Returns:
            Dictionary containing assessment results
        """
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        logger.info(f"📋 Assessing patient ID: {patient_id}")
        
        # Prepare contexts upfront to avoid scope issues later
        # Enhanced guidelines context combines optional general context with matched guidelines
        enhanced_guidelines_context = self._prepare_guidelines_context(guidelines_context, matched_guidelines)
        
        # Trials must not be used in assessment; enforce empty context
        matched_trials_context = None
        
        # Check for existing cached assessment (model-aware, centralized cache)
        assessment_file = os.path.join(output_dir, f"patient_{patient_id}_assessment.json")
        cache_key = self._get_cache_key(patient_data, guidelines_context, matched_guidelines)
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached and cached.get('assessment_text') and cached.get('clinical_summary'):
                # Also save a copy into the current output_dir for audit trail
                try:
                    with open(assessment_file, 'w', encoding='utf-8') as f:
                        json.dump(cached, f, indent=2, ensure_ascii=False)
                    logger.info(f"📄 Cached assessment copied to run folder: {assessment_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed copying cached assessment to run folder: {e}")
                return cached
            # Fallback: reuse legacy run-local assessment file as cache and migrate
            if os.path.exists(assessment_file):
                try:
                    with open(assessment_file, 'r', encoding='utf-8') as f:
                        legacy = json.load(f)
                    if legacy.get('assessment_text') and legacy.get('clinical_summary'):
                        logger.info("♻️ Using legacy run-local assessment as cache and migrating to centralized cache")
                        legacy.setdefault('metadata', {})
                        legacy['metadata'].setdefault('timestamp', datetime.now().isoformat())
                        legacy['metadata']['model_used'] = legacy['metadata'].get('model_used', self.model)
                        legacy['metadata']['cache_reused'] = True
                        legacy['metadata']['cached_at'] = datetime.now().isoformat()
                        # Save to centralized cache for future runs
                        self._save_to_cache(cache_key, legacy)
                        return legacy
                    else:
                        logger.info("ℹ️ Legacy assessment file exists but incomplete; will regenerate")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load legacy assessment file: {e}")
        
        # Create assessment prompt
        assessment_prompt = self._create_assessment_prompt(patient_data, enhanced_guidelines_context, matched_trials_context)
        
        # Save the prompt for debugging/monitoring
        prompt_file = os.path.join(output_dir, f"patient_{patient_id}_assessment_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write("=== PATIENT ASSESSMENT PROMPT ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 50 + "\n\n")
            f.write(assessment_prompt)
        
        logger.info(f"📄 Assessment prompt saved to: {prompt_file}")
        
        # Call LLM for assessment
        try:
            logger.info("🔗 Calling LLM API for assessment...")
            assessment_response = self._call_llm(assessment_prompt)
            
            # Save the raw API response
            response_file = os.path.join(output_dir, f"patient_{patient_id}_assessment_raw_response.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("=== RAW LLM RESPONSE ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Model: {self.model}\n")
                f.write("=" * 50 + "\n\n")
                f.write(assessment_response)
            
            logger.info(f"📄 Raw response saved to: {response_file}")
            
            assessment_result = self._parse_assessment_response(assessment_response)
            
            # Add metadata
            assessment_result['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'patient_id': patient_data.get('ID', 'unknown'),
                'guidelines_provided': guidelines_context is not None,
                'matched_guidelines_used': matched_guidelines is not None,
                'matched_guidelines_count': len(matched_guidelines.get('relevant_guidelines', [])) if matched_guidelines else 0,
                'matched_trials_used': False,
                'matched_trials_count': 0,
                'prompt_file': prompt_file,
                'response_file': response_file,
                'cache_reused': False,  # This is a new assessment, not cached
                'cached_at': None
            }
            
            # Save assessment to file and to centralized cache
            with open(assessment_file, 'w', encoding='utf-8') as f:
                json.dump(assessment_result, f, indent=2, ensure_ascii=False)
            self._save_to_cache(cache_key, assessment_result)
            
            logger.info(f"✅ Assessment completed and saved to: {assessment_file}")
            
            return assessment_result
            
        except Exception as e:
            logger.error(f"❌ Error during patient assessment: {e}")
            raise
    
    def _prepare_guidelines_context(self, guidelines_context: Optional[str] = None, 
                                   matched_guidelines: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Prepare enhanced guidelines context by combining general guidelines with matched guidelines
        
        Args:
            guidelines_context: General guidelines context
            matched_guidelines: Matched guidelines from guidelines_matcher
            
        Returns:
            Enhanced guidelines context string
        """
        context_parts = []
        
        # Add general guidelines context if provided
        if guidelines_context:
            context_parts.append("GENERAL MEDICAL GUIDELINES:")
            context_parts.append(guidelines_context.strip())
            context_parts.append("")
        
        # Add matched guidelines if provided
        if matched_guidelines and matched_guidelines.get('relevant_guidelines'):
            context_parts.append("PATIENT-SPECIFIC MATCHED GUIDELINES:")
            context_parts.append("The following guidelines have been automatically identified as relevant to this patient:")
            context_parts.append("")
            
            for i, guideline in enumerate(matched_guidelines['relevant_guidelines'], 1):
                guideline_title = guideline.get('guideline_title', 'Unknown Guideline')
                relevance_reason = guideline.get('relevance_reason', 'No reason provided')
                relevance_score = guideline.get('relevance_score', 0)
                
                context_parts.append(f"{i}. **{guideline_title}** (Relevance Score: {relevance_score:.1f})")
                context_parts.append(f"   Clinical Relevance: {relevance_reason}")
                context_parts.append("")
                
                # Include the full guideline content if available
                guideline_content = guideline.get('guideline_content', '')
                if guideline_content:
                    context_parts.append(f"   Full Guideline Content:")
                    context_parts.append(f"   {guideline_content}")
                    context_parts.append("")
            
            guidelines_count = len(matched_guidelines['relevant_guidelines'])
            guidelines_evaluated = matched_guidelines.get('metadata', {}).get('guidelines_evaluated', 'unknown')
            context_parts.append(f"Note: {guidelines_count} relevant guidelines identified from {guidelines_evaluated} total guidelines evaluated.")
        
        return "\n".join(context_parts) if context_parts else None

    def _prepare_matched_trials_context(self, matched_trials: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare matched trials context including inclusion/exclusion criteria and published results
        
        Args:
            matched_trials: Matched trials from trial_matcher
            
        Returns:
            Formatted trials context string
        """
        if not matched_trials or not matched_trials.get('relevant_trials'):
            return "No specific matched trials provided."
        
        context_parts = []
        context_parts.append("MATCHED CLINICAL TRIALS:")
        context_parts.append("The following clinical trials have been identified as potentially relevant to this patient:")
        context_parts.append("")
        
        for i, trial in enumerate(matched_trials['relevant_trials'], 1):
            trial_title = trial.get('title', trial.get('official_title', trial.get('brief_title', 'Unknown Trial')))
            nct_id = trial.get('nct_id', 'Unknown NCT')
            relevance_score = trial.get('relevance_score', 0)
            overall_status = trial.get('overall_status', 'Unknown')
            
            context_parts.append(f"{i}. **{trial_title}** ({nct_id})")
            context_parts.append(f"   Status: {overall_status}")
            context_parts.append(f"   Relevance Score: {relevance_score:.2f}")
            context_parts.append("")
            
            # Include inclusion criteria
            inclusion_criteria = trial.get('eligibility', {}).get('criteria', {}).get('textblock', '')
            if inclusion_criteria:
                context_parts.append("   **Inclusion/Exclusion Criteria:**")
                # For production model with 1M context, include full criteria
                if self.model == "google/gemini-2.5-pro":
                    # Full content, no truncation for production model
                    context_parts.append(f"   {inclusion_criteria}")
                else:
                    # Truncate if too long for test models with limited context
                    if len(inclusion_criteria) > 1000:
                        inclusion_criteria = inclusion_criteria[:1000] + "... [truncated]"
                    context_parts.append(f"   {inclusion_criteria}")
                context_parts.append("")
            
            # Include published results if available
            published_results = self._extract_published_results(trial)
            if published_results:
                context_parts.append("   **Published Results:**")
                context_parts.append(f"   {published_results}")
                context_parts.append("")
            else:
                context_parts.append("   **Published Results:** No published results available.")
                context_parts.append("")
            
            # Include primary outcome measures
            primary_outcome = trial.get('primary_outcome')
            if primary_outcome and isinstance(primary_outcome, list) and len(primary_outcome) > 0:
                outcome_measure = primary_outcome[0].get('measure', 'Not specified')
                context_parts.append(f"   **Primary Outcome:** {outcome_measure}")
                context_parts.append("")
        
        trials_count = len(matched_trials['relevant_trials'])
        trials_evaluated = matched_trials.get('metadata', {}).get('trials_evaluated', 'unknown')
        context_parts.append(f"Note: {trials_count} relevant trials identified from {trials_evaluated} total trials evaluated.")
        
        return "\n".join(context_parts)
    
    def _extract_published_results(self, trial: Dict[str, Any]) -> str:
        """
        Extract published results from trial data
        
        Args:
            trial: Trial data dictionary
            
        Returns:
            Formatted published results string
        """
        # Try to extract from publication_analysis
        pub_analysis = trial.get('publication_analysis', {})
        if pub_analysis:
            # Check for online search results
            online_results = pub_analysis.get('online_search_results', {})
            if online_results:
                pubmed_results = online_results.get('pubmed', {})
                if pubmed_results:
                    publications = pubmed_results.get('publications', [])
                    if publications:
                        # Get the first publication's content
                        first_pub = publications[0]
                        
                        # Try full_content first, then abstract_text
                        content = first_pub.get('full_content', '')
                        if not content:
                            content = first_pub.get('abstract_text', '')
                        
                        if content:
                            # For production model with 1M context, include full content
                            if self.model == "google/gemini-2.5-pro":
                                # Full content, no truncation for production model
                                publication_summary = content
                            else:
                                # Truncate if too long for test models with limited context
                                if len(content) > 500:
                                    content = content[:500] + "... [truncated]"
                                publication_summary = content
                            return publication_summary
                        
                        # If no content, at least show title and basic info
                        title = first_pub.get('title', 'Unknown Title')
                        authors = first_pub.get('authors', 'Unknown Authors')
                        return f"Publication: {title} by {authors}"
        
        # Check for results_first_posted
        results_first_posted = trial.get('results_first_posted')
        if results_first_posted:
            return f"Results first posted: {results_first_posted}"
        
        # Check for study_results
        study_results = trial.get('study_results')
        if study_results:
            return "Study results available (detailed analysis required)"
        
        return ""

    def _create_assessment_prompt(self, patient_data: Dict[str, Any], guidelines_context: Optional[str] = None, 
                                 matched_trials_context: Optional[str] = None) -> str:
        """Create the assessment prompt using the loaded prompt template with size management."""
        
        # Format patient data
        patient_info = []
        for key, value in patient_data.items():
            # IMPORTANT: Exclude expert_recommendation and source - we don't want to give the LLM the answer!
            if key.lower() in ['id', 'expert_recommendation', 'source']:
                continue
            if value and str(value).strip():
                patient_info.append(f"- {key}: {value}")
        patient_data_string = "\n".join(patient_info)
        
        # Prepare matched guidelines context
        matched_guidelines = guidelines_context if guidelines_context else "No specific guidelines provided."
        
        # Trials must not be used in assessment; force empty string even if context is passed
        matched_trials = ""
        
        # Create initial prompt
        initial_prompt = self.prompt_template.format(
            patient_data_string=patient_data_string,
            matched_guidelines=matched_guidelines,
            matched_trials=matched_trials
        )
        
        # Check prompt size and truncate if necessary
        prompt_size = len(initial_prompt)
        
        # Get model-specific context limit (convert tokens to characters, rough estimate: 1 token ≈ 4 characters)
        model_token_limit = MODEL_CONTEXT_LIMITS.get(self.model, 128_000)  # Default to 128K tokens
        max_size = int(model_token_limit * 3.5)  # Conservative estimate: 3.5 chars per token
        
        logger.info(f"📏 Model: {self.model}, Token limit: {model_token_limit:,}, Character limit: {max_size:,}")
        
        if prompt_size > max_size:
            logger.warning(f"⚠️ Prompt too large ({prompt_size:,} chars), considering truncation of guidelines content...")
            
            # Avoid truncation for production model with 1M context; truncate only on smaller models
            if self.model != "google/gemini-2.5-pro" and guidelines_context and len(guidelines_context) > 50000:
                # Keep first 30k characters of guidelines and add truncation note
                truncated_guidelines = guidelines_context[:30000] + "\n\n[GUIDELINES CONTENT TRUNCATED DUE TO SIZE LIMITS - SHOWING FIRST 30K CHARACTERS ONLY]"
                
                # Recreate prompt with truncated guidelines
                final_prompt = self.prompt_template.format(
                    patient_data_string=patient_data_string,
                    matched_guidelines=truncated_guidelines,
                    matched_trials=matched_trials
                )
                
                final_size = len(final_prompt)
                logger.info(f"📏 Prompt size reduced from {prompt_size:,} to {final_size:,} characters")
                return final_prompt
            
            # If still too large, truncate further
            if len(initial_prompt) > max_size:
                logger.warning(f"⚠️ Prompt still too large, applying additional truncation...")
                truncated_prompt = initial_prompt[:max_size] + "\n\n[PROMPT TRUNCATED DUE TO SIZE LIMITS]"
                logger.info(f"📏 Prompt truncated to {len(truncated_prompt):,} characters")
                return truncated_prompt
        
        logger.info(f"📏 Prompt size: {prompt_size:,} characters (within limits)")
        return initial_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API for assessment with enhanced error handling"""
        
        # Check prompt size
        prompt_size = len(prompt)
        logger.info(f"🔗 Calling LLM API for assessment... (prompt size: {prompt_size:,} characters)")
        
        if prompt_size > 150000:  # 150k character warning threshold
            logger.warning(f"⚠️ Large prompt detected ({prompt_size:,} characters) - this may cause timeouts")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/piakoller/netTubo",
            "X-Title": "Patient Assessor"
        }
        
        # No max_tokens - allow unlimited output length for all models
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1  # Low temperature for consistent evaluation across pipeline
        }
        
        # Add model-specific parameters for Gemini models
        if "gemini" in self.model.lower():
            # Gemini models might need different parameters
            data.update({
                "stream": False,  # Ensure no streaming
                "top_p": 0.95,    # Add top_p for Gemini models
            })
        
        # Log the request details for debugging
        logger.info(f"🔧 Request payload: model={self.model}, temp={data['temperature']}, unlimited output")
        if "gemini" in self.model.lower():
            logger.info(f"🔧 Gemini-specific params: stream={data.get('stream')}, top_p={data.get('top_p')}")
        
        # Add timeout and retry logic
        max_retries = 4  # Increased for better reliability
        timeout_seconds = 120  # 2 minute timeout
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔗 LLM API call attempt {attempt + 1}/{max_retries + 1}...")
                response = requests.post(
                    self.base_url, 
                    headers=headers, 
                    json=data, 
                    timeout=timeout_seconds
                )
                
                logger.info(f"📡 API Response: Status {response.status_code}, Content-Length: {len(response.content)}")
                
                if response.status_code != 200:
                    error_msg = f"LLM API call failed: {response.status_code} - {response.text}"
                    logger.error(f"❌ {error_msg}")
                    
                    # Log specific error details for debugging
                    try:
                        error_response = response.json()
                        if "error" in error_response:
                            error_details = error_response["error"]
                            logger.error(f"🔍 API Error Details: {error_details}")
                            
                            # Check for specific model-related errors
                            error_message = str(error_details).lower()
                            if "model" in error_message:
                                logger.error(f"🚨 Model-specific error detected. Current model: {self.model}")
                                logger.error(f"💡 Try switching USE_PRODUCTION_MODEL to False in config.py")
                            elif "quota" in error_message or "limit" in error_message:
                                logger.error(f"🚨 Quota/limit error detected. Check your OpenRouter account.")
                            elif "auth" in error_message or "key" in error_message:
                                logger.error(f"🚨 Authentication error. Check your API key permissions.")
                    except:
                        pass  # Continue with original error handling
                    
                    if attempt < max_retries:
                        logger.info(f"🔄 Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                        continue
                    else:
                        raise Exception(error_msg)
                
                try:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    if not content or content.strip() == "":
                        error_msg = f"Empty response from LLM API (attempt {attempt + 1})"
                        logger.warning(f"⚠️ {error_msg}")
                        if attempt < max_retries:
                            logger.info(f"🔄 Retrying in 5 seconds...")
                            import time
                            time.sleep(5)
                            continue
                        else:
                            logger.error(f"❌ All attempts failed - empty response")
                            return ""  # Return empty string rather than raising exception
                    
                    logger.info(f"✅ LLM API call successful (response length: {len(content)} characters)")
                    return content
                    
                except (KeyError, ValueError) as e:
                    error_msg = f"Failed to parse LLM response: {e}"
                    logger.error(f"❌ {error_msg}")
                    if attempt < max_retries:
                        logger.info(f"🔄 Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                        continue
                    else:
                        raise Exception(error_msg)
                
            except requests.exceptions.Timeout:
                error_msg = f"LLM API call timed out after {timeout_seconds} seconds (attempt {attempt + 1})"
                logger.warning(f"⏱️ {error_msg}")
                if attempt < max_retries:
                    logger.info(f"🔄 Retrying with longer timeout...")
                    timeout_seconds += 60  # Increase timeout for retry
                    continue
                else:
                    logger.error(f"❌ All attempts failed - timeout")
                    return ""  # Return empty string rather than raising exception
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error during LLM API call: {e}"
                logger.error(f"❌ {error_msg}")
                if attempt < max_retries:
                    logger.info(f"🔄 Retrying in 10 seconds...")
                    import time
                    time.sleep(10)
                    continue
                else:
                    raise Exception(error_msg)
        
        # Should not reach here, but just in case
        logger.error(f"❌ Unexpected error in LLM API call")
        return ""
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM assessment response from JSON or <assessment> tags"""
        
        try:
            # First, try to parse as JSON (preferred format)
            import json
            import re
            
            try:
                # Try to extract JSON from markdown code blocks if present
                json_content = response.strip()
                
                # Check if response is wrapped in markdown code blocks
                if "```json" in json_content and "```" in json_content:
                    # Extract content between ```json and ```
                    json_match = re.search(r'```json\s*(.*?)\s*```', json_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                        logger.info("🔍 Extracted JSON from markdown code blocks")
                
                # Now try to parse the JSON
                json_response = json.loads(json_content)
                if "assessment" in json_response:
                    assessment_content = json_response["assessment"]
                    logger.info("✅ Successfully parsed JSON response with 'assessment' key")
                    
                    # Create structured response with the assessment content
                    return {
                        "assessment_text": assessment_content,
                        "clinical_summary": {
                            "disease_status": self._extract_field_from_text(assessment_content, "disease status"),
                            "stage": self._extract_field_from_text(assessment_content, "stage"),
                            "key_features": self._extract_list_from_text(assessment_content, "key features")
                        },
                        "clinical_needs": {
                            "primary_needs": self._extract_list_from_text(assessment_content, "clinical needs"),
                            "urgency_level": "medium"  # Default since not specified in simple format
                        },
                        "evidence_priorities": {
                            "high_priority": self._extract_list_from_text(assessment_content, "evidence"),
                            "trial_types_needed": ["clinical trials addressing current management gaps"]
                        },
                        "trial_eligibility": {
                            "favorable_factors": self._extract_list_from_text(assessment_content, "favorable factors"),
                            "potential_barriers": self._extract_list_from_text(assessment_content, "barriers")
                        },
                        "recommendations_for_trial_search": self._extract_list_from_text(assessment_content, "recommendations"),
                        "assessment_confidence": "medium",
                        "raw_response": response,
                        "parsing_method": "JSON"
                    }
            except (json.JSONDecodeError, KeyError) as e:
                # JSON parsing failed, continue to XML tag parsing
                logger.debug(f"JSON parsing failed: {e}")
                pass
            
            # Fallback: Try to extract content from <assessment> tags
            start_tag = '<assessment>'
            end_tag = '</assessment>'
            
            start_idx = response.find(start_tag)
            end_idx = response.find(end_tag)
            
            if start_idx != -1 and end_idx != -1:
                # Extract the assessment content
                assessment_content = response[start_idx + len(start_tag):end_idx].strip()
                logger.info("✅ Successfully parsed XML response with <assessment> tags")
                
                # Create structured response with the assessment content
                # Add basic structure for compatibility with downstream components
                return {
                    "assessment_text": assessment_content,
                    "clinical_summary": {
                        "disease_status": self._extract_field_from_text(assessment_content, "disease status"),
                        "stage": self._extract_field_from_text(assessment_content, "stage"),
                        "key_features": self._extract_list_from_text(assessment_content, "key features")
                    },
                    "clinical_needs": {
                        "primary_needs": self._extract_list_from_text(assessment_content, "clinical needs"),
                        "urgency_level": "medium"  # Default since not specified in simple format
                    },
                    "evidence_priorities": {
                        "high_priority": self._extract_list_from_text(assessment_content, "evidence"),
                        "trial_types_needed": ["clinical trials addressing current management gaps"]
                    },
                    "trial_eligibility": {
                        "favorable_factors": self._extract_list_from_text(assessment_content, "favorable factors"),
                        "potential_barriers": self._extract_list_from_text(assessment_content, "barriers")
                    },
                    "recommendations_for_trial_search": self._extract_list_from_text(assessment_content, "recommendations"),
                    "assessment_confidence": "medium",
                    "raw_response": response,
                    "parsing_method": "XML"
                }
            else:
                # If no assessment tags found, use the whole response
                logger.warning("⚠️ No JSON 'assessment' key or <assessment> tags found, using full response")
                return {
                    "assessment_text": response.strip(),
                    "clinical_summary": {
                        "disease_status": "Unable to parse",
                        "stage": "Unknown",
                        "key_features": []
                    },
                    "clinical_needs": {
                        "primary_needs": ["Assessment and management guidance needed"],
                        "urgency_level": "medium"
                    },
                    "evidence_priorities": {
                        "high_priority": ["Clinical guidance for management decisions"],
                        "trial_types_needed": ["clinical trials"]
                    },
                    "trial_eligibility": {
                        "favorable_factors": [],
                        "potential_barriers": []
                    },
                    "recommendations_for_trial_search": ["Search for relevant clinical trials"],
                    "assessment_confidence": "low",
                    "raw_response": response,
                    "parsing_note": "No assessment tags found",
                    "parsing_method": "fallback"
                }
                
        except Exception as e:
            logger.warning(f"⚠️ Assessment parsing failed: {e}")
            return {
                "assessment_text": response.strip(),
                "parsing_error": f"Parsing error: {str(e)}",
                "clinical_summary": {
                    "disease_status": "Parsing failed",
                    "stage": "Unknown", 
                    "key_features": []
                },
                "clinical_needs": {
                    "primary_needs": ["Assessment parsing failed"],
                    "urgency_level": "low"
                },
                "evidence_priorities": {
                    "high_priority": ["Unable to determine priorities"],
                    "trial_types_needed": ["clinical trials"]
                },
                "trial_eligibility": {
                    "favorable_factors": [],
                    "potential_barriers": ["Assessment parsing failed"]
                },
                "recommendations_for_trial_search": ["Manual review required"],
                "assessment_confidence": "low",
                "raw_response": response
            }
    
    def _extract_field_from_text(self, text: str, field_name: str) -> str:
        """Extract a specific field from assessment text using simple pattern matching"""
        import re
        
        # Try to find the field in the text using various patterns
        patterns = [
            rf"{field_name}:\s*([^\n\r]+)",
            rf"{field_name.title()}:\s*([^\n\r]+)",
            rf"{field_name.upper()}:\s*([^\n\r]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def _extract_list_from_text(self, text: str, field_name: str) -> list:
        """Extract a list of items related to a field from assessment text"""
        import re
        
        # Try to find lists in the text related to the field
        lines = text.split('\n')
        items = []
        
        # Look for lines that mention the field and subsequent bullet points or numbered items
        field_found = False
        for line in lines:
            line = line.strip()
            
            # Check if this line mentions our field
            if field_name.lower() in line.lower():
                field_found = True
                # Check if the line itself contains items after a colon
                if ':' in line:
                    after_colon = line.split(':', 1)[1].strip()
                    if after_colon:
                        # Split by commas and clean up
                        items.extend([item.strip() for item in after_colon.split(',') if item.strip()])
                continue
            
            # If we found the field, look for bullet points or numbered items
            if field_found:
                if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                    # Extract the item text
                    item_text = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet
                    item_text = re.sub(r'^\d+\.\s*', '', item_text)  # Remove numbers
                    if item_text:
                        items.append(item_text)
                elif line and not line.startswith(' ') and ':' in line:
                    # New section started, stop looking
                    break
        
        # If no specific items found, try to extract general concepts from the text
        if not items and field_name.lower() in ['clinical needs', 'recommendations']:
            # Extract some general items based on common keywords
            keywords = {
                'clinical needs': ['treatment', 'therapy', 'management', 'assessment', 'monitoring'],
                'recommendations': ['consider', 'evaluate', 'assess', 'monitor', 'follow'],
                'evidence': ['trial', 'study', 'evidence', 'data'],
                'barriers': ['contraindication', 'barrier', 'limitation', 'concern'],
                'favorable factors': ['eligible', 'suitable', 'appropriate', 'favorable']
            }
            
            field_keywords = keywords.get(field_name.lower(), [])
            for keyword in field_keywords:
                if keyword.lower() in text.lower():
                    items.append(f"{keyword.title()} needed based on assessment")
        
        return items[:5] if items else []  # Limit to 5 items
    
    def get_assessment_summary(self, assessment: Dict[str, Any]) -> str:
        """Get a human-readable summary of the assessment"""
        
        summary_lines = []
        summary_lines.append("=== PATIENT ASSESSMENT SUMMARY ===")
        summary_lines.append(f"Patient ID: {assessment.get('metadata', {}).get('patient_id', 'unknown')}")
        summary_lines.append(f"Assessment Time: {assessment.get('metadata', {}).get('timestamp', 'unknown')}")
        summary_lines.append(f"Assessment Confidence: {assessment.get('assessment_confidence', 'unknown')}")
        summary_lines.append("")
        
        # Main assessment text
        assessment_text = assessment.get('assessment_text', 'No assessment text available')
        summary_lines.append("CLINICAL ASSESSMENT:")
        summary_lines.append("-" * 50)
        
        # Format assessment text with proper line breaks
        assessment_lines = assessment_text.split('\n')
        for line in assessment_lines:
            if line.strip():
                summary_lines.append(line.strip())
            else:
                summary_lines.append("")
        
        summary_lines.append("")
        summary_lines.append("-" * 50)
        
        # Clinical summary (extracted fields)
        clinical = assessment.get('clinical_summary', {})
        if clinical.get('disease_status') != 'Not specified':
            summary_lines.append("")
            summary_lines.append("EXTRACTED CLINICAL INFORMATION:")
            summary_lines.append(f"  Disease Status: {clinical.get('disease_status', 'Unknown')}")
            summary_lines.append(f"  Stage: {clinical.get('stage', 'Unknown')}")
        
        # Show parsing information if there were issues
        if assessment.get('parsing_error'):
            summary_lines.append("")
            summary_lines.append("PARSING NOTES:")
            summary_lines.append(f"  {assessment.get('parsing_error')}")
        
        if assessment.get('parsing_note'):
            summary_lines.append("")
            summary_lines.append("PARSING NOTES:")
            summary_lines.append(f"  {assessment.get('parsing_note')}")
        
        return "\n".join(summary_lines)
    
    def clear_assessment_cache(self, patient_id: str, output_dir: str = ".") -> bool:
        """
        Clear cached assessment for a specific patient
        
        Args:
            patient_id: ID of the patient whose cache should be cleared
            output_dir: Directory containing the cached assessment
            
        Returns:
            True if cache was cleared, False if no cache existed
        """
        removed_any = False
        # Remove run-local file
        assessment_file = os.path.join(output_dir, f"patient_{patient_id}_assessment.json")
        if os.path.exists(assessment_file):
            try:
                os.remove(assessment_file)
                logger.info(f"🗑️ Cleared run-local assessment for patient {patient_id}")
                removed_any = True
            except Exception as e:
                logger.error(f"❌ Failed to clear run-local assessment for patient {patient_id}: {e}")
        # Remove centralized cached files for this patient
        try:
            for p in self.cache_dir.glob(f"patient_{patient_id}_assessment_*.json"):
                try:
                    p.unlink()
                    removed_any = True
                    logger.info(f"🗑️ Removed cached assessment file: {p}")
                except Exception as se:
                    logger.warning(f"⚠️ Failed to remove cached file {p}: {se}")
        except Exception as e:
            logger.warning(f"⚠️ Error scanning assessment cache dir: {e}")
        if not removed_any:
            logger.info(f"ℹ️ No cached assessment found for patient {patient_id}")
        return removed_any
