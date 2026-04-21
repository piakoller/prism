#!/usr/bin/env python3
"""
Therapy Recommender Module

This module generates therapy recommendations based on patient assessment and selected trials.
It synthesizes the evidence from matched trials to provide actionable clinical recommendations.
"""

import os
import json
import logging
import requests
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL, MODEL_CONFIGS

logger = logging.getLogger(__name__)

class TherapyRecommender:
    """Handles therapy recommendation generation based on assessment and trial matching"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, prompt_file: Optional[str] = None):
        """
        Initialize the Therapy Recommender
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: LLM model to use for recommendations (defaults to config.OPENROUTER_MODEL)
            prompt_file: Path to the therapy recommendation prompt file (defaults to standard location)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        # Set default prompt file path
        if prompt_file is None:
            # Look for prompt file in the prompts directory
            base_dir = Path(__file__).parent.parent.parent
            # prompt_file = str(base_dir / "prompts" / "therapy_recommendation_v2_english.txt")
            prompt_file = str(base_dir / "prompts" / "5_therapy_recommendation.txt")
        
        self.prompt_file = Path(prompt_file)
        self.prompt_template = self._load_prompt_template()
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please check your configuration.")
        
        logger.info(f"💡 Therapy Recommender initialized with model: {self.model}")
        logger.info(f"📄 Using prompt file: {self.prompt_file}")
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the text file."""
        if not self.prompt_file.exists():
            error_msg = f"❌ CRITICAL ERROR: Prompt file not found: {self.prompt_file}"
            logger.error(error_msg)
            print(error_msg)
            print("Please ensure the prompt file exists before running the therapy recommender.")
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
    
    def generate_recommendation(self, assessment: Dict[str, Any], trial_analysis_result: Dict[str, Any], 
                              guidelines_context: Optional[str] = None, patient_data: Optional[Dict[str, Any]] = None, 
                              original_trials_data: Optional[Dict[str, Any]] = None, output_dir: str = ".") -> Dict[str, Any]:
        """
        Generate therapy recommendation based on assessment and pre-analyzed trial data
        
        Args:
            assessment: Patient assessment results
            trial_analysis_result: Pre-analyzed trial data from TrialAnalyzer
            guidelines_context: Optional guidelines context
            patient_data: Complete patient data (for including full patient information)
            original_trials_data: Original trial evidence with full details
            output_dir: Directory to save recommendation
            
        Returns:
            Dictionary containing therapy recommendation
        """
        patient_id = assessment.get('metadata', {}).get('patient_id', 'unknown')
        logger.info(f"💊 Generating therapy recommendation for patient {patient_id}")
        
        # Create recommendation prompt using pre-analyzed trial data
        recommendation_prompt = self._create_recommendation_prompt(assessment, trial_analysis_result, guidelines_context, patient_data, original_trials_data)
        
        # Save the prompt for debugging/monitoring
        prompt_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write("=== THERAPY RECOMMENDATION PROMPT ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 50 + "\n\n")
            f.write(recommendation_prompt)
        
        logger.info(f"📄 Therapy recommendation prompt saved to: {prompt_file}")
        
        try:
            # Call LLM for recommendation
            logger.info("🔗 Calling LLM API for therapy recommendation...")
            recommendation_response = self._call_llm(recommendation_prompt)
            
            # Save the raw API response
            response_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_raw_response.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("=== THERAPY RECOMMENDATION RAW RESPONSE ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Model: {self.model}\n")
                f.write("=" * 50 + "\n\n")
                f.write(recommendation_response)
            
            logger.info(f"📄 Raw response saved to: {response_file}")
            
            # Save detailed trial analysis reasoning (from the analyzer stage)
            reasoning_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_reasoning.txt")
            self._save_trial_analysis_reasoning(trial_analysis_result, reasoning_file, patient_id)
            logger.info(f"📄 Trial analysis reasoning saved to: {reasoning_file}")
            
            recommendation_result = self._parse_recommendation_response(recommendation_response)
            
            # Save LLM's therapy recommendation thought process  
            llm_reasoning_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_reasoning.txt")
            self._save_llm_recommendation_reasoning(recommendation_result, llm_reasoning_file, patient_id)
            logger.info(f"📄 LLM therapy recommendation reasoning saved to: {llm_reasoning_file}")
            
            # Add metadata
            recommendation_result['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'patient_id': patient_id,
                'trials_considered': trial_analysis_result.get('trials_analyzed', 0),
                'trials_included': trial_analysis_result.get('trials_included', 0),
                'trials_excluded': trial_analysis_result.get('trials_excluded', 0),
                'assessment_confidence': assessment.get('assessment_confidence', 'unknown'),
                'analysis_complete': trial_analysis_result.get('analysis_complete', False),
                'guidelines_provided': guidelines_context is not None,
                'prompt_file': prompt_file,
                'response_file': response_file,
                'reasoning_file': reasoning_file,
                'llm_reasoning_file': llm_reasoning_file
            }
            
            # Save recommendation to file
            recommendation_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation.json")
            with open(recommendation_file, 'w', encoding='utf-8') as f:
                json.dump(recommendation_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Therapy recommendation completed and saved to: {recommendation_file}")
            
            return recommendation_result
            
        except Exception as e:
            logger.error(f"❌ Error during therapy recommendation: {e}")
            raise
    
    def _create_recommendation_prompt(self, assessment: Dict[str, Any], trial_analysis_result: Dict[str, Any], 
                                    guidelines_context: Optional[str] = None, patient_data: Optional[Dict[str, Any]] = None,
                                    original_trials_data: Optional[Dict[str, Any]] = None) -> str:
        """Create the therapy recommendation prompt using pre-analyzed trial data."""
        
        # Format patient assessment and patient data
        patient_assessment = self._format_assessment_and_patient_data(assessment, patient_data)
        
        # Format trial analysis results with full evidence for included trials
        trial_analysis_results = self._format_trial_analysis_results_with_evidence(trial_analysis_result, original_trials_data)
        
        # Format guidelines context
        formatted_guidelines = guidelines_context if guidelines_context else "No specific guidelines provided."
        
        # Use the loaded prompt template and format it with the data
        return self.prompt_template.format(
            patient_assessment=patient_assessment,
            trial_analysis_results=trial_analysis_results,
            guidelines_context=formatted_guidelines
        )
    
    def _format_assessment_and_patient_data(self, assessment: Dict[str, Any], patient_data: Optional[Dict[str, Any]] = None) -> str:
        """Format both the patient assessment and complete patient information for the prompt."""
        
        sections = []
        
        # Include complete patient information (excluding expert_recommendation)
        if patient_data:
            sections.append("COMPLETE PATIENT INFORMATION:")
            patient_info = []
            for key, value in patient_data.items():
                if key.lower() not in ['expert_recommendation', 'source'] and value and str(value).strip():
                    patient_info.append(f"- {key}: {value}")
            sections.append("\n".join(patient_info))
            sections.append("")
        
        # Include patient assessment
        if 'assessment_text' in assessment:
            assessment_text = assessment['assessment_text']
            
            sections.append("CLINICAL ASSESSMENT:")
            sections.append(assessment_text)
        else:
            # Fallback to old structured format
            clinical_summary = assessment.get('clinical_summary', {})
            treatment_history = assessment.get('treatment_history', {})
            clinical_needs = assessment.get('clinical_needs', {})
            evidence_priorities = assessment.get('evidence_priorities', {})
            trial_eligibility = assessment.get('trial_eligibility', {})
            
            sections.append("PATIENT CLINICAL ASSESSMENT:")
            sections.append(f"- Disease Status: {clinical_summary.get('disease_status', 'Unknown')}")
            sections.append(f"- Stage: {clinical_summary.get('stage', 'Unknown')}")
            sections.append(f"- Key Features: {clinical_summary.get('key_features', [])}")
            sections.append(f"- Previous Treatments: {treatment_history.get('previous_treatments', [])}")
            sections.append(f"- Current Therapy: {treatment_history.get('current_therapy', 'Unknown')}")
            sections.append(f"- Primary Clinical Needs: {clinical_needs.get('primary_needs', [])}")
            sections.append(f"- Urgency Level: {clinical_needs.get('urgency_level', 'Unknown')}")
            sections.append(f"- Evidence Priorities: {evidence_priorities.get('high_priority', [])}")
            sections.append(f"- Trial Eligibility Factors: {trial_eligibility.get('favorable_factors', [])}")
            sections.append(f"- Potential Barriers: {trial_eligibility.get('potential_barriers', [])}")
        
        return "\n".join(sections)
    
    def _format_assessment(self, assessment: Dict[str, Any]) -> str:
        """Format the patient assessment for the prompt."""
        
        # If we have assessment_text (from new format), use it
        if 'assessment_text' in assessment:
            assessment_text = assessment['assessment_text']
            
            formatted_assessment = f"""
CLINICAL ASSESSMENT:
{assessment_text}
"""
            return formatted_assessment
        
        # Fallback to old structured format
        clinical_summary = assessment.get('clinical_summary', {})
        treatment_history = assessment.get('treatment_history', {})
        clinical_needs = assessment.get('clinical_needs', {})
        evidence_priorities = assessment.get('evidence_priorities', {})
        trial_eligibility = assessment.get('trial_eligibility', {})
        
        return f"""
PATIENT CLINICAL ASSESSMENT:
- Disease Status: {clinical_summary.get('disease_status', 'Unknown')}
- Stage: {clinical_summary.get('stage', 'Unknown')}
- Key Features: {clinical_summary.get('key_features', [])}
- Previous Treatments: {treatment_history.get('previous_treatments', [])}
- Current Therapy: {treatment_history.get('current_therapy', 'Unknown')}
- Primary Clinical Needs: {clinical_needs.get('primary_needs', [])}
- Urgency Level: {clinical_needs.get('urgency_level', 'Unknown')}
- Evidence Priorities: {evidence_priorities.get('high_priority', [])}
- Trial Eligibility Factors: {trial_eligibility.get('favorable_factors', [])}
- Potential Barriers: {trial_eligibility.get('potential_barriers', [])}
"""
    
    def _format_trials(self, selected_trials: List[Dict[str, Any]]) -> str:
        """Format the selected trials for the prompt."""
        
        if not selected_trials:
            return "No clinical trials were matched for this patient."
        
        trial_summaries = []
        for i, trial in enumerate(selected_trials, 1):
            # Handle both trial data formats (original trial + evaluation info)
            evaluation = trial.get('evaluation', {})
            
            # Extract trial identifiers with fallbacks
            nct_id = trial.get('NCT ID', trial.get('nct_id', trial.get('NCT Number', 'Unknown')))
            title = trial.get('Study Title', trial.get('title', trial.get('Official Title', trial.get('Brief Title', 'Unknown'))))
            interventions = trial.get('Interventions', trial.get('interventions', trial.get('Intervention', 'Unknown')))
            phase = trial.get('Phase', trial.get('phase', trial.get('Study Phase', 'Unknown')))
            status = trial.get('Study Status', trial.get('status', trial.get('Overall Status', 'Unknown')))
            
            # Extract evaluation information
            relevance_explanation = evaluation.get('explanation', 'No explanation provided')
            
            # Try to extract published results information from various sources
            results_info = "No specific results provided"
            
            # Check publication_analysis for detailed results
            pub_analysis = trial.get('publication_analysis', {})
            if pub_analysis:
                # Check online search results for clinical data
                online_results = pub_analysis.get('online_search_results', {})
                if isinstance(online_results, dict) and 'pubmed' in online_results:
                    pubmed_data = online_results['pubmed']
                    if pubmed_data.get('publications'):
                        # Look for clinical results in first publication with detailed content
                        for pub in pubmed_data['publications']:
                            if pub.get('has_results_keywords') and pub.get('full_content'):
                                content = pub['full_content']
                                # Extract key clinical results
                                if any(term in content.lower() for term in ['median progression-free survival', 'hazard ratio', 'hr 0.', 'pfs']):
                                    # Extract relevant result sections
                                    results_parts = []
                                    content_lower = content.lower()
                                    
                                    # Look for PFS data
                                    if 'median progression-free survival' in content_lower:
                                        start = content_lower.find('median progression-free survival')
                                        if start != -1:
                                            snippet = content[start:start+300]
                                            results_parts.append(snippet.split('.')[0] + '.')
                                    
                                    # Look for hazard ratio data
                                    if 'hazard ratio' in content_lower or 'hr 0.' in content_lower:
                                        hr_start = content_lower.find('hazard ratio')
                                        if hr_start == -1:
                                            hr_start = content_lower.find('hr 0.')
                                        if hr_start != -1:
                                            snippet = content[hr_start:hr_start+200]
                                            results_parts.append(snippet.split('.')[0] + '.')
                                    
                                    if results_parts:
                                        results_info = ' '.join(results_parts)
                                        break
            
            # Fallback to basic results fields if available
            if results_info == "No specific results provided":
                if 'results' in trial or 'Results' in trial:
                    results_info = trial.get('results', trial.get('Results', results_info))
            
            trial_summary = f"""
TRIAL {i}: {nct_id}
Title: {title}
Interventions: {interventions}
Phase: {phase}
Status: {status}
Evaluation: RELEVANT
Rationale: {relevance_explanation}
Published Results: {results_info}
"""
            trial_summaries.append(trial_summary)
        
        return "\n".join(trial_summaries)
    
    def _format_trials_with_criteria(self, selected_trials: List[Dict[str, Any]]) -> str:
        """Format the selected trials with inclusion/exclusion criteria for the prompt."""
        
        if not selected_trials:
            return "No clinical trials were matched for this patient."
        
        trial_summaries = []
        for i, trial in enumerate(selected_trials, 1):
            # Handle both trial data formats (original trial + evaluation info)
            evaluation = trial.get('evaluation', {})
            
            # Extract trial identifiers with fallbacks for multiple field name formats
            nct_id = trial.get('NCT ID', trial.get('nct_id', trial.get('NCT Number', 'Unknown')))
            title = trial.get('Study Title', trial.get('title', trial.get('Official Title', trial.get('Brief Title', 'Unknown'))))
            interventions = trial.get('Interventions', trial.get('interventions', trial.get('Intervention', 'Unknown')))
            phase = trial.get('Phase', trial.get('phase', trial.get('Study Phase', 'Unknown')))
            status = trial.get('Study Status', trial.get('status', trial.get('Overall Status', 'Unknown')))
            
            # Extract evaluation information
            relevance_explanation = evaluation.get('explanation', 'No explanation provided')
            relevance_score = evaluation.get('relevance_score', 'Not scored')
            trial_relevance_reason = trial.get('relevance_reason', relevance_explanation)
            
            # Format comprehensive trial selection reasoning
            selection_reasoning = self._format_trial_selection_reasoning(trial, evaluation)
            
            # Extract inclusion/exclusion criteria
            eligibility_criteria = "No eligibility criteria provided"
            
            # Try different field formats for eligibility criteria
            eligibility = trial.get('eligibility', trial.get('Eligibility', {}))
            if eligibility:
                criteria = eligibility.get('criteria', eligibility.get('Criteria', {}))
                if criteria:
                    textblock = criteria.get('textblock', criteria.get('Textblock', ''))
                    if textblock:
                        # For production model with 1M context, include full criteria
                        if self.model == "google/gemini-2.5-pro":
                            eligibility_criteria = textblock  # Full content, no truncation
                        else:
                            # For test models with limited context, limit the criteria length
                            if len(textblock) > 1000:
                                eligibility_criteria = textblock[:1000] + "... [truncated]"
                            else:
                                eligibility_criteria = textblock
            
            # If no structured criteria found, try other fields
            if eligibility_criteria == "No eligibility criteria provided":
                for field in ['eligibility_criteria', 'Eligibility_Criteria', 'criteria', 'Criteria']:
                    if field in trial and trial[field]:
                        criteria_text = trial[field]
                        # For production model with 1M context, include full criteria
                        if self.model == "google/gemini-2.5-pro":
                            eligibility_criteria = criteria_text  # Full content, no truncation
                        else:
                            # For test models with limited context, limit the criteria length
                            if len(criteria_text) > 1000:
                                eligibility_criteria = criteria_text[:1000] + "... [truncated]"
                            else:
                                eligibility_criteria = criteria_text
                        break
            
            # Try to extract published results information
            results_info = "No specific results provided"
            
            # Check publication_analysis for detailed results
            pub_analysis = trial.get('publication_analysis', {})
            if pub_analysis:
                # Check online search results for clinical data
                online_results = pub_analysis.get('online_search_results', {})
                if isinstance(online_results, dict) and 'pubmed' in online_results:
                    pubmed_data = online_results['pubmed']
                    if pubmed_data.get('publications'):
                        # Look for clinical results in first publication with detailed content
                        for pub in pubmed_data['publications']:
                            if pub.get('has_results_keywords') and pub.get('full_content'):
                                content = pub['full_content']
                                # Extract key clinical results
                                if any(term in content.lower() for term in ['median progression-free survival', 'hazard ratio', 'hr 0.', 'pfs']):
                                    # Extract relevant result sections
                                    results_parts = []
                                    content_lower = content.lower()
                                    
                                    # Look for PFS data
                                    if 'median progression-free survival' in content_lower:
                                        start = content_lower.find('median progression-free survival')
                                        if start != -1:
                                            snippet = content[start:start+300]
                                            results_parts.append(snippet.split('.')[0] + '.')
                                    
                                    # Look for hazard ratio data
                                    if 'hazard ratio' in content_lower or 'hr 0.' in content_lower:
                                        hr_start = content_lower.find('hazard ratio')
                                        if hr_start == -1:
                                            hr_start = content_lower.find('hr 0.')
                                        if hr_start != -1:
                                            snippet = content[hr_start:hr_start+200]
                                            results_parts.append(snippet.split('.')[0] + '.')
                                    
                                    if results_parts:
                                        results_info = ' '.join(results_parts)
                                        break
            
            # Fallback to basic results fields if available
            if results_info == "No specific results provided":
                if 'results' in trial or 'Results' in trial:
                    results_info = trial.get('results', trial.get('Results', results_info))
            
            trial_summary = f"""
TRIAL {i}: {nct_id}
Title: {title}
Interventions: {interventions}
Phase: {phase}
Status: {status}
Evaluation: RELEVANT

TRIAL SELECTION REASONING:
{selection_reasoning}

Inclusion/Exclusion Criteria:
{eligibility_criteria}

Published Results: {results_info}
"""
            trial_summaries.append(trial_summary)
        
        return "\n".join(trial_summaries)
    
    def _format_trial_selection_reasoning(self, trial: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
        """Format comprehensive trial selection reasoning for transparency"""
        
        reasoning_parts = []
        
        # Primary evaluation reasoning
        primary_explanation = evaluation.get('explanation', 'No primary explanation provided')
        reasoning_parts.append(f"Primary Evaluation: {primary_explanation}")
        
        # Additional metadata if available
        relevance_score = evaluation.get('relevance_score')
        if relevance_score is not None:
            reasoning_parts.append(f"Relevance Score: {relevance_score}")
        
        # Trial metadata that influenced selection
        trial_relevance = trial.get('relevance_reason')
        if trial_relevance and trial_relevance != primary_explanation:
            reasoning_parts.append(f"Additional Context: {trial_relevance}")
        
        # Evaluation criteria details if available
        eval_criteria = evaluation.get('criteria_met', {})
        if eval_criteria:
            reasoning_parts.append("Criteria Assessment:")
            for criterion, status in eval_criteria.items():
                reasoning_parts.append(f"  - {criterion}: {status}")
        
        # Clinical alignment factors
        disease_alignment = evaluation.get('disease_alignment')
        treatment_alignment = evaluation.get('treatment_alignment') 
        evidence_quality = evaluation.get('evidence_quality')
        
        if disease_alignment or treatment_alignment or evidence_quality:
            reasoning_parts.append("Clinical Alignment Factors:")
            if disease_alignment:
                reasoning_parts.append(f"  - Disease/Biology: {disease_alignment}")
            if treatment_alignment:
                reasoning_parts.append(f"  - Treatment Context: {treatment_alignment}")
            if evidence_quality:
                reasoning_parts.append(f"  - Evidence Quality: {evidence_quality}")
        
        # Any specific patient-trial matching rationale
        patient_match = evaluation.get('patient_match_rationale')
        if patient_match:
            reasoning_parts.append(f"Patient-Trial Match: {patient_match}")
        
        # Trial characteristics that made it relevant
        key_features = evaluation.get('key_relevant_features', [])
        if key_features:
            reasoning_parts.append(f"Key Relevant Features: {', '.join(key_features)}")
        
        # Potential limitations or considerations
        limitations = evaluation.get('limitations', evaluation.get('considerations'))
        if limitations:
            reasoning_parts.append(f"Considerations: {limitations}")
        
        return "\n".join(reasoning_parts)
    
    def _format_trial_analysis_results(self, trial_analysis_result: Dict[str, Any]) -> str:
        """Format pre-analyzed trial data for the therapy recommendation prompt"""
        
        if trial_analysis_result.get('status') != 'success':
            return f"Trial analysis failed: {trial_analysis_result.get('error_message', 'Unknown error')}"
        
        trial_analyses = trial_analysis_result.get('trial_analyses', [])
        if not trial_analyses:
            return "No trials were analyzed for this patient."
        
        # Filter for trials recommended for inclusion
        included_trials = [t for t in trial_analyses if t.get('recommendation') == 'INCLUDE']
        excluded_trials = [t for t in trial_analyses if t.get('recommendation') == 'EXCLUDE']
        
        formatted_parts = []
        
        # Summary
        formatted_parts.append(f"TRIAL ANALYSIS SUMMARY:")
        formatted_parts.append(f"- Total trials analyzed: {len(trial_analyses)}")
        formatted_parts.append(f"- Trials recommended for inclusion: {len(included_trials)}")
        formatted_parts.append(f"- Trials recommended for exclusion: {len(excluded_trials)}")
        formatted_parts.append("")
        
        # Validation warnings if any
        warnings = trial_analysis_result.get('validation_warnings', [])
        if warnings:
            formatted_parts.append("ANALYSIS WARNINGS:")
            for warning in warnings:
                formatted_parts.append(f"- {warning}")
            formatted_parts.append("")
        
        # Trials recommended for inclusion (detailed)
        if included_trials:
            formatted_parts.append("TRIALS RECOMMENDED FOR INCLUSION:")
            formatted_parts.append("=" * 50)
            
            for i, trial in enumerate(included_trials, 1):
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                relevance = trial.get('relevance_score', 'Unknown')
                # Use 'rationale' field which contains the analysis reasoning
                reasoning = trial.get('rationale', 'No reasoning provided')
                justification = trial.get('rationale', 'No justification provided')  # Same as reasoning for now
                
                formatted_parts.append(f"TRIAL {i}: {nct}")
                formatted_parts.append(f"Name: {name}")
                formatted_parts.append(f"Relevance Score: {relevance}")
                formatted_parts.append(f"Analysis Reasoning:")
                formatted_parts.append(f"{reasoning}")
                formatted_parts.append(f"Inclusion Justification: {justification}")
                formatted_parts.append("-" * 40)
                formatted_parts.append("")
        
        # Brief summary of excluded trials
        if excluded_trials:
            formatted_parts.append("TRIALS EXCLUDED FROM RECOMMENDATION:")
            formatted_parts.append("=" * 50)
            
            for trial in excluded_trials:
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                relevance = trial.get('relevance_score', 'Unknown')
                # Use 'rationale' field which contains the exclusion reasoning
                justification = trial.get('rationale', 'No justification provided')
                
                formatted_parts.append(f"- {nct} ({name}) - Relevance: {relevance}")
                formatted_parts.append(f"  Exclusion reason: {justification}")
        
        return "\n".join(formatted_parts)

    def _format_trial_analysis_results_with_evidence(self, trial_analysis_result: Dict[str, Any], original_trials_data: Optional[Dict[str, Any]] = None) -> str:
        """Format pre-analyzed trial data with full evidence for included trials"""
        
        if trial_analysis_result.get('status') != 'success':
            return f"Trial analysis failed: {trial_analysis_result.get('error_message', 'Unknown error')}"
        
        trial_analyses = trial_analysis_result.get('trial_analyses', [])
        if not trial_analyses:
            return "No trials were analyzed for this patient."
        
        # Filter for trials recommended for inclusion
        included_trials = [t for t in trial_analyses if t.get('recommendation') == 'INCLUDE']
        excluded_trials = [t for t in trial_analyses if t.get('recommendation') == 'EXCLUDE']
        
        # Create lookup dictionary for original trial evidence
        original_trials_lookup = {}
        if original_trials_data and 'matched_trials' in original_trials_data:
            for trial in original_trials_data['matched_trials']:
                # Handle different NCT number field names
                nct_number = trial.get('nct_number', '') or trial.get('nct_id', '')
                if nct_number:
                    original_trials_lookup[nct_number] = trial
        
        formatted_parts = []
        
        # Summary
        formatted_parts.append(f"TRIAL ANALYSIS SUMMARY:")
        formatted_parts.append(f"- Total trials analyzed: {len(trial_analyses)}")
        formatted_parts.append(f"- Trials recommended for inclusion: {len(included_trials)}")
        formatted_parts.append(f"- Trials recommended for exclusion: {len(excluded_trials)}")
        formatted_parts.append("")
        
        # Validation warnings if any
        warnings = trial_analysis_result.get('validation_warnings', [])
        if warnings:
            formatted_parts.append("ANALYSIS WARNINGS:")
            for warning in warnings:
                formatted_parts.append(f"- {warning}")
            formatted_parts.append("")
        
        # Trials recommended for inclusion (detailed with full evidence)
        if included_trials:
            formatted_parts.append("TRIALS RECOMMENDED FOR INCLUSION:")
            formatted_parts.append("=" * 50)
            
            for i, trial in enumerate(included_trials, 1):
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                relevance = trial.get('relevance_score', 'Unknown')
                reasoning = trial.get('rationale', 'No reasoning provided')
                
                formatted_parts.append(f"TRIAL {i}: {nct}")
                formatted_parts.append(f"Name: {name}")
                formatted_parts.append(f"Relevance Score: {relevance}")
                formatted_parts.append(f"Analysis Reasoning:")
                formatted_parts.append(f"{reasoning}")
                formatted_parts.append(f"Inclusion Justification: {reasoning}")  # Same as reasoning for now
                
                # Add full original trial evidence if available
                if nct in original_trials_lookup:
                    original_trial = original_trials_lookup[nct]
                    formatted_parts.append("")
                    formatted_parts.append("FULL TRIAL EVIDENCE:")
                    
                    # Basic trial information
                    formatted_parts.append(f"Phase: {original_trial.get('phase', 'Unknown')}")
                    formatted_parts.append(f"Status: {original_trial.get('status', 'Unknown')}")
                    formatted_parts.append(f"Condition: {original_trial.get('condition', 'Unknown')}")
                    formatted_parts.append(f"Intervention: {original_trial.get('intervention', 'Unknown')}")
                    formatted_parts.append(f"Brief Summary: {original_trial.get('brief_summary', 'No summary available')}")
                    formatted_parts.append(f"Primary Outcome: {original_trial.get('primary_outcome', 'Unknown')}")
                    formatted_parts.append(f"Secondary Outcomes: {original_trial.get('secondary_outcomes', 'Unknown')}")
                    
                    # Evidence and publications
                    evidence = original_trial.get('evidence_and_publications', {})
                    if evidence:
                        formatted_parts.append("")
                        formatted_parts.append("EVIDENCE AND PUBLICATIONS:")
                        
                        # Listed publications
                        listed_pubs = evidence.get('listed_publications', [])
                        if listed_pubs:
                            formatted_parts.append("LISTED PUBLICATIONS:")
                            for pub in listed_pubs:
                                formatted_parts.append(f"Citation: {pub.get('citation', 'No citation')}")
                        
                        # Publication status
                        pub_status = evidence.get('publication_status', {})
                        if pub_status:
                            formatted_parts.append("PUBLICATION STATUS:")
                            formatted_parts.append(f"Has Listed Publications: {pub_status.get('has_listed_publications', False)} | Has Results Publications: {pub_status.get('has_results_publications', False)} | Total Publications Found: {pub_status.get('total_publications_found', 0)}")
                        
                        # PubMed results - include ALL results, not just the first one
                        pubmed_results = evidence.get('pubmed_results', [])
                        if pubmed_results:
                            formatted_parts.append("PUBMED RESULTS:")
                            for idx, result in enumerate(pubmed_results, 1):
                                formatted_parts.append(f"PUBMED RESULT {idx}:")
                                formatted_parts.append(f"Title: {result.get('title', 'No title')}")
                                formatted_parts.append(f"Authors: {result.get('authors', 'No authors')}")
                                formatted_parts.append(f"Journal: {result.get('journal', 'No journal')}")
                                
                                # Include full abstract without truncation
                                abstract = result.get('abstract', 'No abstract')
                                formatted_parts.append(f"Abstract: {abstract}")
                                
                                # Include full key content without truncation
                                key_content = result.get('key_content', '')
                                if key_content:
                                    formatted_parts.append(f"Key Content: {key_content}")
                                
                                # Include full content if available
                                full_content = result.get('full_content', '')
                                if full_content:
                                    formatted_parts.append(f"Full Content: {full_content}")
                                
                                formatted_parts.append("")  # Add spacing between results
                        
                        # Congress abstracts - include all
                        congress_abstracts = evidence.get('congress_abstracts', [])
                        if congress_abstracts:
                            formatted_parts.append("CONGRESS ABSTRACTS:")
                            for idx, abstract in enumerate(congress_abstracts, 1):
                                formatted_parts.append(f"CONGRESS ABSTRACT {idx}:")
                                formatted_parts.append(f"Title: {abstract.get('title', 'No title')}")
                                formatted_parts.append(f"Authors: {abstract.get('authors', 'No authors')}")
                                formatted_parts.append(f"Conference: {abstract.get('conference', 'No conference')}")
                                formatted_parts.append(f"Year: {abstract.get('year', 'No year')}")
                                
                                # Include full abstract text without truncation
                                abstract_text = abstract.get('abstract_text', 'No abstract text')
                                formatted_parts.append(f"Abstract Text: {abstract_text}")
                                
                                # Include key content if available
                                key_content = abstract.get('key_content', '')
                                if key_content:
                                    formatted_parts.append(f"Key Content: {key_content}")
                                
                                formatted_parts.append("")  # Add spacing between abstracts
                        
                        # Online search results - include all relevant data
                        online_search = evidence.get('online_search_results', {})
                        if online_search:
                            formatted_parts.append("ONLINE SEARCH RESULTS:")
                            for source, source_data in online_search.items():
                                formatted_parts.append(f"SOURCE: {source.upper()}")
                                if isinstance(source_data, dict):
                                    # Include all available fields from online search
                                    for field, value in source_data.items():
                                        if value and field not in ['query', 'search_date']:  # Skip metadata
                                            if isinstance(value, list):
                                                if value and len(value) > 0:
                                                    formatted_parts.append(f"{field.replace('_', ' ').title()}:")
                                                    for item in value:
                                                        if isinstance(item, dict):
                                                            for subfield, subvalue in item.items():
                                                                if subvalue:
                                                                    formatted_parts.append(f"  {subfield}: {subvalue}")
                                                        else:
                                                            formatted_parts.append(f"  {item}")
                                            else:
                                                formatted_parts.append(f"{field.replace('_', ' ').title()}: {value}")
                                formatted_parts.append("")
                        
                        # Has posted results and results data
                        has_results = evidence.get('has_posted_results', False)
                        formatted_parts.append(f"HAS POSTED RESULTS: {has_results}")

                    # Also include top-level publications if present on the trial object
                    top_level_pubs = original_trial.get('publications', [])
                    if top_level_pubs:
                        formatted_parts.append("")
                        formatted_parts.append("PUBLICATIONS:")
                        for pub in top_level_pubs:
                            if isinstance(pub, dict):
                                citation = pub.get('citation') or pub.get('title') or str(pub)
                            else:
                                citation = str(pub)
                            formatted_parts.append(f"Citation: {citation}")
                        
                        # Include clinical trial results if available
                        results = original_trial.get('results', {})
                        if results:
                            formatted_parts.append("")
                            formatted_parts.append("CLINICAL TRIAL RESULTS:")
                            for result_field, result_value in results.items():
                                if result_value:
                                    if isinstance(result_value, dict):
                                        formatted_parts.append(f"{result_field.replace('_', ' ').title()}:")
                                        for subfield, subvalue in result_value.items():
                                            if subvalue:
                                                formatted_parts.append(f"  {subfield}: {subvalue}")
                                    else:
                                        formatted_parts.append(f"{result_field.replace('_', ' ').title()}: {result_value}")
                        
                        # Include any additional evidence fields that might be present
                        for evidence_field in ['publication_analysis', 'safety_data', 'efficacy_data', 'biomarker_data']:
                            field_data = original_trial.get(evidence_field, {})
                            if field_data and isinstance(field_data, dict):
                                formatted_parts.append("")
                                formatted_parts.append(f"{evidence_field.replace('_', ' ').upper()}:")
                                for subfield, subvalue in field_data.items():
                                    if subvalue and subfield not in ['evidence_and_publications']:  # Avoid duplication
                                        if isinstance(subvalue, (dict, list)):
                                            formatted_parts.append(f"{subfield}: {subvalue}")
                                        else:
                                            formatted_parts.append(f"{subfield}: {subvalue}")
                
                formatted_parts.append("-" * 40)
                formatted_parts.append("")
        
        # Brief summary of excluded trials
        if excluded_trials:
            formatted_parts.append("TRIALS EXCLUDED FROM RECOMMENDATION:")
            formatted_parts.append("=" * 50)
            
            for trial in excluded_trials:
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                relevance = trial.get('relevance_score', 'Unknown')
                justification = trial.get('rationale', 'No justification provided')
                
                formatted_parts.append(f"- {nct} ({name}) - Relevance: {relevance}")
                formatted_parts.append(f"  Exclusion reason: {justification}")
        
        return "\n".join(formatted_parts)

    def _save_trial_analysis_reasoning(self, trial_analysis_result: Dict[str, Any], reasoning_file: str, patient_id: str) -> None:
        """Save trial analysis reasoning from the first stage for audit trail"""
        
        with open(reasoning_file, 'w', encoding='utf-8') as f:
            f.write("=== TRIAL ANALYSIS REASONING REPORT ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Analysis Status: {trial_analysis_result.get('status', 'Unknown')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary
            f.write("ANALYSIS SUMMARY:\n")
            f.write(f"- Trials analyzed: {trial_analysis_result.get('trials_analyzed', 0)}\n")
            f.write(f"- Trials included: {trial_analysis_result.get('trials_included', 0)}\n")
            f.write(f"- Trials excluded: {trial_analysis_result.get('trials_excluded', 0)}\n")
            f.write(f"- Analysis complete: {trial_analysis_result.get('analysis_complete', False)}\n")
            f.write("\n")
            
            # Validation warnings
            warnings = trial_analysis_result.get('validation_warnings', [])
            if warnings:
                f.write("VALIDATION WARNINGS:\n")
                for warning in warnings:
                    f.write(f"⚠️ {warning}\n")
                f.write("\n")
            
            # Detailed trial analyses
            trial_analyses = trial_analysis_result.get('trial_analyses', [])
            if trial_analyses:
                f.write("DETAILED TRIAL ANALYSES:\n")
                f.write("=" * 50 + "\n\n")
                
                for i, trial in enumerate(trial_analyses, 1):
                    nct = trial.get('nct_number', 'Unknown')
                    name = trial.get('trial_name', 'Unknown')
                    relevance = trial.get('relevance_score', 'Unknown')
                    recommendation = trial.get('recommendation', 'Unknown')
                    reasoning = trial.get('reasoning', 'No reasoning provided')
                    justification = trial.get('recommendation_justification', 'No justification provided')
                    
                    f.write(f"TRIAL {i}: {nct}\n")
                    f.write(f"Name: {name}\n")
                    f.write(f"Relevance Score: {relevance}\n")
                    f.write(f"Recommendation: {recommendation}\n")
                    f.write(f"Reasoning:\n{reasoning}\n")
                    f.write(f"Justification: {justification}\n")
                    f.write("-" * 40 + "\n\n")
            
            # Raw response if available
            if 'raw_response' in trial_analysis_result:
                f.write("RAW ANALYSIS RESPONSE:\n")
                f.write("=" * 50 + "\n")
                f.write(trial_analysis_result['raw_response'])

    def _save_trial_selection_reasoning(self, matching_result: Dict[str, Any], reasoning_file: str, patient_id: str) -> None:
        """Save detailed trial selection reasoning to a dedicated file for clinical audit"""
        
        with open(reasoning_file, 'w', encoding='utf-8') as f:
            f.write("=== TRIAL SELECTION REASONING REPORT ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 60 + "\n\n")
            
            relevant_trials = matching_result.get('relevant_trials', [])
            f.write(f"SUMMARY: {len(relevant_trials)} trial(s) selected for therapy recommendation\n\n")
            
            # Overall matching process summary
            total_evaluated = matching_result.get('total_trials_evaluated', 'Unknown')
            initial_matches = matching_result.get('initial_matches', 'Unknown')
            final_selections = len(relevant_trials)
            
            f.write("MATCHING PROCESS OVERVIEW:\n")
            f.write(f"- Total trials evaluated: {total_evaluated}\n")
            f.write(f"- Initial matches: {initial_matches}\n") 
            f.write(f"- Final selections: {final_selections}\n")
            f.write(f"- Selection confidence: {matching_result.get('confidence_level', 'Unknown')}\n\n")
            
            # Detailed reasoning for each selected trial
            if relevant_trials:
                f.write("DETAILED TRIAL SELECTION REASONING:\n")
                f.write("=" * 50 + "\n\n")
                
                for i, trial in enumerate(relevant_trials, 1):
                    nct_id = trial.get('NCT ID', trial.get('nct_id', trial.get('NCT Number', 'Unknown')))
                    title = trial.get('Study Title', trial.get('title', trial.get('Official Title', 'Unknown')))
                    
                    f.write(f"TRIAL {i}: {nct_id}\n")
                    f.write(f"Title: {title}\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get evaluation data
                    evaluation = trial.get('evaluation', {})
                    
                    # Format comprehensive reasoning
                    detailed_reasoning = self._format_trial_selection_reasoning(trial, evaluation)
                    f.write(f"{detailed_reasoning}\n")
                    
                    # Add any original trial matching response if available
                    if 'raw_response' in evaluation:
                        f.write(f"\nOriginal LLM Response:\n{evaluation['raw_response']}\n")
                    
                    # Add relevance metadata
                    relevance_score = trial.get('relevance_score')
                    if relevance_score:
                        f.write(f"\nRelevance Score: {relevance_score}\n")
                    
                    f.write("\n" + "=" * 50 + "\n\n")
            else:
                f.write("NO TRIALS SELECTED\n")
                f.write("Reason: No trials met the relevance criteria for this patient\n\n")
            
            # Add any overall matching notes
            matching_notes = matching_result.get('notes', matching_result.get('summary'))
            if matching_notes:
                f.write("OVERALL MATCHING NOTES:\n")
                f.write(f"{matching_notes}\n\n")
            
            # Add trial matching process details if available
            process_details = matching_result.get('process_details', {})
            if process_details:
                f.write("MATCHING PROCESS DETAILS:\n")
                for key, value in process_details.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
    
    def generate_trial_selection_summary(self, matching_result: Dict[str, Any]) -> str:
        """Generate a concise summary of trial selection reasoning for clinical review"""
        
        relevant_trials = matching_result.get('relevant_trials', [])
        total_trials = len(relevant_trials)
        
        if total_trials == 0:
            return "No relevant trials identified for this patient."
        
        summary_lines = []
        summary_lines.append(f"TRIAL SELECTION SUMMARY ({total_trials} trial{'s' if total_trials != 1 else ''})")
        summary_lines.append("=" * 50)
        
        for i, trial in enumerate(relevant_trials, 1):
            nct_id = trial.get('NCT ID', trial.get('nct_id', 'Unknown'))
            title = trial.get('Study Title', trial.get('title', 'Unknown'))
            evaluation = trial.get('evaluation', {})
            
            # Extract key reasoning points
            explanation = evaluation.get('explanation', 'No explanation provided')
            relevance_score = evaluation.get('relevance_score', 'Not scored')
            
            # Truncate explanation for summary
            short_explanation = explanation[:150] + "..." if len(explanation) > 150 else explanation
            
            summary_lines.append(f"\n{i}. {nct_id} (Score: {relevance_score})")
            summary_lines.append(f"   {title[:80]}...")
            summary_lines.append(f"   Rationale: {short_explanation}")
        
        return "\n".join(summary_lines)
    
    def _save_llm_recommendation_reasoning(self, recommendation_result: Dict[str, Any], reasoning_file: str, patient_id: str) -> None:
        """Save the LLM's therapy recommendation reasoning for understanding its thought process"""
        
        with open(reasoning_file, 'w', encoding='utf-8') as f:
            f.write("=== LLM THERAPY RECOMMENDATION REASONING ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PURPOSE: This file captures the LLM's reasoning about therapy recommendations\n")
            f.write("based on the pre-analyzed trial data.\n\n")
            
            # Extract the therapy recommendation reasoning
            therapy_reasoning = recommendation_result.get('therapy_recommendation_text', '')
            rationale = recommendation_result.get('rationale_text', '')
            
            if therapy_reasoning:
                f.write("LLM'S THERAPY RECOMMENDATION:\n")
                f.write("=" * 50 + "\n")
                f.write(f"{therapy_reasoning}\n\n")
            
            if rationale:
                f.write("LLM'S RATIONALE:\n")
                f.write("=" * 50 + "\n")
                f.write(f"{rationale}\n\n")
                
                # Extract trial mentions from the rationale
                import re
                nct_mentions = re.findall(r'NCT\d+', rationale)
                if nct_mentions:
                    f.write("TRIALS REFERENCED IN RECOMMENDATION:\n")
                    for nct_id in set(nct_mentions):  # Remove duplicates
                        f.write(f"- {nct_id}\n")
                    f.write("\n")
                
            else:
                f.write("❌ NO EXPLICIT TRIAL SELECTION REASONING PROVIDED\n")
                f.write("The LLM did not provide explicit reasoning about trial selection.\n")
                f.write("This may indicate the prompt needs to be updated to enforce this requirement.\n\n")
            
            # Also capture what trials were mentioned in the therapy recommendation itself
            therapy_text = recommendation_result.get('therapy_recommendation_text', '')
            rationale_text = recommendation_result.get('rationale_text', '')
            
            combined_text = f"{therapy_text}\n{rationale_text}"
            import re
            all_nct_mentions = re.findall(r'NCT\d+', combined_text)
            
            if all_nct_mentions:
                f.write("TRIALS MENTIONED IN THERAPY RECOMMENDATION:\n")
                f.write("=" * 50 + "\n")
                for nct_id in set(all_nct_mentions):
                    f.write(f"- {nct_id}\n")
                f.write("\n")
                
                # Count mentions to see which trials were emphasized
                trial_counts = {}
                for nct_id in all_nct_mentions:
                    trial_counts[nct_id] = trial_counts.get(nct_id, 0) + 1
                
                f.write("TRIAL EMPHASIS (mention frequency):\n")
                for nct_id, count in sorted(trial_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"- {nct_id}: mentioned {count} time{'s' if count != 1 else ''}\n")
                f.write("\n")
            
            # Include the full therapy recommendation and rationale for context
            f.write("FULL THERAPY RECOMMENDATION CONTEXT:\n")
            f.write("=" * 50 + "\n")
            if therapy_text:
                f.write("THERAPY RECOMMENDATION:\n")
                f.write(f"{therapy_text}\n\n")
            
            if rationale_text:
                f.write("RATIONALE:\n")
                f.write(f"{rationale_text}\n\n")
            
            # Note any parsing issues
            if recommendation_result.get('parsing_error'):
                f.write("PARSING ISSUES:\n")
                f.write(f"Error: {recommendation_result['parsing_error']}\n\n")
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API for therapy recommendation"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/piakoller/prism",
            "X-Title": "Therapy Recommender"
        }
        
        logger.info(f"🔧 Using unlimited output tokens for model: {self.model}")

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
            data.update({
                "stream": False,  # Ensure no streaming
                "top_p": 0.95,    # Add top_p for Gemini models
            })
        
        logger.info(f"🔗 Calling LLM API for therapy recommendation...")
        response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"LLM API call failed: {response.status_code} - {response.text}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _parse_recommendation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM recommendation response from XML-like tags"""
        
        try:
            logger.info(f"🔍 Parsing recommendation response (length: {len(response)} chars)")
            
            # Try to extract content from therapy_recommendation, rationale, and trial_selection_reasoning tags
            trial_selection_reasoning = self._extract_tag_content(response, 'trial_selection_reasoning')
            therapy_rec = self._extract_tag_content(response, 'therapy_recommendation')
            rationale = self._extract_tag_content(response, 'rationale')
            
            if therapy_rec or rationale or trial_selection_reasoning:
                logger.info("✅ Successfully extracted content from XML-like tags")
                
                # Validate trial selection reasoning for duplicates
                if trial_selection_reasoning:
                    validation_warnings = self._validate_trial_selection_reasoning(trial_selection_reasoning)
                    if validation_warnings:
                        logger.warning(f"⚠️ Trial selection reasoning validation warnings: {validation_warnings}")
                
                # Create structured response with the extracted content
                # Add basic structure for compatibility with downstream components
                return {
                    "trial_selection_reasoning": trial_selection_reasoning,
                    "therapy_recommendation_text": therapy_rec,
                    "rationale_text": rationale,
                    "recommendation_summary": {
                        "primary_recommendation": self._extract_primary_recommendation(therapy_rec),
                        "recommendation_strength": "medium",  # Default since not specified in simple format
                        "evidence_level": "medium",
                        "urgency": "routine"
                    },
                    "therapy_recommendations": self._extract_therapy_options(therapy_rec),
                    "trial_opportunities": self._extract_trial_info(therapy_rec),
                    "evidence_confidence": "medium",
                    "raw_response": response
                }
            else:
                # If no tags found, treat as JSON fallback
                logger.warning("⚠️ No therapy_recommendation, rationale, or trial_selection_reasoning tags found, trying JSON parsing")
                return self._parse_json_fallback(response)
                
        except Exception as e:
            logger.error(f"❌ Recommendation parsing failed: {e}")
            logger.error(f"Response preview (first 1000 chars): {response[:1000]}")
            return {
                "trial_selection_reasoning": "Parsing failed",
                "therapy_recommendation_text": response.strip(),
                "rationale_text": "Parsing failed",
                "parsing_error": f"Parsing error: {str(e)}",
                "recommendation_summary": {
                    "primary_recommendation": "Parsing failed",
                    "recommendation_strength": "weak",
                    "evidence_level": "low"
                },
                "evidence_confidence": "low",
                "raw_response": response
            }
    
    def _extract_tag_content(self, response: str, tag_name: str) -> str:
        """Extract content from XML-like tags."""
        start_tag = f'<{tag_name}>'
        end_tag = f'</{tag_name}>'
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            return response[start_idx + len(start_tag):end_idx].strip()
        
        return ""
    
    def _extract_primary_recommendation(self, therapy_text: str) -> str:
        """Extract the primary recommendation from therapy text."""
        if not therapy_text:
            return "No recommendation provided"
        
        # Take the first sentence or first 100 characters as primary recommendation
        sentences = therapy_text.split('.')
        if sentences and sentences[0].strip():
            return sentences[0].strip()[:200] + ("..." if len(sentences[0]) > 200 else "")
        
        return therapy_text[:200] + ("..." if len(therapy_text) > 200 else "")
    
    def _extract_therapy_options(self, therapy_text: str) -> List[Dict[str, Any]]:
        """Extract therapy options from therapy text."""
        if not therapy_text:
            return []
        
        # Simple extraction - look for trial IDs and therapy names
        import re
        
        # Look for NCT IDs
        nct_matches = re.findall(r'NCT\d+', therapy_text)
        
        options = []
        if nct_matches:
            for nct_id in nct_matches[:3]:  # Limit to 3
                options.append({
                    "option_type": "trial_participation",
                    "therapy_name": f"Clinical trial {nct_id}",
                    "nct_id": nct_id,
                    "rationale": "Based on therapy recommendation text",
                    "priority_level": "medium"
                })
        else:
            # Generic option if no specific trials found
            options.append({
                "option_type": "standard_care",
                "therapy_name": "Standard care based on guidelines",
                "rationale": therapy_text[:200] + ("..." if len(therapy_text) > 200 else ""),
                "priority_level": "medium"
            })
        
        return options
    
    def _extract_trial_info(self, therapy_text: str) -> Dict[str, Any]:
        """Extract trial information from therapy text."""
        import re
        
        nct_matches = re.findall(r'NCT\d+', therapy_text)
        
        recommended_trials = []
        for nct_id in nct_matches[:3]:  # Limit to 3
            recommended_trials.append({
                "nct_id": nct_id,
                "trial_name": f"Trial {nct_id}",
                "why_recommended": "Mentioned in therapy recommendation",
                "potential_impact": "As described in recommendation text"
            })
        
        return {
            "recommended_trials": recommended_trials,
            "trial_search_suggestions": ["Consider additional trials based on assessment"] if not recommended_trials else []
        }
    
    def _parse_json_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback JSON parsing for compatibility."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                logger.info(f"🔍 Attempting to parse JSON response (length: {len(json_str)} chars)")
                recommendation = json.loads(json_str)
                return recommendation
            else:
                logger.warning(f"⚠️ Could not find JSON braces in response (length: {len(response)} chars)")
                return {
                    "raw_response": response,
                    "parsing_error": "Could not extract JSON from response",
                    "recommendation_summary": {
                        "primary_recommendation": "Unable to parse",
                        "recommendation_strength": "weak",
                        "evidence_level": "low"
                    },
                    "evidence_confidence": "low"
                }
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed at line {getattr(e, 'lineno', 'unknown')}, col {getattr(e, 'colno', 'unknown')}: {e}")
            logger.error(f"Response preview (first 500 chars): {response[:500]}")
            return {
                "raw_response": response,
                "parsing_error": f"JSON decode error at line {getattr(e, 'lineno', 'unknown')}, col {getattr(e, 'colno', 'unknown')}: {str(e)}",
                "recommendation_summary": {
                    "primary_recommendation": "Parsing failed",
                    "recommendation_strength": "weak",
                    "evidence_level": "low"
                },
                "evidence_confidence": "low"
            }
    
    def get_recommendation_summary(self, recommendation: Dict[str, Any]) -> str:
        """Get a human-readable summary of the therapy recommendation"""
        
        summary_lines = []
        summary_lines.append("=== THERAPY RECOMMENDATION SUMMARY ===")
        summary_lines.append(f"Patient ID: {recommendation.get('metadata', {}).get('patient_id', 'unknown')}")
        summary_lines.append(f"Recommendation Time: {recommendation.get('metadata', {}).get('timestamp', 'unknown')}")
        summary_lines.append(f"Evidence Confidence: {recommendation.get('evidence_confidence', 'unknown')}")
        summary_lines.append("")
        
        # Trial selection reasoning (new section)
        trial_reasoning = recommendation.get('trial_selection_reasoning', '')
        if trial_reasoning:
            summary_lines.append("LLM TRIAL SELECTION REASONING:")
            summary_lines.append("-" * 50)
            
            # Format trial reasoning with proper line breaks (truncated for summary)
            reasoning_lines = trial_reasoning.split('\n')
            for line in reasoning_lines[:10]:  # Show first 10 lines
                if line.strip():
                    summary_lines.append(line.strip())
                else:
                    summary_lines.append("")
            
            if len(reasoning_lines) > 10:
                summary_lines.append("... [truncated - see full reasoning file]")
            
            summary_lines.append("")
            summary_lines.append("-" * 50)
        
        # Main therapy recommendation text
        therapy_text = recommendation.get('therapy_recommendation_text', '')
        if therapy_text:
            summary_lines.append("THERAPY RECOMMENDATION:")
            summary_lines.append("-" * 50)
            
            # Format therapy text with proper line breaks
            therapy_lines = therapy_text.split('\n')
            for line in therapy_lines:
                if line.strip():
                    summary_lines.append(line.strip())
                else:
                    summary_lines.append("")
            
            summary_lines.append("")
            summary_lines.append("-" * 50)
        
        # Rationale text
        rationale_text = recommendation.get('rationale_text', '')
        if rationale_text:
            summary_lines.append("")
            summary_lines.append("RATIONALE:")
            summary_lines.append("-" * 50)
            
            # Format rationale text with proper line breaks
            rationale_lines = rationale_text.split('\n')
            for line in rationale_lines:
                if line.strip():
                    summary_lines.append(line.strip())
                else:
                    summary_lines.append("")
            
            summary_lines.append("")
            summary_lines.append("-" * 50)
        
        # Primary recommendation (extracted)
        rec_summary = recommendation.get('recommendation_summary', {})
        if rec_summary.get('primary_recommendation') and rec_summary.get('primary_recommendation') != 'No recommendation provided':
            summary_lines.append("")
            summary_lines.append("EXTRACTED PRIMARY RECOMMENDATION:")
            summary_lines.append(f"  {rec_summary.get('primary_recommendation', 'Unknown')}")
            summary_lines.append(f"  Strength: {rec_summary.get('recommendation_strength', 'Unknown')}")
            summary_lines.append(f"  Evidence Level: {rec_summary.get('evidence_level', 'Unknown')}")
        
        # Trial opportunities (extracted)
        trial_opps = recommendation.get('trial_opportunities', {})
        recommended_trials = trial_opps.get('recommended_trials', [])
        if recommended_trials:
            summary_lines.append("")
            summary_lines.append("IDENTIFIED TRIALS:")
            for trial in recommended_trials:
                summary_lines.append(f"  - {trial.get('nct_id', 'Unknown')}: {trial.get('trial_name', 'Unknown')}")
                summary_lines.append(f"    Rationale: {trial.get('why_recommended', 'Unknown')}")
        
        # Show parsing information if there were issues
        if recommendation.get('parsing_error'):
            summary_lines.append("")
            summary_lines.append("PARSING NOTES:")
            summary_lines.append(f"  {recommendation.get('parsing_error')}")
        
        return "\n".join(summary_lines)

    def _validate_trial_selection_reasoning(self, reasoning_text: str) -> List[str]:
        """
        Validate trial selection reasoning for common issues like duplicate trial mentions
        
        Args:
            reasoning_text: The trial selection reasoning text to validate
            
        Returns:
            List of warning messages if issues are found
        """
        warnings = []
        
        if not reasoning_text:
            return warnings
        
        # Extract NCT numbers mentioned in the text
        import re
        nct_pattern = r'NCT\d{8}'
        nct_mentions = re.findall(nct_pattern, reasoning_text, re.IGNORECASE)
        
        # Check for duplicate NCT mentions
        nct_counts = {}
        for nct in nct_mentions:
            nct_upper = nct.upper()
            nct_counts[nct_upper] = nct_counts.get(nct_upper, 0) + 1
        
        # Find duplicates
        duplicates = {nct: count for nct, count in nct_counts.items() if count > 1}
        if duplicates:
            warnings.append(f"Duplicate NCT mentions found: {duplicates}")
        
        # Check for potential typos in NCT numbers (too short/long)
        invalid_nct_pattern = r'NCT\d{1,7}(?!\d)|NCT\d{9,}'
        invalid_ncts = re.findall(invalid_nct_pattern, reasoning_text, re.IGNORECASE)
        if invalid_ncts:
            warnings.append(f"Potentially invalid NCT numbers (wrong length): {invalid_ncts}")
        
        # Check for trials mentioned in both USED and NOT USED sections
        used_section = ""
        not_used_section = ""
        
        # Split sections
        if "TRIALS USED TO INFORM PRIMARY RECOMMENDATION:" in reasoning_text:
            parts = reasoning_text.split("TRIALS USED TO INFORM PRIMARY RECOMMENDATION:")
            if len(parts) > 1:
                after_used = parts[1]
                if "TRIALS NOT USED IN PRIMARY RECOMMENDATION:" in after_used:
                    used_and_not_parts = after_used.split("TRIALS NOT USED IN PRIMARY RECOMMENDATION:")
                    used_section = used_and_not_parts[0]
                    if len(used_and_not_parts) > 1:
                        not_used_section = used_and_not_parts[1]
                else:
                    used_section = after_used
        
        if used_section and not_used_section:
            used_ncts = set(re.findall(nct_pattern, used_section, re.IGNORECASE))
            not_used_ncts = set(re.findall(nct_pattern, not_used_section, re.IGNORECASE))
            
            # Find trials in both sections (case-insensitive)
            used_ncts_upper = {nct.upper() for nct in used_ncts}
            not_used_ncts_upper = {nct.upper() for nct in not_used_ncts}
            
            overlap = used_ncts_upper.intersection(not_used_ncts_upper)
            if overlap:
                warnings.append(f"Trials mentioned in both USED and NOT USED sections: {overlap}")
        
        return warnings
