#!/usr/bin/env python3
"""
Agentic Patient Assessment an             # Use config defaults for models
        assessment_model = assessment_model or OPENROUTER_MODEL
        matching_model = matching_model or OPENROUTER_MODEL
        analysis_model = analysis_model or OPENROUTER_MODEL
        recommendation_model = recommendation_model or OPENROUTER_MODEL
        guidelines_model = guidelines_model or OPENROUTER_MODEL
        
        # Initialize all components
        self.assessor = PatientAssessor(api_key=self.api_key, model=assessment_model)
        self.trial_matcher = TrialMatcher(api_key=self.api_key, model=matching_model)
        self.trial_analyzer = TrialAnalyzer(api_key=self.api_key, model=analysis_model)
        self.therapy_recommender = TherapyRecommender(api_key=self.api_key, model=recommendation_model)
        self.guidelines_assessor = GuidelinesAssessor(api_key=self.api_key, model=guidelines_model)
        
        logger.info(f"🚀 Agentic Assessment Workflow initialized")
        logger.info(f"📋 Components: Patient Assessment, Trial Matching, Trial Analysis, Therapy Recommendations, Guidelines Matching")e all components
        self.assessor = PatientAssessor(api_key=self.api_key, model=assessment_model)
        self.trial_matcher = TrialMatcher(api_key=self.api_key, model=matching_model)
        self.therapy_recommender = TherapyRecommender(api_key=self.api_key, model=recommendation_model)
        self.guidelines_assessor = GuidelinesAssessor(api_key=self.api_key, model=guidelines_model)
        
        logger.info(f"🚀 Agentic Assessment Workflow initialized")
        logger.info(f"📋 Components: Patient Assessment, Trial Matching, Therapy Recommendations, Guidelines Matching")
        logger.info(f"🤖 Using model: {OPENROUTER_MODEL}")Matching System

This module orchestrates the complete workflow:
1. Initial patient assessment using LLM with guidelines
2. Trial matching based on assessment needs
3. Final therapy recommendation with selected trials
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL
from agentic_assessment.patient_assessor import PatientAssessor
from agentic_assessment.trial_matcher import TrialMatcher
from agentic_assessment.trial_analyzer import TrialAnalyzer
from agentic_assessment.therapy_recommender import TherapyRecommender
from agentic_assessment.guidelines_matcher import GuidelinesAssessor
from agentic_assessment.recommendation_validator import RecommendationValidator
from agentic_assessment.smart_regeneration import SmartRegenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgenticWorkflow:
    """Main orchestrator for the agentic assessment workflow"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 assessment_model: Optional[str] = None,
                 matching_model: Optional[str] = None,
                 analysis_model: Optional[str] = None,
                 recommendation_model: Optional[str] = None,
                 guidelines_model: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 trial_data_path: Optional[str] = None):
        """
        Initialize the agentic workflow
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            assessment_model: Model for patient assessment (defaults to config.OPENROUTER_MODEL)
            matching_model: Model for trial matching (defaults to config.OPENROUTER_MODEL)
            analysis_model: Model for trial analysis (defaults to config.OPENROUTER_MODEL)
            recommendation_model: Model for therapy recommendations (defaults to config.OPENROUTER_MODEL)
            guidelines_model: Model for guidelines matching (defaults to config.OPENROUTER_MODEL)
            output_dir: Output directory for results (optional)
            trial_data_path: Path to trial data file (optional)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        # Use config defaults for models
        assessment_model = assessment_model or OPENROUTER_MODEL
        matching_model = matching_model or OPENROUTER_MODEL
        analysis_model = analysis_model or OPENROUTER_MODEL
        recommendation_model = recommendation_model or OPENROUTER_MODEL
        guidelines_model = guidelines_model or OPENROUTER_MODEL

        # Set output_dir and trial_data_path attributes
        self.output_dir = output_dir or "."
        self.trial_data_path = trial_data_path
        
        # Initialize all components
        self.assessor = PatientAssessor(api_key=self.api_key, model=assessment_model)
        self.trial_matcher = TrialMatcher(api_key=self.api_key, model=matching_model)
        self.trial_analyzer = TrialAnalyzer(api_key=self.api_key, model=analysis_model)
        self.therapy_recommender = TherapyRecommender(api_key=self.api_key, model=recommendation_model)
        self.guidelines_assessor = GuidelinesAssessor(api_key=self.api_key, model=guidelines_model)
        self.validator = RecommendationValidator(api_key=self.api_key, model=recommendation_model)
        self.regenerator = SmartRegenerator(self)
        
        logger.info(f"🚀 Agentic Assessment Workflow initialized")
        logger.info(f"📋 Components: Patient Assessment, Trial Matching, Trial Analysis, Therapy Recommendations, Guidelines Matching, Validation, Smart Regeneration")
    
    def _create_output_dir(self) -> str:
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"agentic_results/agentic_run_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def run_single_patient(self, patient_data: dict, trial_data_path: str,
                          guidelines_context: Optional[str] = "", guidelines_dir: Optional[str] = None,
                          output_dir: str = ".", use_cache: bool = True, 
                          force_regenerate: bool = False) -> dict:
        """
        Run the complete workflow for a single patient including guidelines matching
        
        Args:
            patient_data: Dictionary containing patient information
            trial_data_path: Path to trial data file
            guidelines_context: Optional guidelines context string
            guidelines_dir: Directory containing guideline files for matching
            output_dir: Directory to save results
            use_cache: Whether to use cached results if available (default: True)
            force_regenerate: Whether to force regeneration of all results (default: False)
            
        Returns:
            Dictionary containing all workflow results
        """
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        logger.info(f"🚀 Starting complete workflow for patient {patient_id}")
        
        if force_regenerate and use_cache:
            logger.info("🔄 Force regenerate enabled - clearing existing cache...")
            self.assessor.clear_assessment_cache(patient_id, output_dir)
            self.trial_analyzer.clear_trial_analysis_cache(patient_id, output_dir)
        
        workflow_results = {
            'patient_id': patient_id,
            'status': 'in_progress',
            'timestamp': datetime.now().isoformat(),
            'output_directory': output_dir,
            'files_generated': [],
            'caching_enabled': use_cache,
            'force_regenerate': force_regenerate
        }
        
        progress_file = os.path.join(output_dir, f"patient_{patient_id}_workflow_progress.json")
        
        def update_progress(step, status, details=None):
            """Update progress file with current step status"""
            workflow_results['current_step'] = step
            workflow_results['step_status'] = status
            workflow_results['last_updated'] = datetime.now().isoformat()
            if details:
                workflow_results['step_details'] = details
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2, ensure_ascii=False)
        
        update_progress("workflow_start", "initializing", "Starting complete patient workflow")
        
        try:
            # Step 1: Guidelines Matching (if guidelines directory provided)
            guidelines_result = None
            if guidelines_dir and os.path.exists(guidelines_dir):
                update_progress("guidelines_matching", "running", "Matching relevant medical guidelines")
                logger.info("📚 Step 1: Matching relevant medical guidelines...")
                guidelines_result = self.guidelines_assessor.find_relevant_guidelines(
                    patient_data=patient_data,
                    guidelines_dir=guidelines_dir,
                    output_dir=output_dir
                )
                workflow_results['guidelines_result'] = guidelines_result
                logger.info(f"✅ Guidelines matching completed: {guidelines_result.get('guidelines_found', 0)} relevant guidelines found")
                update_progress("guidelines_matching", "completed", f"Found {guidelines_result.get('guidelines_found', 0)} relevant guidelines")
            
            enhanced_guidelines_context = guidelines_context or ""
            
            # Step 2: Comprehensive Patient Assessment (with matched guidelines only, no trials)
            update_progress("patient_assessment", "running", "Performing comprehensive patient assessment")
            logger.info("📋 Step 2: Performing comprehensive patient assessment...")
            assessment_result = self.assessor.assess_patient(
                patient_data=patient_data,
                guidelines_context=guidelines_context,
                matched_guidelines=guidelines_result,
                matched_trials=None,
                output_dir=output_dir,
                use_cache=use_cache
            )
            workflow_results['assessment_result'] = assessment_result
            
            if 'metadata' in assessment_result and 'prompt_file' in assessment_result['metadata']:
                workflow_results['files_generated'].extend([
                    assessment_result['metadata']['prompt_file'],
                    assessment_result['metadata']['response_file'],
                    os.path.join(output_dir, f"patient_{patient_id}_assessment.json")
                ])
            update_progress("patient_assessment", "completed", f"Assessment saved: {assessment_result.get('metadata', {}).get('prompt_file', 'assessment files')}")
            
            # Step 3: Trial Matching Based on Assessment
            update_progress("trial_matching", "running", "Matching trials based on patient assessment")
            logger.info("🎯 Step 3: Matching trials based on patient assessment...")
            trial_matching_result = self.trial_matcher.match_trials(
                assessment=assessment_result,
                trial_data_path=trial_data_path,
                patient_data=patient_data,
                output_dir=output_dir
            )
            workflow_results['trial_matching_result'] = trial_matching_result
            
            matching_files = []
            if 'evaluation_results' in trial_matching_result:
                for eval_result in trial_matching_result['evaluation_results']:
                    if 'prompt_file' in eval_result:
                        matching_files.append(eval_result['prompt_file'])
                    if 'response_file' in eval_result:
                        matching_files.append(eval_result['response_file'])
            matching_files.append(os.path.join(output_dir, f"patient_{patient_id}_trial_matching.json"))
            workflow_results['files_generated'].extend(matching_files)
            update_progress("trial_matching", "completed", f"Evaluated {len(trial_matching_result.get('evaluation_results', []))} trials")
            
            # Step 4: Trial Analysis (NEW TWO-STAGE APPROACH)
            update_progress("trial_analysis", "running", "Analyzing matched trials for patient-specific relevance")
            logger.info("🔬 Step 4: Analyzing matched trials for patient-specific relevance...")
            
            relevant_trials_data = {
                "trials": trial_matching_result.get("relevant_trials", []),
                "summary": trial_matching_result.get("summary", {}),
                "metadata": trial_matching_result.get("metadata", {})
            }
            logger.info(f"📊 Passing {len(relevant_trials_data['trials'])} relevant trials to analyzer")
            
            trial_analysis_result = self.trial_analyzer.analyze_trials(
                patient_assessment=assessment_result,
                trials_data=relevant_trials_data,
                patient_id=patient_id,
                output_dir=output_dir,
                use_cache=use_cache
            )
            workflow_results['trial_analysis_result'] = trial_analysis_result
            
            analysis_files = []
            if trial_analysis_result.get('metadata'):
                if 'prompt_file' in trial_analysis_result['metadata']:
                    analysis_files.append(trial_analysis_result['metadata']['prompt_file'])
                if 'response_file' in trial_analysis_result['metadata']:
                    analysis_files.append(trial_analysis_result['metadata']['response_file'])
                if 'results_file' in trial_analysis_result['metadata']:
                    analysis_files.append(trial_analysis_result['metadata']['results_file'])
            workflow_results['files_generated'].extend(analysis_files)
            update_progress("trial_analysis", "completed", f"Analyzed {trial_analysis_result.get('trials_analyzed', 0)} trials")
            
            # Step 5: Final Therapy Recommendation (Updated to use trial analysis results)
            update_progress("therapy_recommendation", "running", "Creating comprehensive therapy recommendation")
            logger.info("💊 Step 5: Creating comprehensive therapy recommendation with analyzed trial data...")
            
            if guidelines_result and guidelines_result.get('relevant_guidelines'):
                full_guidelines_sections = []
                for idx, g in enumerate(guidelines_result['relevant_guidelines'], 1):
                    title = g.get('guideline_title', 'Unknown Guideline')
                    reason = g.get('relevance_reason', 'No explanation provided')
                    content = g.get('guideline_content', '')
                    full_guidelines_sections.append(
                        "\n".join([
                            f"===== GUIDELINE {idx}: {title} =====",
                            f"Relevance: {reason}",
                            "--- BEGIN FULL CONTENT ---",
                            content if content else "[No content available]",
                            "--- END FULL CONTENT ---",
                            ""
                        ])
                    )
                full_guidelines_text = "\n".join(full_guidelines_sections)

                header = "FULL MATCHED GUIDELINES (embedded for LLM; exact text follows)\n" + ("=" * 60) + "\n"
                if enhanced_guidelines_context:
                    enhanced_guidelines_context += f"\n\n{header}{full_guidelines_text}"
                else:
                    enhanced_guidelines_context = f"{header}{full_guidelines_text}"

                try:
                    guidelines_full_path = os.path.join(output_dir, f"patient_{patient_id}_matched_guidelines_full.txt")
                    with open(guidelines_full_path, 'w', encoding='utf-8') as gf:
                        gf.write("=== FULL MATCHED GUIDELINES ===\n")
                        gf.write(f"Patient ID: {patient_id}\n")
                        gf.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        gf.write("\n")
                        gf.write(full_guidelines_text)
                    logger.info(f"📄 Full matched guidelines saved to: {guidelines_full_path}")
                    workflow_results['files_generated'].append(guidelines_full_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save full matched guidelines file: {e}")
            
            recommendation_result = self.therapy_recommender.generate_recommendation(
                assessment=assessment_result,
                trial_analysis_result=trial_analysis_result,
                guidelines_context=enhanced_guidelines_context,
                original_trials_data={"matched_trials": trial_matching_result.get("relevant_trials", [])},
                patient_data=patient_data,
                output_dir=output_dir
            )
            workflow_results['recommendation_result'] = recommendation_result
            
            if 'metadata' in recommendation_result and 'prompt_file' in recommendation_result['metadata']:
                workflow_results['files_generated'].extend([
                    recommendation_result['metadata']['prompt_file'],
                    recommendation_result['metadata']['response_file'],
                    os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation.json")
                ])
            update_progress("therapy_recommendation", "completed", f"Therapy recommendation generated: {recommendation_result.get('metadata', {}).get('recommendation_file', 'recommendation files')}")
            
            # Step 6: Validate Recommendation (NEW)
            update_progress("validation", "running", "Validating therapy recommendation")
            logger.info("🔍 Step 6: Validating therapy recommendation...")
            
            validation_result = self.validator.validate_recommendation(
                recommendation_result=recommendation_result,
                assessment_result=assessment_result,
                trial_analysis_result=trial_analysis_result,
                guidelines_result=guidelines_result or {},
                patient_id=patient_id,
                output_dir=output_dir,
                original_trials_data={"matched_trials": trial_matching_result.get("relevant_trials", [])}
            )
            workflow_results['validation_result'] = validation_result
            
            validation_files = [
                os.path.join(output_dir, f"patient_{patient_id}_validation_results.json"),
                os.path.join(output_dir, f"patient_{patient_id}_validation_summary.txt")
            ]
            if validation_result.get('validation_steps', {}).get('semantic', {}).get('prompt_file'):
                validation_files.extend([
                    validation_result['validation_steps']['semantic']['prompt_file'],
                    validation_result['validation_steps']['semantic']['response_file']
                ])
            workflow_results['files_generated'].extend(validation_files)
            
            MAX_REGENERATION_ITERATIONS = 2
            
            if validation_result.get('requires_regeneration', False):
                logger.warning("⚠️ VALIDATION FAILED - Initiating smart regeneration...")
                logger.warning(f"❌ {len(validation_result.get('regeneration_reasons', []))} issues identified")
                
                current_validation = validation_result
                regeneration_iteration = 0
                validation_history = []
                
                while regeneration_iteration < MAX_REGENERATION_ITERATIONS and current_validation.get('requires_regeneration', False):
                    regeneration_iteration += 1
                    logger.info(f"🔄 Regeneration attempt {regeneration_iteration}/{MAX_REGENERATION_ITERATIONS}")
                    
                    validation_history.append(current_validation)
                    
                    update_progress("regeneration", "running", f"Regeneration iteration {regeneration_iteration}/{MAX_REGENERATION_ITERATIONS}")
                    
                    regenerated_results = self.regenerator.analyze_validation_and_regenerate(
                        validation_result=current_validation,
                        patient_data=patient_data,
                        trial_data_path=trial_data_path,
                        guidelines_context=enhanced_guidelines_context,
                        guidelines_dir=guidelines_dir if guidelines_dir is not None else "",
                        output_dir=output_dir,
                        workflow_results=workflow_results,
                        validation_history=validation_history  # Pass cumulative history
                    )
                    
                    if not regenerated_results:
                        logger.error(f"❌ Regeneration iteration {regeneration_iteration} failed")
                        workflow_results['regeneration_status'] = 'failed'
                        break
                    
                    workflow_results = regenerated_results
                    logger.info(f"✅ Regeneration iteration {regeneration_iteration} completed")
                    
                    logger.info(f"🔍 Validating regeneration iteration {regeneration_iteration}...")
                    update_progress("revalidation", "running", f"Validating iteration {regeneration_iteration}")
                    
                    revalidation_result = self.validator.validate_recommendation(
                        recommendation_result=workflow_results['recommendation_result'],
                        assessment_result=workflow_results['assessment_result'],
                        trial_analysis_result=workflow_results['trial_analysis_result'],
                        guidelines_result=workflow_results.get('guidelines_result') or {},
                        patient_id=patient_id,
                        output_dir=output_dir
                    )
                    
                    # Save revalidation with iteration suffix
                    revalidation_file = os.path.join(output_dir, f"patient_{patient_id}_validation_results_iter{regeneration_iteration}.json")
                    revalidation_summary = os.path.join(output_dir, f"patient_{patient_id}_validation_summary_iter{regeneration_iteration}.txt")
                    
                    with open(revalidation_file, 'w', encoding='utf-8') as f:
                        json.dump(revalidation_result, f, indent=2, ensure_ascii=False)
                    
                    with open(revalidation_summary, 'w', encoding='utf-8') as f:
                        f.write(f"=== VALIDATION ITERATION {regeneration_iteration} ===\n\n")
                        f.write(self.validator._create_validation_summary(revalidation_result))
                    
                    workflow_results[f'validation_iter{regeneration_iteration}'] = revalidation_result
                    workflow_results['files_generated'].extend([revalidation_file, revalidation_summary])
                    
                    logger.info(f"💾 Validation summary (iteration {regeneration_iteration}) saved to: {revalidation_summary}")
                    
                    current_validation = revalidation_result
                    
                    if not revalidation_result.get('requires_regeneration', False):
                        logger.info(f"✅ Regeneration successful after {regeneration_iteration} iteration(s)!")
                        workflow_results['regeneration_status'] = 'success'
                        workflow_results['regeneration_iterations'] = regeneration_iteration
                        update_progress("revalidation", "passed", f"Validation passed after {regeneration_iteration} iteration(s)")
                        break
                    else:
                        remaining_issues = len(revalidation_result.get('regeneration_reasons', []))
                        logger.warning(f"⚠️ Iteration {regeneration_iteration}: {remaining_issues} issues remaining")
                        
                        if regeneration_iteration >= MAX_REGENERATION_ITERATIONS:
                            logger.warning(f"⚠️ Maximum regeneration iterations ({MAX_REGENERATION_ITERATIONS}) reached")
                            workflow_results['regeneration_status'] = 'needs_manual_review'
                            workflow_results['regeneration_iterations'] = regeneration_iteration
                            update_progress("revalidation", "max_iterations", f"Requires manual review after {regeneration_iteration} iterations")
                
                update_progress("regeneration", "completed", f"Regeneration status: {workflow_results.get('regeneration_status', 'unknown')}")
            else:
                workflow_results['regeneration_required'] = False
                workflow_results['regeneration_status'] = 'not_needed'
                logger.info("✅ Validation passed - no regeneration needed")
                update_progress("validation", "passed", "Validation successful - no regeneration needed")
            
            workflow_results['status'] = 'completed'
            update_progress("workflow_complete", "completed", f"All steps completed successfully. {len(workflow_results['files_generated'])} files generated.")
            
            workflow_file = os.path.join(output_dir, f"patient_{patient_id}_complete_workflow.json")
            with open(workflow_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Complete workflow finished for patient {patient_id}")
            logger.info(f"📄 Complete results saved to: {workflow_file}")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"❌ Error in workflow for patient {patient_id}: {e}")
            workflow_results['status'] = 'failed'
            workflow_results['error'] = str(e)
            update_progress("workflow_error", "failed", f"Error: {str(e)}")
            return workflow_results
    
    def _create_regeneration_context(self, validation_result: Dict[str, Any]) -> str:
        """Create context string with validation feedback for regeneration"""
        
        feedback_parts = [
            "=" * 70,
            "VALIDATION FEEDBACK FOR REGENERATION",
            "=" * 70,
            "",
            "The previous recommendation did not pass validation. Please address the following issues:",
            ""
        ]
        
        # Add critical issues
        all_issues = []
        for step_name, step_result in validation_result.get('validation_steps', {}).items():
            critical_issues = step_result.get('critical_issues', [])
            if critical_issues:
                all_issues.extend([f"[{step_name.upper()}] {issue}" for issue in critical_issues])
        
        if all_issues:
            feedback_parts.append("CRITICAL ISSUES TO ADDRESS:")
            for issue in all_issues:
                feedback_parts.append(f"  ❌ {issue}")
            feedback_parts.append("")
        
        # Add detailed reasoning from LLM validation
        semantic_validation = validation_result.get('validation_steps', {}).get('semantic', {})
        if semantic_validation.get('detailed_reasoning'):
            feedback_parts.append("DETAILED VALIDATION REASONING:")
            feedback_parts.append(semantic_validation['detailed_reasoning'])
            feedback_parts.append("")
        
        # Add improvement recommendations
        if semantic_validation.get('improvement_recommendations'):
            feedback_parts.append("SPECIFIC IMPROVEMENTS NEEDED:")
            feedback_parts.append(semantic_validation['improvement_recommendations'])
            feedback_parts.append("")
        
        feedback_parts.extend([
            "=" * 70,
            "Please regenerate the recommendation addressing ALL issues above.",
            "=" * 70,
            ""
        ])
        
        return "\n".join(feedback_parts)
    
    def run_batch_patients(self, patients_file: str, trial_data_path: str,
                          guidelines_context: Optional[str] = None, guidelines_dir: Optional[str] = None,
                          output_dir: str = "batch_results", use_cache: bool = True,
                          force_regenerate: bool = False) -> list:
        """
        Run the workflow for multiple patients
        
        Args:
            patients_file: Path to Excel file with patient data
            trial_data_path: Path to trial data file
            guidelines_context: Optional guidelines context string
            guidelines_dir: Directory containing guideline files
            output_dir: Base directory for all results
            use_cache: Whether to use cached results if available (default: True)
            force_regenerate: Whether to force regeneration of all results (default: False)
            
        Returns:
            List of results for each patient
        """
        import pandas as pd
        
        logger.info(f"📊 Starting batch workflow for patients in: {patients_file}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(output_dir, f"batch_run_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        if patients_file.endswith('.xlsx'):
            df_patients = pd.read_excel(patients_file)
        else:
            df_patients = pd.read_csv(patients_file)
        
        patients = df_patients.to_dict('records')
        batch_results = []
        
        for i, patient in enumerate(patients, 1):
            patient_id = patient.get('ID', patient.get('id', f'patient_{i}'))
            logger.info(f"🏥 Processing patient {i}/{len(patients)} (ID: {patient_id})")
            
            try:
                patient_output_dir = os.path.join(batch_output_dir, f"patient_{patient_id}")
                os.makedirs(patient_output_dir, exist_ok=True)
                
                result = self.run_single_patient(
                    patient_data=patient,
                    trial_data_path=trial_data_path,
                    guidelines_context=guidelines_context,
                    guidelines_dir=guidelines_dir,
                    output_dir=patient_output_dir,
                    use_cache=use_cache,
                    force_regenerate=force_regenerate
                )
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Error processing patient {patient_id}: {e}")
                batch_results.append({
                    'patient_id': patient_id,
                    'status': 'failed',
                    'error': str(e)
                })
        
        batch_summary = {
            'batch_metadata': {
                'timestamp': datetime.now().isoformat(),
                'patients_file': patients_file,
                'trial_data_path': trial_data_path,
                'guidelines_dir': guidelines_dir,
                'output_directory': batch_output_dir,
                'total_patients': len(patients),
                'successful_assessments': len([r for r in batch_results if r.get('status') == 'completed']),
                'failed_assessments': len([r for r in batch_results if r.get('status') == 'failed'])
            },
            'patient_results': batch_results
        }
        
        batch_summary_file = os.path.join(batch_output_dir, 'batch_workflow_summary.json')
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Batch workflow completed!")
        logger.info(f"📊 {batch_summary['batch_metadata']['successful_assessments']}/{batch_summary['batch_metadata']['total_patients']} patients processed successfully")
        logger.info(f"📄 Batch summary saved to: {batch_summary_file}")
        
        return batch_results



    def run_full_workflow(self, patient_data: dict, guidelines_context: str = "") -> dict:
        """
        Run the complete agentic workflow
        
        Args:
            patient_data: Dictionary containing patient information
            guidelines_context: Optional guidelines context string
            
        Returns:
            Dictionary containing all results from the workflow
        """
        logger.info("🔍 Starting agentic assessment workflow...")
        
        assessment_result = self.assessor.assess_patient(
            patient_data=patient_data,
            guidelines_context=guidelines_context,
            output_dir=self.output_dir,
            use_cache=True
        )
        
        trial_data_path = self.trial_data_path if self.trial_data_path is not None else ""
        trial_matching_result = self.trial_matcher.match_trials(
            assessment=assessment_result,
            trial_data_path=trial_data_path,
            patient_data=patient_data,
            output_dir=self.output_dir
        )
        
        trial_analysis_result = self.trial_analyzer.analyze_trials(
            patient_assessment=assessment_result,
            trials_data=trial_matching_result,
            patient_id=patient_data.get('ID', 'unknown'),
            output_dir=self.output_dir,
            use_cache=True
        )
        
        therapy_result = self.therapy_recommender.generate_recommendation(
            assessment=assessment_result,
            trial_analysis_result=trial_analysis_result,
            original_trials_data=trial_matching_result,
            guidelines_context=guidelines_context,
            patient_data=patient_data,
            output_dir=self.output_dir
        )
        
        final_results = {
            'workflow_metadata': {
                'timestamp': datetime.now().isoformat(),
                'output_directory': self.output_dir,
                'patient_id': patient_data.get('ID', 'unknown'),
                'workflow_version': '2.0_two_stage'
            },
            'step_1_assessment': assessment_result,
            'step_2_trial_matching': trial_matching_result,
            'step_3_trial_analysis': trial_analysis_result,
            'step_4_therapy_recommendation': therapy_result
        }
        
        results_file = os.path.join(self.output_dir, 'complete_workflow_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Agentic workflow completed successfully!")
        logger.info(f"📄 Complete results saved to: {results_file}")
        
        return final_results
    
    def run_batch_workflow(self, patients_file: str, guidelines_context: Optional[str] = None) -> list:
        """
        Run the workflow for multiple patients
        
        Args:
            patients_file: Path to Excel file with patient data
            guidelines_context: Optional guidelines context string
            
        Returns:
            List of results for each patient
        """
        import pandas as pd
        
        logger.info(f"📊 Starting batch workflow for patients in: {patients_file}")
        
        df_patients = pd.read_excel(patients_file)
        patients = df_patients.to_dict('records')
        
        batch_results = []
        
        for i, patient in enumerate(patients, 1):
            logger.info(f"🏥 Processing patient {i}/{len(patients)} (ID: {patient.get('ID', 'unknown')})")
            
            try:
                patient_output_dir = os.path.join(self.output_dir, f"patient_{patient.get('ID', i)}")
                os.makedirs(patient_output_dir, exist_ok=True)
                
                workflow_instance = AgenticWorkflow(patient_output_dir)
                result = workflow_instance.run_full_workflow(patient, guidelines_context if guidelines_context is not None else "")
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Error processing patient {patient.get('ID', i)}: {e}")
                batch_results.append({
                    'error': str(e),
                    'patient_id': patient.get('ID', i)
                })
        
        batch_summary_file = os.path.join(self.output_dir, 'batch_workflow_summary.json')
        batch_summary = {
            'batch_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patients': len(patients),
                'successful_assessments': len([r for r in batch_results if 'error' not in r]),
                'failed_assessments': len([r for r in batch_results if 'error' in r])
            },
            'results': batch_results
        }
        
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Batch workflow completed!")
        logger.info(f"📄 Batch summary saved to: {batch_summary_file}")
        
        return batch_results

def main():
    """Main entry point for the agentic workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agentic Patient Assessment and Trial Matching')
    parser.add_argument('--patients', required=True, help='Path to Excel file with patient data')
    parser.add_argument('--trials', required=True, help='Path to trial data JSON file')
    parser.add_argument('--guidelines', help='Path to a directory containing guidelines files.')
    parser.add_argument('--output', help='Output directory (default: timestamped directory is created)')
    parser.add_argument('--single', help='Process only patient with this ID from the patient file')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching (force regeneration of all results)')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regeneration of all results (clear existing cache)')
    
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    force_regenerate = args.force_regenerate
    
    if force_regenerate and not use_cache:
        logger.warning("⚠️ Both --no-cache and --force-regenerate specified. Using --no-cache (disable caching completely).")
        force_regenerate = False
    
    workflow = AgenticWorkflow()
    
    if not use_cache:
        logger.info("💾 Caching disabled - all results will be regenerated")
    elif force_regenerate:
        logger.info("🔄 Force regenerate enabled - existing cache will be cleared")
    else:
        logger.info("💾 Caching enabled - will reuse existing results when available")
    
    output_dir = args.output
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"batch_results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    guidelines_context = None
    
    if args.single:
        # Process single patient
        import pandas as pd
        logger.info(f"🔍 Processing single patient with ID: {args.single}")
        try:
            df_patients = pd.read_excel(args.patients)
            # Convert single ID to appropriate type (try int first, then string)
            try:
                single_id = int(args.single)
            except ValueError:
                single_id = args.single
            
            patient_records = df_patients[df_patients['ID'] == single_id].to_dict('records')
            if not patient_records:
                logger.error(f"Patient with ID '{args.single}' not found in {args.patients}")
                sys.exit(1)
            patient = patient_records[0]
            
            patient_output_dir = os.path.join(output_dir, f"patient_{args.single}")
            os.makedirs(patient_output_dir, exist_ok=True)

            workflow.run_single_patient(
                patient_data=patient,
                trial_data_path=args.trials,
                guidelines_dir=args.guidelines,
                output_dir=patient_output_dir,
                use_cache=use_cache,
                force_regenerate=force_regenerate
            )
        except Exception as e:
            logger.error(f"❌ An error occurred processing patient {args.single}: {e}", exc_info=True)
            sys.exit(1)

    else:
        logger.info(f"📊 Processing batch of patients from: {args.patients}")
        try:
            workflow.run_batch_patients(
                patients_file=args.patients,
                trial_data_path=args.trials,
                guidelines_dir=args.guidelines,
                output_dir=output_dir,
                use_cache=use_cache,
                force_regenerate=force_regenerate
            )
        except Exception as e:
            logger.error(f"❌ An error occurred during the batch run: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
