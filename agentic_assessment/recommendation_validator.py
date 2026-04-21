#!/usr/bin/env python3
"""
Recommendation Validator Module

This module performs hybrid validation (rule-based + LLM-based) of therapy recommendations
to ensure consistency, completeness, safety, and evidence alignment.
"""

import os
import re
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)


class RecommendationValidator:
    """Validates therapy recommendations using hybrid rule-based and LLM-based approaches"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Recommendation Validator
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: LLM model to use for validation (defaults to config.OPENROUTER_MODEL)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please check your configuration.")
        
        logger.info(f"🔍 Recommendation Validator initialized with model: {self.model}")
    
    def validate_recommendation(
        self,
        recommendation_result: Dict[str, Any],
        assessment_result: Dict[str, Any],
        trial_analysis_result: Dict[str, Any],
        guidelines_result: Dict[str, Any],
        patient_id: str,
        output_dir: str = ".",
        original_trials_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete validation of therapy recommendation
        
        Args:
            recommendation_result: The therapy recommendation to validate
            assessment_result: Patient assessment results
            trial_analysis_result: Trial analysis results
            guidelines_result: Guidelines matching results
            patient_id: Patient identifier
            output_dir: Directory to save validation results
            original_trials_data: Original trial data with full evidence (for detailed validation)
            
        Returns:
            Dictionary containing validation results and decision to regenerate
        """
        logger.info(f"🔍 Starting validation for patient {patient_id}")
        
        validation_results = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'validation_steps': {},
            'overall_status': 'pending',
            'requires_regeneration': False,
            'regeneration_reasons': [],
            'validation_summary': ''
        }
        
        structural_validation = self._validate_structure(
            recommendation_result,
            assessment_result,
            trial_analysis_result
        )
        validation_results['validation_steps']['structural'] = structural_validation
        
        evidence_validation = self._validate_evidence_consistency(
            recommendation_result,
            trial_analysis_result
        )
        validation_results['validation_steps']['evidence_consistency'] = evidence_validation
        
        semantic_validation = self._validate_semantics_with_llm(
            recommendation_result,
            assessment_result,
            trial_analysis_result,
            guidelines_result,
            patient_id,
            output_dir,
            original_trials_data
        )
        validation_results['validation_steps']['semantic'] = semantic_validation
        
        validation_results = self._determine_regeneration_need(validation_results)
        validation_results['validation_summary'] = self._create_validation_summary(validation_results)
        self._save_validation_results(validation_results, patient_id, output_dir)
        
        # Log final status
        if validation_results['requires_regeneration']:
            logger.warning(f"⚠️ Validation FAILED - Recommendation requires regeneration")
            logger.warning(f"❌ Reasons: {', '.join(validation_results['regeneration_reasons'])}")
        else:
            logger.info(f"✅ Validation PASSED - Recommendation is acceptable")
        
        return validation_results
    
    def _validate_structure(
        self,
        recommendation_result: Dict[str, Any],
        assessment_result: Dict[str, Any],
        trial_analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rule-based validation of recommendation structure and completeness"""
        
        issues = []
        warnings = []
        
        # Check required sections exist
        therapy_rec_text = recommendation_result.get('therapy_recommendation_text', '')
        rationale_text = recommendation_result.get('rationale_text', '')
        
        if not therapy_rec_text or len(therapy_rec_text.strip()) < 100:
            issues.append("Therapy recommendation text is missing or too short (< 100 chars)")
        
        if not rationale_text or len(rationale_text.strip()) < 100:
            issues.append("Rationale text is missing or too short (< 100 chars)")
        
        # Check if recommendation addresses urgent needs
        assessment_urgency = assessment_result.get('urgency_level', '')
        if assessment_urgency in ['urgent', 'high']:
            if 'urgent' not in therapy_rec_text.lower() and 'immediate' not in therapy_rec_text.lower():
                warnings.append(f"Patient assessment indicates {assessment_urgency} urgency, but recommendation doesn't explicitly address urgency")
        
        # Check if trial opportunities are discussed when trials are available
        trial_analyses = trial_analysis_result.get('trial_analyses', [])
        included_trials = [t for t in trial_analyses if t.get('recommendation') == 'INCLUDE']
        
        if len(included_trials) > 0:
            nct_mentions = len(re.findall(r'NCT\d{8}', therapy_rec_text + rationale_text))
            if nct_mentions == 0:
                warnings.append(f"{len(included_trials)} trials were marked for inclusion, but no NCT numbers are mentioned in the recommendation")
        
        metadata = recommendation_result.get('metadata', {})
        if not metadata.get('timestamp'):
            issues.append("Missing timestamp in recommendation metadata")
        
        return {
            'status': 'fail' if issues else 'pass',
            'critical_issues': issues,
            'warnings': warnings,
            'checks_performed': [
                'Required sections present',
                'Minimum content length',
                'Urgency alignment',
                'Trial discussion completeness',
                'Metadata completeness'
            ]
        }
    
    def _validate_evidence_consistency(
        self,
        recommendation_result: Dict[str, Any],
        trial_analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that all cited evidence actually exists in the source data"""
        
        issues = []
        warnings = []
        
        # Extract all NCT numbers from recommendation text
        therapy_rec_text = recommendation_result.get('therapy_recommendation_text', '')
        rationale_text = recommendation_result.get('rationale_text', '')
        full_text = therapy_rec_text + " " + rationale_text
        
        cited_ncts = set(re.findall(r'NCT\d{8}', full_text))
        
        trial_analyses = trial_analysis_result.get('trial_analyses', [])
        available_ncts = set(t.get('nct_number', '') for t in trial_analyses if t.get('nct_number'))
        
        # Hallucination check: cited NCTs that don't exist in evidence
        hallucinated_ncts = cited_ncts - available_ncts
        if hallucinated_ncts:
            issues.append(f"Recommendation cites trials that don't exist in the evidence: {', '.join(sorted(hallucinated_ncts))}")
        
        included_ncts = set(t.get('nct_number', '') for t in trial_analyses 
                           if t.get('recommendation') == 'INCLUDE')
        excluded_ncts = set(t.get('nct_number', '') for t in trial_analyses 
                           if t.get('recommendation') == 'EXCLUDE')
        
        cited_excluded = cited_ncts & excluded_ncts
        if cited_excluded:
            warnings.append(f"Recommendation cites trials that were marked for EXCLUSION: {', '.join(sorted(cited_excluded))}")
        
        # Check if included trials are actually discussed
        not_discussed = included_ncts - cited_ncts
        if len(not_discussed) > 0 and len(included_ncts) <= 5:  # Only flag if small number of trials
            warnings.append(f"Trials marked for INCLUSION but not discussed: {', '.join(sorted(list(not_discussed)[:3]))}")
        
        return {
            'status': 'fail' if issues else 'pass',
            'critical_issues': issues,
            'warnings': warnings,
            'evidence_stats': {
                'trials_cited': len(cited_ncts),
                'trials_available': len(available_ncts),
                'trials_included': len(included_ncts),
                'trials_excluded': len(excluded_ncts),
                'hallucinated_citations': len(hallucinated_ncts)
            },
            'checks_performed': [
                'Trial citation accuracy',
                'Hallucination detection',
                'Inclusion/exclusion consistency',
                'Coverage of included trials'
            ]
        }
    
    def _validate_semantics_with_llm(
        self,
        recommendation_result: Dict[str, Any],
        assessment_result: Dict[str, Any],
        trial_analysis_result: Dict[str, Any],
        guidelines_result: Dict[str, Any],
        patient_id: str,
        output_dir: str,
        original_trials_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use LLM to validate semantic consistency and clinical reasoning"""
        
        validation_prompt = self._create_validation_prompt(
            recommendation_result,
            assessment_result,
            trial_analysis_result,
            guidelines_result,
            original_trials_data
        )
        
        prompt_file = os.path.join(output_dir, f"patient_{patient_id}_validation_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write("=== RECOMMENDATION VALIDATION PROMPT ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Patient ID: {patient_id}\n")
            f.write(f"Model: {self.model}\n")
            f.write("=" * 50 + "\n\n")
            f.write(validation_prompt)
        
        try:
            validation_response = self._call_llm_for_validation(validation_prompt)
            
            response_file = os.path.join(output_dir, f"patient_{patient_id}_validation_response.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write("=== RECOMMENDATION VALIDATION RESPONSE ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Model: {self.model}\n")
                f.write("=" * 50 + "\n\n")
                f.write(validation_response)
            
            parsed_validation = self._parse_validation_response(validation_response)
            parsed_validation['prompt_file'] = prompt_file
            parsed_validation['response_file'] = response_file
            
            return parsed_validation
            
        except Exception as e:
            logger.error(f"❌ Error in LLM-based validation: {e}")
            return {
                'status': 'error',
                'critical_issues': [f"Validation failed due to error: {str(e)}"],
                'warnings': [],
                'detailed_reasoning': '',
                'checks_performed': []
            }
    
    def _create_validation_prompt(
        self,
        recommendation_result: Dict[str, Any],
        assessment_result: Dict[str, Any],
        trial_analysis_result: Dict[str, Any],
        guidelines_result: Dict[str, Any],
        original_trials_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create detailed validation prompt for LLM"""
        
        therapy_rec = recommendation_result.get('therapy_recommendation_text', '')
        rationale = recommendation_result.get('rationale_text', '')
        
        assessment_summary = self._summarize_assessment(assessment_result)
        
        # Use full evidence if available, otherwise fallback to summary
        if original_trials_data:
            trial_summary = self._format_trial_analysis_with_full_evidence(trial_analysis_result, original_trials_data)
        else:
            trial_summary = self._summarize_trials(trial_analysis_result)
        
        guidelines_summary = self._summarize_guidelines(guidelines_result)
        
        prompt_template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'prompts',
            '6_recommendation_validator.txt'
        )
        
        try:
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            logger.debug(f"📄 Loaded validation prompt template from: {prompt_template_path}")
            
            prompt = prompt_template.format(
                assessment_summary=assessment_summary,
                trial_summary=trial_summary,
                guidelines_summary=guidelines_summary,
                therapy_rec=therapy_rec,
                rationale=rationale
            )
            
        except FileNotFoundError:
            error_msg = f"❌ CRITICAL ERROR: Validation prompt template not found at {prompt_template_path}"
            logger.error(error_msg)
            print(error_msg)
            print("Please ensure the validation prompt file exists before running the validator.")
            raise FileNotFoundError(f"Validation prompt template not found: {prompt_template_path}")
        except Exception as e:
            logger.error(f"❌ Error loading prompt template: {e}")
            raise
        
        return prompt
    
    def _summarize_assessment(self, assessment_result: Dict[str, Any]) -> str:
        """Create concise summary of patient assessment"""
        
        lines = []
        
        # Try to use assessment_text if available (contains rich clinical detail)
        assessment_text = assessment_result.get('assessment_text', '')
        
        if assessment_text:
            lines.append("=== DETAILED CLINICAL ASSESSMENT ===")
            lines.append("")
            lines.append(assessment_text)
            lines.append("")
            
        disease_status = assessment_result.get('disease_status', {})
        clinical_summary = assessment_result.get('clinical_summary', {})
        
        if clinical_summary and isinstance(clinical_summary, dict):
            disease_status = clinical_summary.get('disease_status', disease_status)
        
        if disease_status and disease_status != "Not specified":
            if isinstance(disease_status, dict):
                lines.append("Disease Summary:")
                if disease_status.get('primary_diagnosis', 'N/A') != 'N/A':
                    lines.append(f"  - Diagnosis: {disease_status.get('primary_diagnosis')}")
                if disease_status.get('tumor_grade', 'N/A') != 'N/A':
                    lines.append(f"  - Grade: {disease_status.get('tumor_grade')}")
                if disease_status.get('disease_stage', 'N/A') != 'N/A':
                    lines.append(f"  - Stage: {disease_status.get('disease_stage')}")
        
        urgency = assessment_result.get('urgency_level', 'N/A')
        clinical_needs = assessment_result.get('clinical_needs', {})
        if isinstance(clinical_needs, dict):
            urgency = clinical_needs.get('urgency_level', urgency)
        
        if urgency and urgency != 'N/A':
            lines.append(f"Urgency Level: {urgency}")
        
        confidence = assessment_result.get('assessment_confidence', '')
        if confidence:
            lines.append(f"Assessment Confidence: {confidence}")
        
        return "\n".join(lines) if lines else "No assessment summary available"
    
    def _format_trial_analysis_with_full_evidence(self, trial_analysis_result: Dict[str, Any], original_trials_data: Dict[str, Any]) -> str:
        """Format trial analysis with full evidence including publications - imports from therapy_recommender"""
        # Import the therapy recommender method to ensure consistency
        from therapy_recommender import TherapyRecommender
        temp_recommender = TherapyRecommender(api_key=self.api_key, model=self.model)
        return temp_recommender._format_trial_analysis_results_with_evidence(trial_analysis_result, original_trials_data)
        
        return "\n".join(formatted_parts)
    
    def _summarize_trials(self, trial_analysis_result: Dict[str, Any]) -> str:
        """Create concise summary of trial analysis (fallback when no full evidence available)"""
        
        trial_analyses = trial_analysis_result.get('trial_analyses', [])
        
        included = [t for t in trial_analyses if t.get('recommendation') == 'INCLUDE']
        excluded = [t for t in trial_analyses if t.get('recommendation') == 'EXCLUDE']
        
        lines = [
            f"Total Trials Analyzed: {len(trial_analyses)}",
            f"Trials Marked for INCLUSION: {len(included)}",
            f"Trials Marked for EXCLUSION: {len(excluded)}",
            ""
        ]
        
        if included:
            lines.append("INCLUDED TRIALS:")
            for trial in included[:10]:  # Limit to first 10
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                relevance = trial.get('relevance_score', 'N/A')
                lines.append(f"  - {nct}: {name} (Relevance: {relevance})")
        
        return "\n".join(lines)
    
    def _summarize_guidelines(self, guidelines_result: Dict[str, Any]) -> str:
        """Create concise summary of matched guidelines"""
        
        if not guidelines_result or not guidelines_result.get('relevant_guidelines'):
            return "No guidelines matched"
        
        guidelines = guidelines_result.get('relevant_guidelines', [])
        
        lines = [f"Matched Guidelines: {len(guidelines)}", ""]
        
        for g in guidelines[:5]:  # Limit to first 5
            title = g.get('guideline_title', 'Unknown')
            relevance = g.get('relevance_score', 'N/A')
            lines.append(f"  - {title} (Relevance: {relevance})")
        
        return "\n".join(lines)
    
    def _call_llm_for_validation(self, prompt: str) -> str:
        """Call LLM API for validation"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/piakoller/prism",
            "X-Title": "Recommendation Validator"
        }
        
        # No max_tokens - allow unlimited output length for all models
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1
        }
        
        # Add model-specific parameters for Gemini models
        if "gemini" in self.model.lower():
            payload.update({
                "stream": False,
                "top_p": 0.95,
            })
        
        logger.info(f"🌐 Calling {self.model} for validation...")
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        validation_text = result['choices'][0]['message']['content']
        logger.info(f"✅ Received validation response ({len(validation_text)} chars)")
        
        return validation_text
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM validation response"""
        
        decision_match = re.search(r'<validation_decision>\s*(.*?)\s*</validation_decision>', 
                                   response, re.DOTALL | re.IGNORECASE)
        decision = decision_match.group(1).strip().upper() if decision_match else "RECONSIDER"
        
        issues_match = re.search(r'<critical_issues>\s*(.*?)\s*</critical_issues>', 
                                 response, re.DOTALL | re.IGNORECASE)
        issues_text = issues_match.group(1).strip() if issues_match else ""
        critical_issues = self._parse_list_items(issues_text) if issues_text != "None identified." else []
        
        warnings_match = re.search(r'<warnings>\s*(.*?)\s*</warnings>', 
                                   response, re.DOTALL | re.IGNORECASE)
        warnings_text = warnings_match.group(1).strip() if warnings_match else ""
        warnings = self._parse_list_items(warnings_text) if warnings_text != "None identified." else []
        
        reasoning_match = re.search(r'<detailed_reasoning>\s*(.*?)\s*</detailed_reasoning>', 
                                    response, re.DOTALL | re.IGNORECASE)
        detailed_reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        improvements_match = re.search(r'<recommendations_for_improvement>\s*(.*?)\s*</recommendations_for_improvement>', 
                                       response, re.DOTALL | re.IGNORECASE)
        improvements = improvements_match.group(1).strip() if improvements_match else ""
        
        return {
            'status': 'fail' if decision == 'RECONSIDER' else 'pass',
            'decision': decision,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'detailed_reasoning': detailed_reasoning,
            'improvement_recommendations': improvements,
            'raw_response': response,
            'checks_performed': [
                'Completeness validation',
                'Accuracy validation',
                'Consistency validation',
                'Safety validation',
                'Actionability validation'
            ]
        }
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Parse bulleted or numbered list items from text"""
        
        items = []
        
        # Try to find list items with common markers
        for line in text.split('\n'):
            line = line.strip()
            # Match bullets, numbers, dashes
            if re.match(r'^[-•*]\s+', line) or re.match(r'^\d+[\.)]\s+', line):
                # Remove the marker
                item = re.sub(r'^[-•*\d\.)]+\s+', '', line)
                if item:
                    items.append(item)
            elif line and not items:
                # If no list markers found, treat whole text as one item
                items.append(text.strip())
                break
        
        return items if items else [text.strip()] if text.strip() else []
    
    def _determine_regeneration_need(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if recommendation needs regeneration based on validation results"""
        
        regeneration_reasons = []
        
        structural = validation_results['validation_steps'].get('structural', {})
        if structural.get('critical_issues'):
            regeneration_reasons.extend([
                f"Structural: {issue}" for issue in structural['critical_issues']
            ])
        
        evidence = validation_results['validation_steps'].get('evidence_consistency', {})
        if evidence.get('critical_issues'):
            regeneration_reasons.extend([
                f"Evidence: {issue}" for issue in evidence['critical_issues']
            ])
        
        semantic = validation_results['validation_steps'].get('semantic', {})
        if semantic.get('decision') == 'RECONSIDER':
            regeneration_reasons.extend([
                f"Semantic: {issue}" for issue in semantic.get('critical_issues', [])
            ])
        
        # Determine overall status
        if regeneration_reasons:
            validation_results['requires_regeneration'] = True
            validation_results['regeneration_reasons'] = regeneration_reasons
            validation_results['overall_status'] = 'failed'
        else:
            validation_results['requires_regeneration'] = False
            validation_results['overall_status'] = 'passed'
        
        return validation_results
    
    def _create_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """Create human-readable validation summary"""
        
        lines = [
            "=" * 70,
            "THERAPY RECOMMENDATION VALIDATION REPORT",
            "=" * 70,
            f"Patient ID: {validation_results['patient_id']}",
            f"Timestamp: {validation_results['timestamp']}",
            f"Overall Status: {validation_results['overall_status'].upper()}",
            f"Requires Regeneration: {'YES' if validation_results['requires_regeneration'] else 'NO'}",
            ""
        ]
        
        # Structural validation
        lines.append("--- STRUCTURAL VALIDATION ---")
        structural = validation_results['validation_steps'].get('structural', {})
        lines.append(f"Status: {structural.get('status', 'unknown').upper()}")
        if structural.get('critical_issues'):
            lines.append("Critical Issues:")
            for issue in structural['critical_issues']:
                lines.append(f"  ❌ {issue}")
        if structural.get('warnings'):
            lines.append("Warnings:")
            for warning in structural['warnings']:
                lines.append(f"  ⚠️  {warning}")
        if not structural.get('critical_issues') and not structural.get('warnings'):
            lines.append("  ✅ No issues found")
        lines.append("")
        
        # Evidence validation
        lines.append("--- EVIDENCE CONSISTENCY VALIDATION ---")
        evidence = validation_results['validation_steps'].get('evidence_consistency', {})
        lines.append(f"Status: {evidence.get('status', 'unknown').upper()}")
        stats = evidence.get('evidence_stats', {})
        if stats:
            lines.append(f"Trials cited: {stats.get('trials_cited', 0)}")
            lines.append(f"Trials available: {stats.get('trials_available', 0)}")
            lines.append(f"Hallucinated citations: {stats.get('hallucinated_citations', 0)}")
        if evidence.get('critical_issues'):
            lines.append("Critical Issues:")
            for issue in evidence['critical_issues']:
                lines.append(f"  ❌ {issue}")
        if evidence.get('warnings'):
            lines.append("Warnings:")
            for warning in evidence['warnings']:
                lines.append(f"  ⚠️  {warning}")
        if not evidence.get('critical_issues') and not evidence.get('warnings'):
            lines.append("  ✅ No issues found")
        lines.append("")
        
        # Semantic validation
        lines.append("--- SEMANTIC VALIDATION (LLM-BASED) ---")
        semantic = validation_results['validation_steps'].get('semantic', {})
        lines.append(f"Decision: {semantic.get('decision', 'UNKNOWN')}")
        lines.append(f"Status: {semantic.get('status', 'unknown').upper()}")
        if semantic.get('critical_issues'):
            lines.append("Critical Issues:")
            for issue in semantic['critical_issues']:
                lines.append(f"  ❌ {issue}")
        if semantic.get('warnings'):
            lines.append("Warnings:")
            for warning in semantic['warnings']:
                lines.append(f"  ⚠️  {warning}")
        if semantic.get('detailed_reasoning'):
            lines.append("")
            lines.append("Detailed Reasoning:")
            lines.append(semantic['detailed_reasoning'])
        if semantic.get('improvement_recommendations') and semantic.get('improvement_recommendations') != "No improvements needed.":
            lines.append("")
            lines.append("Recommendations for Improvement:")
            lines.append(semantic['improvement_recommendations'])
        if not semantic.get('critical_issues') and not semantic.get('warnings'):
            lines.append("  ✅ No issues found")
        lines.append("")
        
        # Regeneration decision
        lines.append("=" * 70)
        if validation_results['requires_regeneration']:
            lines.append("⚠️  REGENERATION REQUIRED")
            lines.append("=" * 70)
            lines.append("Reasons:")
            for reason in validation_results['regeneration_reasons']:
                lines.append(f"  • {reason}")
        else:
            lines.append("✅ VALIDATION PASSED - NO REGENERATION NEEDED")
            lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _save_validation_results(
        self,
        validation_results: Dict[str, Any],
        patient_id: str,
        output_dir: str
    ):
        """Save validation results to files with iteration support"""
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            iteration = 0
            base_json_file = os.path.join(output_dir, f"patient_{patient_id}_validation_results.json")
            
            if os.path.exists(base_json_file):
                iteration = 1
                while os.path.exists(os.path.join(output_dir, f"patient_{patient_id}_validation_results_iter{iteration}.json")):
                    iteration += 1
            
            if iteration == 0:
                json_file = base_json_file
                summary_file = os.path.join(output_dir, f"patient_{patient_id}_validation_summary.txt")
            else:
                json_file = os.path.join(output_dir, f"patient_{patient_id}_validation_results_iter{iteration}.json")
                summary_file = os.path.join(output_dir, f"patient_{patient_id}_validation_summary_iter{iteration}.txt")
            
            validation_results['iteration'] = iteration
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Validation results (iteration {iteration}) saved to: {json_file}")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"=== VALIDATION ITERATION {iteration} ===\n\n")
                f.write(validation_results['validation_summary'])
            
            logger.info(f"📄 Validation summary (iteration {iteration}) saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving validation results: {e}")
