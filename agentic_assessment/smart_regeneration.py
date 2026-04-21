#!/usr/bin/env python3
"""
Smart Regeneration Module

This module intelligently regenerates parts of the workflow based on validation feedback.
It determines what needs to be regenerated and reuses cached results where possible.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SmartRegenerator:
    """Handles intelligent regeneration of workflow components based on validation feedback"""
    
    def __init__(self, workflow):
        """
        Initialize the Smart Regenerator
        
        Args:
            workflow: AgenticWorkflow instance with all components initialized
        """
        self.workflow = workflow
        logger.info("🔄 Smart Regenerator initialized")
    
    def analyze_validation_and_regenerate(
        self,
        validation_result: Dict[str, Any],
        patient_data: dict,
        trial_data_path: str,
        guidelines_context: Optional[str] = None,
        guidelines_dir: Optional[str] = None,
        output_dir: str = ".",
        workflow_results: Optional[Dict[str, Any]] = None,
        validation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze validation results and regenerate appropriate components
        
        Args:
            validation_result: Validation results from validator
            patient_data: Original patient data
            trial_data_path: Path to trial data
            guidelines_context: Guidelines context string
            guidelines_dir: Guidelines directory
            output_dir: Output directory
            workflow_results: Complete workflow results with cached data
            validation_history: List of all validation results from previous iterations
            
        Returns:
            Updated workflow results with regenerated components
        """
        patient_id = validation_result.get('patient_id', 'unknown')
        
        if not validation_result.get('requires_regeneration', False):
            logger.info(f"✅ Patient {patient_id}: No regeneration needed")
            return workflow_results
        
        logger.info(f"🔄 Patient {patient_id}: Regeneration required")
        logger.info(f"📋 Analyzing {len(validation_result.get('regeneration_reasons', []))} issues...")
        
        # Analyze what needs to be regenerated with cumulative history
        regeneration_plan = self._create_regeneration_plan(validation_result, validation_history)
        
        logger.info(f"📋 Regeneration plan: {regeneration_plan['description']}")
        logger.info(f"🔄 Components to regenerate: {', '.join(regeneration_plan['components'])}")
        
        # Execute regeneration
        if workflow_results is None:
            logger.error("❌ No workflow results provided - cannot regenerate")
            return None
        
        regenerated_results = self._execute_regeneration_plan(
            regeneration_plan,
            workflow_results,
            patient_data,
            output_dir
        )
        
        return regenerated_results
    
    def _create_regeneration_plan(self, validation_result: Dict[str, Any], validation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze validation issues and create a regeneration plan
        
        Args:
            validation_result: Current validation result
            validation_history: List of all previous validation results for cumulative tracking
        
        Returns:
            Dict with 'level', 'components', 'description', and 'issues'
        """
        # Collect all issues from current and previous iterations (cumulative)
        issues = validation_result.get('regeneration_reasons', [])
        all_improvement_recommendations = []
        
        # Aggregate unresolved issues from validation history
        if validation_history:
            logger.info(f"📊 Aggregating issues from {len(validation_history)} previous validation(s)...")
            for idx, prev_validation in enumerate(validation_history):
                prev_issues = prev_validation.get('regeneration_reasons', [])
                # Add historical issues that are still relevant
                for issue in prev_issues:
                    if issue not in issues:  # Avoid duplicates
                        issues.append(f"[Persisting from iter {idx}] {issue}")
                
                # Collect improvement recommendations from all iterations
                prev_semantic = prev_validation.get('validation_steps', {}).get('semantic', {})
                prev_recommendations = prev_semantic.get('improvement_recommendations', '')
                if prev_recommendations and prev_recommendations != "No improvements needed.":
                    all_improvement_recommendations.append(f"--- From Iteration {idx} ---\n{prev_recommendations}")
        
        logger.info(f"📋 Total cumulative issues to address: {len(issues)}")
        
        validation_steps = validation_result.get('validation_steps', {})
        semantic_validation = validation_steps.get('semantic', {})
        
        # Add current iteration's improvement recommendations
        current_recommendations = semantic_validation.get('improvement_recommendations', '')
        if current_recommendations and current_recommendations != "No improvements needed.":
            all_improvement_recommendations.append(f"--- Current Iteration ---\n{current_recommendations}")
        
        # Combine all improvement recommendations
        combined_recommendations = "\n\n".join(all_improvement_recommendations) if all_improvement_recommendations else ''
        
        # Categorize issues
        safety_issues = []
        accuracy_issues = []
        completeness_issues = []
        consistency_issues = []
        
        for issue in issues:
            issue_lower = issue.lower()
            if 'safety' in issue_lower or 'contraindication' in issue_lower or 'risk' in issue_lower:
                safety_issues.append(issue)
            elif 'hallucin' in issue_lower or 'doesn\'t exist' in issue_lower or 'incorrect' in issue_lower:
                accuracy_issues.append(issue)
            elif 'missing' in issue_lower or 'incomplete' in issue_lower or 'doesn\'t address' in issue_lower:
                completeness_issues.append(issue)
            elif 'contradict' in issue_lower or 'inconsistent' in issue_lower or 'mismatch' in issue_lower:
                consistency_issues.append(issue)
        
        # Determine regeneration level
        
        # Level 1: Only recommendation needs regeneration (most common)
        if accuracy_issues or consistency_issues or (completeness_issues and not safety_issues):
            return {
                'level': 'recommendation_only',
                'components': ['therapy_recommendation'],
                'description': 'Regenerate therapy recommendation with cumulative validation feedback',
                'issues': {
                    'accuracy': accuracy_issues,
                    'consistency': consistency_issues,
                    'completeness': completeness_issues
                },
                'use_cache': True,
                'validation_feedback': combined_recommendations
            }
        
        # Level 2: Recommendation + Trial Analysis (if safety issues suggest wrong trial selection)
        if safety_issues:
            # Check if safety issues are about trial selection vs. implementation details
            trial_selection_issue = any('trial' in issue.lower() and 
                                       ('wrong' in issue.lower() or 'inappropriate' in issue.lower())
                                       for issue in safety_issues)
            
            if trial_selection_issue:
                return {
                    'level': 'trial_analysis_and_recommendation',
                    'components': ['trial_analysis', 'therapy_recommendation'],
                    'description': 'Regenerate trial analysis and recommendation due to trial selection issues',
                    'issues': {
                        'safety': safety_issues,
                        'accuracy': accuracy_issues
                    },
                    'use_cache': True,  # Still use cache for assessment, matching, guidelines
                    'validation_feedback': combined_recommendations
                }
            else:
                # Safety issues with implementation, not trial selection
                return {
                    'level': 'recommendation_only',
                    'components': ['therapy_recommendation'],
                    'description': 'Regenerate recommendation addressing cumulative safety concerns',
                    'issues': {
                        'safety': safety_issues
                    },
                    'use_cache': True,
                    'validation_feedback': combined_recommendations
                }
        
        # Default: recommendation only
        return {
            'level': 'recommendation_only',
            'components': ['therapy_recommendation'],
            'description': 'Regenerate therapy recommendation with cumulative validation feedback',
            'issues': {'all_issues': issues},
            'use_cache': True,
            'validation_feedback': combined_recommendations
        }
    
    def _execute_regeneration_plan(
        self,
        plan: Dict[str, Any],
        workflow_results: Dict[str, Any],
        patient_data: dict,
        output_dir: str
    ) -> Dict[str, Any]:
        """Execute the regeneration plan"""
        
        patient_id = workflow_results.get('patient_id', 'unknown')
        
        logger.info(f"🔄 Executing regeneration plan: {plan['description']}")
        
        # Get previous recommendation for context
        previous_recommendation = workflow_results.get('recommendation_result', {})
        
        # Create regeneration context from validation feedback and previous attempt
        regeneration_context = self._create_enhanced_regeneration_context(
            plan['validation_feedback'],
            plan['issues'],
            previous_recommendation
        )
        
        # Save regeneration context for audit
        regen_context_file = os.path.join(output_dir, f"patient_{patient_id}_regeneration_context.txt")
        with open(regen_context_file, 'w', encoding='utf-8') as f:
            f.write(regeneration_context)
        
        workflow_results['regeneration_context_file'] = regen_context_file
        workflow_results['files_generated'].append(regen_context_file)
        
        # Extract cached results to reuse
        assessment_result = workflow_results.get('assessment_result')
        trial_matching_result = workflow_results.get('trial_matching_result', {})
        trial_analysis_result = workflow_results.get('trial_analysis_result', {})
        guidelines_result = workflow_results.get('guidelines_result', {})
        
        # Validate required cached results exist
        if assessment_result is None:
            logger.error(f"❌ Patient {patient_id}: Missing assessment_result - cannot regenerate")
            return workflow_results
        
        # Regenerate based on plan level
        if plan['level'] == 'recommendation_only':
            logger.info("🔄 Regenerating therapy recommendation only (reusing all cached results)...")
            
            regenerated_recommendation = self._regenerate_recommendation_with_feedback(
                assessment_result=assessment_result,
                trial_analysis_result=trial_analysis_result,
                trial_matching_result=trial_matching_result,
                guidelines_result=guidelines_result or {},
                patient_data=patient_data,
                output_dir=output_dir,
                regeneration_context=regeneration_context
            )
            
            # Store original and regenerated versions
            workflow_results['recommendation_result_original'] = workflow_results.get('recommendation_result')
            workflow_results['recommendation_result'] = regenerated_recommendation
            workflow_results['regeneration_iteration'] = 1
            
        elif plan['level'] == 'trial_analysis_and_recommendation':
            logger.info("🔄 Regenerating trial analysis AND recommendation...")
            
            # First regenerate trial analysis with feedback
            regenerated_trial_analysis = self._regenerate_trial_analysis_with_feedback(
                assessment_result=assessment_result,
                trial_matching_result=trial_matching_result,
                patient_id=patient_id,
                output_dir=output_dir,
                regeneration_context=regeneration_context
            )
            
            # Then regenerate recommendation with new trial analysis
            regenerated_recommendation = self._regenerate_recommendation_with_feedback(
                assessment_result=assessment_result,
                trial_analysis_result=regenerated_trial_analysis,
                trial_matching_result=trial_matching_result,
                guidelines_result=guidelines_result or {},
                patient_data=patient_data,
                output_dir=output_dir,
                regeneration_context=regeneration_context
            )
            
            # Store originals and regenerated versions
            workflow_results['trial_analysis_result_original'] = workflow_results.get('trial_analysis_result')
            workflow_results['trial_analysis_result'] = regenerated_trial_analysis
            workflow_results['recommendation_result_original'] = workflow_results.get('recommendation_result')
            workflow_results['recommendation_result'] = regenerated_recommendation
            workflow_results['regeneration_iteration'] = 1
        
        # Mark regeneration complete
        workflow_results['regeneration_performed'] = True
        workflow_results['regeneration_plan'] = plan
        workflow_results['regeneration_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"✅ Regeneration complete for patient {patient_id}")
        
        return workflow_results
    
    def _create_enhanced_regeneration_context(
        self,
        improvement_recommendations: str,
        issues: Dict[str, List[str]],
        previous_recommendation: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create detailed regeneration context for the LLM with strong evidence grounding constraints"""
        
        lines = [
            "=" * 80,
            "CRITICAL: REGENERATION REQUIRED - CUMULATIVE VALIDATION ISSUES",
            "=" * 80,
            "",
            "⚠️  ATTENTION: This is a REGENERATION with cumulative feedback from ALL previous attempts.",
            "The issues below include BOTH new issues AND unresolved issues from prior iterations.",
            "You MUST address EVERY issue listed - not just the newest ones.",
            ""
        ]
        
        # Add previous recommendation if available
        if previous_recommendation:
            previous_text = previous_recommendation.get('therapy_recommendation_text', '')
            if previous_text:
                lines.extend([
                    "=" * 80,
                    "PREVIOUS RECOMMENDATION THAT FAILED VALIDATION",
                    "=" * 80,
                    "",
                    previous_text,
                    "",
                    "⚠️  The above recommendation had validation issues. Review it carefully and",
                    "correct the specific problems identified below while preserving correct sections.",
                    ""
                ])
        
        lines.extend([
            "=" * 80,
            "ALL VALIDATION ISSUES (CUMULATIVE)",
            "=" * 80,
            ""
        ])
        
        # Count total issues for emphasis
        total_issues = sum(len(category_issues) for category_issues in issues.values())
        lines.append(f"📊 Total issues to address: {total_issues}")
        lines.append("")
        
        # Add categorized issues
        for category, category_issues in issues.items():
            if category_issues:
                lines.append(f"### {category.upper()} ISSUES:")
                for issue in category_issues:
                    lines.append(f"  ❌ {issue}")
                lines.append("")
        
        # Add improvement recommendations
        if improvement_recommendations and improvement_recommendations != "No improvements needed.":
            lines.extend([
                "=" * 80,
                "SPECIFIC GUIDANCE FOR REGENERATION (FROM ALL ITERATIONS)",
                "=" * 80,
                "",
                improvement_recommendations,
                ""
            ])
        
        lines.extend([
            "=" * 80,
            "REQUIREMENTS FOR THE REGENERATED RECOMMENDATION",
            "=" * 80,
            "",
            "1. Address EVERY issue listed above - both new and persisting issues",
            "2. Base all recommendations on the evidence provided (trials, guidelines, assessment)",
            "3. Only cite trials and data that appear in the evidence sections above",
            "4. For standard therapies without trial evidence, cite the relevant guidelines",
            "5. Ensure clinical safety and appropriateness for this specific patient",
            "6. Maintain consistency with the patient's clinical status and history",
            "",
            "=" * 80,
            "NOW GENERATE THE CORRECTED RECOMMENDATION",
            "=" * 80,
            ""
        ])
        
        return "\n".join(lines)
    
    def _regenerate_recommendation_with_feedback(
        self,
        assessment_result: Dict[str, Any],
        trial_analysis_result: Dict[str, Any],
        trial_matching_result: Dict[str, Any],
        guidelines_result: Dict[str, Any],
        patient_data: dict,
        output_dir: str,
        regeneration_context: str
    ) -> Dict[str, Any]:
        """Regenerate therapy recommendation with validation feedback"""
        
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        
        logger.info(f"💊 Regenerating therapy recommendation for patient {patient_id}...")
        
        # Prepare enhanced guidelines context with feedback
        enhanced_guidelines_context = self._prepare_enhanced_guidelines_context(
            guidelines_result,
            patient_id,
            output_dir
        )
        
        # Append regeneration context AFTER guidelines (better evidence flow)
        full_context = f"{enhanced_guidelines_context}\n\n{regeneration_context}"
        
        # Generate new recommendation with feedback
        regenerated_result = self.workflow.therapy_recommender.generate_recommendation(
            assessment=assessment_result,
            trial_analysis_result=trial_analysis_result,
            guidelines_context=full_context,
            original_trials_data={"matched_trials": trial_matching_result.get("relevant_trials", [])},
            patient_data=patient_data,
            output_dir=output_dir
        )
        
        # Determine iteration number by checking for existing regenerated files
        iteration = 1
        while os.path.exists(os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_regenerated_iter{iteration}.json")):
            iteration += 1
        
        # Save JSON with iteration-specific suffix
        regen_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_regenerated_iter{iteration}.json")
        with open(regen_file, 'w', encoding='utf-8') as f:
            # Add iteration metadata
            regenerated_result['regeneration_iteration'] = iteration
            json.dump(regenerated_result, f, indent=2, ensure_ascii=False)
        
        # Save prompt as TXT
        prompt_file = regenerated_result.get('metadata', {}).get('prompt_file')
        if prompt_file and os.path.exists(prompt_file):
            prompt_iter_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_prompt_iter{iteration}.txt")
            with open(prompt_file, 'r', encoding='utf-8') as src:
                prompt_content = src.read()
            with open(prompt_iter_file, 'w', encoding='utf-8') as dst:
                dst.write(f"=== REGENERATION ITERATION {iteration} - THERAPY RECOMMENDATION PROMPT ===\n")
                dst.write(f"Timestamp: {datetime.now().isoformat()}\n")
                dst.write("=" * 80 + "\n\n")
                dst.write(prompt_content)
        
        # Save raw response as TXT
        response_file = regenerated_result.get('metadata', {}).get('response_file')
        if response_file and os.path.exists(response_file):
            response_iter_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_response_iter{iteration}.txt")
            with open(response_file, 'r', encoding='utf-8') as src:
                response_content = src.read()
            with open(response_iter_file, 'w', encoding='utf-8') as dst:
                dst.write(f"=== REGENERATION ITERATION {iteration} - THERAPY RECOMMENDATION RESPONSE ===\n")
                dst.write(f"Timestamp: {datetime.now().isoformat()}\n")
                dst.write("=" * 80 + "\n\n")
                dst.write(response_content)
        
        # Save recommendation text as TXT
        recommendation_text = regenerated_result.get('therapy_recommendation_text', '')
        if recommendation_text:
            recommendation_iter_file = os.path.join(output_dir, f"patient_{patient_id}_therapy_recommendation_text_iter{iteration}.txt")
            with open(recommendation_iter_file, 'w', encoding='utf-8') as f:
                f.write(f"=== REGENERATION ITERATION {iteration} - THERAPY RECOMMENDATION ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write("=" * 80 + "\n\n")
                f.write(recommendation_text)
        
        logger.info(f"✅ Regenerated recommendation (iteration {iteration}) saved to: {regen_file}")
        
        return regenerated_result
    
    def _regenerate_trial_analysis_with_feedback(
        self,
        assessment_result: Dict[str, Any],
        trial_matching_result: Dict[str, Any],
        patient_id: str,
        output_dir: str,
        regeneration_context: str
    ) -> Dict[str, Any]:
        """Regenerate trial analysis with validation feedback"""
        
        logger.info(f"🔬 Regenerating trial analysis for patient {patient_id}...")
        
        # Extract relevant trials
        relevant_trials_data = {
            "trials": trial_matching_result.get("relevant_trials", []),
            "summary": trial_matching_result.get("summary", {}),
            "metadata": trial_matching_result.get("metadata", {})
        }
        
        # For now, just regenerate without cache
        # TODO: Could enhance trial_analyzer to accept feedback context
        regenerated_analysis = self.workflow.trial_analyzer.analyze_trials(
            patient_assessment=assessment_result,
            trials_data=relevant_trials_data,
            patient_id=patient_id,
            output_dir=output_dir,
            use_cache=False  # Force regeneration
        )
        
        # Determine iteration number by checking for existing regenerated files
        iteration = 1
        while os.path.exists(os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_regenerated_iter{iteration}.json")):
            iteration += 1
        
        # Save JSON with iteration-specific suffix
        regen_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_regenerated_iter{iteration}.json")
        with open(regen_file, 'w', encoding='utf-8') as f:
            # Add iteration metadata
            regenerated_analysis['regeneration_iteration'] = iteration
            json.dump(regenerated_analysis, f, indent=2, ensure_ascii=False)
        
        # Save prompt as TXT
        prompt_file = regenerated_analysis.get('metadata', {}).get('prompt_file')
        if prompt_file and os.path.exists(prompt_file):
            prompt_iter_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_prompt_iter{iteration}.txt")
            with open(prompt_file, 'r', encoding='utf-8') as src:
                prompt_content = src.read()
            with open(prompt_iter_file, 'w', encoding='utf-8') as dst:
                dst.write(f"=== REGENERATION ITERATION {iteration} - TRIAL ANALYSIS PROMPT ===\n")
                dst.write(f"Timestamp: {datetime.now().isoformat()}\n")
                dst.write("=" * 80 + "\n\n")
                dst.write(prompt_content)
        
        # Save raw response as TXT
        response_file = regenerated_analysis.get('metadata', {}).get('response_file')
        if response_file and os.path.exists(response_file):
            response_iter_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_response_iter{iteration}.txt")
            with open(response_file, 'r', encoding='utf-8') as src:
                response_content = src.read()
            with open(response_iter_file, 'w', encoding='utf-8') as dst:
                dst.write(f"=== REGENERATION ITERATION {iteration} - TRIAL ANALYSIS RESPONSE ===\n")
                dst.write(f"Timestamp: {datetime.now().isoformat()}\n")
                dst.write("=" * 80 + "\n\n")
                dst.write(response_content)
        
        logger.info(f"✅ Regenerated trial analysis (iteration {iteration}) saved to: {regen_file}")
        
        return regenerated_analysis
    
    def _prepare_enhanced_guidelines_context(
        self,
        guidelines_result: Dict[str, Any],
        patient_id: str,
        output_dir: str
    ) -> str:
        """Prepare enhanced guidelines context"""
        
        if not guidelines_result or not guidelines_result.get('relevant_guidelines'):
            return ""
        
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
        header = "FULL MATCHED GUIDELINES\n" + ("=" * 60) + "\n"
        
        return f"{header}{full_guidelines_text}"
