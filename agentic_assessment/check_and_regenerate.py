#!/usr/bin/env python3
"""
Check Validation Status and Regenerate Recommendations

This script checks if a patient's recommendation needs regeneration based on validation results,
and performs smart regeneration reusing cached workflow results.

Usage:
    python check_and_regenerate.py --results-dir "batch_results/run_20251203_164656/patient_1"
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENROUTER_API_KEY
from agentic_assessment.agentic_workflow import AgenticWorkflow
from agentic_assessment.smart_regeneration import SmartRegenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_workflow_results(results_dir: str) -> dict:
    """Load all workflow results from directory"""
    
    # Find patient ID from directory name or files
    patient_id = None
    dir_name = os.path.basename(results_dir)
    if dir_name.startswith('patient_'):
        patient_id = dir_name.replace('patient_', '')
    
    if not patient_id:
        # Try to find from files
        for file in os.listdir(results_dir):
            if file.endswith('_complete_workflow.json'):
                patient_id = file.replace('patient_', '').replace('_complete_workflow.json', '')
                break
    
    if not patient_id:
        raise ValueError(f"Could not determine patient ID from directory: {results_dir}")
    
    logger.info(f"📂 Loading results for patient {patient_id}")
    
    # Load all result files
    workflow_results = {
        'patient_id': patient_id,
        'output_directory': results_dir
    }
    
    # Load complete workflow results if exists
    complete_workflow_file = os.path.join(results_dir, f"patient_{patient_id}_complete_workflow.json")
    if os.path.exists(complete_workflow_file):
        with open(complete_workflow_file, 'r', encoding='utf-8') as f:
            workflow_results.update(json.load(f))
        logger.info(f"✅ Loaded complete workflow results")
    
    # Load individual result files
    result_files = {
        'assessment_result': f"patient_{patient_id}_assessment.json",
        'trial_matching_result': f"patient_{patient_id}_trial_matching.json",
        'trial_analysis_result': f"patient_{patient_id}_trial_analysis.json",
        'recommendation_result': f"patient_{patient_id}_therapy_recommendation.json",
        'guidelines_result': f"patient_{patient_id}_guidelines_match.json",
        'validation_result': f"patient_{patient_id}_validation_results.json"
    }
    
    for key, filename in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                workflow_results[key] = json.load(f)
            logger.info(f"✅ Loaded {key}")
        else:
            logger.warning(f"⚠️ File not found: {filename}")
    
    return workflow_results


def check_validation_status(results_dir: str) -> dict:
    """Check if validation results indicate regeneration is needed"""
    
    workflow_results = load_workflow_results(results_dir)
    patient_id = workflow_results['patient_id']
    
    # Check for validation results
    validation_result = workflow_results.get('validation_result')
    
    if not validation_result:
        logger.error(f"❌ No validation results found for patient {patient_id}")
        return {
            'patient_id': patient_id,
            'validation_found': False,
            'requires_regeneration': False,
            'status': 'no_validation_results'
        }
    
    requires_regeneration = validation_result.get('requires_regeneration', False)
    regeneration_reasons = validation_result.get('regeneration_reasons', [])
    overall_status = validation_result.get('overall_status', 'UNKNOWN')
    
    logger.info(f"📊 Validation Status for Patient {patient_id}:")
    logger.info(f"   Overall Status: {overall_status}")
    logger.info(f"   Requires Regeneration: {'YES' if requires_regeneration else 'NO'}")
    
    if requires_regeneration:
        logger.warning(f"   ❌ {len(regeneration_reasons)} issues identified:")
        for i, reason in enumerate(regeneration_reasons, 1):
            logger.warning(f"      {i}. {reason[:100]}...")
    else:
        logger.info(f"   ✅ Validation passed - no issues found")
    
    return {
        'patient_id': patient_id,
        'validation_found': True,
        'requires_regeneration': requires_regeneration,
        'status': overall_status,
        'issues_count': len(regeneration_reasons),
        'workflow_results': workflow_results,
        'validation_result': validation_result
    }


def regenerate_recommendation(
    workflow_results: dict,
    validation_result: dict,
    trials_path: str,
    guidelines_dir: str = None
) -> dict:
    """Regenerate recommendation using smart regeneration"""
    
    patient_id = workflow_results['patient_id']
    results_dir = workflow_results['output_directory']
    
    logger.info(f"🔄 Starting smart regeneration for patient {patient_id}...")
    
    # Initialize workflow
    workflow = AgenticWorkflow()
    
    # Extract patient data from assessment
    patient_data = workflow_results.get('assessment_result', {}).get('patient_data', {})
    if not patient_data:
        # Try to reconstruct from available data
        patient_data = {
            'ID': patient_id
        }
        logger.warning("⚠️ Limited patient data available - using minimal context")
    
    # Perform smart regeneration
    regenerated_results = workflow.regenerator.analyze_validation_and_regenerate(
        validation_result=validation_result,
        patient_data=patient_data,
        trial_data_path=trials_path,
        guidelines_context=None,  # Will be loaded from cached results
        guidelines_dir=guidelines_dir,
        output_dir=results_dir,
        workflow_results=workflow_results
    )
    
    if regenerated_results:
        logger.info("✅ Smart regeneration completed successfully")
        
        # Save updated workflow results
        updated_workflow_file = os.path.join(results_dir, f"patient_{patient_id}_complete_workflow_regenerated.json")
        with open(updated_workflow_file, 'w', encoding='utf-8') as f:
            json.dump(regenerated_results, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Updated workflow results saved to: {updated_workflow_file}")
        
        # Re-validate
        logger.info("🔍 Re-validating regenerated recommendation...")
        revalidation_result = workflow.validator.validate_recommendation(
            recommendation_result=regenerated_results['recommendation_result'],
            assessment_result=regenerated_results['assessment_result'],
            trial_analysis_result=regenerated_results['trial_analysis_result'],
            guidelines_result=regenerated_results.get('guidelines_result'),
            patient_id=patient_id,
            output_dir=results_dir
        )
        
        # Save revalidation
        revalidation_file = os.path.join(results_dir, f"patient_{patient_id}_validation_results_regenerated.json")
        with open(revalidation_file, 'w', encoding='utf-8') as f:
            json.dump(revalidation_result, f, indent=2, ensure_ascii=False)
        
        revalidation_summary_file = os.path.join(results_dir, f"patient_{patient_id}_validation_summary_regenerated.txt")
        with open(revalidation_summary_file, 'w', encoding='utf-8') as f:
            f.write(workflow.validator._create_validation_summary(revalidation_result))
        
        logger.info(f"📄 Re-validation results saved to: {revalidation_file}")
        
        if revalidation_result.get('requires_regeneration', False):
            logger.warning("⚠️ Regenerated recommendation still has issues")
            logger.warning(f"❌ {len(revalidation_result.get('regeneration_reasons', []))} remaining issues")
            return {
                'success': True,
                'status': 'improved_but_issues_remain',
                'revalidation_result': revalidation_result,
                'regenerated_results': regenerated_results
            }
        else:
            logger.info("✅ Regenerated recommendation passed validation!")
            return {
                'success': True,
                'status': 'validation_passed',
                'revalidation_result': revalidation_result,
                'regenerated_results': regenerated_results
            }
    else:
        logger.error("❌ Smart regeneration failed")
        return {
            'success': False,
            'status': 'regeneration_failed'
        }


def main():
    parser = argparse.ArgumentParser(
        description='Check validation status and regenerate recommendations if needed'
    )
    parser.add_argument(
        '--results-dir',
        required=True,
        help='Directory containing patient results (e.g., batch_results/run_20251203_164656/patient_1)'
    )
    parser.add_argument(
        '--trials',
        help='Path to trial data JSON file (required for regeneration)',
        default=None
    )
    parser.add_argument(
        '--guidelines',
        help='Path to guidelines directory (optional)',
        default=None
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check validation status without regenerating'
    )
    
    args = parser.parse_args()
    
    # Validate results directory exists
    if not os.path.exists(args.results_dir):
        logger.error(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    logger.info(f"🔍 Checking validation status for: {args.results_dir}")
    logger.info("=" * 80)
    
    # Check validation status
    status = check_validation_status(args.results_dir)
    
    logger.info("=" * 80)
    
    if not status['validation_found']:
        logger.error("❌ Cannot proceed - no validation results found")
        logger.error("💡 Run validation first using: python test_validation.py <results-dir>")
        sys.exit(1)
    
    if not status['requires_regeneration']:
        logger.info("✅ No regeneration needed - validation passed!")
        sys.exit(0)
    
    # Check if regeneration is requested
    if args.check_only:
        logger.info("ℹ️  Check-only mode - skipping regeneration")
        logger.warning(f"⚠️ Recommendation requires regeneration ({status['issues_count']} issues)")
        logger.info("💡 To regenerate, run without --check-only and provide --trials path")
        sys.exit(0)
    
    # Validate trials path is provided
    if not args.trials:
        logger.error("❌ Cannot regenerate without --trials path")
        logger.error("💡 Provide trial data path: --trials <path-to-trials.json>")
        sys.exit(1)
    
    if not os.path.exists(args.trials):
        logger.error(f"❌ Trials file not found: {args.trials}")
        sys.exit(1)
    
    # Perform regeneration
    logger.info("🔄 Starting regeneration process...")
    logger.info("=" * 80)
    
    result = regenerate_recommendation(
        workflow_results=status['workflow_results'],
        validation_result=status['validation_result'],
        trials_path=args.trials,
        guidelines_dir=args.guidelines
    )
    
    logger.info("=" * 80)
    
    if result['success']:
        if result['status'] == 'validation_passed':
            logger.info("🎉 SUCCESS: Regenerated recommendation passed validation!")
        else:
            logger.warning("⚠️ PARTIAL SUCCESS: Recommendation improved but issues remain")
            logger.warning("💡 Review revalidation results and consider manual review")
    else:
        logger.error("❌ FAILED: Regeneration unsuccessful")
        sys.exit(1)


if __name__ == "__main__":
    main()
