#!/usr/bin/env python3
"""
Trial Matcher Module

This module handles matching relevant clinical trials based on patient assessment.
It uses the assessment results to identify and rank the most relevant trials.
"""

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL, TRIALS_CACHE_DIR

logger = logging.getLogger(__name__)

class TrialMatcher:
    """Handles trial matching based on patient assessment"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, prompt_file: Optional[str] = None):
        """
        Initialize the Trial Matcher
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: LLM model to use for trial matching (defaults to config.OPENROUTER_MODEL)
            prompt_file: Path to trial matching prompt file
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        # Set default prompt file path
        if prompt_file is None:
            # Use the main prompts folder at the project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.prompt_file = os.path.join(project_root, "prompts", "3_trial_matching.txt")
        else:
            self.prompt_file = prompt_file
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please check your configuration.")
        
        # Load the prompt template
        self.prompt_template = self._load_prompt_template()
        
        logger.info(f"🎯 Trial Matcher initialized with model: {self.model}")
        logger.info(f"📋 Using prompt file: {self.prompt_file}")
        
        # Initialize cache directory
        self.cache_dir = Path(TRIALS_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, patient_data: Dict[str, Any], assessment_data: Dict[str, Any], trials_data: Dict[str, Any]) -> str:
        """Generate a cache key for trial matching."""
        import hashlib
        
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        
        # TEMPORARY: Force use of specific cached trials for patient 2 testing
        if str(patient_id) == '2':
            return "patient_2_trials_53c3a956"
        
        # # TEMPORARY: Force use of existing cache for patient 1 to avoid regeneration costs
        # if str(patient_id) == '1':
        #     return "patient_1_trials_83c3e4a9"
        
        # Create a hash of key inputs for cache consistency
        assessment_text = assessment_data.get('assessment_text', '')
        trials_count = len(trials_data.get('studies', []))
        
        cache_input = f"{patient_id}_{assessment_text[:100]}_{trials_count}"
        cache_hash = hashlib.md5(cache_input.encode()).hexdigest()[:8]
        return f"patient_{patient_id}_trials_{cache_hash}"
    
    def _get_cache_filepath(self, cache_key: str) -> Path:
        """Get the full cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}_trials.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load trial matching results from cache if available."""
        cache_file = self._get_cache_filepath(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Special logging for patient 2 test cache
                if cache_key == "patient_2_trials_53c3a956":
                    logger.info(f"🧪 TESTING: Using specific cached trials for patient 2 from: {cache_file}")
                    logger.info(f"📊 Cache contains {len(cached_data.get('relevant_trials', []))} relevant trials")
                
                # Verify cache is for the same model (be lenient with older caches missing metadata)
                cached_model = (
                    cached_data.get('metadata', {}).get('model_used')
                    or cached_data.get('model_used')
                )

                if cached_model is None:
                    # Backward-compat: migrate old cache by injecting metadata with current model
                    logger.info("🧩 Cache missing model metadata; accepting as compatible and migrating metadata")
                    cached_data.setdefault('metadata', {})
                    cached_data['metadata'].setdefault('timestamp', datetime.now().isoformat())
                    cached_data['metadata']['model_used'] = self.model
                    # Write back migrated cache safely
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as wf:
                            json.dump(cached_data, wf, indent=2, ensure_ascii=False)
                        logger.info(f"🛠️ Migrated cache metadata written to: {cache_file}")
                    except Exception as se:
                        logger.warning(f"⚠️ Failed to migrate cache metadata: {se}")
                    logger.info(f"📂 Loaded trial matching from cache: {cache_file}")
                    return cached_data
                
                if cached_model == self.model:
                    logger.info(f"📂 Loaded trial matching from cache: {cache_file}")
                    return cached_data
                else:
                    logger.info(f"🔄 Cache model mismatch ({cached_model} vs {self.model}), will regenerate")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save trial matching results to cache."""
        try:
            cache_file = self._get_cache_filepath(cache_key)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved trial matching to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to cache: {e}")
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            error_msg = f"❌ Trial matching prompt file not found: {self.prompt_file}"
            print(error_msg)
            logger.error(error_msg)
            raise FileNotFoundError(f"Trial matching prompt file not available at: {self.prompt_file}")
        except Exception as e:
            error_msg = f"❌ Error reading trial matching prompt file: {e}"
            print(error_msg)
            logger.error(error_msg)
            raise
    
    def _format_patient_clinical_info(self, assessment: Dict[str, Any]) -> str:
        """Format patient assessment into clinical information for the prompt"""
        
        clinical_info = []
        
        # Main assessment text (most important clinical information)
        assessment_text = assessment.get('assessment_text', '')
        if assessment_text:
            clinical_info.append("Clinical Assessment:")
            clinical_info.append(assessment_text)
            clinical_info.append("")
        
        # Patient summary (if available)
        patient_summary = assessment.get('patient_summary', {})
        if patient_summary:
            clinical_info.append("Patient Summary:")
            for key, value in patient_summary.items():
                clinical_info.append(f"  {key}: {value}")
            clinical_info.append("")
        
        # Clinical summary (structured information)
        clinical_summary = assessment.get('clinical_summary', {})
        if clinical_summary:
            clinical_info.append("Clinical Summary:")
            for key, value in clinical_summary.items():
                if value and value != "Not specified":
                    clinical_info.append(f"  {key}: {value}")
            clinical_info.append("")
        
        # Clinical needs
        clinical_needs = assessment.get('clinical_needs', {})
        if clinical_needs:
            clinical_info.append("Clinical Needs:")
            for key, value in clinical_needs.items():
                clinical_info.append(f"  {key}: {value}")
            clinical_info.append("")
        
        # Evidence priorities
        evidence_priorities = assessment.get('evidence_priorities', {})
        if evidence_priorities:
            clinical_info.append("Evidence Priorities:")
            for key, value in evidence_priorities.items():
                clinical_info.append(f"  {key}: {value}")
            clinical_info.append("")
        
        # Trial eligibility factors
        trial_eligibility = assessment.get('trial_eligibility', {})
        if trial_eligibility:
            clinical_info.append("Trial Eligibility Factors:")
            for key, value in trial_eligibility.items():
                clinical_info.append(f"  {key}: {value}")
            clinical_info.append("")
        
        # Recommendations
        recommendations = assessment.get('recommendations_for_trial_search', [])
        if recommendations:
            clinical_info.append("Trial Search Recommendations:")
            for rec in recommendations:
                clinical_info.append(f"  - {rec}")
        
        return "\n".join(clinical_info)
    
    def _format_comprehensive_patient_info(self, assessment: Dict[str, Any], patient_data: Optional[Dict[str, Any]] = None) -> str:
        """Format comprehensive patient information combining assessment and original data"""
        
        clinical_info = []
        
        # Start with original patient data if available (more detailed)
        if patient_data:
            clinical_info.append("=== ORIGINAL PATIENT DATA ===")
            for key, value in patient_data.items():
                if value and str(value).strip() and key not in ['ID', 'expert_recommendation', 'source']:
                    # Format clinical information nicely
                    if key == 'clinical_information':
                        clinical_info.append("Clinical Information:")
                        clinical_info.append(str(value))
                    elif key == 'question_for_tumorboard':
                        clinical_info.append("Question for Tumorboard:")
                        clinical_info.append(str(value))
                    else:
                        clinical_info.append(f"{key}: {value}")
            clinical_info.append("")
        
        # Add assessment results if available
        if assessment:
            clinical_info.append("=== ASSESSMENT RESULTS ===")
            
            # Main assessment text (most important clinical information)
            assessment_text = assessment.get('assessment_text', '')
            if assessment_text and assessment_text.strip():
                clinical_info.append("Clinical Assessment:")
                clinical_info.append(assessment_text)
                clinical_info.append("")
                
                # If we have a good assessment text, skip the structured fields 
                # which are often empty or contain default values
                logger.info("✅ Found detailed assessment text, skipping minimal structured fields")
                
            else:
                # Only include structured fields if we don't have good assessment text
                logger.info("⚠️ No detailed assessment text found, including structured fields")
                
                # Patient summary (if available)
                patient_summary = assessment.get('patient_summary', {})
                if patient_summary:
                    clinical_info.append("Patient Summary:")
                    for key, value in patient_summary.items():
                        clinical_info.append(f"  {key}: {value}")
                    clinical_info.append("")
                
                # Clinical summary (structured information) - only if not minimal/empty
                clinical_summary = assessment.get('clinical_summary', {})
                if clinical_summary and self._has_meaningful_content(clinical_summary):
                    clinical_info.append("Clinical Summary:")
                    for key, value in clinical_summary.items():
                        if value and value not in ["Not specified", "Unable to parse", "Unknown"]:
                            clinical_info.append(f"  {key}: {value}")
                    clinical_info.append("")
                
                # Clinical needs - only if not default/generic
                clinical_needs = assessment.get('clinical_needs', {})
                if clinical_needs and self._has_meaningful_content(clinical_needs):
                    clinical_info.append("Clinical Needs:")
                    for key, value in clinical_needs.items():
                        if not self._is_generic_content(value):
                            clinical_info.append(f"  {key}: {value}")
                    clinical_info.append("")
                
                # Evidence priorities - only if not default/generic
                evidence_priorities = assessment.get('evidence_priorities', {})
                if evidence_priorities and self._has_meaningful_content(evidence_priorities):
                    clinical_info.append("Evidence Priorities:")
                    for key, value in evidence_priorities.items():
                        if not self._is_generic_content(value):
                            clinical_info.append(f"  {key}: {value}")
                    clinical_info.append("")
                
                # Trial eligibility factors - only if not empty
                trial_eligibility = assessment.get('trial_eligibility', {})
                if trial_eligibility and self._has_meaningful_content(trial_eligibility):
                    clinical_info.append("Trial Eligibility Factors:")
                    for key, value in trial_eligibility.items():
                        if value:  # Only include non-empty lists
                            clinical_info.append(f"  {key}: {value}")
                    clinical_info.append("")
            
            # Always include recommendations if they exist and are not generic
            recommendations = assessment.get('recommendations_for_trial_search', [])
            if recommendations and not self._is_generic_content(recommendations):
                clinical_info.append("Trial Search Recommendations:")
                for rec in recommendations:
                    clinical_info.append(f"  - {rec}")
        
        return "\n".join(clinical_info)
    
    def _has_meaningful_content(self, data: Dict[str, Any]) -> bool:
        """Check if structured data has meaningful (non-default) content"""
        if not data:
            return False
        
        for key, value in data.items():
            if value and value not in ["Unknown", "Unable to parse", "Not specified", [], {}]:
                if not self._is_generic_content(value):
                    return True
        return False
    
    def _is_generic_content(self, value) -> bool:
        """Check if content is generic/default rather than specific clinical information"""
        if not value:
            return True
            
        generic_phrases = [
            "Assessment and management guidance needed",
            "Clinical guidance for management decisions", 
            "clinical trials",
            "Search for relevant clinical trials",
            "medium",
            "Treatment decision support"
        ]
        
        if isinstance(value, str):
            return value in generic_phrases
        elif isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], str):
                return value[0] in generic_phrases
            return len(value) == 0
        
        return False
    
    def _format_patient_data_for_matching(self, patient_data: Dict[str, Any]) -> str:
        """Format raw patient data into clinical information for initial trial matching"""
        
        clinical_info = []
        
        # Basic patient information
        clinical_info.append("Patient Information:")
        for key, value in patient_data.items():
            # Exclude ID and expert_recommendation to avoid data leakage
            if value and str(value).strip() and key not in ['ID', 'expert_recommendation', 'source']:
                clinical_info.append(f"  {key}: {value}")
        clinical_info.append("")
        
        # Highlight key clinical elements that are important for trial matching
        diagnosis = patient_data.get('diagnosis', patient_data.get('Diagnosis', ''))
        if diagnosis:
            clinical_info.append(f"Primary Diagnosis: {diagnosis}")
        
        stage = patient_data.get('stage', patient_data.get('Stage', ''))
        if stage:
            clinical_info.append(f"Disease Stage: {stage}")
        
        prior_treatment = patient_data.get('prior_treatment', patient_data.get('Previous_Treatment', ''))
        if prior_treatment:
            clinical_info.append(f"Prior Treatment: {prior_treatment}")
        
        question = patient_data.get('question_for_tumorboard', patient_data.get('Question_for_Tumorboard', ''))
        if question:
            clinical_info.append(f"Clinical Question: {question}")
        
        return "\n".join(clinical_info)
    
    def _create_individual_evaluation_prompt(self, clinical_info: str, question_for_tumorboard: str, trial: Dict[str, Any]) -> str:
        """Create evaluation prompt for a single trial"""
        
        # Extract trial information with proper field mapping
        nct_id = trial.get('nct_id', 'Unknown')
        title = trial.get('title', 'Unknown')
        phase = trial.get('phase', 'Unknown')
        status = trial.get('status', 'Unknown')
        
        # Handle intervention field (note: field is 'intervention', not 'interventions')
        intervention = trial.get('intervention', trial.get('interventions', 'Unknown'))
        if isinstance(intervention, list):
            intervention = '; '.join(intervention) if intervention else 'Unknown'
        
        # Handle eligibility criteria
        eligibility_criteria = trial.get('eligibility_criteria', 'Unknown')
        
        # Handle brief summary
        summary = trial.get('brief_summary', trial.get('summary', 'Unknown'))
        
        # Handle detailed description
        detailed_description = trial.get('detailed_description', '')
        
        # Extract publication results data
        publication_citations = self._extract_publication_citations(trial)
        publication_findings = self._extract_publication_findings(trial)
        external_evidence = self._extract_external_evidence(trial)
        
        # For now, we'll use placeholder values for publication data since we don't have it
        publications_summary = self._format_publications_summary(trial)
        
        # Use the loaded prompt template and substitute placeholders
        prompt = self.prompt_template.format(
            clinical_information=clinical_info,
            question_for_tumorboard=question_for_tumorboard,
            nct_id=nct_id,
            title=title,
            phase=phase,
            status=status,
            intervention=intervention,
            eligibility_criteria=eligibility_criteria,
            summary=summary,
            publications_summary=publications_summary,
            publication_citations=publication_citations,
            publication_findings=publication_findings,
            external_evidence=external_evidence
        )
        
        return prompt
    
    def _extract_publication_citations(self, trial: Dict[str, Any]) -> str:
        """Extract publication citations from trial data"""
        citations = []
        
        # Get listed publications
        publications = trial.get('publications', [])
        for pub in publications:
            if isinstance(pub, dict) and pub.get('citation'):
                citations.append(f"• {pub['citation']}")
        
        # Get online search publications
        pub_analysis = trial.get('publication_analysis', {})
        online_results = pub_analysis.get('online_search_results', {})
        
        # PubMed publications
        pubmed_data = online_results.get('pubmed', {})
        pubmed_pubs = pubmed_data.get('publications', [])
        for pub in pubmed_pubs:
            if isinstance(pub, dict) and pub.get('title'):
                pmid = pub.get('pmid', '')
                title = pub.get('title', '')
                pmid_text = f" (PMID: {pmid})" if pmid else ""
                citations.append(f"• {title}{pmid_text}")
        
        # Onclive articles
        onclive_data = online_results.get('onclive', {})
        onclive_articles = onclive_data.get('articles', [])
        for article in onclive_articles:
            if isinstance(article, dict) and article.get('title'):
                title = article.get('title', '')
                url = article.get('url', '')
                url_text = f" ({url})" if url else ""
                citations.append(f"• {title}{url_text} [Onclive]")
        
        # Congress abstracts
        congress_data = online_results.get('congress_abstracts', {})
        congress_abstracts = congress_data.get('abstracts', [])
        for abstract in congress_abstracts:
            if isinstance(abstract, dict) and abstract.get('title'):
                title = abstract.get('title', '')
                citations.append(f"• {title} [Congress Abstract]")
        
        # Google Scholar articles
        scholar_data = online_results.get('google_scholar', {})
        scholar_articles = scholar_data.get('articles', [])
        for article in scholar_articles:
            if isinstance(article, dict) and article.get('title'):
                title = article.get('title', '')
                authors = article.get('authors', '')
                journal = article.get('journal', '')
                citation_text = title
                if authors:
                    citation_text += f" - {authors}"
                if journal:
                    citation_text += f" ({journal})"
                citations.append(f"• {citation_text} [Google Scholar]")
        
        # Check for any other publication sources dynamically
        for source_name, source_data in online_results.items():
            if source_name not in ['pubmed', 'onclive', 'congress_abstracts', 'google_scholar']:
                if isinstance(source_data, dict):
                    # Try different possible field names for articles/publications
                    for field_name in ['articles', 'publications', 'abstracts', 'papers']:
                        articles = source_data.get(field_name, [])
                        if articles:
                            for article in articles:
                                if isinstance(article, dict) and article.get('title'):
                                    title = article.get('title', '')
                                    citations.append(f"• {title} [{source_name.title()}]")
        
        return '\n'.join(citations) if citations else 'No publications found'
    
    def _extract_publication_findings(self, trial: Dict[str, Any]) -> str:
        """Extract key findings from publication abstracts and content"""
        findings = []
        
        # Get online search publications with abstracts/content
        pub_analysis = trial.get('publication_analysis', {})
        online_results = pub_analysis.get('online_search_results', {})
        
        # PubMed publications
        pubmed_data = online_results.get('pubmed', {})
        pubmed_pubs = pubmed_data.get('publications', [])
        
        for pub in pubmed_pubs:
            if isinstance(pub, dict):
                # Extract abstract
                abstract = pub.get('abstract_text', '')
                if abstract:
                    findings.append(f"PubMed Abstract: {abstract}")
                
                # Extract key content if available
                full_content = pub.get('full_content', '')
                if full_content and len(full_content) > len(abstract):
                    # Include the full content rather than truncated excerpts
                    findings.append(f"PubMed Full Content: {full_content}")
                elif len(abstract) == 0 and full_content:
                    # If no abstract but has content, include full content
                    findings.append(f"PubMed Full Content: {full_content}")
        
        # Onclive articles
        onclive_data = online_results.get('onclive', {})
        onclive_articles = onclive_data.get('articles', [])
        
        for article in onclive_articles:
            if isinstance(article, dict):
                # Extract full content from Onclive
                full_content = article.get('full_content', '')
                content_summary = article.get('content_summary', '')
                title = article.get('title', 'Onclive Article')
                
                if full_content:
                    findings.append(f"Onclive Article ({title}): {full_content}")
                elif content_summary:
                    findings.append(f"Onclive Article Summary ({title}): {content_summary}")
        
        # Congress abstracts
        congress_data = online_results.get('congress_abstracts', {})
        congress_abstracts = congress_data.get('abstracts', [])
        
        for abstract in congress_abstracts:
            if isinstance(abstract, dict):
                abstract_text = abstract.get('abstract_text', '')
                title = abstract.get('title', 'Congress Abstract')
                if abstract_text:
                    findings.append(f"Congress Abstract ({title}): {abstract_text}")
        
        # Google Scholar articles
        scholar_data = online_results.get('google_scholar', {})
        scholar_articles = scholar_data.get('articles', [])
        
        for article in scholar_articles:
            if isinstance(article, dict):
                # Extract content from Google Scholar
                abstract_text = article.get('abstract_text', '')
                full_content = article.get('full_content', '')
                content_summary = article.get('content_summary', '')
                title = article.get('title', 'Google Scholar Article')
                
                if full_content:
                    findings.append(f"Google Scholar Article ({title}): {full_content}")
                elif abstract_text:
                    findings.append(f"Google Scholar Abstract ({title}): {abstract_text}")
                elif content_summary:
                    findings.append(f"Google Scholar Summary ({title}): {content_summary}")
        
        # Check for any other publication sources dynamically
        for source_name, source_data in online_results.items():
            if source_name not in ['pubmed', 'onclive', 'congress_abstracts', 'google_scholar']:
                if isinstance(source_data, dict):
                    # Try different possible field names for articles/publications
                    for field_name in ['articles', 'publications', 'abstracts', 'papers']:
                        articles = source_data.get(field_name, [])
                        if articles:
                            for article in articles:
                                if isinstance(article, dict):
                                    # Extract content from any other source
                                    title = article.get('title', f'{source_name.title()} Article')
                                    full_content = article.get('full_content', '')
                                    abstract_text = article.get('abstract_text', '')
                                    content_summary = article.get('content_summary', '')
                                    
                                    if full_content:
                                        findings.append(f"{source_name.title()} Article ({title}): {full_content}")
                                    elif abstract_text:
                                        findings.append(f"{source_name.title()} Abstract ({title}): {abstract_text}")
                                    elif content_summary:
                                        findings.append(f"{source_name.title()} Summary ({title}): {content_summary}")
        
        return '\n\n'.join(findings) if findings else 'No detailed findings available'
    
    def _extract_external_evidence(self, trial: Dict[str, Any]) -> str:
        """Extract additional evidence from external sources"""
        evidence = []
        
        # Get outcomes from the trial itself
        primary_outcome = trial.get('primary_outcome', '')
        secondary_outcome = trial.get('secondary_outcome', '')
        
        if primary_outcome:
            evidence.append(f"Primary outcome: {primary_outcome}")
        if secondary_outcome:
            evidence.append(f"Secondary outcomes: {secondary_outcome}")
        
        # Get web results summary
        pub_analysis = trial.get('publication_analysis', {})
        web_summary = pub_analysis.get('web_results_summary', '')
        if web_summary:
            evidence.append(f"Web search summary: {web_summary}")
        
        # Get source overview
        source_overview = pub_analysis.get('source_overview', '')
        if source_overview:
            evidence.append(f"Sources checked: {source_overview}")
        
        # Get external sources found
        external_sources = pub_analysis.get('external_sources_found', [])
        if external_sources:
            evidence.append(f"External sources: {', '.join(external_sources)}")
        
        # Get any additional analysis notes
        analysis_notes = pub_analysis.get('analysis_notes', '')
        if analysis_notes:
            evidence.append(f"Analysis notes: {analysis_notes}")
        
        return '\n'.join(evidence) if evidence else 'No additional external evidence'
    
    def _format_publications_summary(self, trial: Dict[str, Any]) -> str:
        """Format publication summary information"""
        pub_analysis = trial.get('publication_analysis', {})
        
        summary_parts = []
        
        # Publications count
        pub_count = trial.get('publications_count', 0)
        has_pubs = trial.get('has_publications', False)
        
        if has_pubs and pub_count > 0:
            summary_parts.append(f"{pub_count} publication(s) found")
        
        # Analysis notes
        analysis_notes = pub_analysis.get('analysis_notes', '')
        if analysis_notes:
            summary_parts.append(f"Analysis: {analysis_notes}")
        
        # External sources
        external_sources = pub_analysis.get('external_sources_found', [])
        if external_sources:
            summary_parts.append(f"External sources: {', '.join(external_sources)}")
        
        return '; '.join(summary_parts) if summary_parts else 'No publication data available'
    
    def _parse_yes_no_response(self, response: str, trial: Dict[str, Any], trial_index: int) -> Dict[str, Any]:
        """Parse the YES/NO response from the LLM with enhanced format validation"""
        
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                
                # Validate expected format for trial matching
                if 'relevant' in evaluation and 'explanation' in evaluation:
                    # Correct format - add metadata and return
                    evaluation['trial_index'] = trial_index
                    evaluation['nct_id'] = trial.get('NCT ID', trial.get('nct_id', 'Unknown'))
                    evaluation['trial_title'] = trial.get('Study Title', trial.get('title', 'Unknown'))
                    return evaluation
                
                elif 'assessment' in evaluation:
                    # Wrong format - got patient assessment instead of trial matching
                    logger.error(f"🚨 Wrong response format for trial {trial_index}: Got patient assessment instead of trial matching")
                    return {
                        'trial_index': trial_index,
                        'nct_id': trial.get('NCT ID', trial.get('nct_id', 'Unknown')),
                        'trial_title': trial.get('Study Title', trial.get('title', 'Unknown')),
                        'relevant': 'ERROR',
                        'explanation': 'LLM returned patient assessment format instead of trial matching format',
                        'raw_response': response,
                        'parsing_error': 'Wrong JSON schema: expected {relevant, explanation}, got {assessment}'
                    }
                
                else:
                    # Unknown format
                    logger.error(f"🚨 Unexpected JSON format for trial {trial_index}: {list(evaluation.keys())}")
                    return {
                        'trial_index': trial_index,
                        'nct_id': trial.get('NCT ID', trial.get('nct_id', 'Unknown')),
                        'trial_title': trial.get('Study Title', trial.get('title', 'Unknown')),
                        'relevant': 'ERROR',
                        'explanation': f'Unexpected JSON format. Got keys: {list(evaluation.keys())}',
                        'raw_response': response,
                        'parsing_error': f'Unexpected JSON schema: {list(evaluation.keys())}'
                    }
            else:
                # If no JSON found, create error response
                logger.error(f"🚨 No JSON found in response for trial {trial_index}")
                return {
                    'trial_index': trial_index,
                    'nct_id': trial.get('NCT ID', trial.get('nct_id', 'Unknown')),
                    'trial_title': trial.get('Study Title', trial.get('title', 'Unknown')),
                    'relevant': 'ERROR',
                    'explanation': 'Could not extract JSON from response',
                    'raw_response': response,
                    'parsing_error': 'Could not extract JSON from response'
                }
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parsing failed for trial {trial_index}: {e}")
            return {
                'trial_index': trial_index,
                'nct_id': trial.get('NCT ID', trial.get('nct_id', 'Unknown')),
                'trial_title': trial.get('Study Title', trial.get('title', 'Unknown')),
                'relevant': 'ERROR',
                'explanation': f'JSON parsing failed: {str(e)}',
                'raw_response': response,
                'parsing_error': f'JSON decode error: {str(e)}'
            }
        logger.info(f"📋 Using prompt file: {self.prompt_file}")
    
    def load_trial_data(self, trial_data_path: str) -> List[Dict[str, Any]]:
        """Load trial data from JSON file"""
        
        logger.info(f"📚 Loading trial data from: {trial_data_path}")
        
        try:
            if trial_data_path.endswith('.json'):
                with open(trial_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle nested JSON structure with 'studies' key
                if isinstance(data, dict) and 'studies' in data:
                    logger.info("ℹ️ Found 'studies' key, loading trials from this key.")
                    trials = data['studies']
                elif isinstance(data, list):
                    trials = data
                elif isinstance(data, dict):
                    trials = [v for v in data.values() if isinstance(v, dict)]
                else:
                    raise TypeError(f"Unsupported JSON structure in {trial_data_path}")
                    
            elif trial_data_path.endswith('.xlsx'):
                df = pd.read_excel(trial_data_path)
                trials = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported file format: {trial_data_path}")
            
            # Ensure trials is a list of dictionaries
            trials_list = [t for t in trials if isinstance(t, dict)]
            # Explicitly cast each element to Dict[str, Any] for type safety
            trials_list_typed: List[Dict[str, Any]] = [{str(k): v for k, v in t.items()} for t in trials_list]
            logger.info(f"✅ Loaded {len(trials_list_typed)} trials from {trial_data_path}")
            return trials_list_typed
            
        except Exception as e:
            logger.error(f"❌ Error loading trial data: {e}")
            raise
    
    def match_trials_for_patient(self, patient_data: Dict[str, Any], trial_data_path: str, 
                                output_dir: str = ".") -> Dict[str, Any]:
        """
        Initial trial matching based on raw patient data (before assessment)
        
        Args:
            patient_data: Raw patient data dictionary
            trial_data_path: Path to trial data file
            output_dir: Directory to save matching results
            
        Returns:
            Dictionary containing initial matching results with all relevant trials
        """
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        logger.info(f"🔍 Initial trial matching for patient {patient_id}")
        
        # Load trial data
        trials = self.load_trial_data(trial_data_path)
        
        # Format patient data for matching
        clinical_info = self._format_patient_data_for_matching(patient_data)
        
        # Get question for tumorboard
        question_for_tumorboard = patient_data.get('question_for_tumorboard', 
                                                 patient_data.get('Question_for_Tumorboard', 
                                                 'Treatment recommendation needed'))
        
        # Evaluate trials (evaluate all trials for comprehensive initial matching)
        relevant_trials = []
        evaluation_results = []
        
        logger.info(f"🔄 Evaluating all {len(trials)} trials for initial matching...")
        
        # Evaluate all trials for initial matching (no artificial limit)
        trials_to_evaluate = trials
        
        # Create trial_matching subfolder
        trial_matching_dir = os.path.join(output_dir, "trial_matching")
        os.makedirs(trial_matching_dir, exist_ok=True)
        
        for i, trial in enumerate(trials_to_evaluate):
            # Get trial info for logging
            trial_title = trial.get('title', trial.get('brief_title', trial.get('official_title', 'Unknown')))
            trial_nct = trial.get('nct_id', f'trial_{i}')
            
            logger.info(f"📊 Evaluating trial {i+1}/{len(trials_to_evaluate)}: {trial_title[:60]}... ({trial_nct})")
            
            try:
                # Create evaluation prompt
                prompt = self._create_individual_evaluation_prompt(clinical_info, question_for_tumorboard, trial)
                
                # Save prompt in trial_matching subfolder
                trial_nct = trial.get('nct_id', f'trial_{i}')
                prompt_file = os.path.join(trial_matching_dir, f"patient_{patient_id}_initial_trial_{trial_nct}_prompt.txt")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== INITIAL TRIAL EVALUATION PROMPT ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Patient ID: {patient_id}\n")
                    f.write(f"Trial NCT ID: {trial_nct}\n")
                    f.write(f"Model: {self.model}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(prompt)
                
                # Get LLM response
                response = self._call_llm(prompt)
                
                # Save response in trial_matching subfolder
                response_file = os.path.join(trial_matching_dir, f"patient_{patient_id}_initial_trial_{trial_nct}_response.txt")
                with open(response_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== INITIAL TRIAL EVALUATION RESPONSE ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Patient ID: {patient_id}\n")
                    f.write(f"Trial NCT ID: {trial_nct}\n")
                    f.write(f"Model: {self.model}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(response)
                
                # Parse response
                evaluation_result = self._parse_yes_no_response(response, trial, i)
                evaluation_result['prompt_file'] = prompt_file
                evaluation_result['response_file'] = response_file
                evaluation_results.append(evaluation_result)
                
                # If trial is relevant, add to relevant_trials
                if evaluation_result.get('is_relevant', False):
                    trial_with_metadata = trial.copy()
                    trial_with_metadata['relevance_score'] = evaluation_result.get('relevance_score', 0.5)
                    trial_with_metadata['relevance_reason'] = evaluation_result.get('reasoning', 'Initial matching based on patient data')
                    relevant_trials.append(trial_with_metadata)
                    logger.info(f"✅ Trial {trial_nct} marked as relevant")
                else:
                    logger.info(f"❌ Trial {trial_nct} not relevant")
                    
            except Exception as e:
                logger.error(f"❌ Error evaluating trial {i}: {e}")
                continue
        
        # Sort relevant trials by relevance score and take top results
        relevant_trials.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"📊 Found {len(relevant_trials)} relevant trials - including all in results")
        
        # Create final results
        matching_result = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model,
            'trials_evaluated': len(evaluation_results),
            'relevant_trials': relevant_trials,
            'evaluation_results': evaluation_results,
            'matching_type': 'initial_patient_based',
            'metadata': {
                'total_trials_available': len(trials),
                'trials_evaluated': len(trials_to_evaluate),
                'relevant_trials_found': len(relevant_trials)
            }
        }
        
        # Save matching results
        results_file = os.path.join(output_dir, f"patient_{patient_id}_initial_trial_matching.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(matching_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Initial trial matching completed: {len(relevant_trials)} relevant trials found")
        logger.info(f"📄 Results saved to: {results_file}")
        
        return matching_result
    
    def match_trials(self, assessment: Dict[str, Any], trial_data_path: str, 
                    patient_data: Optional[Dict[str, Any]] = None, output_dir: str = ".") -> Dict[str, Any]:
        """
        Match relevant trials based on patient assessment with model-specific caching
        
        Args:
            assessment: Patient assessment results
            trial_data_path: Path to trial data file
            patient_data: Original patient data (optional, for additional context)
            output_dir: Directory to save matching results
            
        Returns:
            Dictionary containing matching results with all relevant trials
        """
        patient_id = assessment.get('metadata', {}).get('patient_id', 'unknown')
        logger.info(f"🔍 Evaluating trials for patient {patient_id}")
        
        # Load trial data for cache key generation
        trials = self.load_trial_data(trial_data_path)
        
        # Try to load from cache first
        cache_key = self._get_cache_key(patient_data or {}, assessment, {'studies': trials})
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result is not None:
            # Check if cached result has limitations that our current code doesn't have
            summary = cached_result.get('summary', {})
            trials_returned = summary.get('trials_returned', 0)
            relevant_trials_found = summary.get('relevant_trials_found', 0)
            
            # If the cached result was limited but found more trials, invalidate cache
            if relevant_trials_found > trials_returned and relevant_trials_found > 10:
                logger.info(f"🔄 Cached result was limited ({trials_returned}/{relevant_trials_found} trials), re-evaluating with updated logic")
            else:
                logger.info(f"✅ Using cached trial matching for patient {patient_id} with model {self.model}")
                
                # Still save to output directory for current run
                output_file = os.path.join(output_dir, f"patient_{patient_id}_trials.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cached_result, f, indent=2, ensure_ascii=False)
                logger.info(f"📄 Results copied to: {output_file}")
                
                return cached_result
        
        # Evaluate each trial individually
        relevant_trials = []
        evaluation_results = []
        
        # Prepare patient clinical information for prompt
        # Include both assessment results and original patient data for comprehensive context
        clinical_info = self._format_comprehensive_patient_info(assessment, patient_data)
        
        # Get question for tumorboard from original patient data if available, otherwise from assessment
        question_for_tumorboard = ""
        if patient_data and 'question_for_tumorboard' in patient_data:
            question_for_tumorboard = patient_data['question_for_tumorboard']
        else:
            # Fallback to assessment data or generic
            question_for_tumorboard = assessment.get('question_for_tumorboard', 
                                                    assessment.get('clinical_needs', {}).get('primary_needs', 'Treatment decision support'))
            
        # Convert to string if it's a list
        if isinstance(question_for_tumorboard, list):
            question_for_tumorboard = '; '.join(str(q) for q in question_for_tumorboard)
        
        # Create trial_matching subfolder
        trial_matching_dir = os.path.join(output_dir, "trial_matching")
        os.makedirs(trial_matching_dir, exist_ok=True)
        
        logger.info(f"📝 Evaluating {len(trials)} trials individually...")
        
        for i, trial in enumerate(trials):
            try:
                # Create individual evaluation prompt
                evaluation_prompt = self._create_individual_evaluation_prompt(
                    clinical_info, question_for_tumorboard, trial
                )
                
                # Save the prompt for this trial in trial_matching subfolder
                trial_nct = trial.get('NCT ID', trial.get('nct_id', f'trial_{i+1}'))
                prompt_file = os.path.join(trial_matching_dir, f"trial_matching_{trial_nct}_prompt.txt")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write("=== TRIAL MATCHING PROMPT ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Trial: {i+1}/{len(trials)}\n")
                    f.write(f"NCT ID: {trial_nct}\n")
                    f.write(f"Model: {self.model}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(evaluation_prompt)
                
                # Call LLM for this specific trial
                response = self._call_llm(evaluation_prompt)
                
                # Save the raw response in trial_matching subfolder
                response_file = os.path.join(trial_matching_dir, f"trial_matching_{trial_nct}_response.txt")
                with open(response_file, 'w', encoding='utf-8') as f:
                    f.write("=== TRIAL MATCHING RAW RESPONSE ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Trial: {i+1}/{len(trials)}\n")
                    f.write(f"NCT ID: {trial_nct}\n")
                    f.write(f"Model: {self.model}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(response)
                
                evaluation = self._parse_yes_no_response(response, trial, i)
                evaluation['prompt_file'] = prompt_file
                evaluation['response_file'] = response_file
                evaluation_results.append(evaluation)
                
                # If relevant, add to relevant trials list
                if evaluation.get('relevant') == 'YES':
                    trial_with_evaluation = trial.copy()
                    trial_with_evaluation['evaluation'] = evaluation
                    relevant_trials.append(trial_with_evaluation)
                    logger.info(f"✅ Trial {i+1}/{len(trials)}: {trial.get('NCT ID', trial.get('nct_id', 'Unknown'))} - RELEVANT")
                else:
                    logger.info(f"❌ Trial {i+1}/{len(trials)}: {trial.get('NCT ID', trial.get('nct_id', 'Unknown'))} - NOT RELEVANT")
                    
            except Exception as e:
                logger.error(f"❌ Error evaluating trial {i+1}: {e}")
                evaluation_results.append({
                    'trial_index': i,
                    'nct_id': trial.get('NCT ID', trial.get('nct_id', 'Unknown')),
                    'relevant': 'ERROR',
                    'explanation': f'Evaluation failed: {str(e)}',
                    'error': str(e)
                })
        
        # Sort relevant trials by some criteria (could be enhanced later)
        relevant_trials.sort(key=lambda x: x.get('Phase', 'Unknown'), reverse=True)
        
        # Create final result
        matching_result = {
            'relevant_trials': relevant_trials,  # Include all relevant trials
            'all_evaluations': evaluation_results,
            'summary': {
                'total_trials_evaluated': len(evaluation_results),
                'relevant_trials_found': len(relevant_trials),
                'trials_returned': len(relevant_trials)
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'patient_id': patient_id,
                'total_trials_considered': len(trials),
                'trial_data_source': trial_data_path
            }
        }

        # Backward compatibility for callers expecting 'evaluation_results'
        matching_result['evaluation_results'] = evaluation_results
        
        # Save matching results to file
        matching_file = os.path.join(output_dir, f"patient_{patient_id}_trial_matching.json")
        with open(matching_file, 'w', encoding='utf-8') as f:
            json.dump(matching_result, f, indent=2, ensure_ascii=False)
        
        # Save to cache for future use
        self._save_to_cache(cache_key, matching_result)
        
        logger.info(f"✅ Trial evaluation completed. Found {len(relevant_trials)} relevant trials out of {len(evaluation_results)} evaluated.")
        logger.info(f"📁 Results saved to: {matching_file}")
        logger.info(f"💾 Results cached for model: {self.model}")
        
        return matching_result
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API for trial evaluation with enhanced error handling"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/piakoller/prism",
            "X-Title": "Trial Matcher"
        }
        
        # No max_tokens limit - allow unrestricted output length
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1  # Low temperature for consistent evaluation
        }
        
        # Add model-specific parameters for Gemini models (like in patient_assessor)
        if "gemini" in self.model.lower():
            data.update({
                "stream": False,  # Ensure no streaming
                "top_p": 0.95,    # Add top_p for Gemini models
            })
        
        # Add timeout and retry logic
        max_retries = 4  # Increased from 2 to reduce empty responses
        timeout_seconds = 120
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔗 Trial matching API call attempt {attempt + 1}/{max_retries + 1} (no token limit)...")
                response = requests.post(
                    self.base_url, 
                    headers=headers, 
                    json=data, 
                    timeout=timeout_seconds
                )
                
                logger.info(f"📡 API Response: Status {response.status_code}, Content-Length: {len(response.content)}")
                
                if response.status_code != 200:
                    error_msg = f"Trial matching API call failed: {response.status_code} - {response.text}"
                    logger.error(f"❌ {error_msg}")
                    
                    # Log specific error details for debugging
                    try:
                        error_response = response.json()
                        if "error" in error_response:
                            error_details = error_response["error"]
                            logger.error(f"🔍 API Error Details: {error_details}")
                    except:
                        pass
                    
                    if attempt < max_retries:
                        logger.info(f"🔄 Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                        continue
                    else:
                        return error_msg
                
                try:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    if not content or content.strip() == "":
                        error_msg = f"Empty response from trial matching API (attempt {attempt + 1})"
                        logger.warning(f"⚠️ {error_msg}")
                        if attempt < max_retries:
                            wait_time = 5 * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s, 40s
                            logger.info(f"🔄 Retrying in {wait_time} seconds...")
                            import time
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"❌ All attempts failed - empty response")
                            return ""
                    
                    logger.info(f"✅ Trial matching API call successful (response length: {len(content)} characters)")
                    # Small delay to prevent rate limiting on subsequent calls
                    import time
                    time.sleep(0.5)
                    return content
                    
                except (KeyError, ValueError) as e:
                    error_msg = f"Failed to parse trial matching response: {e}"
                    logger.error(f"❌ {error_msg}")
                    if attempt < max_retries:
                        wait_time = 5 * (2 ** attempt)  # Exponential backoff
                        logger.info(f"🔄 Retrying in {wait_time} seconds...")
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        return error_msg
                
            except requests.exceptions.Timeout:
                error_msg = f"Trial matching API call timed out after {timeout_seconds} seconds (attempt {attempt + 1})"
                logger.warning(f"⏱️ {error_msg}")
                if attempt < max_retries:
                    logger.info(f"🔄 Retrying with longer timeout...")
                    timeout_seconds += 60
                    continue
                else:
                    logger.error(f"❌ All attempts failed - timeout")
                    return error_msg
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error during trial matching API call: {e}"
                logger.error(f"❌ {error_msg}")
                if attempt < max_retries:
                    logger.info(f"🔄 Retrying in 10 seconds...")
                    import time
                    time.sleep(10)
                    continue
                else:
                    logger.error(f"❌ All attempts failed - network error")
                    return error_msg
        # If all retries exhausted and no return occurred, return empty string
        return ""
    
    def get_matching_summary(self, matching_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the trial matching results"""
        
        summary_lines = []
        summary_lines.append("=== TRIAL EVALUATION SUMMARY ===")
        summary_lines.append(f"Patient ID: {matching_result.get('metadata', {}).get('patient_id', 'unknown')}")
        summary_lines.append(f"Evaluation Time: {matching_result.get('metadata', {}).get('timestamp', 'unknown')}")
        summary_lines.append(f"Total Trials Evaluated: {matching_result.get('summary', {}).get('total_trials_evaluated', 'unknown')}")
        summary_lines.append(f"Relevant Trials Found: {matching_result.get('summary', {}).get('relevant_trials_found', 'unknown')}")
        summary_lines.append(f"Trials Returned: {matching_result.get('summary', {}).get('trials_returned', 'unknown')}")
        summary_lines.append("")
        
        # Relevant trials
        relevant_trials = matching_result.get('relevant_trials', [])
        if relevant_trials:
            summary_lines.append("RELEVANT TRIALS:")
            for i, trial in enumerate(relevant_trials):
                nct_id = trial.get('NCT ID', trial.get('nct_id', 'Unknown'))
                title = trial.get('Study Title', trial.get('title', 'Unknown'))
                evaluation = trial.get('evaluation', {})
                
                summary_lines.append(f"  {i+1}. {nct_id}")
                summary_lines.append(f"     Title: {title[:100]}...")
                summary_lines.append(f"     Phase: {trial.get('Phase', 'Unknown')}")
                summary_lines.append(f"     Status: {trial.get('Study Status', 'Unknown')}")
                
                # Evaluation explanation
                explanation = evaluation.get('explanation', 'No explanation provided')
                summary_lines.append(f"     Rationale: {explanation}...")
                summary_lines.append("")
        else:
            summary_lines.append("NO RELEVANT TRIALS FOUND")
            summary_lines.append("")
        
        # Summary of all evaluations
        all_evaluations = matching_result.get('all_evaluations', [])
        if all_evaluations:
            yes_count = sum(1 for eval in all_evaluations if eval.get('relevant') == 'YES')
            no_count = sum(1 for eval in all_evaluations if eval.get('relevant') == 'NO')
            error_count = sum(1 for eval in all_evaluations if eval.get('relevant') == 'ERROR')
            
            summary_lines.append("EVALUATION BREAKDOWN:")
            summary_lines.append(f"  YES (Relevant): {yes_count}")
            summary_lines.append(f"  NO (Not Relevant): {no_count}")
            summary_lines.append(f"  ERROR (Evaluation Failed): {error_count}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
