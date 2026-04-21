#!/usr/bin/env python3
"""
Trial Analyzer Module

This module systematically analyzes clinical trial evidence for relevance to specific patients.
It provides structured evaluation of each trial to prevent hallucination and ensure comprehensive analysis.
"""

import os
import json
import logging
import requests
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL, MODEL_CONFIGS, TRIAL_ANALYSIS_CACHE_DIR

logger = logging.getLogger(__name__)

class TrialAnalyzer:
    """Handles systematic analysis of clinical trials for patient-specific relevance"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, prompt_file: Optional[Union[str, Path]] = None):
        """
        Initialize the Trial Analyzer
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: LLM model to use for analysis (defaults to config.OPENROUTER_MODEL)
            prompt_file: Path to the trial analysis prompt file (defaults to standard location)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        if prompt_file is None:
            base_dir = Path(__file__).parent.parent.parent
            prompt_file = base_dir / "prompts" / "4_trial_analysis.txt"

        if isinstance(prompt_file, Path):
            self.prompt_file = prompt_file
        else:
            self.prompt_file = Path(str(prompt_file))
        self.prompt_template = self._load_prompt_template()
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please check your configuration.")
        
        logger.info(f"🔬 Trial Analyzer initialized with model: {self.model}")
        logger.info(f"📄 Using prompt file: {self.prompt_file}")

        try:
            self.cache_dir = Path(TRIAL_ANALYSIS_CACHE_DIR)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 Trial analysis cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize trial analysis cache dir: {e}")

    def _get_cache_key(self, patient_assessment: Dict[str, Any], trials_data: Dict[str, Any], patient_id: str) -> str:
        """Generate a stable cache key using only invariant inputs."""
        import hashlib
        
        assessment_text = ""
        if isinstance(patient_assessment, dict):
            assessment_text = patient_assessment.get('assessment_text', '')
            if not assessment_text:
                assessment_text = patient_assessment.get('clinical_summary', '')
        
        nct_ids = []
        trials_list = []
        if isinstance(trials_data, dict):
            if 'trials' in trials_data:
                trials_list = trials_data['trials']
            elif 'relevant_trials' in trials_data:
                trials_list = trials_data['relevant_trials']
        
        for trial in trials_list:
            if isinstance(trial, dict) and 'nct_id' in trial:
                nct_ids.append(trial['nct_id'])
        
        nct_ids_sorted = sorted(set(nct_ids))
        trials_signature = '|'.join(nct_ids_sorted)
        
        base = f"{patient_id}|{self.model}|{assessment_text[:500]}|{trials_signature}"
        digest = hashlib.md5(base.encode('utf-8')).hexdigest()[:12]
        return f"patient_{patient_id}_{digest}"

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}_analysis.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(cache_key)
        if path.exists():
            try:
                data = json.load(open(path, 'r', encoding='utf-8'))
                meta = data.get('metadata', {})
                if meta.get('model_used') == self.model:
                    logger.info(f"📂 Loaded trial analysis from cache: {path}")
                    return data
                else:
                    logger.info(f"🔄 Cache model mismatch ({meta.get('model_used')} != {self.model}); regenerating")
            except Exception as e:
                logger.warning(f"⚠️ Failed to read cache {path}: {e}")
        
        # Migration fallback: check for any existing cache files for this patient and model
        try:
            pattern = f"patient_{cache_key.split('_')[1]}_*_analysis.json"  # Extract patient_id from cache_key
            for existing_cache in self.cache_dir.glob(pattern):
                try:
                    data = json.load(open(existing_cache, 'r', encoding='utf-8'))
                    meta = data.get('metadata', {})
                    if meta.get('model_used') == self.model and data.get('status') == 'success':
                        logger.info(f"♻️ Migrating existing cache file for reuse: {existing_cache}")
                        if 'metadata' not in data:
                            data['metadata'] = {}
                        data['metadata']['cache_reused'] = True
                        data['metadata']['cached_at'] = datetime.now().isoformat()
                        data['metadata']['migrated_from'] = str(existing_cache)
                        self._save_to_cache(cache_key, data)
                        return data
                except Exception as e:
                    logger.debug(f"Could not migrate cache file {existing_cache}: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Error during cache migration search: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        try:
            path = self._cache_path(cache_key)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved trial analysis to cache: {path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save trial analysis cache: {e}")

    def _load_prompt_template(self) -> str:
        """Load the trial analysis prompt template from file"""
        try:
            if not self.prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
            
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                template = f.read()
            
            logger.info(f"✅ Loaded trial analysis prompt template ({len(template)} chars)")
            return template
            
        except Exception as e:
            logger.error(f"❌ Error loading prompt template: {e}")
            raise

    def analyze_trials(self, patient_assessment: Dict[str, Any], trials_data: Dict[str, Any], 
                      patient_id: str, output_dir: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze clinical trials for patient-specific relevance
        
        Args:
            patient_assessment: Patient assessment data
            trials_data: Clinical trials evidence data
            patient_id: Patient identifier
            output_dir: Directory to save analysis results (optional)
            use_cache: Whether to use cached analysis if available (default: True)
            
        Returns:
            Dict containing structured trial analysis results
        """
        try:
            logger.info(f"🔬 Starting trial analysis for patient {patient_id}")
            
            # Check for existing cached analysis (model-aware, content-hashed)
            if use_cache:
                cache_key = self._get_cache_key(patient_assessment, trials_data, patient_id)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    # Update metadata to show reuse
                    if 'metadata' not in cached:
                        cached['metadata'] = {}
                    cached['metadata']['cached_at'] = datetime.now().isoformat()
                    cached['metadata']['cache_reused'] = True
                    return cached
            
            # Format patient assessment for prompt
            assessment_text = self._format_patient_assessment(patient_assessment)
            
            # Format trials evidence for prompt  
            trials_evidence = self._format_trials_evidence(trials_data)
            
            # Count input trials for validation
            input_trial_count = self._count_input_trials(trials_data)
            logger.info(f"📊 Input contains {input_trial_count} trials to analyze")
            
            # Create the analysis prompt
            prompt = self._create_analysis_prompt(assessment_text, trials_evidence)
            
            # Save the prompt for debugging/monitoring
            if output_dir:
                prompt_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_prompt.txt")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write("=== TRIAL ANALYSIS PROMPT ===\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Patient ID: {patient_id}\n")
                    f.write(f"Model: {self.model}\n")
                    f.write(f"Input Trial Count: {input_trial_count}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(prompt)
                
                logger.info(f"📄 Trial analysis prompt saved to: {prompt_file}")
            
            # Call LLM for analysis
            response = self._call_llm_for_analysis(prompt)
            
            # Parse and validate the analysis response
            analysis_result = self._parse_analysis_response(response, input_trial_count)
            
            analysis_result['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'patient_id': patient_id,
                'input_trial_count': input_trial_count,
                'cache_reused': False,
                'cached_at': None
            }
            
            if output_dir:
                analysis_result['metadata']['prompt_file'] = prompt_file
            
            # Save analysis results if output directory provided
            if output_dir:
                self._save_analysis_results(analysis_result, patient_id, output_dir)

            # Save to model-aware cache
            if use_cache:
                try:
                    cache_key = self._get_cache_key(patient_assessment, trials_data, patient_id)
                    self._save_to_cache(cache_key, analysis_result)
                except Exception as e:
                    logger.warning(f"⚠️ Could not cache trial analysis: {e}")
            
            logger.info(f"✅ Trial analysis completed for patient {patient_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error in trial analysis: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "analysis_complete": False,
                "trials_analyzed": 0,
                "trials_included": 0,
                "trials_excluded": 0
            }

    def _format_patient_assessment(self, assessment: Dict[str, Any]) -> str:
        """Format patient assessment for the prompt"""
        try:
            if isinstance(assessment, dict):
                # Extract key clinical information
                patient_info = []
                
                # Basic demographics and diagnosis
                if 'patient_id' in assessment:
                    patient_info.append(f"Patient ID: {assessment['patient_id']}")
                
                if 'diagnosis' in assessment:
                    patient_info.append(f"Diagnosis: {assessment['diagnosis']}")
                
                if 'tumor_characteristics' in assessment:
                    tumor_chars = assessment['tumor_characteristics']
                    if isinstance(tumor_chars, dict):
                        for key, value in tumor_chars.items():
                            patient_info.append(f"{key}: {value}")
                    else:
                        patient_info.append(f"Tumor characteristics: {tumor_chars}")
                
                # Clinical status and symptoms
                if 'clinical_status' in assessment:
                    patient_info.append(f"Clinical status: {assessment['clinical_status']}")
                
                if 'symptoms' in assessment:
                    patient_info.append(f"Symptoms: {assessment['symptoms']}")
                
                # Previous treatments
                if 'previous_treatments' in assessment:
                    patient_info.append(f"Previous treatments: {assessment['previous_treatments']}")
                
                # Include any assessment text if available
                if 'assessment_text' in assessment:
                    patient_info.append(f"Assessment: {assessment['assessment_text']}")
                
                return "\n".join(patient_info)
            else:
                return str(assessment)
                
        except Exception as e:
            logger.warning(f"⚠️ Error formatting patient assessment: {e}")
            return str(assessment)

    def _format_trials_evidence(self, trials_data: Dict[str, Any]) -> str:
        """Format trials evidence for the prompt with comprehensive evidence extraction"""
        try:
            trials_list = []
            
            # Handle different trial data structures
            if isinstance(trials_data, dict):
                if 'trials' in trials_data:
                    trials_list = trials_data['trials']
                elif 'relevant_trials' in trials_data:
                    trials_list = trials_data['relevant_trials']
                else:
                    # Try to find trials in other common keys
                    for key in trials_data:
                        if isinstance(trials_data[key], list) and len(trials_data[key]) > 0:
                            # Check if first item looks like a trial (has nct_id)
                            if isinstance(trials_data[key][0], dict) and 'nct_id' in trials_data[key][0]:
                                trials_list = trials_data[key]
                                break
            
            if trials_list:
                # Format each trial with key information
                formatted_trials = []
                
                for trial in trials_list:
                    trial_text = []
                    
                    # NCT number (most important)
                    if 'nct_id' in trial:
                        trial_text.append(f"NCT Number: {trial['nct_id']}")
                    
                    # Trial title/name
                    if 'title' in trial:
                        trial_text.append(f"Title: {trial['title']}")
                    elif 'study_name' in trial:
                        trial_text.append(f"Study Name: {trial['study_name']}")
                    
                    # Study phase and design
                    if 'phase' in trial:
                        trial_text.append(f"Phase: {trial['phase']}")
                    if 'study_type' in trial:
                        trial_text.append(f"Study Type: {trial['study_type']}")
                    if 'status' in trial:
                        trial_text.append(f"Status: {trial['status']}")
                    
                    # Population and conditions
                    if 'condition' in trial:
                        trial_text.append(f"Condition: {trial['condition']}")
                    elif 'conditions' in trial:
                        trial_text.append(f"Conditions: {trial['conditions']}")
                    if 'population' in trial:
                        trial_text.append(f"Population: {trial['population']}")
                    if 'intervention' in trial:
                        trial_text.append(f"Intervention: {trial['intervention']}")
                    
                    # Study details
                    if 'brief_summary' in trial and trial['brief_summary']:
                        trial_text.append(f"Brief Summary: {trial['brief_summary']}")
                    if 'primary_outcome' in trial and trial['primary_outcome']:
                        trial_text.append(f"Primary Outcome: {trial['primary_outcome']}")
                    if 'secondary_outcome' in trial and trial['secondary_outcome']:
                        trial_text.append(f"Secondary Outcomes: {trial['secondary_outcome']}")
                    
                    # COMPREHENSIVE EVIDENCE EXTRACTION
                    evidence_sections = []
                    
                    # 1. Listed publications (formal peer-reviewed)
                    if 'publications' in trial and trial['publications']:
                        pubs = []
                        for pub in trial['publications']:
                            if isinstance(pub, dict):
                                pub_info = []
                                if 'citation' in pub:
                                    pub_info.append(f"Citation: {pub['citation']}")
                                if 'pmid' in pub:
                                    pub_info.append(f"PMID: {pub['pmid']}")
                                if 'type' in pub:
                                    pub_info.append(f"Type: {pub['type']}")
                                pubs.append(" | ".join(pub_info))
                            else:
                                pubs.append(str(pub))
                        if pubs:
                            evidence_sections.append(f"LISTED PUBLICATIONS:\n" + "\n".join(pubs))
                    
                    # 2. Publication analysis and web search results
                    if 'publication_analysis' in trial:
                        pub_analysis = trial['publication_analysis']
                        
                        # Summary of publication status
                        pub_status = []
                        if 'has_listed_publications' in pub_analysis:
                            pub_status.append(f"Has Listed Publications: {pub_analysis['has_listed_publications']}")
                        if 'has_results_publications' in pub_analysis:
                            pub_status.append(f"Has Results Publications: {pub_analysis['has_results_publications']}")
                        if 'total_publications_found' in pub_analysis:
                            pub_status.append(f"Total Publications Found: {pub_analysis['total_publications_found']}")
                        if pub_status:
                            evidence_sections.append(f"PUBLICATION STATUS:\n" + " | ".join(pub_status))
                        
                        # Online search results with actual content
                        if 'online_search_results' in pub_analysis:
                            search_results = pub_analysis['online_search_results']
                            
                            # PubMed results with abstract and full content
                            if 'pubmed' in search_results and search_results['pubmed'].get('publications'):
                                pubmed_pubs = []
                                for pub in search_results['pubmed']['publications']:
                                    if isinstance(pub, dict):
                                        pub_parts = []
                                        
                                        # Basic citation info
                                        title = pub.get('title', 'No title')
                                        authors = pub.get('authors', 'No authors')
                                        journal = pub.get('journal', 'No journal')
                                        pub_parts.append(f"Title: {title}")
                                        pub_parts.append(f"Authors: {authors}")
                                        pub_parts.append(f"Journal: {journal}")
                                        
                                        # Abstract text (very important for evidence)
                                        abstract = pub.get('abstract_text', '')
                                        if abstract:
                                            # Include full abstract without truncation
                                            pub_parts.append(f"Abstract: {abstract}")
                                        
                                        # Full content if available
                                        full_content = pub.get('full_content', '')
                                        if full_content:
                                            # Include full content without truncation
                                            pub_parts.append(f"Key Content: {full_content}")
                                        
                                        pubmed_pubs.append("\n".join(pub_parts))
                                
                                if pubmed_pubs:
                                    evidence_sections.append(f"PUBMED RESULTS:\n" + "\n\n".join(pubmed_pubs))
                            
                            # OncoLive articles with comprehensive content
                            if 'onclive' in search_results and search_results['onclive'].get('articles'):
                                onclive_articles = []
                                for article in search_results['onclive']['articles']:
                                    if isinstance(article, dict):
                                        article_parts = []
                                        title = article.get('title', 'No title')
                                        article_parts.append(f"Title: {title}")
                                        
                                        # Include URL if available
                                        url = article.get('url', '')
                                        if url:
                                            article_parts.append(f"URL: {url}")
                                        
                                        # Content summary
                                        summary = article.get('content_summary', '')
                                        if summary:
                                            article_parts.append(f"Content Summary: {summary}")
                                            
                                        # Full content
                                        full_content = article.get('full_content', '')
                                        if full_content:
                                            article_parts.append(f"Full Content: {full_content}")

                                        onclive_articles.append("\n".join(article_parts))
                                
                                if onclive_articles:
                                    evidence_sections.append(f"ONCLIVE RESULTS:\n" + "\n\n".join(onclive_articles))
                            

                            # Congress abstracts
                            if 'congress_abstracts' in search_results:
                                # Handle both direct list and nested structure
                                congress_data = search_results['congress_abstracts']
                                abstracts_list = []
                                
                                if isinstance(congress_data, list):
                                    abstracts_list = congress_data
                                elif isinstance(congress_data, dict) and congress_data.get('abstracts'):
                                    abstracts_list = congress_data['abstracts']
                                
                                if abstracts_list:
                                    congress_abstract_texts = []
                                    for abstract in abstracts_list:
                                        if isinstance(abstract, dict):
                                            abstract_parts = []
                                            title = abstract.get('title', 'No title')
                                            congress = abstract.get('congress', 'N/A')
                                            abstract_parts.append(f"Congress: {congress} | Title: {title}")
                                            
                                            content = abstract.get('content', '')
                                            if content:
                                                abstract_parts.append(f"Additional Content: {content}")
                                            
                                            congress_abstract_texts.append("\n".join(abstract_parts))
                                        else:
                                            congress_abstract_texts.append(str(abstract))
                                    
                                    if congress_abstract_texts:
                                        evidence_sections.append(f"CONGRESS ABSTRACTS:\n" + "\n\n".join(congress_abstract_texts))
                    
                    # 3. Results data if available
                    if 'results' in trial and trial['results']:
                        # Format results to be more readable
                        results_text = json.dumps(trial['results'], indent=2)
                        evidence_sections.append(f"TRIAL RESULTS:\n{results_text}")
                    
                    # 4. Has posted results flag
                    if 'has_posted_results' in trial:
                        evidence_sections.append(f"HAS POSTED RESULTS: {trial['has_posted_results']}")
                    
                    # Add evidence sections to trial text
                    if evidence_sections:
                        trial_text.append("\nEVIDENCE AND PUBLICATIONS:")
                        trial_text.extend(evidence_sections)
                    else:
                        trial_text.append("\nEVIDENCE AND PUBLICATIONS: No published results or evidence found")
                    
                    formatted_trials.append("\n".join(trial_text))
                
                return "\n\n" + "="*80 + "\n\n".join(formatted_trials)
            else:
                return str(trials_data)
                
        except Exception as e:
            logger.warning(f"⚠️ Error formatting trials evidence: {e}")
            return str(trials_data)

    def _count_input_trials(self, trials_data: Dict[str, Any]) -> int:
        """Count the number of trials in the input data"""
        try:
            if isinstance(trials_data, dict):
                # Handle different trial data structures
                if 'trials' in trials_data:
                    return len(trials_data['trials'])
                elif 'relevant_trials' in trials_data:
                    return len(trials_data['relevant_trials'])
                else:
                    # Try to find trials in other common keys
                    for key in trials_data:
                        if isinstance(trials_data[key], list) and len(trials_data[key]) > 0:
                            # Check if first item looks like a trial (has nct_id)
                            if isinstance(trials_data[key][0], dict) and 'nct_id' in trials_data[key][0]:
                                return len(trials_data[key])
            
            # Fallback: Try to count NCT numbers in the text
            nct_pattern = r'NCT\d{8}'
            nct_matches = re.findall(nct_pattern, str(trials_data), re.IGNORECASE)
            return len(set(nct_match.upper() for nct_match in nct_matches))
        except Exception as e:
            logger.warning(f"⚠️ Error counting input trials: {e}")
            return 0

    def _create_analysis_prompt(self, patient_assessment: str, trials_evidence: str) -> str:
        """Create the complete analysis prompt"""
        return self.prompt_template.format(
            patient_assessment=patient_assessment,
            clinical_trials_evidence=trials_evidence
        )

    def _call_llm_for_analysis(self, prompt: str) -> str:
        """Call the LLM API for trial analysis"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "HTTP-Referer": "https://github.com/piakoller/prism",
                "X-Title": "Trial Analyzer"
            }
            
            # No max_tokens - allow unlimited output length for all models
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            # Add model-specific parameters for Gemini models
            if "gemini" in self.model.lower():
                data.update({
                    "stream": False,
                    "top_p": 0.95,
                })
            
            logger.info(f"🔗 Calling LLM API for trial analysis...")
            response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
            
            if response.status_code != 200:
                raise Exception(f"LLM API call failed: {response.status_code} - {response.text}")
            
            # Parse JSON safely; some upstream errors might return non-JSON bodies
            try:
                result = response.json()
            except Exception as je:
                snippet = response.text[:500] if response.text else "<empty body>"
                logger.error(
                    f"❌ Failed to parse LLM response as JSON: {je}. Status: {response.status_code}. Body starts with: {snippet}"
                )
                raise Exception(f"Invalid JSON response from LLM provider (status {response.status_code})")
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"❌ LLM API call failed: {e}")
            raise

    def _parse_analysis_response(self, response: str, expected_trial_count: int) -> Dict[str, Any]:
        """Parse and validate the trial analysis response"""
        try:
            logger.info(f"🔍 Parsing trial analysis response (length: {len(response)} chars)")
            
            # Extract individual trial analyses
            trial_analyses = self._extract_trial_analyses(response)
            
            # Extract summary information
            summary_info = self._extract_summary_info(response)
            
            # Validate completeness
            validation_warnings = self._validate_analysis_completeness(
                trial_analyses, expected_trial_count, summary_info
            )
            
            # Structure the result
            result = {
                "status": "success",
                "analysis_complete": len(validation_warnings) == 0,
                "trials_analyzed": len(trial_analyses),
                "trials_included": len([t for t in trial_analyses if t.get('recommendation') == 'INCLUDE']),
                "trials_excluded": len([t for t in trial_analyses if t.get('recommendation') == 'EXCLUDE']),
                "expected_trial_count": expected_trial_count,
                "trial_analyses": trial_analyses,
                "summary": summary_info,
                "validation_warnings": validation_warnings,
                "raw_response": response
            }
            
            if validation_warnings:
                logger.warning(f"⚠️ Analysis validation warnings: {validation_warnings}")
            else:
                logger.info("✅ Trial analysis validation successful")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing analysis response: {e}")
            return {
                "status": "error", 
                "error_message": str(e),
                "analysis_complete": False,
                "trials_analyzed": 0,
                "trials_included": 0,
                "trials_excluded": 0,
                "raw_response": response
            }

    def _extract_trial_analyses(self, response: str) -> List[Dict[str, Any]]:
        """Extract individual trial analyses from the response"""
        trial_analyses = []
        
        try:
            # Split response into trial analysis blocks
            trial_blocks = response.split('<trial_analysis>')
            
            for block in trial_blocks[1:]:  # Skip first element (before first <trial_analysis>)
                # Find the end of this trial block
                end_marker = block.find('</trial_analysis>')
                if end_marker != -1:
                    trial_content = block[:end_marker].strip()
                else:
                    trial_content = block.split('---')[0].strip()  # Use --- as fallback separator
                
                # Parse this trial analysis
                trial_analysis = self._parse_single_trial_analysis(trial_content)
                if trial_analysis:
                    trial_analyses.append(trial_analysis)
            
            logger.info(f"📊 Extracted {len(trial_analyses)} trial analyses")
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting trial analyses: {e}")
        
        return trial_analyses

    def _parse_single_trial_analysis(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse a single trial analysis block based on v2 prompt format"""
        try:
            analysis = {}
            
            # Extract NCT number
            nct_match = re.search(r'\*\*NCT NUMBER:\*\*\s*([A-Z]*\d{8})', content, re.IGNORECASE)
            if nct_match:
                analysis['nct_number'] = nct_match.group(1).upper()
            
            # Extract trial name
            name_match = re.search(r'\*\*TRIAL NAME:\*\*\s*(.+?)(?=\n\*\*|\n---|\n<|$)', content, re.IGNORECASE | re.DOTALL)
            if name_match:
                analysis['trial_name'] = name_match.group(1).strip()
            
            # Extract decision (INCLUDE/EXCLUDE)
            decision_match = re.search(r'\*\*DECISION:\*\*\s*(INCLUDE|EXCLUDE)', content, re.IGNORECASE)
            if decision_match:
                analysis['decision'] = decision_match.group(1).upper()
                # For backward compatibility with 'recommendation'
                analysis['recommendation'] = analysis['decision']

            # Extract rationale for the decision
            rationale_match = re.search(r'\*\*RATIONALE:\*\*\s*(.+?)(?:\n---|</trial_analysis>|$)', content, re.DOTALL | re.IGNORECASE)
            if rationale_match:
                analysis['rationale'] = rationale_match.group(1).strip()
            
            # Only return analysis if we have the essential fields
            if 'nct_number' in analysis and 'decision' in analysis:
                return analysis
            else:
                logger.warning(f"⚠️ Incomplete trial analysis: missing NCT number or decision. Content: {content[:100]}...")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error parsing single trial analysis: {e}")
            return None

    def _extract_summary_info(self, response: str) -> Dict[str, Any]:
        """Extract summary information from the response"""
        summary = {}
        
        try:
            # Look for summary section
            summary_match = re.search(r'<summary>\s*(.+?)\s*</summary>', response, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary_content = summary_match.group(1)
                
                # Extract counts
                total_match = re.search(r'\*\*TOTAL TRIALS ANALYZED:\s*(\d+)\*\*', summary_content, re.IGNORECASE)
                if total_match:
                    summary['total_analyzed'] = int(total_match.group(1))
                
                included_match = re.search(r'\*\*TRIALS TO INCLUDE:\s*(\d+)\*\*', summary_content, re.IGNORECASE)
                if included_match:
                    summary['recommended_inclusion'] = int(included_match.group(1))
                
                excluded_match = re.search(r'\*\*TRIALS TO EXCLUDE:\s*(\d+)\*\*', summary_content, re.IGNORECASE)
                if excluded_match:
                    summary['recommended_exclusion'] = int(excluded_match.group(1))
        
        except Exception as e:
            logger.warning(f"⚠️ Error extracting summary info: {e}")
        
        return summary

    def _validate_analysis_completeness(self, trial_analyses: List[Dict[str, Any]], 
                                       expected_count: int, summary: Dict[str, Any]) -> List[str]:
        """Validate that the analysis is complete and accurate"""
        warnings = []
        
        try:
            # Check trial count
            analyzed_count = len(trial_analyses)
            if analyzed_count != expected_count:
                warnings.append(f"Trial count mismatch: analyzed {analyzed_count}, expected {expected_count}")
            
            # Check for duplicate NCT numbers
            nct_numbers = [t.get('nct_number') for t in trial_analyses if t.get('nct_number')]
            unique_ncts = set(nct_numbers)
            if len(nct_numbers) != len(unique_ncts):
                duplicates = [nct for nct in unique_ncts if nct_numbers.count(nct) > 1]
                warnings.append(f"Duplicate NCT numbers found: {duplicates}")
            
            # Check for invalid NCT numbers
            invalid_ncts = [nct for nct in nct_numbers if not (isinstance(nct, str) and re.match(r'^NCT\d{8}$', nct))]
            if invalid_ncts:
                warnings.append(f"Invalid NCT numbers: {invalid_ncts}")
            
            # Check summary consistency
            if summary:
                if 'total_analyzed' in summary and summary['total_analyzed'] != analyzed_count:
                    warnings.append(f"Summary count mismatch: summary says {summary['total_analyzed']}, found {analyzed_count}")
            
            # Check for missing recommendations
            missing_rec = [t.get('nct_number', 'Unknown') for t in trial_analyses 
                          if not t.get('recommendation') or t.get('recommendation') not in ['INCLUDE', 'EXCLUDE']]
            if missing_rec:
                warnings.append(f"Missing or invalid recommendations for: {missing_rec}")
        
        except Exception as e:
            warnings.append(f"Validation error: {e}")
        
        return warnings

    def _save_analysis_results(self, analysis_result: Dict[str, Any], patient_id: str, output_dir: str):
        """Save trial analysis results to files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save structured analysis results
            analysis_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis.json")
            
            # Save raw response
            raw_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_raw_response.txt")
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"=== TRIAL ANALYSIS RAW RESPONSE ===\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Model: {self.model}\n")
                f.write("=" * 50 + "\n\n")
                f.write(analysis_result.get('raw_response', ''))
            
            # Save human-readable summary
            summary_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(self._create_analysis_summary(analysis_result, patient_id))
            
            # Update metadata with file paths
            if 'metadata' not in analysis_result:
                analysis_result['metadata'] = {}
            analysis_result['metadata'].update({
                'results_file': analysis_file,
                'response_file': raw_file,
                'summary_file': summary_file
            })
            
            # Save the updated analysis results with metadata
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Trial analysis results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error saving analysis results: {e}")

    def _create_analysis_summary(self, analysis_result: Dict[str, Any], patient_id: str) -> str:
        """Create a human-readable summary of the trial analysis"""
        summary_lines = []
        
        summary_lines.append(f"TRIAL ANALYSIS SUMMARY - PATIENT {patient_id}")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Analysis Status: {analysis_result.get('status', 'Unknown')}")
        summary_lines.append(f"Analysis Complete: {analysis_result.get('analysis_complete', False)}")
        summary_lines.append(f"Trials Analyzed: {analysis_result.get('trials_analyzed', 0)}")
        summary_lines.append(f"Trials to Include: {analysis_result.get('trials_included', 0)}")
        summary_lines.append(f"Trials to Exclude: {analysis_result.get('trials_excluded', 0)}")
        summary_lines.append("")
        
        # Validation warnings
        warnings = analysis_result.get('validation_warnings', [])
        if warnings:
            summary_lines.append("VALIDATION WARNINGS:")
            for warning in warnings:
                summary_lines.append(f"  ⚠️ {warning}")
            summary_lines.append("")
        
        # Trials recommended for inclusion
        trial_analyses = analysis_result.get('trial_analyses', [])
        included_trials = [t for t in trial_analyses if t.get('decision') == 'INCLUDE']
        
        if included_trials:
            summary_lines.append("TRIALS TO INCLUDE IN RECOMMENDATION:")
            for trial in included_trials:
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                rationale = trial.get('rationale', 'No rationale provided.')
                summary_lines.append(f"  ✅ {nct}: {name}")
                summary_lines.append(f"     Rationale: {rationale}")
            summary_lines.append("")
        
        # Trials recommended for exclusion
        excluded_trials = [t for t in trial_analyses if t.get('decision') == 'EXCLUDE']
        
        if excluded_trials:
            summary_lines.append("TRIALS TO EXCLUDE FROM RECOMMENDATION:")
            for trial in excluded_trials:
                nct = trial.get('nct_number', 'Unknown')
                name = trial.get('trial_name', 'Unknown')
                rationale = trial.get('rationale', 'No rationale provided.')
                summary_lines.append(f"  ❌ {nct}: {name}")
                summary_lines.append(f"     Rationale: {rationale}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def clear_trial_analysis_cache(self, patient_id: str, output_dir: str = ".") -> bool:
        """
        Clear cached trial analysis for a specific patient.
        Clears both old per-output_dir cache file and model-aware content cache entries.
        """
        cleared = False
        # Clear legacy per-output cache file
        analysis_file = os.path.join(output_dir, f"patient_{patient_id}_trial_analysis.json")
        if os.path.exists(analysis_file):
            try:
                os.remove(analysis_file)
                logger.info(f"🗑️ Cleared legacy cached trial analysis for patient {patient_id}")
                cleared = True
            except Exception as e:
                logger.error(f"❌ Failed to clear legacy trial analysis cache for patient {patient_id}: {e}")
        # Model-aware cache: remove any files that start with patient key
        try:
            if hasattr(self, 'cache_dir') and self.cache_dir.exists():
                for fp in self.cache_dir.glob(f"patient_{patient_id}_*_analysis.json"):
                    try:
                        fp.unlink()
                        cleared = True
                    except Exception as e:
                        logger.warning(f"⚠️ Could not delete cache file {fp}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Error scanning cache dir for cleanup: {e}")
        if not cleared:
            logger.info(f"ℹ️ No trial analysis cache found for patient {patient_id}")
        return cleared
