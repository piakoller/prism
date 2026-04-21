#!/usr/bin/env python3
"""
Guidelines Matcher Module for Agentic Assessment

This module provides functionality to match patients with relevant medical guidelines
using LLM evaluation. It can be used standalone or integrated with the agentic workflow.
"""

import os
import json
import requests
import time
import logging
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL, USE_PRODUCTION_MODEL, MODEL_CONTEXT_LIMITS, GUIDELINES_CACHE_DIR

logger = logging.getLogger(__name__)

@dataclass
class GuidelineMatch:
    """Represents a guideline match for a patient."""
    guideline_file: str
    guideline_title: str
    guideline_content: str
    relevance_score: float
    relevance_reason: str
    content_length: int
    
    @property
    def guideline_name(self) -> str:
        """Extract a readable name from the filename."""
        name = Path(self.guideline_file).stem
        name = name.replace('J Neuroendocrinology - ', '')
        name = name.replace(' - ', ' | ')
        return name

class GuidelinesAssessor:
    """Matches patients with relevant medical guidelines using LLM evaluation."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, prompt_file: Optional[str] = None):
        """
        Initialize the Guidelines Assessor
        
        Args:
            api_key: OpenRouter API key (defaults to config.OPENROUTER_API_KEY)
            model: LLM model to use for guideline matching (defaults to config.OPENROUTER_MODEL)
            prompt_file: Path to the guideline matching prompt file (defaults to standard location)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        if prompt_file is None:
            base_dir = Path(__file__).parent.parent.parent
            prompt_file = str(base_dir / "prompts" / "1_guideline_matching.txt")
        
        self.prompt_file = Path(prompt_file)
        self.prompt_template = self._load_prompt_template()
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please check your configuration.")
        
        logger.info(f"📋 Guidelines Assessor initialized with model: {self.model}")
        logger.info(f"📄 Using prompt file: {self.prompt_file}")
        
        self.cache_dir = Path(GUIDELINES_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, patient_data: Dict[str, Any], guidelines_dir: str) -> str:
        """Generate a cache key for a patient and guidelines directory."""
        import hashlib
        
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        patient_str = str(sorted(patient_data.items()))
        cache_input = f"{patient_id}_{guidelines_dir}_{patient_str}"
        cache_hash = hashlib.md5(cache_input.encode()).hexdigest()[:8]
        return f"patient_{patient_id}_{cache_hash}"
    
    def _get_cache_filepath(self, cache_key: str) -> Path:
        """Get the full cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}_guidelines.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load guidelines matching results from cache if available."""
        cache_file = self._get_cache_filepath(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                cached_model = cached_data.get('metadata', {}).get('model_used')
                if cached_model == self.model:
                    logger.info(f"📂 Loaded guidelines from cache: {cache_file}")
                    return cached_data
                else:
                    logger.info(f"🔄 Cache model mismatch ({cached_model} vs {self.model}), will regenerate")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save guidelines matching results to cache."""
        try:
            cache_file = self._get_cache_filepath(cache_key)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved guidelines to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to cache: {e}")
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the text file."""
        if not self.prompt_file.exists():
            error_msg = f"❌ CRITICAL ERROR: Prompt file not found: {self.prompt_file}"
            logger.error(error_msg)
            print(error_msg)
            print("Please ensure the guideline matching prompt file exists before running the guidelines assessor.")
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
    
    def load_guidelines(self, guidelines_dir: str) -> List[Dict[str, Any]]:
        """Load all markdown guidelines from the guidelines directory."""
        guidelines = []
        guidelines_path = Path(guidelines_dir)
        
        if not guidelines_path.exists():
            logger.error(f"Guidelines directory not found: {guidelines_path}")
            return guidelines
        
        enet_guidelines_path = guidelines_path / "2-0" / "ENET_Guidelines" / "mds"
        esmo_guideline_path = guidelines_path / "2-0" / "Gastroenteropancreatic Neuroendocrine Neoplasms ESMO Clinical Practice Guidelines for Diagnosis, Treatment and Follow-up.md"
        
        if enet_guidelines_path.exists():
            md_files = list(enet_guidelines_path.glob("*.md"))
            logger.info(f"Found {len(md_files)} ENET guideline files in {enet_guidelines_path}")
            
            for md_file in md_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    guideline = {
                        'file': md_file.name,
                        'title': self._extract_title_from_content(content, md_file.name),
                        'content': content,
                        'content_length': len(content),
                        'full_path': str(md_file),
                        'guideline_type': 'ENET'
                    }
                    guidelines.append(guideline)
                    logger.debug(f"Loaded ENET guideline: {md_file.name} ({len(content)} chars)")
                    
                except Exception as e:
                    logger.error(f"Error loading ENET guideline {md_file}: {e}")
                    continue
        else:
            logger.warning(f"ENET guidelines directory not found: {enet_guidelines_path}")
        
        if esmo_guideline_path.exists():
            try:
                with open(esmo_guideline_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                guideline = {
                    'file': esmo_guideline_path.name,
                    'title': self._extract_title_from_content(content, esmo_guideline_path.name),
                    'content': content,
                    'content_length': len(content),
                    'full_path': str(esmo_guideline_path),
                    'guideline_type': 'ESMO'
                }
                guidelines.append(guideline)
                logger.info(f"Loaded ESMO guideline: {esmo_guideline_path.name} ({len(content)} chars)")
                
            except Exception as e:
                logger.error(f"Error loading ESMO guideline {esmo_guideline_path}: {e}")
        else:
            logger.warning(f"ESMO guideline not found: {esmo_guideline_path}")
        
        return guidelines
    
    def _extract_title_from_content(self, content: str, filename: str) -> str:
        """Extract title from markdown content or use filename as fallback."""
        lines = content.split('\n')
        for line in lines[:10]:
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback to cleaned filename
        title = Path(filename).stem
        title = title.replace('J Neuroendocrinology - ', '')
        title = title.replace(' - ', ' | ')
        return title
    
    def evaluate_guideline_relevance(self, patient_data: Dict[str, Any], 
                                   guideline_info: Dict[str, Any]) -> Tuple[float, str]:
        """
        Use LLM to evaluate if a guideline is relevant for a patient.
        
        Returns:
            Tuple of (relevance_score, explanation)
        """
        case_id = patient_data.get('ID', patient_data.get('id', 'Unknown'))
        clinical_information = self._extract_clinical_info(patient_data)
        question = patient_data.get('question_for_tumorboard', 
                                  patient_data.get('question', 'Clinical guidance needed'))
        
        guideline_title = guideline_info.get('title', 'Unknown Guideline')
        guideline_content = guideline_info.get('content', 'No content available')
        
        prompt = self._create_guideline_evaluation_prompt(
            case_id, clinical_information, question, guideline_title, guideline_content
        )
        
        try:
            llm_response = self._call_llm(prompt)
            
            if "ERROR:" in llm_response:
                logger.error(f"LLM evaluation failed: {llm_response}")
                return 0.0, "LLM evaluation failed"
            
            return self._parse_llm_response(llm_response, guideline_title)
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"
    
    def _extract_clinical_info(self, patient_data: Dict[str, Any]) -> str:
        """Extract clinical information from patient data in a standardized format."""
        clinical_fields = [
            'clinical_information', 'Primary Site', 'Histology', 'Grade', 'Stage',
            'Ki-67', 'Chromogranin A', 'Previous Treatments', 'Current Status',
            'Performance Status', 'Age', 'Gender'
        ]
        
        clinical_info = []
        for field in clinical_fields:
            value = patient_data.get(field)
            if value and str(value).strip() and str(value).lower() != 'nan':
                clinical_info.append(f"{field}: {value}")
        
        return "; ".join(clinical_info) if clinical_info else "Limited clinical information available"
    
    def _create_guideline_evaluation_prompt(self, case_id: str, clinical_info: str, 
                                          question: str, guideline_title: str, 
                                          guideline_content: str) -> str:
        """Create the prompt for guideline evaluation using the loaded prompt template."""
        
        current_model = OPENROUTER_MODEL
        model_context_limit = MODEL_CONTEXT_LIMITS.get(current_model, 30000)
        
        # Production models with 1M context: use full content without truncation
        if current_model in ["google/gemini-3-pro-preview"]:
            logger.info(f"Using production model: {current_model} - NO TRUNCATION (full content included)")
        else:
            # Free models with limited context: apply truncation
            # Estimate: 1 token ≈ 4 chars; reserve space for prompt template (~5K) and patient data (~2K)
            max_guideline_chars = model_context_limit // 2
            
            logger.info(f"Using test model: {current_model}, max guideline chars: {max_guideline_chars}")
            
            if len(guideline_content) > max_guideline_chars:
                logger.warning(f"Truncating guideline content from {len(guideline_content)} to {max_guideline_chars} chars for model context limit")
                guideline_content = guideline_content[:max_guideline_chars] + "\n\n[CONTENT TRUNCATED FOR MODEL CONTEXT LIMIT]"
        
        formatted_prompt = self.prompt_template.format(
            case_id=case_id,
            clinical_information=clinical_info,
            question=question,
            guideline_title=guideline_title,
            guideline_content=guideline_content
        )
        
        # Warn if total prompt length may exceed model context (4 chars ≈ 1 token)
        max_total_chars = model_context_limit * 4
        if len(formatted_prompt) > max_total_chars:
            logger.warning(f"Prompt length {len(formatted_prompt)} chars may exceed model context limit of ~{max_total_chars} chars")
        
        return formatted_prompt
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API with error handling and retries."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/piakoller/netTubo",
            "X-Title": "Guideline Matcher"
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
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
                
                if response.status_code == 200:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0]["message"]["content"]
                        if content:
                            return content
                        logger.warning("Empty content in API response")
                else:
                    logger.warning(f"API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"LLM API request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return "ERROR: LLM API call failed after all retries"
    
    def _parse_llm_response(self, response: str, guideline_title: str) -> Tuple[float, str]:
        """Parse the LLM response to extract relevance score and explanation."""
        relevance_score = 0.0
        explanation = "No explanation provided"
        
        try:
            # Try to parse as JSON first
            parsed_response = self._extract_json_from_response(response)
            
            if parsed_response:
                relevant = parsed_response.get("relevant", "").upper()
                if "YES" in relevant:
                    relevance_score = 1.0
                elif "NO" in relevant:
                    relevance_score = 0.0
                    
                explanation = parsed_response.get("explanation", "No explanation provided")
                logger.debug(f"Successfully parsed JSON response for {guideline_title}: relevant={relevant}")
                
            else:
                # Fallback to text parsing
                if re.search(r'\bYES\b', response.upper()):
                    relevance_score = 1.0
                elif re.search(r'\bNO\b', response.upper()):
                    relevance_score = 0.0
                
                explanation = self._clean_explanation(response)
                
        except Exception as e:
            logger.debug(f"Response parsing failed for guideline {guideline_title}: {e}")
            explanation = self._clean_explanation(response)
        
        return relevance_score, explanation
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        try:
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end+1]
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
            
            # Try to extract key-value pairs manually
            relevant_match = re.search(r'"relevant"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', response, re.IGNORECASE | re.DOTALL)
            
            if relevant_match or explanation_match:
                result = {}
                if relevant_match:
                    result["relevant"] = relevant_match.group(1)
                if explanation_match:
                    result["explanation"] = explanation_match.group(1)
                return result
                
        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")
        
        return {}
    
    def _clean_explanation(self, text: str) -> str:
        """Clean and format explanation text."""
        if not text:
            return text
            
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        
        # Remove decision markers
        text = re.sub(r'DECISION:\s*(YES|NO)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'REASONING:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'EXPLANATION:\s*', '', text, flags=re.IGNORECASE)
        
        # Clean up spacing
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def find_relevant_guidelines(self, patient_data: Dict[str, Any], 
                               guidelines_dir: str,
                               min_relevance_score: float = 1.0,
                               max_guidelines: int = 10,
                               output_dir: str = ".") -> Dict[str, Any]:
        """
        Find relevant guidelines for a patient with model-specific caching.
        
        Args:
            patient_data: Patient information dictionary
            guidelines_dir: Directory containing guideline files
            min_relevance_score: Minimum score to include guideline (1.0 = relevant only)
            max_guidelines: Maximum number of guidelines to return
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing guidelines matching results
        """
        patient_id = patient_data.get('ID', patient_data.get('id', 'unknown'))
        logger.info(f"📋 Finding relevant guidelines for patient {patient_id}")
        
        # Try to load from cache first
        cache_key = self._get_cache_key(patient_data, guidelines_dir)
        cached_result = self._load_from_cache(cache_key)
        
        if cached_result is not None:
            logger.info(f"✅ Using cached guidelines for patient {patient_id} with model {self.model}")
            
            # Still save to output directory for current run
            output_file = os.path.join(output_dir, f"patient_{patient_id}_guidelines.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cached_result, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Results copied to: {output_file}")
            
            return cached_result
        
        # Load guidelines
        guidelines = self.load_guidelines(guidelines_dir)
        if not guidelines:
            logger.error("No guidelines loaded")
            return {
                'patient_id': patient_id,
                'guidelines_found': 0,
                'relevant_guidelines': [],
                'error': 'No guidelines loaded'
            }
        
        matches = []
        logger.info(f"Evaluating {len(guidelines)} guidelines for patient {patient_id}")
        
        for i, guideline in enumerate(guidelines):
            try:
                guideline_name = guideline['title']
                guideline_type = guideline.get('guideline_type', 'Unknown')
                logger.info(f"Evaluating guideline {i+1}/{len(guidelines)}: {guideline_name} ({guideline_type})")
                
                # Always include ESMO guideline without LLM evaluation
                if guideline_type == 'ESMO':
                    match = GuidelineMatch(
                        guideline_file=guideline['file'],
                        guideline_title=guideline['title'],
                        guideline_content=guideline['content'],
                        relevance_score=1.0,  # Always relevant
                        relevance_reason="ESMO Clinical Practice Guidelines are always included as the primary reference for gastroenteropancreatic neuroendocrine neoplasms management.",
                        content_length=guideline['content_length']
                    )
                    matches.append(match)
                    logger.info(f"Added ESMO guideline: {match.guideline_title} (Always included)")
                    continue
                
                # Evaluate ENET guidelines with LLM
                score, reason = self.evaluate_guideline_relevance(patient_data, guideline)
                
                if score < 0 or score > 1:
                    logger.warning(f"Invalid score {score} for guideline {guideline_name}, setting to 0.0")
                    score = 0.0
                
                if not reason or reason.strip() == "":
                    reason = "No explanation provided"
                
                logger.info(f"Guideline {guideline_name} - Decision: {'YES' if score >= 1.0 else 'NO'} (Score: {score:.1f})")
                
                if score >= min_relevance_score:
                    match = GuidelineMatch(
                        guideline_file=guideline['file'],
                        guideline_title=guideline['title'],
                        guideline_content=guideline['content'],
                        relevance_score=score,
                        relevance_reason=reason,
                        content_length=guideline['content_length']
                    )
                    matches.append(match)
                    logger.info(f"Added relevant guideline: {match.guideline_title} (Score: {score:.3f})")
                
                # Add delay to avoid rate limiting (only for LLM-evaluated guidelines)
                time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error processing guideline {guideline.get('title', 'Unknown')}: {e}")
                continue
        
        # Sort by relevance score
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        matches = matches[:max_guidelines]
        
        # Filter out expert_recommendation and source to avoid data leakage
        filtered_patient_data = {k: v for k, v in patient_data.items() if k.lower() not in ['expert_recommendation', 'source']}
        
        result = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'patient_id': patient_id,
                'guidelines_evaluated': len(guidelines),
                'guidelines_directory': guidelines_dir
            },
            'patient_data': filtered_patient_data,
            'guidelines_found': len(matches),
            'relevant_guidelines': [
                {
                    'guideline_file': match.guideline_file,
                    'guideline_title': match.guideline_title,
                    'guideline_content': match.guideline_content,  # Include full content
                    'relevance_score': match.relevance_score,
                    'relevance_reason': match.relevance_reason,
                    'content_length': match.content_length
                }
                for match in matches
            ]
        }
        
        output_file = os.path.join(output_dir, f"patient_{patient_id}_guidelines.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self._save_to_cache(cache_key, result)
        
        logger.info(f"✅ Guidelines matching completed for patient {patient_id}: {len(matches)} relevant guidelines found")
        logger.info(f"📄 Results saved to: {output_file}")
        logger.info(f"💾 Results cached for model: {self.model}")
        
        return result
    
    def get_guidelines_summary(self, guidelines_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the guidelines matching results."""
        
        summary_lines = []
        summary_lines.append("=== GUIDELINES MATCHING SUMMARY ===")
        summary_lines.append(f"Patient ID: {guidelines_result.get('metadata', {}).get('patient_id', 'unknown')}")
        summary_lines.append(f"Matching Time: {guidelines_result.get('metadata', {}).get('timestamp', 'unknown')}")
        summary_lines.append(f"Guidelines Evaluated: {guidelines_result.get('metadata', {}).get('guidelines_evaluated', 'unknown')}")
        summary_lines.append(f"Relevant Guidelines Found: {guidelines_result.get('guidelines_found', 0)}")
        summary_lines.append("")
        
        relevant_guidelines = guidelines_result.get('relevant_guidelines', [])
        if relevant_guidelines:
            summary_lines.append("Relevant Guidelines:")
            for i, guideline in enumerate(relevant_guidelines):
                title = guideline.get('guideline_title', 'Unknown')
                score = guideline.get('relevance_score', 0)
                reason = guideline.get('relevance_reason', 'No reason provided')
                
                summary_lines.append(f"  {i+1}. {title} (Score: {score:.1f})")
                summary_lines.append(f"     Rationale: {reason}...")
                summary_lines.append("")
        else:
            summary_lines.append("No relevant guidelines found for this patient.")
        
        return "\n".join(summary_lines)
