#!/usr/bin/env python3
"""
Clinical Trials Study Collector

This script queries the ClinicalTrials.gov API to collect all studies that match
the filter criteria (studies with both posted results AND publications) without
LLM evaluation. It saves all matching studies for further analysis.
"""

import sys
import os
import json
import time
import logging
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalTrialsAPI:
    """API wrapper for ClinicalTrials.gov"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize the API wrapper.
        
        Args:
            rate_limit_delay: Delay between API calls in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a request to the ClinicalTrials.gov API.
        
        Args:
            endpoint: API endpoint (e.g., 'studies')
            params: Query parameters
            
        Returns:
            JSON response data or None if failed
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

@dataclass
class StudyRecord:
    """Represents a clinical trial study record."""
    nct_id: str
    title: str
    status: str
    phase: str
    condition: str
    intervention: str
    brief_summary: str
    detailed_description: str
    eligibility_criteria: str
    start_date: str
    completion_date: str
    primary_outcome: str
    secondary_outcome: str
    sponsor: str
    locations: List[str]
    url: str
    has_posted_results: bool
    has_publications: bool
    publications_count: int
    publications: List[Dict]  # Store full publication details
    search_terms_matched: List[str]

class StudyCollector:
    """Collects all studies matching filter criteria without LLM evaluation."""
    def __init__(self, api: ClinicalTrialsAPI, output_dir: Path):
        self.api = api
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
               
    def collect_net_studies_simple(self) -> Tuple[List[StudyRecord], str]:
        """
        Simple API call to collect Neuroendocrine Tumor studies, excluding Phase 1.
        
        Returns:
            Tuple of (all_studies, query_used)
        """
        all_studies = []
        
        # Query for NET studies with results posted from 2020 onwards, excluding Phase 1
        # AREA[HasResults]false
        # query = 'AREA[Condition](Neuroendocrine Tumors OR Carcinoma Neuroendocrine OR Carcinoid Tumor) AND NOT (AREA[Phase]Phase 1) AND AREA[OverallStatus](ACTIVE_NOT_RECRUITING OR COMPLETED) AND (AREA[HasResults]false) AND AREA[PrimaryCompletionDate] RANGE[2020-01-01,MAX]'
        
        # AREA[HasResults]true
        query = 'AREA[Condition](Neuroendocrine Tumors OR Carcinoma Neuroendocrine OR Carcinoid Tumor) AND NOT (AREA[Phase]Phase 1) AND AREA[OverallStatus](ACTIVE_NOT_RECRUITING OR COMPLETED) AND (AREA[HasResults]true)'

        logger.info(f"Using simple API call with query: {query}")
        
        try:
            url = f"{self.api.BASE_URL}/studies"
            params = {
                "format": "json",
                "pageSize": 1000,
                "query.term": query
            }
            
            logger.info(f"Making API call to: {url}")
            logger.info(f"Parameters: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"Full URL: {response.url}")
            
            data = response.json()
            studies = data.get("studies", [])
            
            logger.info(f"Found {len(studies)} studies from API")
            
            for study in studies:
                study_record = self._process_study(study, "simple_net_search")
                if study_record:
                    all_studies.append(study_record)
                    logger.debug(f"Added study: {study_record.nct_id} - {study_record.title[:50]}...")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return [], query
        except Exception as e:
            logger.error(f"Error in simple NET search: {e}")
            return [], query
        
        logger.info(f"Collected {len(all_studies)} studies using simple API call")
        
        total_from_api = len(studies) if 'studies' in locals() else 0
        studies_with_publications = sum(1 for study in all_studies if study.publications_count > 0)
        studies_without_publications = sum(1 for study in all_studies if study.publications_count == 0)
        
        if studies_with_publications > 0:
            logger.info(f"Studies with publications: {studies_with_publications}")
        if studies_without_publications > 0:
            logger.info(f"Studies without publications: {studies_without_publications}")
        
        return all_studies, query
    
    def _process_study(self, study: Dict, search_term: str) -> Optional[StudyRecord]:
        """Process a single study and convert to StudyRecord."""
        try:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            interventions_module = protocol.get("armsInterventionsModule", {})
            eligibility_module = protocol.get("eligibilityModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            contacts_module = protocol.get("contactsLocationsModule", {})
            description_module = protocol.get("descriptionModule", {})
            references_module = protocol.get("referencesModule", {})
            
            nct_id = identification.get("nctId", "")
            if not nct_id:
                return None
            
            # Check publications
            has_publications = False
            publications_count = 0
            publications = []
            
            if references_module:
                references = references_module.get("references", [])
                publications_count = len(references)
                has_publications = publications_count > 0
                
                for ref in references:
                    pub_info = {
                        'citation': ref.get('citation', ''),
                        'pmid': ref.get('pmid', ''),
                        'type': ref.get('type', '')
                    }
                    publications.append(pub_info)
            
            # Check posted results
            has_posted_results = study.get("hasResults", False)
            
            locations = []
            location_facilities = contacts_module.get("locations", [])
            for loc in location_facilities:
                facility = loc.get("facility", "")
                city = loc.get("city", "")
                country = loc.get("country", "")
                if facility:
                    locations.append(f"{facility}, {city}, {country}")
            
            study_record = StudyRecord(
                nct_id=nct_id,
                title=identification.get("briefTitle", ""),
                status=status_module.get("overallStatus", ""),
                phase=design_module.get("phases", ["N/A"])[0] if design_module.get("phases") else "N/A",
                condition="; ".join(conditions_module.get("conditions", [])),
                intervention="; ".join([
                    interv.get("name", "") for interv in interventions_module.get("interventions", [])
                ]),
                brief_summary=description_module.get("briefSummary", ""),
                detailed_description=description_module.get("detailedDescription", ""),
                eligibility_criteria=eligibility_module.get("eligibilityCriteria", ""),
                start_date=status_module.get("startDateStruct", {}).get("date", ""),
                completion_date=status_module.get("completionDateStruct", {}).get("date", ""),
                primary_outcome="; ".join([
                    outcome.get("measure", "") for outcome in outcomes_module.get("primaryOutcomes", [])
                ]),
                secondary_outcome="; ".join([
                    outcome.get("measure", "") for outcome in outcomes_module.get("secondaryOutcomes", [])
                ]),
                sponsor=sponsor_module.get("leadSponsor", {}).get("name", ""),
                locations=locations,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                has_posted_results=has_posted_results,
                has_publications=has_publications,
                publications_count=publications_count,
                publications=publications,
                search_terms_matched=[search_term]
            )
            
            return study_record
            
        except Exception as e:
            logger.error(f"Error processing study: {e}")
            return None
    
    def save_studies(self, studies: List[StudyRecord], filename_prefix: str = "./collected_studies", query: str = None):
        """Save collected studies to JSON and text files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        query_suffix = ""
        if query:
            query_safe = re.sub(r'[^\w\s-]', '', query.replace('AREA[', '').replace(']', '_').replace('(', '').replace(')', ''))
            query_safe = re.sub(r'[-\s]+', '_', query_safe).strip('_')[:50]
            query_suffix = f"_query_{query_safe}"
        
        studies_with_publications = [study for study in studies if study.publications_count > 0]
        studies_without_publications = [study for study in studies if study.publications_count == 0]
        
        json_file = self.output_dir / f"{filename_prefix}{query_suffix}_{timestamp}.json"
        studies_data = []
        
        for study in studies:
            study_dict = asdict(study)
            # Add publication status flag
            study_dict['has_publications_flag'] = study.publications_count > 0
            studies_data.append(study_dict)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "collection_date": datetime.now().isoformat(),
                "search_query": query,
                "total_studies": len(studies),
                "studies_with_publications": len(studies_with_publications),
                "studies_without_publications": len(studies_without_publications),
                "studies": studies_data
            }, f, indent=2, ensure_ascii=False)
        
        txt_file = self.output_dir / f"{filename_prefix}{query_suffix}_{timestamp}.txt"
        self._generate_text_report(studies, txt_file, query)
        
        stats_file = self.output_dir / f"{filename_prefix}{query_suffix}_stats_{timestamp}.json"
        
        logger.info(f"Saved {len(studies)} studies to:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Text: {txt_file}")
        logger.info(f"  Stats: {stats_file}")
    
    def _generate_text_report(self, studies: List[StudyRecord], output_file: Path, query: str = None):
        """Generate a human-readable text report of collected studies."""
        studies_with_publications = [study for study in studies if study.publications_count > 0]
        studies_without_publications = [study for study in studies if study.publications_count == 0]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COLLECTED CLINICAL TRIALS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Studies: {len(studies)}\n")
            f.write(f"Studies with Publications: {len(studies_with_publications)}\n")
            f.write(f"Studies without Publications: {len(studies_without_publications)}\n")
            if query:
                f.write(f"Search Query: {query}\n")
            f.write(f"Filter Criteria: Neuroendocrine tumor studies, excluding Phase 1\n")
            f.write("-"*80 + "\n\n")
            
            for i, study in enumerate(studies, 1):
                f.write(f"{i}. {study.title}\n")
                f.write(f"   NCT ID: {study.nct_id}\n")
                f.write(f"   Status: {study.status}\n")
                f.write(f"   Phase: {study.phase}\n")
                f.write(f"   Condition: {study.condition}\n")
                f.write(f"   Intervention: {study.intervention}\n")
                f.write(f"   Sponsor: {study.sponsor}\n")
                f.write(f"   Start Date: {study.start_date}\n")
                f.write(f"   Completion Date: {study.completion_date}\n")
                f.write(f"   Has Posted Results: {study.has_posted_results}\n")
                f.write(f"   Publications Count: {study.publications_count}\n")
                f.write(f"   Has Publications: {'Yes' if study.publications_count > 0 else 'No'}\n")
                f.write(f"   Search Terms Matched: {', '.join(study.search_terms_matched)}\n")
                f.write(f"   URL: {study.url}\n")
                
                if study.brief_summary:
                    summary = study.brief_summary[:300] + "..." if len(study.brief_summary) > 300 else study.brief_summary
                    f.write(f"   Summary: {summary}\n")
                
                if study.locations:
                    f.write(f"   Locations: {'; '.join(study.locations[:3])}\n")
                    if len(study.locations) > 3:
                        f.write(f"              ... and {len(study.locations) - 3} more locations\n")
                
                f.write("-"*40 + "\n\n")

    
def main():
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "collected_studies"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    collector = StudyCollector(api, OUTPUT_DIR)
    
    logger.info("Starting NET study collection")
    logger.info('Query will be dynamically set in collect_net_studies_simple method')
    try:
        studies, query_used = collector.collect_net_studies_simple()
        
        if studies:
            studies_with_publications = [study for study in studies if study.publications_count > 0]
            studies_without_publications = [study for study in studies if study.publications_count == 0]
            
            logger.info(f"Collected {len(studies)} studies total")
            logger.info(f"Studies with publications: {len(studies_with_publications)}")
            logger.info(f"Studies without publications: {len(studies_without_publications)}")
            
            collector.save_studies(studies, "net_studies_all", query_used)
            
            print(f"\n" + "="*80)
            print("STUDY COLLECTION RESULTS")
            print("="*80)
            print(f"Total studies collected: {len(studies)}")
            print(f"Studies with publications: {len(studies_with_publications)}")
            print(f"Studies without publications: {len(studies_without_publications)}")
            
        if studies:
            logger.info("Study collection completed successfully")
            print("\n" + "="*80)
            print("NET STUDY COLLECTION SUMMARY")
            print("="*80)
            print(f"Total studies: {len(studies)}")
            print(f"Studies with publications: {len(studies_with_publications)}")
            print(f"Studies without publications: {len(studies_without_publications)}")
            print(f'Query used: {query_used}')
            print(f"Results saved to: {OUTPUT_DIR}")
            
            if studies:
                status_counts = {}
                phase_counts = {}
                results_counts = {"has_results": 0, "no_results": 0}
                publication_counts = {"with_publications": 0, "without_publications": 0}
                
                for study in studies:
                    status_counts[study.status] = status_counts.get(study.status, 0) + 1
                    phase_counts[study.phase] = phase_counts.get(study.phase, 0) + 1
                    if study.has_posted_results:
                        results_counts["has_results"] += 1
                    else:
                        results_counts["no_results"] += 1
                    
                    if study.publications_count > 0:
                        publication_counts["with_publications"] += 1
                    else:
                        publication_counts["without_publications"] += 1
                
                print(f"\nStatus distribution:")
                for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {status}: {count}")
                
                print(f"\nPhase distribution:")
                for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {phase}: {count}")
                    
                print(f"\nResults availability:")
                print(f"  Studies with posted results: {results_counts['has_results']}")
                print(f"  Studies without posted results: {results_counts['no_results']}")
                
                print(f"\nPublication availability:")
                print(f"  Studies with publications: {publication_counts['with_publications']}")
                print(f"  Studies without publications: {publication_counts['without_publications']}")
            
            print("="*80)
            
        else:
            print("No studies found matching the criteria.")
            
    except Exception as e:
        logger.error(f"Error during study collection: {e}")
        return

if __name__ == "__main__":
    main()
