#!/usr/bin/env python3
"""
Clinical Trials Study Filter

This script filters collected studies by analyzing publication status:
1. Checks if publications are listed in the study data
2. Analyzes publication content to determine if they contain actual results
3. Searches online for NCT numbers to find additional publications/abstracts
4. Removes studies without any publications or results
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
import urllib.parse

# Import the new online search module
from online_search import OnlineResultChecker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PublicationAnalysis:
    """Analysis results for study publications"""
    has_listed_publications: bool
    listed_publications_count: int
    has_results_publications: bool
    web_search_performed: bool
    additional_publications_found: int
    total_publications_found: int
    publication_sources: List[str]
    analysis_notes: str
    # Add online search results
    online_search_results: Dict
    web_results_summary: str
    external_sources_found: List[str]
    # Add detailed source overview with scraping information
    source_overview: Dict
    
@dataclass
class FilteringResult:
    """Results of the filtering decision for a study"""
    study_kept: bool
    filtering_reason: str
    has_posted_results: bool
    decision_factors: List[str]
    # Add online search decision factors
    online_evidence_found: bool
    external_publications_count: int
    search_confidence: str  # "high", "medium", "low"

class StudyFilter:
    """Filters studies based on publication availability and content analysis."""
    
    def __init__(self, rate_limit_delay: float = 2.0, save_incremental: bool = True, incremental_save_interval: int = 5):
        """
        Initialize the study filter.
        
        Args:
            rate_limit_delay: Delay between web searches in seconds
            save_incremental: Whether to save intermediate results during processing
            incremental_save_interval: How many studies to process before saving intermediate results
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.save_incremental = save_incremental
        self.incremental_save_interval = incremental_save_interval
        
        # Initialize the online search module with content scraping enabled
        self.online_checker = OnlineResultChecker(
            rate_limit_delay=rate_limit_delay,
            scrape_content=True,
            max_content_length=2000
        )
        
    def _save_incremental_results(self, processed_studies: List[Dict], current_index: int, total_studies: int, 
                                 output_dir: Path, base_filename: str, is_hasresults_false: bool = False):
        """
        Save incremental results during processing.
        
        Args:
            processed_studies: List of studies processed so far
            current_index: Current study index being processed
            total_studies: Total number of studies to process
            output_dir: Output directory for files
            base_filename: Base filename for output files
            is_hasresults_false: Whether processing HasResults-false file
        """
        if not self.save_incremental:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_hasresults_false" if is_hasresults_false else ""
            
            # Separate processed studies into kept and removed
            kept_studies = [s for s in processed_studies if s.get('filtering_result', {}).get('study_kept', False)]
            removed_studies = [s for s in processed_studies if not s.get('filtering_result', {}).get('study_kept', False)]
            
            # Calculate current statistics
            current_stats = {
                'processed_so_far': len(processed_studies),
                'total_studies': total_studies,
                'progress_percent': (len(processed_studies) / total_studies * 100) if total_studies > 0 else 0,
                'kept_studies': len(kept_studies),
                'removed_studies': len(removed_studies),
                'last_updated': datetime.now().isoformat(),
                'processing_status': 'IN_PROGRESS' if len(processed_studies) < total_studies else 'COMPLETED'
            }
            
            # Save incremental kept studies
            if kept_studies:
                incremental_kept_file = output_dir / f"{base_filename}{suffix}_INCREMENTAL_kept.json"
                
                incremental_data = {
                    "processing_status": current_stats['processing_status'],
                    "last_updated": current_stats['last_updated'],
                    "progress": f"{current_stats['processed_so_far']}/{current_stats['total_studies']} ({current_stats['progress_percent']:.1f}%)",
                    "filter_date": datetime.now().isoformat(),
                    "filter_version": "2.1_incremental",
                    "special_processing": "HasResults-false enhanced filtering" if is_hasresults_false else "Standard filtering",
                    "incremental_results": {
                        "total_processed": current_stats['processed_so_far'],
                        "studies_kept": current_stats['kept_studies'],
                        "studies_removed": current_stats['removed_studies']
                    },
                    "studies": kept_studies
                }
                
                with open(incremental_kept_file, 'w', encoding='utf-8') as f:
                    json.dump(incremental_data, f, indent=2, ensure_ascii=False)
            
            # Save incremental progress report
            progress_file = output_dir / f"{base_filename}{suffix}_INCREMENTAL_progress.txt"
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("INCREMENTAL FILTERING PROGRESS REPORT\n")
                if is_hasresults_false:
                    f.write("SPECIAL PROCESSING: HasResults-false Dataset\n")
                f.write("="*80 + "\n")
                f.write(f"Last Updated: {current_stats['last_updated']}\n")
                f.write(f"Status: {current_stats['processing_status']}\n")
                f.write(f"Progress: {current_stats['processed_so_far']}/{current_stats['total_studies']} ({current_stats['progress_percent']:.1f}%)\n")
                f.write(f"Studies Kept: {current_stats['kept_studies']}\n")
                f.write(f"Studies Removed: {current_stats['removed_studies']}\n")
                f.write("-"*80 + "\n\n")
                
                if kept_studies:
                    f.write(f"RECENTLY KEPT STUDIES (Last 5):\n")
                    f.write("="*40 + "\n")
                    for study in kept_studies[-5:]:  # Show last 5 kept studies
                        f.write(f"• {study.get('brief_title', study.get('title', 'No title'))}\n")
                        f.write(f"  NCT ID: {study.get('nct_id', '')}\n")
                        analysis = study.get('publication_analysis', {})
                        f.write(f"  Publications: {analysis.get('total_publications_found', 0)}\n")
                        f.write(f"  Summary: {analysis.get('web_results_summary', 'N/A')[:100]}\n")
                        f.write("-"*30 + "\n")
                
                if len(processed_studies) < total_studies:
                    remaining = total_studies - len(processed_studies)
                    est_time_remaining = remaining * 10  # Rough estimate: 10 seconds per study
                    f.write(f"\nESTIMATED REMAINING:\n")
                    f.write(f"Studies left to process: {remaining}\n")
                    f.write(f"Estimated time remaining: ~{est_time_remaining//60} minutes\n")
                    f.write(f"Check this file periodically for updates.\n")
                
        except Exception as e:
            logger.warning(f"Failed to save incremental results: {e}")

    def _rate_limit(self):
        """Apply rate limiting between web searches."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def load_studies(self, input_file: Path) -> List[Dict]:
        """Load studies from JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('studies', [])
        except Exception as e:
            logger.error(f"Error loading studies from {input_file}: {e}")
            return []
    
    def analyze_listed_publications(self, study: Dict) -> Tuple[bool, int, List[str]]:
        """
        Analyze publications listed in the study data.
        
        Returns:
            Tuple of (has_results_publications, publications_count, publication_sources)
        """
        publications = study.get('publications', [])
        publications_count = len(publications)
        
        if publications_count == 0:
            return False, 0, []
        
        # Keywords that indicate actual results (not just protocols or rationale)
        results_keywords = [
            'results', 'outcome', 'efficacy', 'safety', 'response', 'survival',
            'toxicity', 'adverse', 'endpoint', 'analysis', 'findings', 'data',
            'trial results', 'study results', 'interim analysis', 'final analysis',
            'primary endpoint', 'secondary endpoint', 'progression', 'remission'
        ]
        
        # Keywords that suggest protocol/design papers (not results)
        protocol_keywords = [
            'protocol', 'design', 'rationale', 'methodology', 'methods',
            'study design', 'trial design', 'background', 'introduction'
        ]
        
        has_results_publications = False
        publication_sources = []
        
        for pub in publications:
            citation = pub.get('citation', '').lower()
            pub_type = pub.get('type', '').lower()
            pmid = pub.get('pmid', '')
            
            publication_sources.append(f"PMID: {pmid}" if pmid else "Citation listed")
            
            # Skip if it's clearly a protocol paper
            if any(keyword in citation for keyword in protocol_keywords):
                continue
            
            # Check for results indicators
            if any(keyword in citation for keyword in results_keywords):
                has_results_publications = True
                break
            
            # If type indicates results
            if 'result' in pub_type:
                has_results_publications = True
                break
        
        return has_results_publications, publications_count, publication_sources
    
    def search_web_for_publications(self, nct_id: str, study_title: str) -> Tuple[int, List[str], Dict, str, Dict]:
        """
        Search the web for additional publications using the NCT ID and online search module.
        
        Returns:
            Tuple of (additional_publications_found, sources_found, search_results, results_summary, source_overview)
        """
        logger.info(f"Performing online search for {nct_id}")
        
        try:
            # Define progress callback for search sources
            def search_progress(completed, total, source_name):
                print(f" [{source_name}]", end='', flush=True)
            
            # Use the OnlineResultChecker to search for publications
            search_results = self.online_checker.search_for_study_results(nct_id, study_title, progress_callback=search_progress)
            
            # Extract information from the search results with detailed source overview
            additional_count = 0
            sources_found = []
            results_summary = ""
            source_overview = {
                'pubmed': {'count': 0, 'items': [], 'scraped': 0},
                'onclive': {'count': 0, 'items': [], 'scraped': 0},
                'congress_abstracts': {'count': 0, 'items': [], 'scraped': 0},
                'asco': {'count': 0, 'items': [], 'scraped': 0},
                'enets': {'count': 0, 'items': [], 'scraped': 0},
                'annals_oncology': {'count': 0, 'items': [], 'scraped': 0},
                'clinicaltrials_gov': {'count': 0, 'items': [], 'scraped': 0}
            }
            
            if search_results:
                # Check PubMed results
                pubmed_results = search_results.get('pubmed', {})
                if pubmed_results.get('publications_found', 0) > 0:
                    count = pubmed_results['publications_found']
                    additional_count += count
                    publications = pubmed_results.get('publications', [])
                    
                    source_overview['pubmed']['count'] = count
                    source_overview['pubmed']['scraped'] = sum(1 for pub in publications if pub.get('content_scraped', False))
                    
                    for pub in publications:
                        title_snippet = pub['title'][:50] + "..." if len(pub['title']) > 50 else pub['title']
                        scraped_indicator = " [📄]" if pub.get('content_scraped') else ""
                        sources_found.append(f"PubMed: {title_snippet}{scraped_indicator}")
                        source_overview['pubmed']['items'].append({
                            'title': pub['title'],
                            'url': pub.get('url', ''),
                            'pmid': pub.get('pmid', ''),
                            'content_scraped': pub.get('content_scraped', False),
                            'content_length': len(pub.get('full_content', ''))
                        })
                
                # Check congress/conference results (includes ASCO, ENETS, Annals)
                congress_results = search_results.get('congress_abstracts', {})
                if congress_results.get('abstracts_found', 0) > 0:
                    count = congress_results['abstracts_found']
                    additional_count += count
                    abstracts = congress_results.get('abstracts', [])
                    
                    source_overview['congress_abstracts']['count'] = count
                    source_overview['congress_abstracts']['scraped'] = sum(1 for abs in abstracts if abs.get('content_scraped', False))
                    
                    # Break down by specific sources within congress abstracts
                    for abs_item in abstracts:
                        source = abs_item.get('source', 'Congress')
                        title_snippet = abs_item['title'][:50] + "..." if len(abs_item['title']) > 50 else abs_item['title']
                        scraped_indicator = " [📄]" if abs_item.get('content_scraped') else ""
                        sources_found.append(f"{source}: {title_snippet}{scraped_indicator}")
                        
                        # Add to specific source overview
                        source_key = source.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('via_bing', '')
                        if 'asco' in source_key:
                            source_overview['asco']['count'] += 1
                            if abs_item.get('content_scraped'):
                                source_overview['asco']['scraped'] += 1
                            source_overview['asco']['items'].append({
                                'title': abs_item['title'],
                                'url': abs_item.get('url', ''),
                                'content_scraped': abs_item.get('content_scraped', False),
                                'content_length': len(abs_item.get('full_content', ''))
                            })
                        elif 'enets' in source_key:
                            source_overview['enets']['count'] += 1
                            if abs_item.get('content_scraped'):
                                source_overview['enets']['scraped'] += 1
                            source_overview['enets']['items'].append({
                                'title': abs_item['title'],
                                'url': abs_item.get('url', ''),
                                'content_scraped': abs_item.get('content_scraped', False),
                                'content_length': len(abs_item.get('full_content', ''))
                            })
                        elif 'annals' in source_key:
                            source_overview['annals_oncology']['count'] += 1
                            if abs_item.get('content_scraped'):
                                source_overview['annals_oncology']['scraped'] += 1
                            source_overview['annals_oncology']['items'].append({
                                'title': abs_item['title'],
                                'url': abs_item.get('url', ''),
                                'content_scraped': abs_item.get('content_scraped', False),
                                'content_length': len(abs_item.get('full_content', ''))
                            })
                
                # Check Onclive results
                onclive_results = search_results.get('onclive', {})
                if onclive_results.get('articles_found', 0) > 0:
                    count = onclive_results['articles_found']
                    additional_count += count
                    articles = onclive_results.get('articles', [])
                    
                    source_overview['onclive']['count'] = count
                    source_overview['onclive']['scraped'] = sum(1 for art in articles if art.get('content_scraped', False))
                    
                    for art in articles:
                        title_snippet = art['title'][:50] + "..." if len(art['title']) > 50 else art['title']
                        scraped_indicator = " [📄]" if art.get('content_scraped') else ""
                        sources_found.append(f"Onclive: {title_snippet}{scraped_indicator}")
                        source_overview['onclive']['items'].append({
                            'title': art['title'],
                            'url': art.get('url', ''),
                            'content_scraped': art.get('content_scraped', False),
                            'content_length': len(art.get('full_content', ''))
                        })
                
                # Check ClinicalTrials.gov additional info
                ct_results = search_results.get('clinicaltrials_gov', {})
                if ct_results.get('additional_info_found', False):
                    sources_found.append(f"ClinicalTrials.gov: {ct_results.get('info_type', 'Additional information')}")
                    source_overview['clinicaltrials_gov']['count'] = 1
                
                # Create summary with scraping info
                summary_parts = []
                if pubmed_results.get('publications_found', 0) > 0:
                    pub_count = pubmed_results['publications_found']
                    scraped_count = source_overview['pubmed']['scraped']
                    summary_parts.append(f"{pub_count} PubMed publications ({scraped_count} scraped)")
                    
                if congress_results.get('abstracts_found', 0) > 0:
                    abs_count = congress_results['abstracts_found']
                    scraped_count = source_overview['congress_abstracts']['scraped']
                    
                    # Break down congress by specific sources
                    congress_details = []
                    if source_overview['asco']['count'] > 0:
                        congress_details.append(f"ASCO: {source_overview['asco']['count']} ({source_overview['asco']['scraped']} scraped)")
                    if source_overview['enets']['count'] > 0:
                        congress_details.append(f"ENETS: {source_overview['enets']['count']} ({source_overview['enets']['scraped']} scraped)")
                    if source_overview['annals_oncology']['count'] > 0:
                        congress_details.append(f"Annals: {source_overview['annals_oncology']['count']} ({source_overview['annals_oncology']['scraped']} scraped)")
                    
                    if congress_details:
                        summary_parts.append(f"Congress abstracts: {'; '.join(congress_details)}")
                    else:
                        summary_parts.append(f"{abs_count} congress abstracts ({scraped_count} scraped)")
                        
                if onclive_results.get('articles_found', 0) > 0:
                    art_count = onclive_results['articles_found']
                    scraped_count = source_overview['onclive']['scraped']
                    summary_parts.append(f"{art_count} Onclive articles ({scraped_count} scraped)")
                    
                if ct_results.get('additional_info_found', False):
                    summary_parts.append("additional ClinicalTrials.gov info")
                
                results_summary = "; ".join(summary_parts) if summary_parts else "No additional publications found"
            else:
                results_summary = "Online search failed or returned no results"
            
            return additional_count, sources_found, search_results, results_summary, source_overview
            
        except Exception as e:
            logger.error(f"Error in online search for {nct_id}: {e}")
            return 0, [], {}, f"Search failed: {str(e)}", {}
    
    def _extract_year_from_study(self, title: str) -> Optional[int]:
        """Extract year from study title or return None."""
        # This is a placeholder - in practice, you'd use completion dates from the study data
        year_match = re.search(r'20\d{2}', title)
        if year_match:
            return int(year_match.group())
        return None
    
    def analyze_study_publications(self, study: Dict) -> PublicationAnalysis:
        """
        Comprehensive analysis of study publications including online search.
        
        Args:
            study: Study dictionary from loaded data
            
        Returns:
            PublicationAnalysis with detailed results including online search findings
        """
        nct_id = study.get('nct_id', '')
        title = study.get('title', '')
        
        logger.info(f"Analyzing publications for {nct_id}: {title[:50]}...")
        
        # Step 1: Analyze listed publications
        has_results_pubs, listed_count, listed_sources = self.analyze_listed_publications(study)
        
        # Step 2: Perform online search for additional publications
        web_search_performed = True
        additional_count, web_sources, online_results, web_summary, source_overview = self.search_web_for_publications(nct_id, title)
        
        # Step 3: Determine final publication status
        total_publications = listed_count + additional_count
        has_any_results = has_results_pubs or additional_count > 0
        
        all_sources = listed_sources + web_sources
        
        # Step 4: Extract external sources from online search
        external_sources = []
        if online_results:
            # Add specific external sources found
            for source_type, source_data in online_results.items():
                if isinstance(source_data, dict) and source_data.get('publications', []):
                    for pub in source_data['publications']:
                        if pub.get('url'):
                            external_sources.append(f"{source_type}: {pub['url']}")
                        elif pub.get('doi'):
                            external_sources.append(f"{source_type}: DOI {pub['doi']}")
                        elif pub.get('pmid'):
                            external_sources.append(f"{source_type}: PMID {pub['pmid']}")
                if isinstance(source_data, dict) and source_data.get('abstracts', []):
                    for abs_item in source_data['abstracts']:
                        if abs_item.get('url'):
                            external_sources.append(f"{source_type}: {abs_item['url']}")
        
        # Create analysis notes
        notes = []
        if listed_count > 0:
            notes.append(f"Found {listed_count} listed publication(s)")
        if has_results_pubs:
            notes.append("Listed publications appear to contain results")
        if web_search_performed:
            notes.append("Online search performed")
        if additional_count > 0:
            notes.append(f"Found {additional_count} additional publication(s) via online search")
        if web_summary:
            notes.append(f"Web search details: {web_summary}")
        if not has_any_results:
            notes.append("No publications with results found")
        
        analysis_notes = "; ".join(notes) if notes else "No analysis performed"
        
        return PublicationAnalysis(
            has_listed_publications=listed_count > 0,
            listed_publications_count=listed_count,
            has_results_publications=has_results_pubs or additional_count > 0,
            web_search_performed=web_search_performed,
            additional_publications_found=additional_count,
            total_publications_found=total_publications,
            publication_sources=all_sources,
            analysis_notes=analysis_notes,
            online_search_results=online_results or {},
            web_results_summary=web_summary,
            external_sources_found=external_sources,
            source_overview=source_overview or {}
        )
    
    def filter_studies(self, studies: List[Dict], keep_unpublished: bool = False, input_filename: str = "") -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Filter studies based on publication analysis.
        
        Args:
            studies: List of study dictionaries
            keep_unpublished: If True, keep studies without publications for further analysis
            input_filename: Name of the input file to determine filtering strategy
            
        Returns:
            Tuple of (filtered_studies, removed_studies, analysis_summary)
        """
        filtered_studies = []
        removed_studies = []
        
        # Check if we're processing a HasResults-false file
        is_hasresults_false = "HasResults-false" in input_filename
        logger.info(f"Processing file: {input_filename}")
        if is_hasresults_false:
            logger.info("Detected HasResults-false file - applying enhanced filtering for studies without posted results")
        
        analysis_stats = {
            'total_analyzed': 0,
            'with_listed_publications': 0,
            'with_results_publications': 0,
            'with_posted_results': 0,
            'web_searches_performed': 0,
            'additional_publications_found': 0,
            'online_evidence_found': 0,
            'high_confidence_studies': 0,
            'medium_confidence_studies': 0,
            'low_confidence_studies': 0,
            'kept_studies': 0,
            'removed_studies': 0,
            'hasresults_false_processing': is_hasresults_false
        }
        
        total_studies = len(studies)
        logger.info(f"Starting analysis of {total_studies} studies")
        
        # Prepare for incremental saving
        output_dir = None
        base_filename = ""
        if self.save_incremental:
            # Extract output directory info from input filename for incremental saves
            input_path = Path(input_filename) if input_filename else Path("unknown")
            output_dir = input_path.parent / "filtered_results"
            output_dir.mkdir(exist_ok=True)
            base_filename = input_path.stem
            
            # Create initial progress file
            initial_progress_file = output_dir / f"{base_filename}{'_hasresults_false' if is_hasresults_false else ''}_INCREMENTAL_progress.txt"
            with open(initial_progress_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("FILTERING STARTED - INCREMENTAL PROGRESS TRACKING\n")
                f.write("="*80 + "\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Studies to Process: {total_studies}\n")
                f.write(f"Incremental Save Interval: Every {self.incremental_save_interval} studies\n")
                if is_hasresults_false:
                    f.write("Special Processing: HasResults-false Dataset\n")
                f.write("-"*80 + "\n")
                f.write("Processing will begin shortly...\n")
                f.write("This file will be updated periodically with progress.\n")
        
        processed_studies = []
        
        for i, study in enumerate(studies, 1):
            analysis_stats['total_analyzed'] += 1
            
            # Print progress
            progress_percent = (i / total_studies) * 100
            nct_id = study.get('nct_id', 'Unknown')
            print(f"\rProgress: [{i:4d}/{total_studies:4d}] ({progress_percent:5.1f}%) - Analyzing {nct_id}...", end='', flush=True)
            
            # Check if study has posted results on ClinicalTrials.gov
            has_posted_results = study.get('has_posted_results', False)
            if has_posted_results:
                analysis_stats['with_posted_results'] += 1
            
            # Perform publication analysis
            analysis = self.analyze_study_publications(study)
            
            # Update statistics
            if analysis.has_listed_publications:
                analysis_stats['with_listed_publications'] += 1
            if analysis.has_results_publications:
                analysis_stats['with_results_publications'] += 1
            if analysis.web_search_performed:
                analysis_stats['web_searches_performed'] += 1
            if analysis.additional_publications_found > 0:
                analysis_stats['additional_publications_found'] += 1
                analysis_stats['online_evidence_found'] += 1
            
            # Add analysis to study data
            study['publication_analysis'] = asdict(analysis)
            
            # Create detailed filtering result with online search info
            decision_factors = []
            online_evidence_found = analysis.additional_publications_found > 0
            external_publications_count = analysis.additional_publications_found
            
            # Determine search confidence based on multiple factors
            search_confidence = "low"
            if analysis.has_results_publications and has_posted_results:
                search_confidence = "high"
            elif analysis.has_results_publications or has_posted_results or online_evidence_found:
                search_confidence = "medium"
            
            if analysis.has_results_publications:
                decision_factors.append("Has publications with results")
            if has_posted_results:
                decision_factors.append("Has posted results on ClinicalTrials.gov")
            if online_evidence_found:
                decision_factors.append(f"Found {external_publications_count} external publications via online search")
            if keep_unpublished and not (analysis.has_results_publications or has_posted_results or online_evidence_found):
                decision_factors.append("Kept for review (keep_unpublished=True)")
            
            # Enhanced decision logic including online search results
            # Special handling for HasResults-false files
            if is_hasresults_false:
                # For HasResults-false files, we're more aggressive in looking for external evidence
                # since these studies don't have posted results on ClinicalTrials.gov
                should_keep = (
                    analysis.has_results_publications or 
                    online_evidence_found or  # Prioritize online search results for HasResults-false
                    (keep_unpublished and analysis.has_listed_publications)  # Keep studies with any publications if requested
                )
                
                if should_keep and online_evidence_found:
                    decision_factors.append("Found external evidence via online search (HasResults-false file)")
                elif should_keep and analysis.has_results_publications:
                    decision_factors.append("Has listed publications with results (HasResults-false file)")
                elif should_keep and keep_unpublished and analysis.has_listed_publications:
                    decision_factors.append("Has publications - kept for review (HasResults-false file, keep_unpublished=True)")
            else:
                # Standard decision logic for other files
                should_keep = (
                    analysis.has_results_publications or 
                    has_posted_results or 
                    online_evidence_found or  # Include online search results in decision
                    keep_unpublished
                )
            
            if not should_keep:
                if is_hasresults_false:
                    decision_factors.append("No publications with results and no external evidence found (HasResults-false file)")
                else:
                    decision_factors.append("No publications with results, no posted results, and no external evidence found")
            
            filtering_reason = "KEPT" if should_keep else "REMOVED"
            if should_keep:
                filtering_reason += f": {'; '.join(decision_factors)}"
            else:
                filtering_reason += f": {'; '.join(decision_factors)}"
            
            filtering_result = FilteringResult(
                study_kept=should_keep,
                filtering_reason=filtering_reason,
                has_posted_results=has_posted_results,
                decision_factors=decision_factors,
                online_evidence_found=online_evidence_found,
                external_publications_count=external_publications_count,
                search_confidence=search_confidence
            )
            
            # Track confidence levels
            if search_confidence == 'high':
                analysis_stats['high_confidence_studies'] += 1
            elif search_confidence == 'medium':
                analysis_stats['medium_confidence_studies'] += 1
            else:
                analysis_stats['low_confidence_studies'] += 1
            
            # Add filtering result to study data
            study['filtering_result'] = asdict(filtering_result)
            
            if should_keep:
                filtered_studies.append(study)
                analysis_stats['kept_studies'] += 1
            else:
                removed_studies.append(study)
                analysis_stats['removed_studies'] += 1
                logger.info(f"Removing study {study.get('nct_id', '')}: {analysis.analysis_notes}")
            
            # Add to processed studies for incremental saving
            processed_studies.append(study)
            
            # Save incremental results at intervals
            if self.save_incremental and output_dir and (i % self.incremental_save_interval == 0 or i == total_studies):
                print(f" [Saving...]", end='', flush=True)
                self._save_incremental_results(
                    processed_studies, i, total_studies, 
                    output_dir, base_filename, is_hasresults_false
                )
        
        # Print completion message
        print(f"\nCompleted analysis of {total_studies} studies")
        
        # Clean up incremental files if processing completed successfully
        if self.save_incremental and output_dir:
            self._cleanup_incremental_files(output_dir, base_filename, is_hasresults_false)
        
        return filtered_studies, removed_studies, analysis_stats
    
    def _cleanup_incremental_files(self, output_dir: Path, base_filename: str, is_hasresults_false: bool = False):
        """Clean up incremental files after successful completion."""
        try:
            suffix = "_hasresults_false" if is_hasresults_false else ""
            
            # List of incremental files to clean up
            incremental_files = [
                output_dir / f"{base_filename}{suffix}_INCREMENTAL_kept.json",
                output_dir / f"{base_filename}{suffix}_INCREMENTAL_progress.txt"
            ]
            
            for file_path in incremental_files:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Cleaned up incremental file: {file_path.name}")
                    
        except Exception as e:
            logger.warning(f"Failed to clean up incremental files: {e}")

    def save_filtered_results(self, filtered_studies: List[Dict], removed_studies: List[Dict],
                            analysis_stats: Dict, output_dir: Path, original_file: str):
        """Save filtered results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filenames
        base_name = Path(original_file).stem
        
        # Add special suffix for HasResults-false files
        is_hasresults_false = "HasResults-false" in original_file
        suffix = "_hasresults_false" if is_hasresults_false else ""
        
        # Save filtered studies (those with publications)
        if filtered_studies:
            filtered_file = output_dir / f"{base_name}{suffix}_filtered_{timestamp}.json"
            
            # Determine filter criteria description based on file type
            if is_hasresults_false:
                filter_criteria = "Studies from HasResults-false dataset with external publications or evidence found via online search"
            else:
                filter_criteria = "Studies with publications, results, or posted results on ClinicalTrials.gov"
            
            filtered_data = {
                "filter_date": datetime.now().isoformat(),
                "original_file": original_file,
                "filter_criteria": filter_criteria,
                "filter_version": "2.1",
                "special_processing": "HasResults-false enhanced filtering" if is_hasresults_false else "Standard filtering",
                "total_studies": len(filtered_studies),
                "analysis_summary": analysis_stats,
                "filtering_metadata": {
                    "criteria_used": [
                        "Has publications with actual results",
                        "Has posted results on ClinicalTrials.gov" if not is_hasresults_false else "No posted results (HasResults-false file)",
                        "Has external publications found via online search",
                        "Keep unpublished flag (if enabled)"
                    ],
                    "web_search_enabled": True,
                    "online_search_sources": ["PubMed", "Google Scholar", "Congress Abstracts", "ClinicalTrials.gov"],
                    "rate_limit_delay": 2.0,
                    "hasresults_false_processing": is_hasresults_false
                },
                "studies": filtered_studies
            }
            
            with open(filtered_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(filtered_studies)} filtered studies to: {filtered_file}")
        
        # Save removed studies (those without publications)
        if removed_studies:
            removed_file = output_dir / f"{base_name}{suffix}_removed_{timestamp}.json"
            
            # Determine removal reason based on file type
            if is_hasresults_false:
                removal_reason = "No publications with results and no external evidence found via online search (HasResults-false dataset)"
            else:
                removal_reason = "No publications with results and no posted results on ClinicalTrials.gov"
            
            removed_data = {
                "filter_date": datetime.now().isoformat(),
                "original_file": original_file,
                "removal_reason": removal_reason,
                "filter_version": "2.1",
                "special_processing": "HasResults-false enhanced filtering" if is_hasresults_false else "Standard filtering",
                "total_studies": len(removed_studies),
                "analysis_summary": analysis_stats,
                "filtering_metadata": {
                    "criteria_used": [
                        "Has publications with actual results",
                        "Has posted results on ClinicalTrials.gov" if not is_hasresults_false else "No posted results (HasResults-false file)",
                        "Has external publications found via online search",
                        "Keep unpublished flag (if enabled)"
                    ],
                    "web_search_enabled": True,
                    "online_search_sources": ["PubMed", "Google Scholar", "Congress Abstracts", "ClinicalTrials.gov"],
                    "rate_limit_delay": 2.0,
                    "hasresults_false_processing": is_hasresults_false
                },
                "studies": removed_studies
            }
            
            with open(removed_file, 'w', encoding='utf-8') as f:
                json.dump(removed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(removed_studies)} removed studies to: {removed_file}")
        
        # Save analysis report
        report_file = output_dir / f"{base_name}{suffix}_filter_report_{timestamp}.txt"
        self._generate_filter_report(analysis_stats, filtered_studies, removed_studies, report_file, is_hasresults_false)
        
        return filtered_file if filtered_studies else None, removed_file if removed_studies else None
    
    def _generate_filter_report(self, stats: Dict, filtered_studies: List[Dict], 
                              removed_studies: List[Dict], output_file: Path, is_hasresults_false: bool = False):
        """Generate a human-readable filter report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLINICAL TRIALS PUBLICATION FILTER REPORT\n")
            if is_hasresults_false:
                f.write("SPECIAL PROCESSING: HasResults-false Dataset\n")
            f.write("="*80 + "\n")
            f.write(f"Filter Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Studies Analyzed: {stats['total_analyzed']}\n")
            f.write(f"Studies Kept: {stats['kept_studies']}\n")
            f.write(f"Studies Removed: {stats['removed_studies']}\n")
            if is_hasresults_false:
                f.write(f"Special Processing: Enhanced filtering for studies without posted results\n")
            f.write("-"*80 + "\n\n")
            
            # Analysis Statistics
            f.write("ANALYSIS STATISTICS:\n")
            f.write(f"Studies with listed publications: {stats['with_listed_publications']}\n")
            f.write(f"Studies with results publications: {stats['with_results_publications']}\n")
            if not is_hasresults_false:
                f.write(f"Studies with posted results on ClinicalTrials.gov: {stats['with_posted_results']}\n")
            else:
                f.write(f"Studies with posted results on ClinicalTrials.gov: {stats['with_posted_results']} (Expected: 0 for HasResults-false)\n")
            f.write(f"Web searches performed: {stats['web_searches_performed']}\n")
            f.write(f"Studies with online evidence found: {stats['online_evidence_found']}\n")
            f.write(f"High confidence studies: {stats['high_confidence_studies']}\n")
            f.write(f"Medium confidence studies: {stats['medium_confidence_studies']}\n")
            f.write(f"Low confidence studies: {stats['low_confidence_studies']}\n")
            
            if is_hasresults_false:
                f.write(f"\nHASRESULTS-FALSE FILTERING NOTES:\n")
                f.write(f"- These studies don't have posted results on ClinicalTrials.gov\n")
                f.write(f"- Filtering relies heavily on external publications and online search\n")
                f.write(f"- Studies are kept if external evidence is found via web search\n")
            
            f.write("\n" + "-"*80 + "\n\n")
            
            # Studies kept (with publications)
            if filtered_studies:
                f.write(f"KEPT STUDIES ({len(filtered_studies)}):\n")
                f.write("="*50 + "\n")
                for i, study in enumerate(filtered_studies, 1):
                    f.write(f"{i}. {study.get('title', 'No title')}\n")
                    f.write(f"   NCT ID: {study.get('nct_id', '')}\n")
                    f.write(f"   Status: {study.get('status', '')}\n")
                    f.write(f"   Has Posted Results: {study.get('has_posted_results', False)}\n")
                    
                    analysis = study.get('publication_analysis', {})
                    filtering = study.get('filtering_result', {})
                    source_overview = analysis.get('source_overview', {})
                    
                    f.write(f"   Publications Found: {analysis.get('total_publications_found', 0)}\n")
                    f.write(f"   External Publications: {analysis.get('additional_publications_found', 0)}\n")
                    f.write(f"   Online Search Summary: {analysis.get('web_results_summary', 'N/A')}\n")
                    f.write(f"   Search Confidence: {filtering.get('search_confidence', 'N/A')}\n")
                    
                    # Add detailed source breakdown
                    f.write(f"   Source Breakdown:\n")
                    if source_overview:
                        for source, details in source_overview.items():
                            if details.get('count', 0) > 0:
                                count = details['count']
                                scraped = details.get('scraped', 0)
                                source_name = source.replace('_', ' ').title()
                                f.write(f"     - {source_name}: {count} items ({scraped} with scraped content)\n")
                                
                                # Show specific items if available
                                items = details.get('items', [])
                                for i, item in enumerate(items[:3], 1):  # Show first 3 items
                                    title_short = item['title'][:60] + "..." if len(item['title']) > 60 else item['title']
                                    scraped_info = f" [Content: {item.get('content_length', 0)} chars]" if item.get('content_scraped') else " [No content]"
                                    f.write(f"       {i}. {title_short}{scraped_info}\n")
                                if len(items) > 3:
                                    f.write(f"       ... and {len(items) - 3} more items\n")
                    else:
                        f.write(f"     - No external sources found\n")
                    
                    f.write(f"   Analysis: {analysis.get('analysis_notes', 'N/A')}\n")
                    f.write(f"   Filtering Decision: {filtering.get('filtering_reason', 'N/A')}\n")
                    f.write(f"   Sources: {'; '.join(analysis.get('publication_sources', []))}\n")
                    if analysis.get('external_sources_found'):
                        f.write(f"   External Sources: {'; '.join(analysis.get('external_sources_found', []))}\n")
                    f.write("-"*40 + "\n")
                f.write("\n")
            
            # Studies removed (without publications)
            if removed_studies:
                f.write(f"REMOVED STUDIES ({len(removed_studies)}):\n")
                f.write("="*50 + "\n")
                for i, study in enumerate(removed_studies, 1):
                    f.write(f"{i}. {study.get('title', 'No title')}\n")
                    f.write(f"   NCT ID: {study.get('nct_id', '')}\n")
                    f.write(f"   Status: {study.get('status', '')}\n")
                    f.write(f"   Has Posted Results: {study.get('has_posted_results', False)}\n")
                    
                    analysis = study.get('publication_analysis', {})
                    filtering = study.get('filtering_result', {})
                    f.write(f"   Search Confidence: {filtering.get('search_confidence', 'N/A')}\n")
                    f.write(f"   Online Search Summary: {analysis.get('web_results_summary', 'N/A')}\n")
                    f.write(f"   Reason: {analysis.get('analysis_notes', 'No publications found')}\n")
                    f.write(f"   Filtering Decision: {filtering.get('filtering_reason', 'N/A')}\n")
                    f.write("-"*40 + "\n")


def main():
    """Main function to filter studies based on publication analysis."""
    
    if len(sys.argv) != 2:
        print("Usage: python study_filter.py <path_to_studies_json_file>")
        print("Example: python study_filter.py collected_studies/net_studies_all_20240101_120000.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    # Configuration
    OUTPUT_DIR = input_file.parent / "filtered_results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize filter with incremental saving enabled
    study_filter = StudyFilter(
        rate_limit_delay=2.0, 
        save_incremental=True, 
        incremental_save_interval=5  # Save every 5 studies
    )
    
    logger.info(f"Loading studies from: {input_file}")
    
    # Load studies
    studies = study_filter.load_studies(input_file)
    
    if not studies:
        print("No studies found in the input file")
        sys.exit(1)
    
    logger.info(f"Loaded {len(studies)} studies for analysis")
    
    # Filter studies based on publication analysis
    try:
        filtered_studies, removed_studies, analysis_stats = study_filter.filter_studies(
            studies, 
            keep_unpublished=False,  # Set to True if you want to keep unpublished studies for review
            input_filename=str(input_file)  # Pass the filename for special processing
        )
        
        # Save results
        filtered_file, removed_file = study_filter.save_filtered_results(
            filtered_studies, removed_studies, analysis_stats, OUTPUT_DIR, str(input_file)
        )
        
        # Print summary
        print(f"\n" + "="*80)
        print("PUBLICATION FILTER RESULTS")
        if analysis_stats.get('hasresults_false_processing', False):
            print("SPECIAL PROCESSING: HasResults-false Dataset")
        print("="*80)
        print(f"Total studies analyzed: {analysis_stats['total_analyzed']}")
        print(f"Studies with publications (kept): {analysis_stats['kept_studies']}")
        print(f"Studies without publications (removed): {analysis_stats['removed_studies']}")
        if not analysis_stats.get('hasresults_false_processing', False):
            print(f"Studies with posted results on ClinicalTrials.gov: {analysis_stats['with_posted_results']}")
        else:
            print(f"Studies with posted results: {analysis_stats['with_posted_results']} (Expected: 0 for HasResults-false)")
        print(f"Web searches performed: {analysis_stats['web_searches_performed']}")
        print(f"Studies with online evidence found: {analysis_stats['online_evidence_found']}")
        print(f"High confidence studies: {analysis_stats['high_confidence_studies']}")
        print(f"Medium confidence studies: {analysis_stats['medium_confidence_studies']}")
        print(f"Low confidence studies: {analysis_stats['low_confidence_studies']}")
        
        # Add source overview statistics
        print(f"\nSource Overview (with content scraping):")
        print("=" * 50)
        source_stats = {
            'pubmed': {'total': 0, 'scraped': 0},
            'onclive': {'total': 0, 'scraped': 0}, 
            'asco': {'total': 0, 'scraped': 0},
            'enets': {'total': 0, 'scraped': 0},
            'annals_oncology': {'total': 0, 'scraped': 0}
        }
        
        # Collect statistics from all studies
        for study in filtered_studies + removed_studies:
            analysis = study.get('publication_analysis', {})
            source_overview = analysis.get('source_overview', {})
            
            for source, details in source_overview.items():
                if source in source_stats and details.get('count', 0) > 0:
                    source_stats[source]['total'] += details.get('count', 0)
                    source_stats[source]['scraped'] += details.get('scraped', 0)
        
        for source, stats in source_stats.items():
            if stats['total'] > 0:
                source_name = source.replace('_', ' ').title()
                scrape_percent = (stats['scraped'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"{source_name:15}: {stats['total']:3d} items, {stats['scraped']:3d} scraped ({scrape_percent:4.1f}%)")
        
        if analysis_stats.get('hasresults_false_processing', False):
            print(f"\nNote: Enhanced filtering applied for HasResults-false dataset")
            print(f"Studies kept based on external publications and online evidence")
        
        print(f"\nResults saved to: {OUTPUT_DIR}")
        
        if filtered_file:
            print(f"Filtered studies: {filtered_file}")
        if removed_file:
            print(f"Removed studies: {removed_file}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()