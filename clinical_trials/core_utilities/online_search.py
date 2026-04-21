#!/usr/bin/env python3
"""
Online Result Checker for Clinical Trials

This script provides utilities to search external websites using HTTP requests.
Simplified version using requests and BeautifulSoup instead of Selenium.
"""

import sys
import time
import logging
import re
import urllib.parse
from typing import List, Dict, Tuple, Optional
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineResultChecker:
    """
    Performs external web searches for clinical trial results using HTTP requests.
    """
    
    def __init__(self, rate_limit_delay: float = 2.0, scrape_content: bool = True, max_content_length: int = None):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.scrape_content = scrape_content
        self.max_content_length = max_content_length  # None means no limit
        
        # Setup HTTP session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.results_keywords = [
            'results', 'outcome', 'efficacy', 'safety', 'response', 'survival',
            'toxicity', 'adverse', 'endpoint', 'analysis', 'findings', 'data',
            'trial results', 'study results', 'interim analysis', 'final analysis',
            'primary endpoint', 'secondary endpoint', 'progression', 'remission',
            'objective response rate', 'overall survival', 'progression-free survival'
        ]
        
        # Remove the duplicate content scraping settings since they're now in __init__
        
    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _scrape_article_content(self, url: str, source: str) -> Dict[str, str]:
        """
        Scrape full content from an article URL.
        Returns dict with 'content', 'summary', and 'scraped_successfully' keys.
        """
        if not self.scrape_content or not url:
            return {'content': '', 'summary': '', 'scraped_successfully': False}
        
        try:
            logger.info(f"Scraping content from {source}: {url[:100]}...")
            content = self._get_page_content(url)
            
            if not content:
                return {'content': '', 'summary': '', 'scraped_successfully': False}
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
                script.decompose()
            
            # Extract content based on source
            if 'onclive.com' in url:
                content_text = self._extract_onclive_content(soup)
            else:
                # Generic content extraction
                content_text = self._extract_generic_content(soup)
            
            # Clean and limit content
            content_text = re.sub(r'\s+', ' ', content_text.strip())
            
            # Create summary (first 500 characters)
            summary = content_text[:500] + "..." if len(content_text) > 500 else content_text
            
            # Keep full content without truncation
            # Remove the artificial length limit to preserve complete article content
            
            return {
                'content': content_text,
                'summary': summary,
                'scraped_successfully': True
            }
            
        except Exception as e:
            logger.warning(f"Failed to scrape content from {url}: {e}")
            return {'content': '', 'summary': '', 'scraped_successfully': False}

    def _extract_onclive_content(self, soup: BeautifulSoup) -> str:
        """Extract content specifically from Onclive articles."""
        # First try the specific Onclive content div
        onclive_content = soup.select_one('.blockText_blockContent__TbCXh')
        if onclive_content:
            # Remove ads, sidebars, and navigation from the content
            for unwanted in onclive_content.select('.ad, .sidebar, .navigation, .related, .comments'):
                unwanted.decompose()
            
            text_content = onclive_content.get_text(separator=' ', strip=True)
            if len(text_content) > 200:  # Ensure we got substantial content
                return text_content
        
        # Fallback to other common Onclive selectors
        content_selectors = [
            '.article-content',
            '.content-body',
            '.article-body',
            '.entry-content',
            '[class*="article"]',
            '[class*="content"]',
            'main',
            '.post-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove ads, sidebars, and navigation
                for unwanted in content_elem.select('.ad, .sidebar, .navigation, .related, .comments'):
                    unwanted.decompose()
                
                paragraphs = content_elem.find_all(['p', 'div'], recursive=True)
                text_content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                
                if len(text_content) > 200:  # Ensure we got substantial content
                    return text_content
        
        # Fallback to generic extraction
        return self._extract_generic_content(soup)

    def _extract_generic_content(self, soup: BeautifulSoup) -> str:
        """Generic content extraction for any website."""
        # Try common content containers
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.post',
            '.entry',
            '#content',
            '#main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                text = content_elem.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    return text
        
        # Fallback: get all paragraphs from body
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    def _get_page_content(self, url: str) -> Optional[str]:
        """Get page content using HTTP requests."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
            return None

    def _make_request_with_session(self, url: str):
        """Make a request with the session and return response object."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
            return None

    def _has_results_content(self, text: str) -> bool:
        return any(keyword in text.lower() for keyword in self.results_keywords)

    def _calculate_relevance_score(self, text: str, nct_id: str) -> float:
        score = 0.0
        text_lower = text.lower()
        if nct_id.lower() in text_lower:
            score += 10.0
        score += sum(2.0 for keyword in self.results_keywords if keyword in text_lower)
        return score

    def _extract_real_url_from_bing_redirect(self, bing_url: str) -> str:
        """Extract the real URL from Bing's redirect mechanism."""
        if 'u=a1' in bing_url:
            try:
                # Extract the base64-encoded part after u=a1
                u_param = bing_url.split('u=a1')[1].split('&')[0]
                # Decode the base64 string
                import base64
                decoded_url = base64.b64decode(u_param + '==').decode('utf-8', errors='ignore')
                return decoded_url
            except Exception as e:
                logger.debug(f"Failed to decode Bing redirect URL: {e}")
        return bing_url

    def _search_bing(self, nct_id: str, site_domain: str, source_name: str) -> List[Dict]:
        """Search Bing for articles from a specific site domain."""
        articles = []
        
        # Use different search strategies for better results
        if site_domain == 'onclive.com':
            # For onclive, use the simple query that works best
            search_query = f"onclive {nct_id}"
        else:
            # For other sites, use site: prefix
            search_query = f"site:{site_domain} {nct_id}"
        
        bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(search_query)}"
        
        logger.info(f"Searching {source_name} via Bing: {search_query}")
        
        content = self._get_page_content(bing_url)
        if not content:
            return articles
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Parse Bing algorithmic results
        result_items = soup.select('#b_results li.b_algo')
        
        for result in result_items:
            try:
                # Get title and URL from the main link
                title_link = result.select_one('h2 a')
                if not title_link:
                    continue
                    
                title = title_link.get_text(strip=True)
                url = title_link.get('href', '')
                
                # Extract real URL from Bing redirect if needed
                real_url = self._extract_real_url_from_bing_redirect(url)
                
                # Check domain using the real URL
                if site_domain != 'onclive.com' and site_domain not in real_url:
                    continue
                elif site_domain == 'onclive.com' and 'onclive.com' not in real_url:
                    continue
                
                # Get snippet from the result description
                snippet_elem = result.select_one('.b_caption p, .b_caption')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                full_text = f"{title} {snippet}"
                
                # For onclive, be more restrictive - require NCT ID or very specific keywords
                # For other sites, require results content or NCT ID
                if site_domain == 'onclive.com':
                    # First check if NCT ID is in title/snippet for quick relevance
                    has_nct_id = nct_id.lower() in full_text.lower()
                    
                    # If no NCT ID in title/snippet, check for very specific drug/trial terms
                    # that are likely related to the specific trial
                    specific_terms = []
                    if nct_id == 'NCT03049189':
                        specific_terms = ['177lu-edotreotide', 'compete trial', 'edotreotide', 'lutetium']
                    
                    has_specific_terms = any(term in full_text.lower() for term in specific_terms) if specific_terms else False
                    
                    # Basic relevance check - either has NCT ID or specific terms
                    basic_relevance = has_nct_id or has_specific_terms
                    
                    # If basic relevance passes, scrape content to check for NCT ID there
                    if basic_relevance:
                        # Scrape full content to check for NCT ID
                        scraped_data = self._scrape_article_content(real_url, source_name)
                        content_has_nct_id = nct_id.lower() in scraped_data.get('content', '').lower()
                        
                        # Final relevance: NCT ID in title/snippet OR (specific terms AND NCT ID in content)
                        is_relevant = has_nct_id or (has_specific_terms and content_has_nct_id)
                    else:
                        is_relevant = False
                        scraped_data = {'content': '', 'summary': '', 'scraped_successfully': False}
                else:
                    # For other sites, require results content or NCT ID
                    is_relevant = self._has_results_content(full_text) or nct_id.lower() in full_text.lower()
                    # Use the real URL for scraping content
                    scraped_data = self._scrape_article_content(real_url, source_name) if is_relevant else {'content': '', 'summary': '', 'scraped_successfully': False}
                
                if is_relevant:
                    
                    article_data = {
                        'title': title,
                        'url': real_url,  # Use the real URL instead of the Bing redirect
                        'source': f"{source_name} (via Bing)",
                        'abstract_text': snippet,
                        'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                        'full_content': scraped_data['content'],
                        'content_summary': scraped_data['summary'],
                        'content_scraped': scraped_data['scraped_successfully']
                    }
                    
                    articles.append(article_data)
                    
            except Exception as e:
                logger.debug(f"Error processing Bing result: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        logger.info(f"Found {len(unique_articles)} {source_name} articles via Bing")
        return unique_articles

    def search_congress_abstracts(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search for congress abstracts from key sources."""
        logger.info(f"Searching congress abstracts for {nct_id} - No sources configured")
        
        abstracts = []
        # All congress abstract sources have been removed
        
        result_data = {
            'abstracts_found': len(abstracts),
            'abstracts': sorted(abstracts, key=lambda x: x.get('relevance_score', 0), reverse=True),
            'search_successful': True,
            'sources_searched': []
        }
        
        logger.info(f"Found {len(abstracts)} congress abstracts for {nct_id}")
        return len(abstracts) > 0, result_data

    def _search_annals_oncology_direct(self, nct_id: str) -> List[Dict]:
        """Direct search of Annals of Oncology website - REMOVED."""
        logger.info("Annals of Oncology search has been removed")
        return []

    def _search_google_scholar(self, nct_id: str, source_name: str) -> List[Dict]:
        """Search Google Scholar for academic publications."""
        articles = []
        
        # Use different search strategies for better results
        search_queries = [
            f'"{nct_id}" clinical trial',
            f'{nct_id} study results',
            f'{nct_id} efficacy safety'
        ]
        
        for search_query in search_queries:
            scholar_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(search_query)}"
            
            logger.info(f"Searching {source_name} via Google Scholar: {search_query}")
            
            content = self._get_page_content(scholar_url)
            if not content:
                continue
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Parse Google Scholar results
            result_items = soup.select('.gs_ri')
            
            for result in result_items[:5]:  # Limit to first 5 results per query
                try:
                    # Get title and URL from the main link
                    title_link = result.select_one('.gs_rt a, .gs_rt')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '') if title_link.name == 'a' else ""
                    
                    # If no direct URL, try to find PDF or other links
                    if not url:
                        pdf_link = result.select_one('.gs_ggs .gs_ctg2')
                        if pdf_link and pdf_link.get('href'):
                            url = pdf_link.get('href', '')
                    
                    # Get snippet from the result description
                    snippet_elem = result.select_one('.gs_rs')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Get citation info
                    citation_elem = result.select_one('.gs_a')
                    citation = citation_elem.get_text(strip=True) if citation_elem else ""
                    
                    full_text = f"{title} {snippet} {citation}"
                    
                    # Check relevance
                    is_relevant = (self._has_results_content(full_text) or 
                                 nct_id.lower() in full_text.lower() or
                                 any(term in full_text.lower() for term in ['trial', 'study', 'clinical', 'efficacy', 'safety']))
                    
                    if is_relevant and title not in [a['title'] for a in articles]:  # Avoid duplicates
                        # Scrape full content from the article if URL is available
                        scraped_data = {'content': '', 'summary': '', 'scraped_successfully': False}
                        if url and url.startswith('http'):
                            scraped_data = self._scrape_article_content(url, source_name)
                        
                        article_data = {
                            'title': title,
                            'url': url if url else scholar_url,
                            'source': f"{source_name} (via Google Scholar)",
                            'abstract_text': snippet,
                            'citation_info': citation,
                            'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                            'full_content': scraped_data['content'],
                            'content_summary': scraped_data['summary'],
                            'content_scraped': scraped_data['scraped_successfully']
                        }
                        
                        articles.append(article_data)
                        
                except Exception as e:
                    logger.debug(f"Error processing Google Scholar result: {e}")
                    continue
            
            # Add delay between queries to avoid being blocked
            time.sleep(3)
            
            # Stop if we have enough results
            if len(articles) >= 10:
                break
        
        logger.info(f"Found {len(articles)} {source_name} articles via Google Scholar")
        return articles

    def search_onclive_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search Onclive for relevant articles."""
        logger.info(f"Searching Onclive for {nct_id}")
        
        articles = self._search_bing(nct_id, 'onclive.com', 'Onclive')
        
        result_data = {
            'articles_found': len(articles),
            'articles': sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True),
            'search_successful': True
        }
        
        logger.info(f"Found {len(articles)} Onclive articles for {nct_id}")
        return len(articles) > 0, result_data

    def search_google_scholar_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search Google Scholar for academic publications."""
        logger.info(f"Searching Google Scholar for {nct_id}")
        
        articles = self._search_google_scholar(nct_id, 'Google Scholar')
        
        result_data = {
            'articles_found': len(articles),
            'articles': sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True),
            'search_successful': True
        }
        
        logger.info(f"Found {len(articles)} Google Scholar articles for {nct_id}")
        return len(articles) > 0, result_data

    def _create_search_result_dict(self, articles: List[Dict]) -> Dict:
        return {
            'publications_found': len(articles),
            'publications': articles,
            'search_successful': len(articles) > 0
        }

    def search_pubmed_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search PubMed for publications."""
        logger.info(f"Searching PubMed for {nct_id}")
        pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Search strategies in order of preference
        search_strategies = [
            f'("{nct_id}"[ClinicalTrialIdentifier])',
            f'("{nct_id}"[Text Word])',
            f'{nct_id}'
        ]
        
        all_pmids = set()
        for search_term in search_strategies:
            esearch_params = {"db": "pubmed", "term": search_term, "retmax": 10, "sort": "relevance"}
            response = self._make_request_with_session(f"{pubmed_base}esearch.fcgi?{urllib.parse.urlencode(esearch_params)}")
            
            if response:
                soup = BeautifulSoup(response.text, 'xml')
                all_pmids.update(id_tag.text for id_tag in soup.find_all('Id'))
                
            if len(all_pmids) >= 10:
                break
        
        if not all_pmids:
            return False, {'publications_found': 0, 'publications': [], 'search_successful': False}
        
        # Fetch article details
        pmid_list = list(all_pmids)[:10]
        efetch_params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        response = self._make_request_with_session(f"{pubmed_base}efetch.fcgi?{urllib.parse.urlencode(efetch_params)}")
        
        if not response:
            return False, {'publications_found': 0, 'publications': [], 'search_successful': False}
        
        soup = BeautifulSoup(response.text, 'xml')
        publications = []
        
        for article in soup.find_all('PubmedArticle'):
            try:
                abstract_tag = article.find('AbstractText')
                title_tag = article.find('ArticleTitle')
                pmid_tag = article.find('PMID')
                
                abstract_text = abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else ""
                title = title_tag.get_text(strip=True) if title_tag else ""
                pmid = pmid_tag.text if pmid_tag else ""
                
                full_text = f"{title} {abstract_text}"
                
                # For PubMed, we already have the abstract, but we can try to get full text if available
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                scraped_data = self._scrape_article_content(pubmed_url, 'PubMed')
                
                publications.append({
                    'pmid': pmid,
                    'title': title,
                    'url': pubmed_url,
                    'has_results_keywords': self._has_results_content(full_text),
                    'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                    'abstract_text': abstract_text[:400] + "..." if len(abstract_text) > 400 else abstract_text,
                    'full_content': scraped_data['content'] if scraped_data['content'] else abstract_text,
                    'content_summary': scraped_data['summary'] if scraped_data['summary'] else abstract_text[:500],
                    'content_scraped': scraped_data['scraped_successfully']
                })
            except Exception as e:
                logger.warning(f"Error processing PubMed article: {e}")
        
        publications.sort(key=lambda x: x['relevance_score'], reverse=True)
        result_data = {
            'publications_found': len(publications),
            'publications': publications,
            'search_successful': True
        }
        
        logger.info(f"Found {len(publications)} PubMed publications for {nct_id}")
        return len(publications) > 0, result_data

    def set_content_scraping(self, enabled: bool, max_length: int = None):
        """Enable or disable content scraping functionality."""
        self.scrape_content = enabled
        self.max_content_length = max_length  # None means no limit
        length_msg = "unlimited" if max_length is None else str(max_length)
        logger.info(f"Content scraping {'enabled' if enabled else 'disabled'}, max length: {length_msg}")

    def search_for_study_results(self, nct_id: str, study_title: str, progress_callback=None) -> Dict:
        logger.info(f"Comprehensive search for study results: {nct_id}")
        results = {
            'nct_id': nct_id, 'study_title': study_title, 'search_timestamp': time.time(),
            'pubmed': {}, 'congress_abstracts': {}, 'onclive': {}, 'google_scholar': {}
        }
        
        # Track progress through the search sources
        total_sources = 4
        completed_sources = 0
        
        def update_progress(source_name: str):
            nonlocal completed_sources
            completed_sources += 1
            if progress_callback:
                progress_callback(completed_sources, total_sources, source_name)
        
        try:
            _, results['pubmed'] = self.search_pubmed_enhanced(nct_id, study_title)
            update_progress("PubMed")
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            update_progress("PubMed")
            
        try:
            _, results['congress_abstracts'] = self.search_congress_abstracts(nct_id, study_title)
            update_progress("Congress Abstracts")
        except Exception as e:
            logger.error(f"Congress search failed: {e}")
            update_progress("Congress Abstracts")
            
        try:
            _, results['onclive'] = self.search_onclive_enhanced(nct_id, study_title)
            update_progress("Onclive")
        except Exception as e:
            logger.error(f"Onclive search failed: {e}")
            update_progress("Onclive")
            
        try:
            _, results['google_scholar'] = self.search_google_scholar_enhanced(nct_id, study_title)
            update_progress("Google Scholar")
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            update_progress("Google Scholar")
            
        return results

if __name__ == "__main__":
    # Simplified to use HTTP requests only with unlimited content scraping
    checker = OnlineResultChecker(rate_limit_delay=2.0, scrape_content=True, max_content_length=None)
    
    nct_id_example = "NCT03049189"
    title_example = "A Study of 177Lu-Edotreotide Versus Everolimus in GEP-NET (COMPETE)"
    
    print(f"\n--- Checking {nct_id_example} ---")
    results = checker.search_for_study_results(nct_id_example, title_example)
    
    print("\n--- Search Summary ---")
    total_scraped = 0
    total_items = 0
    
    for source, data in results.items():
        if isinstance(data, dict) and 'search_successful' in data:
            count = data.get('articles_found', data.get('publications_found', data.get('abstracts_found', 0)))
            items = data.get('articles') or data.get('publications') or data.get('abstracts')
            
            if items:
                scraped_count = sum(1 for item in items if item.get('content_scraped', False))
                total_scraped += scraped_count
                total_items += len(items)
                
                print(f"{source.replace('_', ' ').title()}: Found {count} results, {scraped_count} scraped.")
                print(f"  - Example: {items[0]['title'][:70]}... ({items[0]['url'][:50]}...)")
                
                if items[0].get('content_scraped'):
                    summary = items[0].get('content_summary', '')
                    print(f"  - Content: {summary[:100]}...")
            else:
                print(f"{source.replace('_', ' ').title()}: Found {count} results.")
    
    print(f"\nOverall: {total_scraped}/{total_items} articles had content successfully scraped.")
    
    # Save results with scraped content
    import json
    with open('example_results_with_content.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Detailed results with scraped content saved to: example_results_with_content.json")