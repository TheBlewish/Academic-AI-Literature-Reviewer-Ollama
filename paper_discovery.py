# paper_discovery.py
# Academic Paper Discovery Engine
#
# CHANGES IN THIS VERSION:
#   1. Semantic Scholar "one-strike" rule — after first 429 in a session,
#      disables SS for the rest of the run. Saves ~10 min of futile retries.
#   2. Unpaywall email validation (from previous fix).
#   3. URL-encoded DOIs for Unpaywall (from previous fix).
#   4. filter_catalog() for relevance filtering.
#   5. All other functionality preserved.
#   6. Removed truncation from paper titles and DOI display.

import os
import json
import time
import logging
import hashlib
import re
import threading
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import quote as url_quote

from colorama import Fore, Style, init
init()

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PaperMetadata:
    title: str
    authors: List[str]
    year: Optional[int]
    abstract: Optional[str]
    doi: Optional[str]
    venue: Optional[str]
    citation_count: Optional[int]
    publication_types: List[str]
    open_access_pdf_url: Optional[str]
    source_api: str
    external_ids: Dict[str, str] = field(default_factory=dict)
    full_text_available: bool = False
    full_text_path: Optional[str] = None
    full_text_content: Optional[str] = None
    acquisition_method: Optional[str] = None
    acquisition_attempted: bool = False

    def __post_init__(self):
        if self.doi:
            self.paper_id = f"doi:{self.doi.lower().strip()}"
        else:
            title_hash = hashlib.md5(self.title.lower().strip().encode()).hexdigest()[:12]
            self.paper_id = f"hash:{title_hash}"

    def to_catalog_entry(self) -> Dict:
        return {
            "paper_id": self.paper_id, "title": self.title,
            "authors": self.authors[:3], "year": self.year,
            "venue": self.venue, "citation_count": self.citation_count,
            "doi": self.doi, "has_full_text": self.full_text_available,
            "abstract_preview": (self.abstract[:300] + "...") if self.abstract and len(self.abstract) > 300 else self.abstract,
            "source_api": self.source_api, "publication_types": self.publication_types,
        }


@dataclass
class SearchResult:
    query: str
    api_name: str
    papers_found: int
    papers: List[PaperMetadata]
    error: Optional[str] = None
    response_time: float = 0.0


@dataclass
class APIStats:
    total_queries: int = 0
    total_results: int = 0
    total_errors: int = 0
    total_full_texts: int = 0
    avg_response_time: float = 0.0
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return (self.total_queries - self.total_errors) / self.total_queries

    @property
    def avg_results_per_query(self) -> float:
        successful = self.total_queries - self.total_errors
        return self.total_results / successful if successful > 0 else 0.0

    def to_summary(self) -> str:
        return (f"queries={self.total_queries}, results={self.total_results}, "
                f"errors={self.total_errors}, full_texts={self.total_full_texts}, "
                f"success_rate={self.success_rate:.0%}, avg={self.avg_results_per_query:.1f}")


# =============================================================================
# API CLIENTS
# =============================================================================

class SemanticScholarClient:
    def __init__(self, config: Dict):
        self.base_url = config.get("base_url", "https://api.semanticscholar.org/graph/v1")
        self.api_key = config.get("api_key", "")
        self.fields = config.get("fields", "title,authors,year,abstract,externalIds,openAccessPdf,citationCount,venue,publicationTypes,publicationDate,journal")
        self.results_per_page = config.get("results_per_page", 20)
        self.max_pages = config.get("max_pages", 3)
        self.rate_limit_delay = config.get("rate_limit_delay", 1.0)
        self._disabled = False  # ONE-STRIKE: disabled after first 429

    def search(self, query: str, limit: int = None) -> SearchResult:
        # ONE-STRIKE: Skip entirely if already rate-limited this session
        if self._disabled:
            return SearchResult(query=query, api_name="semantic_scholar",
                                papers_found=0, papers=[],
                                error="Disabled (rate limited earlier — get free API key at semanticscholar.org)")

        if limit is None:
            limit = self.results_per_page
        start_time = time.time()
        papers, error = [], None
        try:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["x-api-key"] = self.api_key
            total_to_fetch = min(limit, self.results_per_page * self.max_pages)
            offset = 0
            while offset < total_to_fetch:
                params = {"query": query, "fields": self.fields,
                          "offset": offset, "limit": min(self.results_per_page, total_to_fetch - offset)}

                resp = requests.get(f"{self.base_url}/paper/search",
                                    params=params, headers=headers, timeout=30)

                if resp.status_code == 429:
                    if not self.api_key:
                        # ONE-STRIKE: No API key = disable for entire session
                        self._disabled = True
                        error = "Rate limited (no API key). SS disabled for this session."
                        print(f"    {Fore.YELLOW}  SS disabled for session — get free key at semanticscholar.org/product/api{Style.RESET_ALL}")
                        break
                    else:
                        # Has API key — do a single short retry
                        time.sleep(3)
                        resp = requests.get(f"{self.base_url}/paper/search",
                                            params=params, headers=headers, timeout=30)
                        if resp.status_code == 429:
                            error = f"Rate limited even with API key"
                            break

                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    break
                data = resp.json()
                results = data.get("data", [])
                if not results:
                    break
                for item in results:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                offset += len(results)
                if data.get("total", 0) <= offset:
                    break
                if offset < total_to_fetch:
                    time.sleep(self.rate_limit_delay)
        except requests.exceptions.Timeout:
            error = "Timed out"
        except requests.exceptions.ConnectionError:
            error = "Connection failed"
        except Exception as e:
            error = str(e)
        return SearchResult(query=query, api_name="semantic_scholar",
                            papers_found=len(papers), papers=papers,
                            error=error, response_time=time.time() - start_time)

    def _parse(self, item: Dict) -> Optional[PaperMetadata]:
        try:
            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            oa_pdf = item.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url") if oa_pdf else None
            pub_types = item.get("publicationTypes") or []
            venue = item.get("venue", "")
            if not venue:
                venue = (item.get("journal") or {}).get("name", "")
            return PaperMetadata(
                title=item.get("title", "Unknown"), authors=authors, year=item.get("year"),
                abstract=item.get("abstract"), doi=doi, venue=venue,
                citation_count=item.get("citationCount"), publication_types=pub_types,
                open_access_pdf_url=pdf_url, source_api="semantic_scholar", external_ids=ext_ids)
        except Exception as e:
            logger.debug(f"SS parse error: {e}")
            return None


class OpenAlexClient:
    def __init__(self, config: Dict):
        self.base_url = config.get("base_url", "https://api.openalex.org")
        self.email = config.get("email", "academic.researcher@example.com")
        self.results_per_page = config.get("results_per_page", 25)
        self.max_pages = config.get("max_pages", 3)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.2)

    def search(self, query: str, limit: int = None) -> SearchResult:
        if limit is None:
            limit = self.results_per_page
        start_time = time.time()
        papers, error = [], None
        try:
            total_to_fetch = min(limit, self.results_per_page * self.max_pages)
            page = 1
            while len(papers) < total_to_fetch:
                params = {"search": query, "per_page": min(self.results_per_page, total_to_fetch - len(papers)),
                          "page": page, "mailto": self.email, "sort": "cited_by_count:desc",
                          "select": "id,doi,title,authorships,publication_year,primary_location,"
                                    "cited_by_count,type,abstract_inverted_index,open_access,biblio"}
                resp = requests.get(f"{self.base_url}/works", params=params, timeout=30)
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    break
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break
                for item in results:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                if page * self.results_per_page >= data.get("meta", {}).get("count", 0):
                    break
                page += 1
                if len(papers) < total_to_fetch:
                    time.sleep(self.rate_limit_delay)
        except requests.exceptions.Timeout:
            error = "Timed out"
        except requests.exceptions.ConnectionError:
            error = "Connection failed"
        except Exception as e:
            error = str(e)
        return SearchResult(query=query, api_name="openalex",
                            papers_found=len(papers), papers=papers,
                            error=error, response_time=time.time() - start_time)

    def lookup_by_doi(self, doi: str) -> Optional[PaperMetadata]:
        try:
            doi_clean = doi.replace("https://doi.org/", "")
            resp = requests.get(f"{self.base_url}/works/https://doi.org/{doi_clean}",
                                params={"mailto": self.email}, timeout=15)
            if resp.status_code == 200:
                return self._parse(resp.json())
        except Exception:
            pass
        return None

    def _parse(self, item: Dict) -> Optional[PaperMetadata]:
        try:
            authors = [a.get("author", {}).get("display_name", "")
                       for a in item.get("authorships", [])
                       if a.get("author", {}).get("display_name")]
            doi_raw = item.get("doi", "") or ""
            doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
            pdf_url = None
            oa_info = item.get("open_access") or {}
            pdf_url = oa_info.get("oa_url")
            if not pdf_url:
                primary_loc = item.get("primary_location") or {}
                if primary_loc.get("is_oa"):
                    pdf_url = primary_loc.get("pdf_url") or primary_loc.get("landing_page_url")
            abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
            venue = ""
            source = (item.get("primary_location") or {}).get("source") or {}
            venue = source.get("display_name", "")
            ext_ids = {}
            if doi:
                ext_ids["DOI"] = doi
            oa_id = item.get("id", "")
            if oa_id:
                ext_ids["OpenAlex"] = oa_id
            return PaperMetadata(
                title=item.get("title") or "Unknown", authors=authors,
                year=item.get("publication_year"), abstract=abstract, doi=doi,
                venue=venue, citation_count=item.get("cited_by_count"),
                publication_types=[item.get("type", "")] if item.get("type") else [],
                open_access_pdf_url=pdf_url, source_api="openalex", external_ids=ext_ids)
        except Exception as e:
            logger.debug(f"OpenAlex parse error: {e}")
            return None

    def _reconstruct_abstract(self, inv_idx: Optional[Dict]) -> Optional[str]:
        if not inv_idx:
            return None
        try:
            positions = []
            for word, posns in inv_idx.items():
                for p in posns:
                    positions.append((p, word))
            positions.sort()
            return " ".join(w for _, w in positions)
        except Exception:
            return None


class COREClient:
    def __init__(self, config: Dict):
        self.base_url = config.get("base_url", "https://api.core.ac.uk/v3")
        self.api_key = config.get("api_key", "")
        self.results_per_page = config.get("results_per_page", 25)
        self.max_pages = config.get("max_pages", 2)
        self.rate_limit_delay = config.get("rate_limit_delay", 2.5)

    def search(self, query: str, limit: int = None) -> SearchResult:
        if not self.api_key:
            return SearchResult(query=query, api_name="core", papers_found=0, papers=[],
                                error="No CORE API key (free at core.ac.uk/services/api)")
        if limit is None:
            limit = self.results_per_page
        start_time = time.time()
        papers, error = [], None
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
            total_to_fetch = min(limit, self.results_per_page * self.max_pages)
            offset = 0
            while offset < total_to_fetch:
                params = {"q": query, "limit": min(self.results_per_page, total_to_fetch - offset), "offset": offset}
                resp = requests.get(f"{self.base_url}/search/works", params=params, headers=headers, timeout=30)
                if resp.status_code == 429:
                    time.sleep(10)
                    resp = requests.get(f"{self.base_url}/search/works", params=params, headers=headers, timeout=30)
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    break
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break
                for item in results:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                offset += len(results)
                if offset >= data.get("totalHits", 0):
                    break
                if offset < total_to_fetch:
                    time.sleep(self.rate_limit_delay)
        except requests.exceptions.Timeout:
            error = "Timed out"
        except requests.exceptions.ConnectionError:
            error = "Connection failed"
        except Exception as e:
            error = str(e)
        return SearchResult(query=query, api_name="core",
                            papers_found=len(papers), papers=papers,
                            error=error, response_time=time.time() - start_time)

    def _parse(self, item: Dict) -> Optional[PaperMetadata]:
        try:
            authors = []
            for a in item.get("authors", []):
                name = a.get("name", "") if isinstance(a, dict) else str(a)
                if name:
                    authors.append(name)
            doi = item.get("doi")
            if doi:
                doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
            pdf_url = item.get("downloadUrl")
            ext_ids = {}
            if doi:
                ext_ids["DOI"] = doi
            core_id = item.get("id")
            if core_id:
                ext_ids["CORE"] = str(core_id)
            venue = ""
            journals = item.get("journals") or []
            if journals and isinstance(journals[0], dict):
                venue = journals[0].get("title", "")
            if not venue:
                venue = item.get("publisher", "") or ""
            has_inline = bool(item.get("fullText"))
            return PaperMetadata(
                title=item.get("title") or "Unknown", authors=authors,
                year=item.get("yearPublished"), abstract=item.get("abstract"),
                doi=doi, venue=venue, citation_count=item.get("citationCount"),
                publication_types=[], open_access_pdf_url=pdf_url,
                source_api="core", external_ids=ext_ids,
                full_text_available=has_inline,
                full_text_content=item.get("fullText") if has_inline else None,
                acquisition_method="core_inline" if has_inline else None)
        except Exception as e:
            logger.debug(f"CORE parse error: {e}")
            return None


class EuropePMCClient:
    """Europe PMC — free, no auth, 10 req/s. Strong for biomedical + preprints."""
    def __init__(self, config: Dict):
        self.base_url = config.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest")
        self.results_per_page = config.get("results_per_page", 25)
        self.max_pages = config.get("max_pages", 3)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.15)

    def search(self, query: str, limit: int = None) -> SearchResult:
        if limit is None:
            limit = self.results_per_page
        start_time = time.time()
        papers, error = [], None
        try:
            total_to_fetch = min(limit, self.results_per_page * self.max_pages)
            cursor_mark = "*"
            while len(papers) < total_to_fetch:
                params = {
                    "query": query,
                    "resultType": "core",
                    "format": "json",
                    "pageSize": min(self.results_per_page, total_to_fetch - len(papers)),
                    "cursorMark": cursor_mark,
                    "sort": "CITED desc",
                }
                resp = requests.get(f"{self.base_url}/search", params=params, timeout=30)
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    break
                data = resp.json()
                results = data.get("resultList", {}).get("result", [])
                if not results:
                    break
                for item in results:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                next_cursor = data.get("nextCursorMark")
                if not next_cursor or next_cursor == cursor_mark:
                    break
                cursor_mark = next_cursor
                if len(papers) < total_to_fetch:
                    time.sleep(self.rate_limit_delay)
        except requests.exceptions.Timeout:
            error = "Timed out"
        except requests.exceptions.ConnectionError:
            error = "Connection failed"
        except Exception as e:
            error = str(e)
        return SearchResult(query=query, api_name="europe_pmc",
                            papers_found=len(papers), papers=papers,
                            error=error, response_time=time.time() - start_time)

    def _parse(self, item: Dict) -> Optional[PaperMetadata]:
        try:
            authors = []
            author_list = item.get("authorList", {}).get("author", [])
            for a in author_list:
                name = a.get("fullName", "")
                if name:
                    authors.append(name)
            doi = item.get("doi")
            pmid = item.get("pmid")
            pmcid = item.get("pmcid")
            pdf_url = None
            if item.get("isOpenAccess") == "Y" and pmcid:
                pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
            venue = item.get("journalTitle", "")
            ext_ids = {}
            if doi:
                ext_ids["DOI"] = doi
            if pmid:
                ext_ids["PubMed"] = str(pmid)
            if pmcid:
                ext_ids["PubMedCentral"] = pmcid
            citation_count = item.get("citedByCount")
            abstract = item.get("abstractText")
            return PaperMetadata(
                title=item.get("title") or "Unknown", authors=authors,
                year=int(item.get("pubYear")) if item.get("pubYear") else None,
                abstract=abstract, doi=doi, venue=venue,
                citation_count=citation_count,
                publication_types=[item.get("pubType", "")] if item.get("pubType") else [],
                open_access_pdf_url=pdf_url, source_api="europe_pmc",
                external_ids=ext_ids)
        except Exception as e:
            logger.debug(f"Europe PMC parse error: {e}")
            return None


class CrossrefClient:
    """Crossref REST API — free, no auth, mailto for polite pool (10 req/s)."""
    def __init__(self, config: Dict):
        self.base_url = config.get("base_url", "https://api.crossref.org")
        self.email = config.get("email", "academic.researcher@example.com")
        self.results_per_page = config.get("results_per_page", 20)
        self.max_pages = config.get("max_pages", 2)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.15)

    def search(self, query: str, limit: int = None) -> SearchResult:
        if limit is None:
            limit = self.results_per_page
        start_time = time.time()
        papers, error = [], None
        try:
            total_to_fetch = min(limit, self.results_per_page * self.max_pages)
            offset = 0
            while offset < total_to_fetch:
                params = {
                    "query": query,
                    "rows": min(self.results_per_page, total_to_fetch - offset),
                    "offset": offset,
                    "sort": "relevance",
                    "mailto": self.email,
                }
                resp = requests.get(f"{self.base_url}/works", params=params, timeout=30,
                                    headers={"User-Agent": f"AcademicLitReview/1.0 (mailto:{self.email})"})
                if resp.status_code == 429:
                    time.sleep(5)
                    resp = requests.get(f"{self.base_url}/works", params=params, timeout=30,
                                        headers={"User-Agent": f"AcademicLitReview/1.0 (mailto:{self.email})"})
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    break
                data = resp.json()
                items = data.get("message", {}).get("items", [])
                if not items:
                    break
                for item in items:
                    p = self._parse(item)
                    if p:
                        papers.append(p)
                offset += len(items)
                total_results = data.get("message", {}).get("total-results", 0)
                if offset >= total_results:
                    break
                if offset < total_to_fetch:
                    time.sleep(self.rate_limit_delay)
        except requests.exceptions.Timeout:
            error = "Timed out"
        except requests.exceptions.ConnectionError:
            error = "Connection failed"
        except Exception as e:
            error = str(e)
        return SearchResult(query=query, api_name="crossref",
                            papers_found=len(papers), papers=papers,
                            error=error, response_time=time.time() - start_time)

    def _parse(self, item: Dict) -> Optional[PaperMetadata]:
        try:
            authors = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            doi = item.get("DOI")
            # Get title (Crossref returns as list)
            title_list = item.get("title", [])
            title = title_list[0] if title_list else "Unknown"
            # Year from published-print or published-online or created
            year = None
            for date_field in ["published-print", "published-online", "created"]:
                date_parts = item.get(date_field, {}).get("date-parts", [[]])
                if date_parts and date_parts[0] and date_parts[0][0]:
                    year = date_parts[0][0]
                    break
            # Venue
            venue_list = item.get("container-title", [])
            venue = venue_list[0] if venue_list else ""
            # Abstract (Crossref sometimes has it with JATS XML tags)
            abstract = item.get("abstract", "")
            if abstract:
                # Strip JATS XML tags
                abstract = re.sub(r'<[^>]+>', '', abstract).strip()
            # Citation count
            citation_count = item.get("is-referenced-by-count")
            # OA link
            pdf_url = None
            for link in item.get("link", []):
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break
            ext_ids = {}
            if doi:
                ext_ids["DOI"] = doi
            return PaperMetadata(
                title=title, authors=authors, year=year,
                abstract=abstract if abstract else None,
                doi=doi, venue=venue, citation_count=citation_count,
                publication_types=[item.get("type", "")] if item.get("type") else [],
                open_access_pdf_url=pdf_url, source_api="crossref",
                external_ids=ext_ids)
        except Exception as e:
            logger.debug(f"Crossref parse error: {e}")
            return None


class UnpaywallClient:
    def __init__(self, config: Dict):
        self.base_url = config.get("base_url", "https://api.unpaywall.org/v2")
        self.email = config.get("email", "academic.researcher@example.com")
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)
        self._email_valid = "example.com" not in self.email
        if not self._email_valid:
            print(f"{Fore.YELLOW}  \u26a0 Unpaywall: email is 'example.com' — will reject requests.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}    Fix: export UNPAYWALL_EMAIL=\"your.real@email.com\"{Style.RESET_ALL}")

    def find_open_access(self, doi: str) -> Dict:
        result = {"pdf_url": None, "all_locations": [], "is_oa": False, "error": None}
        if not doi:
            result["error"] = "No DOI"
            return result
        if not self._email_valid:
            result["error"] = "Set UNPAYWALL_EMAIL env var to a real email"
            return result
        try:
            doi_clean = doi.strip().replace("https://doi.org/", "").replace("http://doi.org/", "")
            doi_encoded = url_quote(doi_clean, safe='')
            resp = requests.get(f"{self.base_url}/{doi_encoded}",
                                params={"email": self.email}, timeout=15)
            if resp.status_code == 404:
                result["error"] = "DOI not in Unpaywall"
                return result
            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}"
                return result
            data = resp.json()
            result["is_oa"] = data.get("is_oa", False)
            best = data.get("best_oa_location") or {}
            if best:
                result["pdf_url"] = best.get("url_for_pdf") or best.get("url")
            for loc in data.get("oa_locations", []):
                result["all_locations"].append({
                    "url": loc.get("url"), "pdf_url": loc.get("url_for_pdf"),
                    "host_type": loc.get("host_type"), "version": loc.get("version")})
            time.sleep(self.rate_limit_delay)
        except requests.exceptions.Timeout:
            result["error"] = "Timed out"
        except Exception as e:
            result["error"] = str(e)
        return result


# =============================================================================
# PAPER DISCOVERY ENGINE
# =============================================================================

class PaperDiscoveryEngine:
    def __init__(self, config: Dict = None):
        if config is None:
            from academic_config import get_search_api_config
            config = get_search_api_config()

        self.clients = {}
        self.api_stats = {}

        for name, cls in [("semantic_scholar", SemanticScholarClient),
                          ("openalex", OpenAlexClient), ("core", COREClient),
                          ("europe_pmc", EuropePMCClient), ("crossref", CrossrefClient)]:
            cfg = config.get(name, {})
            if cfg.get("enabled", True):
                self.clients[name] = cls(cfg)
                self.api_stats[name] = APIStats()

        up_cfg = config.get("unpaywall", {})
        self.unpaywall = UnpaywallClient(up_cfg) if up_cfg.get("enabled", True) else None

        self._catalog_lock = threading.Lock()
        self.paper_catalog: Dict[str, PaperMetadata] = {}
        self.doi_index: Dict[str, str] = {}
        # Normalised-title index: catches preprint/published duplicates that
        # carry different DOIs and would otherwise both enter the catalog.
        self.title_index: Dict[str, str] = {}
        self.papers_dir = "Papers"
        os.makedirs(self.papers_dir, exist_ok=True)

    # =========================================================================
    # SEARCH — returns raw results for LLM to select from
    # =========================================================================

    def search_single_api(self, query: str, api_name: str, limit: int = 20) -> SearchResult:
        """Search one API. Does NOT add to catalog — returns results for LLM selection."""
        client = self.clients.get(api_name)
        if not client:
            return SearchResult(query=query, api_name=api_name, papers_found=0, papers=[],
                                error=f"'{api_name}' not configured")
        result = client.search(query, limit=limit)
        stats = self.api_stats[api_name]
        stats.total_queries += 1
        if result.error:
            stats.total_errors += 1
            stats.last_error = result.error
        else:
            stats.total_results += result.papers_found
        n = stats.total_queries
        stats.avg_response_time = (stats.avg_response_time * (n - 1) + result.response_time) / n
        return result

    def search_all_apis(self, query: str, limit_per_api: int = 20) -> List[PaperMetadata]:
        """
        Search all enabled APIs for a query. Returns DEDUPLICATED list of papers
        but does NOT add to catalog. The LLM will select which ones to keep.
        """
        all_papers: Dict[str, PaperMetadata] = {}
        doi_seen: Dict[str, str] = {}

        for api_name in self.clients:
            q_display = query[:60] + "..." if len(query) > 60 else query
            print(f"    {Fore.CYAN}Searching {api_name}: \"{q_display}\"{Style.RESET_ALL}", flush=True)

            result = self.search_single_api(query, api_name, limit=limit_per_api)

            if result.error:
                print(f"    {Fore.RED}  {api_name}: {result.error}{Style.RESET_ALL}")
                continue

            new_count = 0
            for paper in result.papers:
                # Dedup by DOI
                if paper.doi:
                    doi_clean = paper.doi.lower().strip()
                    if doi_clean in doi_seen:
                        # Merge OA links
                        existing = all_papers[doi_seen[doi_clean]]
                        if paper.open_access_pdf_url and not existing.open_access_pdf_url:
                            existing.open_access_pdf_url = paper.open_access_pdf_url
                        if paper.abstract and not existing.abstract:
                            existing.abstract = paper.abstract
                        continue
                    doi_seen[doi_clean] = paper.paper_id

                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper
                    new_count += 1

            print(f"    {Fore.GREEN}  {result.papers_found} found, {new_count} new "
                  f"({result.response_time:.1f}s){Style.RESET_ALL}")

        return list(all_papers.values())

    # =========================================================================
    # CATALOG MANAGEMENT — papers added only after LLM selection
    # =========================================================================

    def add_selected_papers(self, papers: List[PaperMetadata]) -> int:
        """Add LLM-selected papers to the catalog. Returns count added."""
        added = 0
        for paper in papers:
            if self._add_to_catalog(paper):
                added += 1
        return added

    # Titles shorter than this are too generic to dedup on safely.
    TITLE_DEDUP_MIN_CHARS = 25

    @staticmethod
    def _norm_title_key(title: str) -> str:
        """Normalised title key used to catch preprint/published duplicates."""
        return re.sub(r'[^a-z0-9]+', ' ', (title or "").lower()).strip()

    def _add_to_catalog(self, paper: PaperMetadata) -> bool:
        """Add a paper to the catalog, or merge it into an existing record.

        INVARIANT: doi_index and title_index only ever contain ids that are
        present in paper_catalog. Every dedup decision is made BEFORE any index
        is written, so a merge can never leave an index entry pointing at a
        paper that was never stored. (An earlier version wrote the DOI index
        first and then merged on title, which produced a dangling pointer and a
        KeyError the next time that DOI was seen.)
        """
        with self._catalog_lock:
            if not hasattr(self, "title_index"):
                self.title_index = {}

            doi_clean = paper.doi.lower().strip() if paper.doi else ""
            tkey = self._norm_title_key(paper.title)
            title_dedup_ok = len(tkey) >= self.TITLE_DEDUP_MIN_CHARS

            # ---- 1) DOI dedup -----------------------------------------------
            if doi_clean and doi_clean in self.doi_index:
                existing = self.paper_catalog.get(self.doi_index[doi_clean])
                if existing is not None:
                    self._merge_metadata(existing, paper)
                    return False
                # Defensive: index entry pointing at nothing. Heal it and carry
                # on rather than raising.
                del self.doi_index[doi_clean]

            if paper.paper_id in self.paper_catalog:
                return False

            # ---- 2) Title dedup ---------------------------------------------
            # The same study routinely appears twice with DIFFERENT DOIs — a
            # preprint (bioRxiv / Research Square / Authorea) and the published
            # version, or a journal issuing two DOIs for one item — so DOI
            # dedup alone lets both into the catalog. Each then costs a separate
            # acquisition, quick-read and curation call.
            if title_dedup_ok and tkey in self.title_index:
                existing = self.paper_catalog.get(self.title_index[tkey])
                if existing is not None:
                    self._merge_metadata(existing, paper)
                    # Prefer the record that actually has a retrievable full
                    # text; a published version usually beats the preprint.
                    if (not getattr(existing, "full_text_available", False)
                            and getattr(paper, "full_text_available", False)):
                        existing.full_text_content = paper.full_text_content
                        existing.full_text_available = True
                    # Point this DOI at the SURVIVING record so any later
                    # lookup of it resolves to a real paper.
                    if doi_clean:
                        self.doi_index[doi_clean] = existing.paper_id
                    return False
                del self.title_index[tkey]

            # ---- 3) Store, then index ---------------------------------------
            self.paper_catalog[paper.paper_id] = paper
            if doi_clean:
                self.doi_index[doi_clean] = paper.paper_id
            if title_dedup_ok:
                self.title_index[tkey] = paper.paper_id
            return True

    def _merge_metadata(self, existing: PaperMetadata, new: PaperMetadata):
        if not existing.abstract and new.abstract:
            existing.abstract = new.abstract
        elif new.abstract and existing.abstract and len(new.abstract) > len(existing.abstract):
            existing.abstract = new.abstract
        if new.citation_count and (not existing.citation_count or new.citation_count > existing.citation_count):
            existing.citation_count = new.citation_count
        for k, v in new.external_ids.items():
            if k not in existing.external_ids:
                existing.external_ids[k] = v
        if new.open_access_pdf_url and not existing.open_access_pdf_url:
            existing.open_access_pdf_url = new.open_access_pdf_url
        if new.full_text_available and not existing.full_text_available:
            existing.full_text_available = True
            existing.full_text_content = new.full_text_content
            existing.full_text_path = new.full_text_path
            existing.acquisition_method = new.acquisition_method

    # =========================================================================
    # FULL TEXT ACQUISITION
    # =========================================================================

    def acquire_full_text(self, paper: PaperMetadata) -> bool:
        if paper.full_text_available and paper.full_text_content:
            return True
        if paper.acquisition_attempted:
            return False

        methods_tried = []

        if paper.open_access_pdf_url:
            print(f"      Trying: direct OA link...", end=" ", flush=True)
            methods_tried.append("direct_oa")
            if self._download_and_extract(paper, paper.open_access_pdf_url, "direct_oa"):
                print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                return True
            print(f"{Fore.RED}failed{Style.RESET_ALL}")

        if paper.doi and self.unpaywall:
            # FIXED: Show full DOI — no truncation
            print(f"      Trying: Unpaywall (DOI: {paper.doi})...", end=" ", flush=True)
            methods_tried.append("unpaywall")
            oa = self.unpaywall.find_open_access(paper.doi)
            if oa.get("error"):
                print(f"{Fore.RED}{oa['error']}{Style.RESET_ALL}")
            elif oa["pdf_url"]:
                if self._download_and_extract(paper, oa["pdf_url"], "unpaywall"):
                    print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                    return True
                print(f"{Fore.RED}failed{Style.RESET_ALL}")
                for loc in oa.get("all_locations", []):
                    alt = loc.get("pdf_url") or loc.get("url")
                    if alt and alt != oa.get("pdf_url"):
                        host = loc.get("host_type", "?")
                        print(f"      Trying: Unpaywall alt ({host})...", end=" ", flush=True)
                        methods_tried.append(f"unpaywall_{host}")
                        if self._download_and_extract(paper, alt, f"unpaywall_{host}"):
                            print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                            return True
                        print(f"{Fore.RED}failed{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}no OA version{Style.RESET_ALL}")

        pmcid = paper.external_ids.get("PubMedCentral") or paper.external_ids.get("PMCID")
        if pmcid:
            print(f"      Trying: PubMed Central ({pmcid})...", end=" ", flush=True)
            methods_tried.append("pmc")
            if self._download_and_extract(paper, f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/", "pmc"):
                print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                return True
            print(f"{Fore.RED}failed{Style.RESET_ALL}")

        arxiv_id = paper.external_ids.get("ArXiv")
        if arxiv_id:
            print(f"      Trying: ArXiv ({arxiv_id})...", end=" ", flush=True)
            methods_tried.append("arxiv")
            if self._download_and_extract(paper, f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv"):
                print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                return True
            print(f"{Fore.RED}failed{Style.RESET_ALL}")

        if paper.doi and "openalex" in self.clients:
            print(f"      Trying: OpenAlex cross-lookup...", end=" ", flush=True)
            methods_tried.append("openalex_lookup")
            oa_client = self.clients["openalex"]
            if hasattr(oa_client, 'lookup_by_doi'):
                oa_paper = oa_client.lookup_by_doi(paper.doi)
                if oa_paper and oa_paper.open_access_pdf_url:
                    if self._download_and_extract(paper, oa_paper.open_access_pdf_url, "openalex_crossref"):
                        print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                        return True
                    print(f"{Fore.RED}failed{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}no OA link found{Style.RESET_ALL}")

        if "core" in self.clients and self.clients["core"].api_key:
            short_title = " ".join(paper.title.split()[:8])
            print(f"      Trying: CORE title search...", end=" ", flush=True)
            methods_tried.append("core_title")
            core_result = self.clients["core"].search(short_title, limit=3)
            if not core_result.error:
                for core_paper in core_result.papers:
                    if core_paper.full_text_content:
                        paper.full_text_available = True
                        paper.full_text_content = core_paper.full_text_content
                        paper.acquisition_method = "core_title_search"
                        print(f"{Fore.GREEN}SUCCESS (inline text){Style.RESET_ALL}")
                        return True
                    if core_paper.open_access_pdf_url:
                        if self._download_and_extract(paper, core_paper.open_access_pdf_url, "core_title"):
                            print(f"{Fore.GREEN}SUCCESS{Style.RESET_ALL}")
                            return True
                print(f"{Fore.RED}no match{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}{core_result.error}{Style.RESET_ALL}")

        paper.acquisition_attempted = True
        if not methods_tried:
            print(f"      {Fore.RED}No methods available{Style.RESET_ALL}")
        else:
            print(f"      {Fore.RED}Exhausted {len(methods_tried)} methods: {', '.join(methods_tried)}{Style.RESET_ALL}")
        return False

    def acquire_full_texts_batch(self, papers: List[PaperMetadata] = None,
                                 max_papers: int = None) -> Tuple[int, int]:
        if papers is None:
            papers = list(self.paper_catalog.values())
        if max_papers:
            papers = papers[:max_papers]

        successes, failures = 0, 0
        for i, paper in enumerate(papers):
            if paper.full_text_available:
                successes += 1
                continue
            if paper.acquisition_attempted:
                failures += 1
                continue

            # FIXED: Show full title — no truncation
            print(f"  {Fore.CYAN}[{i+1}/{len(papers)}] Acquiring: {paper.title}{Style.RESET_ALL}")
            if self.acquire_full_text(paper):
                successes += 1
                print(f"  {Fore.GREEN}  \u2713 via {paper.acquisition_method}{Style.RESET_ALL}")
            else:
                failures += 1
                if paper.abstract:
                    print(f"  {Fore.YELLOW}  \u2717 No full text (abstract available){Style.RESET_ALL}")
                else:
                    print(f"  {Fore.RED}  \u2717 No full text, no abstract{Style.RESET_ALL}")

        return successes, failures

    def _download_and_extract(self, paper: PaperMetadata, url: str, method: str) -> bool:
        try:
            resp = requests.get(url, timeout=30, allow_redirects=True, headers={
                "User-Agent": "Academic-Literature-Review-Bot/1.0 (Research)"})
            if resp.status_code != 200:
                return False
            content = resp.content
            if not content or len(content) < 100:
                return False
            content_type = resp.headers.get("Content-Type", "").lower()

            if content[:5] == b'%PDF-' or 'application/pdf' in content_type:
                safe = re.sub(r'[^\w\s-]', '', paper.title[:80]).strip()
                safe = re.sub(r'\s+', '_', safe)
                path = os.path.join(self.papers_dir, f"{safe}_{paper.year or 'n'}.pdf")
                with open(path, 'wb') as f:
                    f.write(content)
                text = self._extract_pdf_text(path)
                if text and len(text.strip()) > 200:
                    paper.full_text_available = True
                    paper.full_text_path = path
                    paper.full_text_content = text
                    paper.acquisition_method = method
                    return True
                try:
                    os.remove(path)
                except OSError:
                    pass
                return False

            if b'<html' in content[:1000].lower() or b'<!doctype' in content[:1000].lower():
                text = self._extract_html_text(content.decode('utf-8', errors='ignore'))
                if text and len(text.strip()) > 500:
                    paper.full_text_available = True
                    paper.full_text_content = text
                    paper.acquisition_method = f"{method}_html"
                    return True
            return False
        except Exception as e:
            logger.debug(f"Download failed {url}: {e}")
            return False

    def _extract_pdf_text(self, path: str) -> Optional[str]:
        try:
            import fitz
            doc = fitz.open(path)
            parts = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(parts)
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                logger.error("No PDF library. pip install PyMuPDF")
                return None
        except Exception as e:
            logger.error(f"PDF error: {e}")
            return None

    def _extract_html_text(self, html: str) -> Optional[str]:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            for t in soup(["script", "style", "nav", "header", "footer"]):
                t.decompose()
            return "\n".join(l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip())
        except ImportError:
            return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html)).strip()

    # =========================================================================
    # CATALOG ACCESSORS
    # =========================================================================

    def get_catalog_summary(self) -> Dict:
        with self._catalog_lock:
            total = len(self.paper_catalog)
            with_ft = sum(1 for p in self.paper_catalog.values() if p.full_text_available)
            with_abs = sum(1 for p in self.paper_catalog.values() if p.abstract)
            by_source = {}
            for p in self.paper_catalog.values():
                by_source[p.source_api] = by_source.get(p.source_api, 0) + 1
            return {"total_papers": total, "with_full_text": with_ft,
                    "with_abstract_only": with_abs - with_ft, "by_source": by_source}

    def get_api_stats_for_llm(self) -> str:
        lines = ["API PERFORMANCE:"]
        for name, stats in self.api_stats.items():
            lines.append(f"  {name} [{'ON' if name in self.clients else 'OFF'}]: {stats.to_summary()}")
            if stats.last_error:
                lines.append(f"    Last error: {stats.last_error}")
        return "\n".join(lines)

    def get_all_papers(self) -> List[PaperMetadata]:
        with self._catalog_lock:
            return list(self.paper_catalog.values())

    def get_full_text_papers(self) -> List[PaperMetadata]:
        with self._catalog_lock:
            return [p for p in self.paper_catalog.values() if p.full_text_available]
