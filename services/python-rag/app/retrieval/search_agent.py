"""Zero-fee web search and deep scraping agent for external context injection."""

import logging
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """Zero-fee web search and web-scraping agent for local model context injection."""

    def search_and_scrape(self, query: str, max_results: int = 3) -> list[str]:
        """Performs DuckDuckGo search and extracts deep text from target URLs.

        Args:
            query: Search query string.
            max_results: Maximum number of search results to fetch.

        Returns:
            list[str]: Formatted source snippets with titles, URLs, and text content.
        """
        snippets = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            for idx, r in enumerate(results):
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")

                # Fetch deeper text if available
                deep_text = self._scrape_url(url)
                content = deep_text if deep_text else body

                snippets.append(f"Source [{idx+1}]: {title}\nURL: {url}\nContent: {content}\n")
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        return snippets

    def _scrape_url(self, url: str) -> str | None:
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
                # Return the first 3 substantive paragraphs joined
                substantive_paras = [p for p in paragraphs if len(p) > 40]
                if substantive_paras:
                    return "\n".join(substantive_paras[:3])
        except Exception:
            pass
        return None
