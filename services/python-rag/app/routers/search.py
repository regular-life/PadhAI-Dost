"""Web search grounding router utilizing DuckDuckGo and HTML scraping."""

from fastapi import APIRouter
from pydantic import BaseModel
from app.retrieval.search_agent import WebSearchAgent

router = APIRouter()


class SearchRequest(BaseModel):
    """Request payload for real-time web search grounding."""

    query: str
    max_results: int = 3


@router.post("/search")
async def web_search(req: SearchRequest):
    """Performs live web search and deep scraping for external grounding.

    Args:
        req: Search query and result count parameters.

    Returns:
        dict: List of scraped text snippets and search results.
    """
    agent = WebSearchAgent()
    results = agent.search_and_scrape(req.query, req.max_results)
    return {"results": results}
