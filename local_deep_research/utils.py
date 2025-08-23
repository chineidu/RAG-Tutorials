import os
from enum import Enum
from typing import Any

import httpx
from langchain_community.utilities import GoogleSerperAPIWrapper, SearxSearchWrapper
from langchain_exa import ExaSearchResults
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from markdownify import markdownify

from .configuration import Configuration
from .model_config import LocalModel
from .settings import refresh_settings

settings = refresh_settings()

CHARS_PER_TOKEN: int = 4

class SearchAPI(str, Enum):
    SEARXNG: str = "searxng"
    EXA: str = "exa"
    TAVILY: str = "tavily"
    GOOGLE: str = "google"

    def __str__(self) -> str:
        """
        Return the string representation of the object.
        """
        return str(self.value)

# utils.py
def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.

    Iteratively removes all occurrences of content enclosed in thinking tokens.

    Args:
        text (str): The text to process

    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start: int = text.find("<think>")
        end: int = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def get_llm(configurable: Configuration) -> ChatOpenAI:
    """Helper function to initialize LLM based on configuration.

    Uses JSON mode if use_tool_calling is False, otherwise regular mode for tool calling.

    Args:
        configurable: Configuration object containing LLM settings

    Returns:
        Configured LLM instance
    """
    if configurable.llm_provider == "lmstudio":
        if configurable.use_tool_calling:
            return ChatOpenAI(
                base_url=settings.LMSTUDIO_URL,  # type: ignore
                model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
                temperature=0,
            )
        return ChatOpenAI(
            base_url=settings.LMSTUDIO_UR,  # type: ignore
            model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
            temperature=0,
        )
    # Default to Ollama
    if configurable.use_tool_calling:
        return ChatOpenAI(
            base_url=settings.OLLAMA_URL,  # type: ignore
            model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
            temperature=0,
        )
    return ChatOpenAI(
        base_url=settings.OLLAMA_URL,  # type: ignore
        model=LocalModel.MISTRAL_7B_INSTRUCT_V0_3_Q4_0,  # type: ignore
        temperature=0,
    )


def format_sources(search_results: dict[str, Any]) -> str:
    """
    Format search results into a markdown-style bulleted list of sources.

    Parameters
    ----------
    search_results : dict[str, Any]
        Dictionary containing search results with a "results" key,
        where each result is a dictionary with "title" and "url".

    Returns
    -------
    str
        Markdown-formatted string listing each source as a bullet point.
    """
    return "\n".join([f"* {src['title']}: {src['url']}" for src in search_results["results"]])


def deduplicate_and_format_sources(
    search_response: dict[str, Any] | list[dict[str, Any]], max_tokens_per_source: int, fetch_full_page: bool = False
) -> str:
    """
    Deduplicate search results by URL and format them into a plain-text source summary.

    Parameters
    ----------
    search_response : dict or list of dict
        Either a dictionary with a "results" key containing a list of result dicts, or a list
        of result dicts (or lists/dicts that contain "results"). Each result dict is expected
        to contain at minimum the keys "url", "title", and "content". Optionally results may
        include "raw_content".
    max_tokens_per_source : int
        Maximum number of tokens to include when including full source content. This is used
        to compute a character limit via CHARS_PER_TOKEN (characters ~= tokens * CHARS_PER_TOKEN).
    fetch_full_page : bool, optional
        If True, include truncated "raw_content" for each source according to the token limit.
        Default is False.

    Returns
    -------
    str
        A formatted plain-text string listing unique sources with their title, URL, the most
        relevant snippet, and optionally truncated full content.

    Raises
    ------
    ValueError
        If `search_response` is neither a dict with a "results" key nor a list of result dicts.

    Notes
    -----
    Deduplication is performed by URL (later items override earlier ones). The character limit
    for full content is computed as `max_tokens_per_source * CHARS_PER_TOKEN`. If a result has
    no "raw_content" and `fetch_full_page` is requested, an empty string will be used and a
    warning is printed.
    """
    # Convert input to list[result]
    if isinstance(search_response, dict):
        if "results" not in search_response:
            raise ValueError("Dict input must contain a 'results' key.")
        sources_list: list[Any] = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.append(response)
    else:
        raise ValueError("Input must either be a dict with `results` or a list of results.")

    # Deduplicate by URL (later entries override)
    unique_sources: dict[str, Any] = {src["url"]: src for src in sources_list if isinstance(src, dict) and "url" in src}

    # Format output
    formatted_text: str = "Sources:\n\n"
    for src in unique_sources.values():
        title = src.get("title", "<no title>")
        url = src.get("url", "<no url>")
        content = src.get("content", "")
        formatted_text += f"Source: {title}\n==\n"
        formatted_text += f"URL: {url}\n==\n"
        formatted_text += f"Most relevant content from source: : {content}\n==\n"
        if fetch_full_page:
            # Calculate rough estimates of characters per token
            char_limit: int = max_tokens_per_source * CHARS_PER_TOKEN
            raw_content: str = src.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {url}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "...</truncated>"
            formatted_text += f"Full source content limited to {max_tokens_per_source:,} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def fetch_raw_content(url: str) -> str | None:
    """
    Fetch HTML content from a URL and convert it to markdown format.

    Parameters
    ----------
    url : str
        The URL to fetch content from.

    Returns
    -------
    str or None
        The fetched content converted to markdown if successful,
        None if any error occurs during fetching or conversion.

    Notes
    -----
    Uses a 10-second timeout to avoid hanging on slow sites or large pages.
    """
    try:
        # Create a client with reasonable timeout
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
        return None


def searxng_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> dict[str, list[dict[str, Any]]]:
    """
    Search the web using SearXNG with improved error handling.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int, optional
        Maximum number of results to return (default is 3).
    fetch_full_page : bool, optional
        If True, fetch and include the full page content for each result (default is False).

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary with a "results" key containing a list of result dictionaries.
        Each result dictionary contains:
            - "title": str, the result title
            - "url": str, the result URL
            - "content": str, the snippet or summary
            - "raw_content": str, the full page content if fetched, otherwise the snippet

    Notes
    -----
    Uses the SearxSearchWrapper for querying SearXNG. Handles incomplete results and fetch errors gracefully.
    """
    host = os.environ.get("SEARXNG_URL", "http://localhost:8080")

    try:
        s = SearxSearchWrapper(searx_host=host)
        results = []
        search_results = s.results(query, num_results=max_results)

        for r in search_results:
            url: str = r.get("link", "")
            title: str = r.get("title", "")
            content: str = r.get("snippet", "")

            if not all([url, title]):
                print(f"Warning: Incomplete result from SearXNG: {r}")
                continue

            raw_content = content
            if fetch_full_page:
                raw_content = fetch_raw_content(url)
                if raw_content is None:
                    raw_content = content

            result = {
                "title": title,
                "url": url,
                "content": content,
                "raw_content": raw_content,
            }
            results.append(result)

        return {"results": results}

    except Exception as e:
        print(f"SearXNG search failed: {str(e)}")
        return {"results": []}


def exa_search(query: str, num_results: int = 3) -> dict[str, list[dict[str, Any]]]:
    """
    Perform a web search using the ExaSearchResults tool.

    Parameters
    ----------
    query : str
        The search query string.
    num_results : int, optional
        Maximum number of results to return (default is 3).

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary with a "results" key containing a list of result dictionaries.
        Each result dictionary contains:
            - "title": str, the result title
            - "url": str, the result URL
            - "content": str, truncated text content
            - "raw_content": str, the full text content

    Notes
    -----
    Uses the ExaSearchResults tool for querying Exa. Handles errors gracefully and truncates content to a character limit.
    """
    character_limit: int = 1_500
    # Initialize the ExaSearchResults tool
    search_tool = ExaSearchResults(exa_api_key=settings.EXA_API_KEY.get_secret_value()) # type: ignore

    try:
        # Perform a search query
        search_results = search_tool._run(  # noqa: SLF001
            query=query,
            num_results=num_results,
            text_contents_options=True,
            highlights=True,
        )
        results: list[dict[str, Any]] = [
            {
                "title": result.title,
                "url": result.url,
                "content": result.text[:character_limit] + "... </truncated>",
                "raw_content": result.text,
            }
            for result in search_results.results  # type: ignore
        ]
        return {"results": results}

    except Exception as e:
        print(f"Exa search failed: {str(e)}")
        return {"results": []}


def tavily_search(query: str, max_results: int = 3) -> dict[str, list[dict[str, Any]]]:
    """
    Perform a web search using the TavilySearch tool.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int, optional
        Maximum number of results to return (default is 3).

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary with a "results" key containing a list of result dictionaries.
        Each result dictionary contains:
            - "title": str, the result title
            - "url": str, the result URL
            - "content": str, the snippet or summary
            - "raw_content": str, the full page content if fetched, otherwise the snippet

    Notes
    -----
    Uses the TavilySearch tool for querying Tavily. Handles errors gracefully.
    """
    search_tool = TavilySearch(max_results=max_results, topic="general")
    try:
        results: list[dict[str, Any]] = search_tool.invoke(query)["results"]
        return {"results": results}

    except Exception as e:
        print(f"Tavily search failed: {str(e)}")
    return {"results": []}


def google_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> dict[str, list[dict[str, Any]]]:
    """
    Perform a web search using the GoogleSerperAPIWrapper.

    Parameters
    ----------
    query : str
        The search query string.
    max_results : int, optional
        Maximum number of results to return (default is 3).
    fetch_full_page : bool, optional
        If True, fetch and include the full page content for each result (default is False).

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary with a "results" key containing a list of result dictionaries.
        Each result dictionary contains:
            - "title": str, the result title
            - "url": str, the result URL
            - "content": str, the snippet or summary
            - "raw_content": str, the full page content if fetched, otherwise the snippet

    Notes
    -----
    Uses the GoogleSerperAPIWrapper for querying Google Serper. Handles errors gracefully.
    """
    search = GoogleSerperAPIWrapper(k=max_results)

    try:
        raw_results = search.results(query)

        results = [
            {
                "title": res["title"],
                "url": res["link"],
                "content": res["snippet"],
                "raw_content": res["snippet"],
            }
            for res in raw_results["organic"]
        ]
        if fetch_full_page:
            for res in results:
                res["raw_content"] = fetch_raw_content(res["url"])
        return {"results": results}

    except Exception as e:
        print(f"Google search failed: {str(e)}")
        return {"results": []}
