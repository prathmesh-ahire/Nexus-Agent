"""
search.py -- NEXUS Web Search & API Key Management
Provides web search capabilities using Google's Gemini API
with search grounding for live web results.

Also provides centralized API key management (load/save from
config/api_keys.json) for all modules that need API credentials.

Originally split across search.py (Phase 29) and search.py (Phase 29),
now unified into a single module for cleaner organisation.
"""

import json
import os
from datetime import datetime

from nexus import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_KEYS_FILE = str(config.API_KEYS_FILE)
API_KEYS_TEMPLATE = str(config.API_KEYS_TEMPLATE)

GEMINI_MODEL = "gemini-2.0-flash"


# ===========================================================================
# Part 1: API Key Management (originally in search.py)
# ===========================================================================

def load_api_key(service_name):
    """
    Read an API key from config/api_keys.json.

    Args:
        service_name: The key name in the JSON file, e.g. 'gemini_api_key'.

    Returns:
        The API key string, or None if not found / file missing / invalid.
    """
    if not os.path.isfile(API_KEYS_FILE):
        return None

    try:
        with open(API_KEYS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    key = data.get(service_name, None)

    # Reject placeholder values
    if key and key.strip() and key not in ("YOUR_GEMINI_API_KEY_HERE", "YOUR_API_KEY_HERE", ""):
        return key.strip()

    return None


def save_api_key(service_name, api_key):
    """
    Save an API key to config/api_keys.json.

    Args:
        service_name: The key name, e.g. 'gemini_api_key'.
        api_key:      The API key string to store.

    Returns:
        True if saved successfully, False on error.
    """
    # Load existing data
    data = {}
    if os.path.isfile(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    data[service_name] = api_key

    try:
        os.makedirs(os.path.dirname(API_KEYS_FILE), exist_ok=True)
        with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except OSError:
        return False


def get_setup_instructions():
    """
    Return user-friendly instructions for setting up the Gemini API key.

    Returns:
        Formatted instruction string.
    """
    return (
        "Gemini API key not configured.\n"
        "\n"
        "To enable web search:\n"
        "  1. Go to https://aistudio.google.com/apikey\n"
        "  2. Create a free API key\n"
        "  3. Add it to config/api_keys.json:\n"
        '     {"gemini_api_key": "YOUR_KEY_HERE"}\n'
        "\n"
        "Or use Settings > API Key in the GUI."
    )


# ===========================================================================
# Part 2: Gemini Web Search
# ===========================================================================

# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------
def search_web_gemini(query):
    """
    Search the web using Google Gemini API with search grounding.

    Args:
        query: The search query string.

    Returns:
        Formatted result string with answer and sources.
    """
    if not query or not query.strip():
        return "Please provide a search query."

    # Check for google-genai package
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return (
            "Google GenAI package not installed.\n"
            "Install with: pip install google-genai\n"
            "Then set your API key in config/api_keys.json"
        )

    # Check for API key
    api_key = load_api_key("gemini_api_key")
    if not api_key:
        return get_setup_instructions()

    # Create client and search
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )

        # Extract and format the response
        return _format_response(response, query)

    except Exception as e:
        return _handle_error(e, query)


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------
def _format_response(response, query):
    """
    Format the Gemini API response into a readable string.

    Args:
        response: The Gemini API response object.
        query:    The original query (for context).

    Returns:
        Formatted string with answer text and source URLs.
    """
    # Extract the answer text
    answer_text = ""
    try:
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                answer_text = candidate.content.parts[0].text
    except (AttributeError, IndexError):
        pass

    if not answer_text:
        return f"No results found for: \"{query}\""

    # Extract source URLs from grounding metadata
    sources = _extract_sources(response)

    # Build formatted output
    lines = []
    lines.append(answer_text.strip())

    if sources:
        lines.append("")
        lines.append("Sources:")
        for i, src in enumerate(sources[:5], 1):  # Max 5 sources
            title = src.get("title", "")
            url = src.get("url", "")
            if title and url:
                lines.append(f"  {i}. {title}")
                lines.append(f"     {url}")
            elif url:
                lines.append(f"  {i}. {url}")

    # Add timestamp
    lines.append("")
    lines.append(f"(searched: {datetime.now().strftime('%Y-%m-%d %H:%M')})")

    return "\n".join(lines)


def _extract_sources(response):
    """
    Extract source URLs from Gemini's grounding metadata.

    Args:
        response: The Gemini API response object.

    Returns:
        List of dicts with 'title' and 'url' keys.
    """
    sources = []

    try:
        if not response.candidates or len(response.candidates) == 0:
            return sources

        candidate = response.candidates[0]

        # Try grounding_metadata
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding:
            # grounding_chunks contain the source info
            chunks = getattr(grounding, "grounding_chunks", None)
            if chunks:
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        sources.append({
                            "title": getattr(web, "title", "") or "",
                            "url": getattr(web, "uri", "") or "",
                        })

    except (AttributeError, TypeError):
        pass

    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for src in sources:
        url = src.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(src)

    return unique_sources


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def _handle_error(error, query):
    """
    Handle Gemini API errors and return user-friendly messages.

    Args:
        error: The exception that occurred.
        query: The original query.

    Returns:
        Error message string.
    """
    error_str = str(error).lower()

    # API key issues
    if "api_key" in error_str or "invalid" in error_str or "401" in error_str:
        return (
            "Invalid Gemini API key.\n"
            "Please check your key in config/api_keys.json\n"
            "Get a new key at: https://aistudio.google.com/apikey"
        )

    # Rate limiting
    if "429" in error_str or "rate" in error_str or "quota" in error_str:
        return (
            "Gemini API rate limit reached.\n"
            "The free tier has a limited number of requests per minute.\n"
            "Please wait a moment and try again."
        )

    # Network issues
    if "connect" in error_str or "network" in error_str or "timeout" in error_str:
        return (
            "Could not connect to Gemini API.\n"
            "Please check your internet connection and try again."
        )

    # Permission / billing
    if "403" in error_str or "permission" in error_str:
        return (
            "Access denied by Gemini API.\n"
            "Your API key may not have the required permissions.\n"
            "Check your key at: https://aistudio.google.com/apikey"
        )

    # Generic fallback
    return (
        f"Web search error: {error}\n"
        f"Query: \"{query}\"\n"
        "Try again or check your API key and internet connection."
    )


# ---------------------------------------------------------------------------
# Structured result (for programmatic use)
# ---------------------------------------------------------------------------
def search_web_structured(query):
    """
    Search the web and return a structured dict result.
    Used internally by other modules that need parsed data.

    Args:
        query: The search query string.

    Returns:
        Dict with keys: 'answer', 'sources', 'timestamp', 'error'
    """
    result = {
        "answer": "",
        "sources": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": None,
    }

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        result["error"] = "google-genai not installed"
        return result

    api_key = load_api_key("gemini_api_key")
    if not api_key:
        result["error"] = "no_api_key"
        return result

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )

        # Extract answer
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                result["answer"] = candidate.content.parts[0].text

        # Extract sources
        result["sources"] = _extract_sources(response)

    except Exception as e:
        result["error"] = str(e)

    return result
