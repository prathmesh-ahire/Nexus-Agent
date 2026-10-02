"""
browser.py — NEXUS Web Data Fetch, Save & Browser Automation
Fetches live data from the web (stock prices, web page text) and saves
it locally as .txt files. Also provides browser automation using Selenium
with the Microsoft Edge WebDriver -- no driver binary ships with NEXUS;
Selenium Manager (built into selenium 4.6+) downloads a version-matched
msedgedriver automatically on first use.
Supports opening URLs, searching the web, and scraping visible text.

Uses:
  - yfinance for stock market data
  - requests + BeautifulSoup for general web scraping
  - selenium for browser automation (open URLs, search, scrape)
  - permission_guard for safe file saving

Originally split across browser.py (Phase 24) and browser.py (Phase 23),
now unified into a single module for cleaner organisation.
"""

import contextlib
import datetime
import os
import re
import time

from nexus import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(config.PROJECT_ROOT)


# ===========================================================================
# Part 1: Browser Automation (originally in browser.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Lazy Selenium import with clear error message
# ---------------------------------------------------------------------------
def _check_selenium():
    """
    Verify selenium is installed and return the pieces needed to drive Edge.

    The driver binary itself is not checked here: Selenium Manager (built into
    selenium 4.6+) downloads a version-matched msedgedriver on demand.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options
        from selenium.webdriver.edge.service import Service
    except ImportError as e:
        raise RuntimeError(
            "Selenium is not installed. Run: pip install selenium"
        ) from e

    return webdriver, Service, Options


# ---------------------------------------------------------------------------
# Internal helper — create an Edge browser instance
# ---------------------------------------------------------------------------
def _create_browser(headless=False):
    """
    Create and return a Selenium Edge browser instance.

    Args:
        headless: If True, run browser in the background (no visible window).
                  If False (default), open a visible browser window.

    Returns:
        A selenium.webdriver.Edge instance.
    """
    webdriver, Service, Options = _check_selenium()

    options = Options()
    if headless:
        options.add_argument("--headless=new")

    # Suppress unnecessary logging
    options.add_argument("--log-level=3")
    options.add_argument("--disable-logging")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # No executable_path: Selenium Manager locates and version-matches the driver.
    service = Service()

    try:
        return webdriver.Edge(service=service, options=options)
    except Exception as e:
        raise RuntimeError(
            f"Failed to start Edge browser: {e}\n"
            "Make sure Microsoft Edge is installed."
        ) from e


# ---------------------------------------------------------------------------
# Public API — open_browser
# ---------------------------------------------------------------------------
def open_browser(url):
    """
    Open Microsoft Edge and navigate to the given URL.

    Args:
        url: The URL to open (e.g. "https://google.com").

    Returns:
        A status message string.
    """
    # Normalise the URL — add https:// if missing
    url = _normalise_url(url)

    try:
        browser = _create_browser(headless=False)
        browser.get(url)
        title = browser.title or url
        return f"Opened Edge browser: {title}\nURL: {url}"
    except Exception as e:
        return f"Could not open browser: {e}"


# ---------------------------------------------------------------------------
# Public API — search_web (Selenium-based)
# ---------------------------------------------------------------------------
def search_web(query):
    """
    Open Edge and perform a web search using Bing.

    Args:
        query: The search query string.

    Returns:
        A status message string.
    """
    if not query or not query.strip():
        return "Please provide a search query."

    # Use Bing (works offline-friendly with Edge, no consent screens)
    search_url = f"https://www.bing.com/search?q={_url_encode(query)}"

    try:
        browser = _create_browser(headless=False)
        browser.get(search_url)
        return f"Searching the web for: \"{query}\"\nOpened in Edge browser."
    except Exception as e:
        return f"Could not perform web search: {e}"


# ---------------------------------------------------------------------------
# Public API — scrape_text
# ---------------------------------------------------------------------------
def scrape_text(url):
    """
    Open a web page in headless Edge and extract the visible text content.
    Useful for reading articles, documentation, or simple web pages.

    Args:
        url: The URL to scrape.

    Returns:
        The extracted text (truncated to ~3000 chars) or an error message.
    """
    url = _normalise_url(url)

    try:
        browser = _create_browser(headless=True)
        browser.get(url)

        # Wait briefly for page to load
        time.sleep(2)

        # Get the page text (body only, strips scripts/styles)
        text = browser.find_element("tag name", "body").text
        browser.quit()

        if not text or not text.strip():
            return f"No readable text found on: {url}"

        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text.strip())

        # Truncate if very long
        if len(text) > 3000:
            text = text[:3000] + "\n\n[... content truncated — page too long ...]"

        return f"Content from: {url}\n{'─' * 40}\n{text}"

    except Exception as e:
        return f"Could not scrape page: {e}"
    finally:
        # Ensure browser is closed even on error
        with contextlib.suppress(Exception):
            browser.quit()


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------
def _normalise_url(url):
    """Add https:// prefix if the URL doesn't have a scheme."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _url_encode(text):
    """Simple URL-encoding for search queries."""
    try:
        from urllib.parse import quote_plus
        return quote_plus(text)
    except ImportError:
        # Fallback — basic encoding
        return text.replace(" ", "+").replace("&", "%26")


# ===========================================================================
# Part 2: Stock Price Fetching (originally in browser.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Indian stock name-to-symbol mapping (common NSE stocks)
# ---------------------------------------------------------------------------
_INDIAN_STOCK_MAP = {
    # Large caps
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfcbank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "icicibank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "wipro": "WIPRO.NS",
    "hcl": "HCLTECH.NS",
    "hcl tech": "HCLTECH.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "itc": "ITC.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "axis bank": "AXISBANK.NS",
    "axis": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "asian paints": "ASIANPAINT.NS",
    "asian paint": "ASIANPAINT.NS",
    "sun pharma": "SUNPHARMA.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "tata power": "TATAPOWER.NS",
    "titan": "TITAN.NS",
    "ultratech": "ULTRACEMCO.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani green": "ADANIGREEN.NS",
    "tech mahindra": "TECHM.NS",
    "mahindra": "M&M.NS",
    "m&m": "M&M.NS",
    "power grid": "POWERGRID.NS",
    "ntpc": "NTPC.NS",
    "ongc": "ONGC.NS",
    "coal india": "COALINDIA.NS",
    "hindalco": "HINDALCO.NS",
    "jsw steel": "JSWSTEEL.NS",
    "dr reddy": "DRREDDY.NS",
    "dr reddys": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "divis": "DIVISLAB.NS",
    "divis lab": "DIVISLAB.NS",
    "indusind": "INDUSINDBK.NS",
    "indusind bank": "INDUSINDBK.NS",
    "nestle": "NESTLEIND.NS",
    "nestle india": "NESTLEIND.NS",
    "britannia": "BRITANNIA.NS",
    "zomato": "ZOMATO.NS",
    "paytm": "PAYTM.NS",
    "nykaa": "NYKAA.NS",
    "irctc": "IRCTC.NS",
    "vedanta": "VEDL.NS",
    "grasim": "GRASIM.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "hero": "HEROMOTOCO.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "eicher": "EICHERMOT.NS",
    "eicher motors": "EICHERMOT.NS",
    # Indices
    "nifty": "^NSEI",
    "nifty 50": "^NSEI",
    "sensex": "^BSESN",
    # US stocks (popular ones)
    "apple": "AAPL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
}


# ---------------------------------------------------------------------------
# Helper — resolve a stock name/symbol to a yfinance symbol
# ---------------------------------------------------------------------------
def _resolve_symbol(name_or_symbol):
    """
    Resolve a user-provided stock name or symbol to a yfinance-compatible symbol.

    Logic:
      1. Check the Indian stock map (case-insensitive)
      2. If it looks like a raw symbol (all uppercase, no spaces), use as-is
      3. Otherwise, try adding .NS suffix (assume Indian NSE stock)
    """
    clean = name_or_symbol.strip()
    lower = clean.lower()

    # Check the mapping first
    if lower in _INDIAN_STOCK_MAP:
        return _INDIAN_STOCK_MAP[lower]

    # If already has a suffix (.NS, .BO, etc.) or is a known US symbol, use as-is
    if "." in clean or clean.startswith("^"):
        return clean.upper()

    # If it's all uppercase and looks like a ticker (no spaces), use as-is
    if clean.upper() == clean and " " not in clean and len(clean) <= 10:
        return clean

    # Default: try as Indian NSE stock
    return clean.upper() + ".NS"


# ---------------------------------------------------------------------------
# Fetch stock prices
# ---------------------------------------------------------------------------
def fetch_stock_prices(symbols_list):
    """
    Fetch current/last-close prices for a list of stock symbols or names.

    Args:
        symbols_list: List of stock names or symbols.
                      e.g. ["RELIANCE", "TCS", "AAPL"] or ["reliance", "infosys"]

    Returns:
        List of dicts: [{"name": ..., "symbol": ..., "price": ..., "currency": ..., "change": ...}, ...]
        On error, returns a list with a single dict containing an "error" key.
    """
    try:
        import yfinance as yf
    except ImportError:
        return [{"error": "yfinance is not installed. Run: pip install yfinance"}]

    results = []

    for raw_name in symbols_list:
        raw_name = raw_name.strip()
        if not raw_name:
            continue

        symbol = _resolve_symbol(raw_name)

        try:
            ticker = yf.Ticker(symbol)

            # Try fast_info first (faster, less data), fall back to info
            try:
                fast = ticker.fast_info
                price = getattr(fast, "last_price", None)
                currency = getattr(fast, "currency", "N/A")
                prev_close = getattr(fast, "previous_close", None)
            except Exception:
                price = None
                currency = "N/A"
                prev_close = None

            # If fast_info didn't work, try the full info dict
            if price is None:
                info = ticker.info
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                currency = info.get("currency", "N/A")
                prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

            if price is None:
                results.append({
                    "name": raw_name.title(),
                    "symbol": symbol,
                    "error": "Price not available (stock may not exist or market is closed)",
                })
                continue

            # Calculate change
            change_str = ""
            if prev_close and prev_close > 0:
                change = price - prev_close
                change_pct = (change / prev_close) * 100
                sign = "+" if change >= 0 else ""
                change_str = f"{sign}{change:.2f} ({sign}{change_pct:.2f}%)"

            results.append({
                "name": raw_name.title(),
                "symbol": symbol,
                "price": round(price, 2),
                "currency": currency,
                "change": change_str,
            })

        except Exception as e:
            results.append({
                "name": raw_name.title(),
                "symbol": symbol,
                "error": str(e),
            })

    return results


def format_stock_results(results):
    """
    Format stock price results into a readable string for display.

    Args:
        results: List of dicts from fetch_stock_prices().

    Returns:
        Formatted string.
    """
    if not results:
        return "No stock data available."

    # Check for top-level error
    if len(results) == 1 and "error" in results[0] and "name" not in results[0]:
        return f"Error: {results[0]['error']}"

    lines = []
    lines.append("")
    lines.append("  Stock Prices")
    lines.append("  " + "─" * 55)

    for r in results:
        if "error" in r:
            lines.append(f"  {r['name']:<18s} ({r['symbol']:<14s})  ⚠ {r['error']}")
        else:
            price_str = f"{r['currency']} {r['price']:,.2f}"
            change_str = f"  {r.get('change', '')}" if r.get("change") else ""
            lines.append(f"  {r['name']:<18s} ({r['symbol']:<14s})  {price_str}{change_str}")

    lines.append("  " + "─" * 55)
    lines.append(f"  Fetched at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Part 3: General Web Page Fetching
# ===========================================================================

def fetch_webpage_text(url):
    """
    Fetch a web page and extract readable text content.

    Args:
        url: The URL to fetch.

    Returns:
        Cleaned text content or error message string.
    """
    try:
        import requests
    except ImportError:
        return "Error: 'requests' is not installed. Run: pip install requests"

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: 'beautifulsoup4' is not installed. Run: pip install beautifulsoup4"

    # Normalise URL
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Fetch the page
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (f"Connection error: Could not connect to {url}\n"
                "Check your internet connection and try again.")
    except requests.exceptions.Timeout:
        return f"Timeout: The page at {url} took too long to respond (>10 seconds)."
    except requests.exceptions.MissingSchema:
        return f"Invalid URL: \"{url}\" — please include a valid web address."
    except requests.exceptions.HTTPError as e:
        return f"HTTP error {response.status_code}: Could not access {url}\n{e}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

    # Check content type — only parse HTML
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return (f"Non-HTML content at {url}\n"
                f"Content-Type: {content_type}\n"
                "This URL does not contain readable web page content.")

    # Parse HTML and extract text
    try:
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, nav, footer, and other non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe", "svg", "form"]):
            tag.decompose()

        # Try to find main content area first
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"role": "main"})
            or soup.find("div", {"id": "content"})
            or soup.find("div", {"class": "content"})
            or soup.body
        )

        if main_content is None:
            return f"No readable content found on: {url}"

        text = main_content.get_text(separator="\n", strip=True)

        if not text.strip():
            return f"No readable text found on: {url}"

        # Clean up excessive whitespace/newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Truncate if very long
        if len(text) > 4000:
            text = text[:4000] + "\n\n[... content truncated — page too long ...]"

        return f"Content from: {url}\n{'─' * 50}\n{text}"

    except Exception as e:
        return f"Error parsing page content: {e}"


# ===========================================================================
# Part 4: Save Fetched Data to File
# ===========================================================================

def save_to_txt(data, filepath, format_type="table"):
    """
    Save fetched data to a readable .txt file.

    Args:
        data:        List of stock dicts OR plain text string.
        filepath:    Where to save (absolute or relative path).
        format_type: "table" for stock data, "text" for plain text.

    Returns:
        Confirmation message string.
    """
    from nexus.security.permissions import is_allowed

    # Resolve filepath
    filepath = os.path.abspath(filepath)

    # Permission check
    if not is_allowed(filepath):
        return (f"Permission denied: Cannot save to {filepath}\n"
                "This location is not in your allowed folders.\n"
                "Add the folder via Settings > Manage Permissions.")

    # Build the content to write
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "\n" + "=" * 60 + "\n"

    if format_type == "table" and isinstance(data, list):
        # Stock data — format as a clean table
        lines = []
        lines.append(f"Data fetched on: {now}")
        lines.append("─" * 60)
        lines.append(f"{'Company':<20s} {'Symbol':<16s} {'Price':<14s} {'Currency':<10s} {'Change'}")
        lines.append("─" * 60)

        for r in data:
            if "error" in r:
                lines.append(f"{r.get('name', 'N/A'):<20s} {r.get('symbol', 'N/A'):<16s} {'ERROR':<14s} {'N/A':<10s} {r['error']}")
            else:
                price_str = f"{r['price']:,.2f}"
                lines.append(
                    f"{r['name']:<20s} {r['symbol']:<16s} {price_str:<14s} {r['currency']:<10s} {r.get('change', '')}"
                )

        lines.append("─" * 60)
        content = "\n".join(lines)

    elif format_type == "text" and isinstance(data, str):
        # Plain text content
        content = f"Data fetched on: {now}\n{'─' * 60}\n{data}"

    else:
        content = f"Data fetched on: {now}\n{'─' * 60}\n{str(data)}"

    # Write to file (append if exists, create if not)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.isfile(filepath):
            # Append with separator
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(separator)
                f.write(content + "\n")
            return (f"Data appended to: {filepath}\n"
                    f"({len(data) if isinstance(data, list) else 1} record(s) saved)")
        # Create new file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        return (f"Data saved to: {filepath}\n"
                f"({len(data) if isinstance(data, list) else 1} record(s) saved)")

    except OSError as e:
        return f"Error saving file: {e}"


def save_to_excel(data, filepath):
    """
    Save fetched data to an Excel file using openpyxl.
    STUB — to be fully implemented in a future version.

    Args:
        data:     List of dicts (stock data).
        filepath: Where to save (.xlsx).

    Returns:
        Status message.
    """
    return ("Excel export is not yet available.\n"
            "Use save_to_txt() to save data as a .txt file instead.")
