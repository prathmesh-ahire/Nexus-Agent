"""
utils.py -- NEXUS Utility Helpers
Shared utility functions used across the NEXUS agent: text chunking
for long documents, text cleaning, formatted output printing, and
file path extraction from natural language input.
"""

import re


# ---------------------------------------------------------------------------
# Text chunking (for long documents that exceed the context window)
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=1500, overlap=100):
    """
    Split text into overlapping chunks of approximately `chunk_size` words.
    Each chunk overlaps with the previous one by `overlap` words to preserve
    context across chunk boundaries.

    Args:
        text:       The full text string to split.
        chunk_size: Maximum number of words per chunk.
        overlap:    Number of overlapping words between consecutive chunks.

    Returns:
        A list of chunk strings. Returns [text] if text is short enough
        to fit in a single chunk.
    """
    if not text or not text.strip():
        return []

    words = text.split()

    # If the text fits in one chunk, return it as-is
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Move forward by (chunk_size - overlap) words
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Text cleaning (for raw extracted text from PDFs, etc.)
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Clean raw extracted text by removing null bytes, collapsing excessive
    whitespace, and stripping leading/trailing spaces.

    Args:
        text: The raw text string to clean.

    Returns:
        The cleaned text string, or empty string if input is None/empty.
    """
    if not text:
        return ""

    # Remove null bytes (common in PDF extraction)
    text = text.replace("\x00", "")

    # Replace multiple consecutive newlines with a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one (but preserve single newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading and trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# File path extraction from natural language
# ---------------------------------------------------------------------------
# Common file extensions NEXUS supports
_SUPPORTED_EXTENSIONS = (
    r"\.txt|\.pdf|\.xlsx|\.xls|\.doc|\.docx|\.csv|\.json|\.md"
)

# Patterns to try, in priority order
_PATH_PATTERNS = [
    # 1. Quoted strings (single or double quotes)
    r'["\']([^"\']+)["\']',

    # 2. Windows absolute paths:  C:\Users\...\file.txt  or  D:/folder/file.pdf
    r'([A-Za-z]:[\\\/][\w\s\\\/.\-]+(?:' + _SUPPORTED_EXTENSIONS + r'))',

    # 3. Windows absolute folder paths:  C:\Users\Documents
    r'([A-Za-z]:[\\\/][\w\s\\\/.\-]+)',

    # 4. Relative paths or filenames with supported extensions
    r'([\w\.\-\\\/]+(?:' + _SUPPORTED_EXTENSIONS + r'))',
]


def extract_filepath(text):
    """
    Attempt to find a file or folder path in a natural language string.
    Tries quoted strings first, then Windows absolute paths, then relative
    paths with known extensions.

    Args:
        text: The user input string to search for a path.

    Returns:
        The extracted path string, or None if no path was found.
    """
    if not text:
        return None

    for pattern in _PATH_PATTERNS:
        match = re.search(pattern, text)
        if match:
            path = match.group(1) if match.lastindex else match.group(0)
            path = path.strip()
            # Skip very short matches that are unlikely to be real paths
            if len(path) > 2:
                return path

    return None


# ---------------------------------------------------------------------------
# Human-readable file sizes
# ---------------------------------------------------------------------------
def format_size(size_bytes):
    """Convert bytes to a human-readable string (B/KB/MB/GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
