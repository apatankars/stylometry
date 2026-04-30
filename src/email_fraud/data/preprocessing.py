"""Email preprocessing pipeline.

Two public entry points:

clean_email_raw(raw, config)
    Takes a full RFC-2822 email string (as read from disk).
    Returns the cleaned body text, or None if the email is unusable.
    This is the primary entry point used by prepare_data.py.

preprocess(body, config)
    Takes an already-extracted body string.
    Applies the configured cleaning steps and returns the cleaned text.
    Kept for backward compatibility and for callers that have already
    extracted the body separately.

All patterns are compiled once at import time.
"""

from __future__ import annotations

import email as _email_stdlib
import email.message
import re
from typing import Optional

from email_fraud.config import PreprocessingConfig

# ftfy ("fixes text for you") corrects garbled Unicode artifacts common in
# Enron data (e.g. â€™ instead of ').  Optional: pipeline degrades gracefully.
try:
    import ftfy as _ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False

# BeautifulSoup for HTML-to-text fallback when no text/plain part exists.
# Optional: falls back to crude tag stripping if not installed.
try:
    from bs4 import BeautifulSoup as _BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reply / forward stripping patterns (coworker's comprehensive set)
# ---------------------------------------------------------------------------

# "--- Original Message ---" and similar hard separators used by Outlook.
_RE_REPLY_CHAIN = re.compile(
    r"(-{3,}\s*Original Message\s*-{3,}.*)",
    re.IGNORECASE | re.DOTALL,
)
# "--- Forwarded by Name/ORG/CORP@CORP ---" blocks from Lotus Notes / Outlook.
_RE_FORWARD_BLOCK = re.compile(
    r"(-{3,}\s*Forwarded by.*?-{3,})",
    re.IGNORECASE | re.DOTALL,
)

# Lotus Notes (the email client used throughout the Enron corpus) has several
# distinctive reply-header formats.  These patterns cover all observed variants.

# Generic name pattern for Lotus Notes sender names (e.g. "Jeffrey K. Skilling")
_LOTUS_NAME = r"(?:[A-Z][a-zA-Z]*\.?[ \t]+){1,5}[A-Z][a-zA-Z]+(?:[/@]\w+)*"

# "From:  user@company  01/23/2001" — Lotus Notes inline From: header in reply
_RE_LOTUS_REPLY = re.compile(
    r"\n[ \t]*From:[ \t]+\S[^\n]*@[^\n]+\d{1,2}/\d{1,2}/\d{2,4}.*",
    re.IGNORECASE | re.DOTALL,
)

# "John Smith\n01/23/2001 10:30 AM" — name on one line, datetime on next
_RE_LOTUS_NAMELINE = re.compile(
    r"\n[ \t]*" + _LOTUS_NAME + r"\n[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]+\d{1,2}:\d{2}[ \t]*(?:AM|PM).*",
    re.DOTALL,
)

# "John Smith on 01/23/2001 ..." — name + "on" + date on the same line
_RE_LOTUS_ON = re.compile(
    r"\n[ \t]*" + _LOTUS_NAME + r"[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*",
    re.DOTALL | re.IGNORECASE,
)

# "\"John Smith\" <john@co.com>\n01/23/2001 10:30 AM" — quoted name variant
_RE_QUOTED_NAME_REPLY = re.compile(
    r'\n[ \t]*"[^"\n]+"[ \t]+<[^>\n]+>\n[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]+\d{1,2}:\d{2}[ \t]*(?:AM|PM).*',
    re.DOTALL,
)

# "John Smith <john@co.com> on 01/23/2001 ..." — name + email + on + date
_RE_EXT_REPLY = re.compile(
    r'\n[ \t]*(?:"[^"\n]+"|[A-Za-z][A-Za-z\s]+?)[ \t]+<[^>\n]+>[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*',
    re.DOTALL,
)

# "user@company.com on 01/23/2001 ..." — bare email address variant
_RE_BARE_REPLY = re.compile(
    r'\n[ \t]*\S+@\S+[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*',
    re.DOTALL,
)

# "From: ...", "To: ...", etc. — forwarded-message header lines inline in body.
# These are distinct from RFC-2822 headers (which are already stripped by the
# email parser).  They appear when a message is forwarded inline.
_RE_INLINE_HEADER = re.compile(
    r"^\s*(From|To|Sent|Date|Subject|Cc|Bcc)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Gmail / standard "On Mon, Jan 23, 2001, John Smith wrote:" reply attribution.
_RE_GMAIL_REPLY = re.compile(
    r"\nOn\s+\w{3},?.+?wrote:.*",
    re.IGNORECASE | re.DOTALL,
)

# "> quoted" lines — the standard email quoting convention.
_RE_QUOTED_LINE = re.compile(r"^>.*$", re.MULTILINE)

# Signature separators: "-- " (RFC 3676), "---", "___", etc.
# Everything after the separator is stripped.
_RE_SIG_SEPARATOR = re.compile(
    r"(\n-- \n|\n--\n|\n_{3,}\n|\n-{3,}\n).*",
    re.DOTALL,
)

# All reply-stripping patterns applied in sequence.
# Order matters: broad/definitive patterns (forwarded blocks, reply chains)
# run first; fine-grained patterns run after.
_REPLY_STRIP_PATTERNS: list[re.Pattern] = [
    _RE_FORWARD_BLOCK,   # "--- Forwarded by ..." blocks
    _RE_REPLY_CHAIN,     # "--- Original Message ---"
    _RE_GMAIL_REPLY,     # "On Mon ... wrote:"
    _RE_LOTUS_REPLY,     # "From:  user@enron.com  01/23/2001"
    _RE_LOTUS_NAMELINE,  # "John Smith\n01/23/2001 10:30 AM"
    _RE_LOTUS_ON,        # "John Smith on 01/23/2001"
    _RE_QUOTED_NAME_REPLY,  # '"John" <j@co.com>\n01/23/2001'
    _RE_EXT_REPLY,       # '"John" <j@co.com> on 01/23/2001'
    _RE_BARE_REPLY,      # "j@co.com on 01/23/2001"
    _RE_INLINE_HEADER,   # "From: ...", "To: ..."
    _RE_QUOTED_LINE,     # "> ..." lines
]


# ---------------------------------------------------------------------------
# Entity / high-variance normalization
# ---------------------------------------------------------------------------

# Substitution table: (pattern, replacement_token).
# Applied in order — URL first so "http://user@host" doesn't match the email
# pattern before the URL pattern has a chance to consume it.
_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = [
    # URLs first so they don't partially match email patterns
    (re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),                     "[URL]"),
    # Standard email addresses (RFC 5321 simplified)
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),   "[EMAIL]"),
    # Enron internal routing addresses (e.g. "J.Smith/HOU/ENRON@ENRON")
    (re.compile(r"\b[A-Za-z][A-Za-z\s]+/[A-Z]+/[A-Z]+@[A-Z]+\b"),           "[EMAIL]"),
    # Dates in multiple formats (written month, MM/DD/YYYY, YYYY-MM-DD)
    (
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{2,4}\b"
            r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
            r"|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b",
            re.IGNORECASE,
        ),
        "[DATE]",
    ),
    # Times like "10:30 AM" or "14:22:00"
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b", re.IGNORECASE), "[TIME]"),
    # Phone numbers in common US formats
    (re.compile(r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b|\b\d{10}\b"),             "[PHONE]"),
    # Enron order IDs, confirmation numbers, etc. (all-caps prefix + digits)
    (re.compile(r"\b[A-Z]{1,4}[-_]?\d{5,}\b|\b\d{6,}\b"),                    "[ORDER_ID]"),
    # US city+state+zip patterns (e.g. "Houston, TX 77002")
    (re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),         "[LOCATION]"),
]

# Used by _is_usable() to count how many placeholder tokens are in the text.
# If >50% of tokens are placeholders, the email is over-normalised and useless.
_RE_PLACEHOLDER = re.compile(
    r"\b(EMAIL|URL|DATE|TIME|PHONE|ORDER_ID|LOCATION)\b"
)

_RE_BLANK_LINES = re.compile(r"\n{3,}")    # collapse 3+ blank lines to 2
_RE_WHITESPACE  = re.compile(r"[ \t]+")    # collapse horizontal whitespace


# ---------------------------------------------------------------------------
# Body extraction
# ---------------------------------------------------------------------------

def _decode_payload(part: email.message.Message) -> Optional[str]:
    """Decode a single MIME part to a Python string.

    Falls back to latin-1 if the declared charset is unknown — latin-1 can
    decode any byte sequence (even if the result is garbled), so this never
    raises UnicodeDecodeError.
    """
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if payload is None:
        return None
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("latin-1", errors="replace")


def _extract_body(msg: email.message.Message) -> Optional[str]:
    """Extract plain text from a parsed email message.

    Prefers text/plain; falls back to HTML (stripped with BeautifulSoup) if
    no plain-text part exists.  Returns None if no usable body is found.

    Why prefer text/plain?
    ----------------------
    HTML emails contain structural markup (tags, CSS) that is noise for
    stylometric analysis.  Plain text preserves the author's actual writing.
    The HTML fallback exists because some Enron emails only have HTML parts.
    """
    html_part = None
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            return _decode_payload(part)
        if part.get_content_type() == "text/html" and html_part is None:
            html_part = part  # stash first HTML part in case no plain text

    if html_part is not None:
        raw_html = _decode_payload(html_part)
        if raw_html:
            if _BS4_AVAILABLE:
                # BeautifulSoup handles malformed HTML gracefully and preserves
                # paragraph structure via the separator parameter.
                return _BeautifulSoup(raw_html, "html.parser").get_text(separator="\n")
            # Crude fallback: strip all tags with a regex — misses whitespace
            # normalization but better than returning raw HTML.
            return re.sub(r"<[^>]+>", " ", raw_html)
    return None


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def _isolate_newest_message(body: str) -> str:
    """Strip all reply / forward blocks, leaving only the newest message.

    Applies all patterns in _REPLY_STRIP_PATTERNS sequentially.  Each pattern
    is a DOTALL regex that matches from the reply marker to the end of the
    string, so sub("", ...) chops everything from that point on.
    """
    for pattern in _REPLY_STRIP_PATTERNS:
        body = pattern.sub("", body)
    return body


def _strip_signatures(body: str) -> str:
    """Remove email signatures that follow standard separator lines.

    Covers RFC 3676 "-- " (dash dash space), bare "--", underscores, and dashes.
    """
    return _RE_SIG_SEPARATOR.sub("", body)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and excessive blank lines."""
    text = _RE_WHITESPACE.sub(" ", text)           # tabs and multi-spaces → single space
    text = _RE_BLANK_LINES.sub("\n\n", text)        # 3+ blank lines → 2
    return text.strip()


def _normalize_high_variance(text: str) -> str:
    """Replace high-variance tokens (URLs, emails, dates, phones, …) with placeholders.

    High-variance tokens are those that carry little stylometric information but
    vary a lot between emails (e.g. specific dates, URLs, order numbers).
    Replacing them with class tokens like [DATE] reduces encoder noise and
    lets the model focus on vocabulary, syntax, and phrasing.
    """
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def _is_usable(body: str, min_chars: int) -> bool:
    """Return True if the body is long enough and not over-normalised.

    Two filters:
    1. Length: body must have at least min_chars characters after cleaning.
       Very short emails (e.g. "Sounds good.") don't carry useful stylometric
       signal and can destabilize contrastive training.
    2. Placeholder ratio: if >50% of tokens are placeholders like [URL] or
       [DATE], the entity_masking step removed too much content and the
       remaining text is mostly boilerplate structure, not the author's voice.
    """
    stripped = body.strip()
    if len(stripped) < min_chars:
        return False
    words = stripped.split()
    if not words:
        return False
    # Count placeholder tokens and check they don't dominate the text.
    token_ratio = len(_RE_PLACEHOLDER.findall(stripped)) / len(words)
    return token_ratio < 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_email_raw(raw: str, config: PreprocessingConfig) -> Optional[str]:
    """Full pipeline from a raw RFC-2822 email string to cleaned body text.

    Entry point for scripts/prepare_data.py.

    Args:
        raw:    Complete RFC-2822 message string (as read from disk).
        config: PreprocessingConfig controlling which steps are applied.

    Returns:
        Cleaned body text, or None if the email is filtered out (too short,
        parse failure, or unusable after cleaning).
    """
    try:
        # Python's stdlib email parser handles the RFC-2822 header/body split,
        # MIME multipart traversal, and charset detection.
        msg = _email_stdlib.message_from_string(raw)
    except Exception:
        return None

    body = _extract_body(msg)
    if body is None:
        return None

    # Fix garbled Unicode artifacts (e.g. â€™ → ') before any other cleaning
    # so downstream regex patterns don't have to handle encoding variants.
    if config.fix_encoding and _FTFY_AVAILABLE:
        body = _ftfy.fix_text(body)

    return preprocess(body, config)


def preprocess(text: str, config: PreprocessingConfig) -> Optional[str]:
    """Apply configured cleaning steps to an already-extracted email body.

    Returns the cleaned text, or None if the result is shorter than
    config.min_body_chars after cleaning (usability filter).

    Steps (each gated by a config flag):
    1. strip_quoted      — isolate newest message; removes Lotus Notes blocks,
                           forwarded headers, Gmail "On ... wrote:", "> " lines.
    2. strip_signatures  — cut text at the first signature separator (-- / ---).
    3. entity_masking    — replace URLs, emails, dates, phones with placeholders.
    4. Whitespace normalisation is always applied.
    5. Truncation to config.max_body_chars is always applied.
    6. Usability check: returns None if result is shorter than min_body_chars.
    """
    if config.strip_quoted:
        text = _isolate_newest_message(text)

    if config.strip_signatures:
        text = _strip_signatures(text)

    if config.entity_masking:
        text = _normalize_high_variance(text)

    # Whitespace normalization runs unconditionally — always needed.
    text = _normalize_whitespace(text)

    # Truncate before the usability check so very long emails don't get
    # filtered out just because the end of the text is unusable.
    if config.max_body_chars and len(text) > config.max_body_chars:
        text = text[:config.max_body_chars]

    # Final usability check: drop emails that are too short or over-normalised.
    if not _is_usable(text, config.min_body_chars):
        return None

    return text


def preprocess_batch(texts: list[str], config: PreprocessingConfig) -> list[str]:
    """Apply preprocess() to a list of email bodies; silently drops None results.

    The filtered-out count is not returned; callers that need it should call
    preprocess() individually.
    """
    results = []
    for t in texts:
        out = preprocess(t, config)
        if out is not None:
            results.append(out)
    return results
