"""
Unified email preprocessing pipeline for Enron and client datasets.
Identical rules are applied to both sources to prevent the model from
learning dataset-specific artifacts instead of authorship signals.
"""

import os
import re
import email
import email.message
import json
import pathlib
import ftfy
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator, Optional

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[assignment]


ENRON_ROOT = pathlib.Path("C:/Users/danil/Documents/enr_data")

SENT_FOLDERS = {"sent", "sent_items", "_sent_mail", "sent_mail"}

MIN_BODY_CHARS = 50
MAX_BODY_CHARS = 4000
BATCH_SIZE     = 500

_RE_REPLY_CHAIN = re.compile(
    r"(-{3,}\s*Original Message\s*-{3,}.*)",
    re.IGNORECASE | re.DOTALL,
)
_RE_FORWARD_BLOCK = re.compile(
    r"(-{3,}\s*Forwarded by.*?-{3,})",
    re.IGNORECASE | re.DOTALL,
)

# Handles: "Janette Elbertson", "Brett R Wiggs", "Amitava Dhar@ENRON", "Name/HOU/ECT@ECT"
_LOTUS_NAME = r"(?:[A-Z][a-zA-Z]*\.?[ \t]+){1,5}[A-Z][a-zA-Z]+(?:[/@]\w+)*"

_RE_LOTUS_REPLY = re.compile(
    r"\n[ \t]*From:[ \t]+\S[^\n]*@[^\n]+\d{1,2}/\d{1,2}/\d{2,4}.*",
    re.IGNORECASE | re.DOTALL,
)
_RE_LOTUS_NAMELINE = re.compile(
    r"\n[ \t]*" + _LOTUS_NAME + r"\n[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]+\d{1,2}:\d{2}[ \t]*(?:AM|PM).*",
    re.DOTALL,
)
_RE_LOTUS_ON = re.compile(
    r"\n[ \t]*" + _LOTUS_NAME + r"[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*",
    re.DOTALL | re.IGNORECASE,
)
_RE_QUOTED_NAME_REPLY = re.compile(
    r'\n[ \t]*"[^"\n]+"[ \t]+<[^>\n]+>\n[ \t]*\d{1,2}/\d{1,2}/\d{2,4}[ \t]+\d{1,2}:\d{2}[ \t]*(?:AM|PM).*',
    re.DOTALL,
)
_RE_EXT_REPLY = re.compile(
    r'\n[ \t]*(?:"[^"\n]+"|[A-Za-z][A-Za-z\s]+?)[ \t]+<[^>\n]+>[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*',
    re.DOTALL,
)
_RE_BARE_REPLY = re.compile(
    r'\n[ \t]*\S+@\S+[ \t]+on[ \t]+\d{1,2}/\d{1,2}/\d{2,4}.*',
    re.DOTALL,
)
_RE_INLINE_HEADER = re.compile(
    r"^\s*(From|To|Sent|Date|Subject|Cc|Bcc)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_GMAIL_REPLY = re.compile(
    r"\nOn\s+\w{3},?.+?wrote:.*",
    re.IGNORECASE | re.DOTALL,
)

_REPLY_STRIP_PATTERNS: list[re.Pattern] = [
    _RE_FORWARD_BLOCK,
    _RE_REPLY_CHAIN,
    _RE_GMAIL_REPLY,
    _RE_LOTUS_REPLY,
    _RE_LOTUS_NAMELINE,
    _RE_LOTUS_ON,
    _RE_QUOTED_NAME_REPLY,
    _RE_EXT_REPLY,
    _RE_BARE_REPLY,
    _RE_INLINE_HEADER,
]

_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_WHITESPACE  = re.compile(r"[ \t]+")

_SUBSTITUTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),                    "URL"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  "EMAIL"),
    (re.compile(r"\b[A-Za-z][A-Za-z\s]+/[A-Z]+/[A-Z]+@[A-Z]+\b"),          "EMAIL"),  # Enron routing
    (
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{2,4}\b"
            r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
            r"|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b",
            re.IGNORECASE,
        ),
        "DATE",
    ),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b", re.IGNORECASE), "TIME"),
    (re.compile(r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b|\b\d{10}\b"),             "PHONE"),
    (re.compile(r"\b[A-Z]{1,4}[-_]?\d{5,}\b|\b\d{6,}\b"),                    "ORDER_ID"),
    (re.compile(r"\b[A-Z][a-z]+,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),         "LOCATION"),
]

# The Enron maildir was created on Unix; files named "1." "2." etc. are valid
# on NTFS but Python's open() normalises away trailing dots via the Windows API.
# We use FindFirstFileW / FindNextFileW directly to retrieve the 8.3 short name
# for each entry, then open the file via that name instead.
try:
    import ctypes
    import ctypes.wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLow", ctypes.wintypes.DWORD), ("dwHigh", ctypes.wintypes.DWORD)]

    class _WIN32_FIND_DATAW(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes",   ctypes.wintypes.DWORD),
            ("ftCreationTime",     _FILETIME),
            ("ftLastAccessTime",   _FILETIME),
            ("ftLastWriteTime",    _FILETIME),
            ("nFileSizeHigh",      ctypes.wintypes.DWORD),
            ("nFileSizeLow",       ctypes.wintypes.DWORD),
            ("dwReserved0",        ctypes.wintypes.DWORD),
            ("dwReserved1",        ctypes.wintypes.DWORD),
            ("cFileName",          ctypes.c_wchar * 260),
            ("cAlternateFileName", ctypes.c_wchar * 14),
        ]

    _k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    _k32.FindFirstFileW.restype  = ctypes.c_void_p
    _k32.FindFirstFileW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(_WIN32_FIND_DATAW)]
    _k32.FindNextFileW.restype   = ctypes.wintypes.BOOL
    _k32.FindNextFileW.argtypes  = [ctypes.c_void_p, ctypes.POINTER(_WIN32_FIND_DATAW)]
    _k32.FindClose.restype       = ctypes.wintypes.BOOL
    _k32.FindClose.argtypes      = [ctypes.c_void_p]
    _INVALID_HANDLE              = ctypes.c_void_p(-1).value

    def _iter_dir(folder_str: str) -> Iterator[tuple[str, str, bool]]:
        data = _WIN32_FIND_DATAW()
        h = _k32.FindFirstFileW(folder_str + "\\*", ctypes.byref(data))
        if h is None or h == _INVALID_HANDLE:
            return
        try:
            while True:
                long  = data.cFileName
                short = data.cAlternateFileName or long
                is_dir = bool(data.dwFileAttributes & 0x10)
                if long not in (".", ".."):
                    yield long, short, is_dir
                if not _k32.FindNextFileW(h, ctypes.byref(data)):
                    break
        finally:
            _k32.FindClose(h)

except AttributeError:
    def _iter_dir(folder_str: str) -> Iterator[tuple[str, str, bool]]:  # type: ignore[misc]
        for entry in pathlib.Path(folder_str).iterdir():
            yield entry.name, entry.name, entry.is_dir()


def parse_raw_email(raw: str) -> Optional[dict]:
    try:
        msg = email.message_from_string(raw)
    except Exception:
        return None

    headers = {
        "message_id": msg.get("Message-ID", "").strip(),
        "date":        msg.get("Date", "").strip(),
        "from":        msg.get("From", "").strip(),
        "to":          msg.get("To", "").strip(),
        "subject":     msg.get("Subject", "").strip(),
        "x_folder":    msg.get("X-Folder", "").strip(),
        "x_origin":    msg.get("X-Origin", "").strip(),
    }

    body = _extract_body(msg)
    if body is None:
        return None

    return {"headers": headers, "raw_body": body}


def _extract_body(msg: email.message.Message) -> Optional[str]:
    # Prefer text/plain; fall back to HTML for modern clients that omit a plain part
    html_part = None
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            return _decode_payload(part)
        if part.get_content_type() == "text/html" and html_part is None:
            html_part = part
    if html_part is not None:
        raw_html = _decode_payload(html_part)
        if raw_html:
            return BeautifulSoup(raw_html, "html.parser").get_text(separator="\n")
    return None


def _decode_payload(part: email.message.Message) -> Optional[str]:
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if payload is None:
        return None
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("latin-1", errors="replace")


def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text)


def isolate_newest_message(body: str) -> str:
    for pattern in _REPLY_STRIP_PATTERNS:
        body = pattern.sub("", body)
    return body


def normalize_whitespace(text: str) -> str:
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def normalize_high_variance(text: str) -> str:
    for pattern, replacement in _SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


_RE_PLACEHOLDER = re.compile(r"\b(EMAIL|URL|DATE|TIME|PHONE|ORDER_ID|LOCATION)\b")

def is_usable(body: str, min_body_chars: int = MIN_BODY_CHARS) -> bool:
    stripped = body.strip()
    if len(stripped) < min_body_chars:
        return False
    words = stripped.split()
    if not words:
        return False
    token_ratio = len(_RE_PLACEHOLDER.findall(stripped)) / len(words)
    return token_ratio < 0.5


def truncate(body: str) -> str:
    return body[:MAX_BODY_CHARS]


def clean_email(raw: str, min_body_chars: int = MIN_BODY_CHARS) -> Optional[dict]:
    """
    Full pipeline applied identically to Enron and client emails.
    Returns a cleaned record dict, or None if the email is filtered out.
    """
    parsed = parse_raw_email(raw)
    if parsed is None:
        return None

    body = parsed["raw_body"]
    body = fix_encoding(body)
    body = isolate_newest_message(body)
    body = normalize_whitespace(body)
    body = normalize_high_variance(body)

    if not is_usable(body, min_body_chars):
        return None

    return {
        "message_id": parsed["headers"]["message_id"],
        "date":        parsed["headers"]["date"],
        "from":        parsed["headers"]["from"],
        "to":          parsed["headers"]["to"],
        "subject":     parsed["headers"]["subject"],
        "x_folder":    parsed["headers"]["x_folder"],
        "x_origin":    parsed["headers"]["x_origin"],
        "body":        truncate(body),
    }


def _collect_enron_paths(root: pathlib.Path, sent_only: bool) -> list[tuple[str, str, str]]:
    items = []
    for person_long, person_short, is_dir in _iter_dir(str(root)):
        if not is_dir:
            continue
        person_path = root / person_short
        for folder_long, folder_short, f_is_dir in _iter_dir(str(person_path)):
            if not f_is_dir:
                continue
            if sent_only and folder_long.lower() not in SENT_FOLDERS:
                continue
            folder_path = person_path / folder_short
            for _, fname_short, is_file_dir in _iter_dir(str(folder_path)):
                if not is_file_dir:
                    items.append((str(folder_path / fname_short), person_long, folder_long))
    return items


def _filter_by_min_count(
    paths: list[tuple[str, str, str]], min_count: int
) -> tuple[list[tuple[str, str, str]], int, int]:
    by_author: dict[str, list] = {}
    for item in paths:
        by_author.setdefault(item[1], []).append(item)
    kept = {a: p for a, p in by_author.items() if len(p) >= min_count}
    n_dropped = len(by_author) - len(kept)
    return [item for p in kept.values() for item in p], len(kept), n_dropped


def _clean_batch(batch: list[tuple[str, str, str]]) -> list[dict]:
    results = []
    for path_str, author, folder in batch:
        try:
            raw = pathlib.Path(path_str).read_text(encoding="latin-1")
        except OSError:
            continue
        cleaned = clean_email(raw)
        if cleaned is None:
            continue
        cleaned["author"] = author
        cleaned["folder"] = folder
        cleaned["source"] = "enron"
        results.append(cleaned)
    return results


def _run_batches(batches: list, workers: int) -> list[dict]:
    wrap = _tqdm if _tqdm else (lambda it, **_: it)
    all_records: list[dict] = []

    if workers == 1:
        it = wrap(map(_clean_batch, batches), total=len(batches), desc="cleaning", unit="batch")
        for batch_result in it:
            all_records.extend(batch_result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            it = wrap(pool.map(_clean_batch, batches, chunksize=4), total=len(batches), desc="cleaning", unit="batch")
            for batch_result in it:
                all_records.extend(batch_result)

    return all_records


def load_enron(
    root: pathlib.Path = ENRON_ROOT,
    sent_only: bool = True,
    min_emails_per_author: int = 100,
    workers: Optional[int] = None,
    exclude_ids: Optional[set] = None,
) -> pd.DataFrame:
    """
    Walk the Enron maildir tree and return a cleaned DataFrame.

    sent_only=True restricts to sent folders so every email is labelled with
    its actual author. min_emails_per_author drops authors with too little data.

    exclude_ids: set of message_id strings to drop after cleaning — pass the
                 IDs from test_set_ids.txt so the test set is never seen during
                 training.
    """
    if workers is None:
        workers = os.cpu_count() or 1

    print("Scanning directory tree…")
    all_paths = _collect_enron_paths(root, sent_only)
    filtered, n_authors, n_dropped = _filter_by_min_count(all_paths, min_emails_per_author)

    print(f"  {n_authors} authors  ({n_dropped} dropped, <{min_emails_per_author} emails)")
    print(f"  {len(filtered):,} files to process  (workers={workers})")

    batches = [filtered[i : i + BATCH_SIZE] for i in range(0, len(filtered), BATCH_SIZE)]
    df = pd.DataFrame(_run_batches(batches, workers))

    if exclude_ids:
        before = len(df)
        df = df[~df["message_id"].isin(exclude_ids)].reset_index(drop=True)
        print(f"  Excluded {before - len(df):,} test-set emails ({len(df):,} remaining)")

    return df


def load_client_emails(records_iter, min_body_chars: int = 15) -> pd.DataFrame:
    """
    Clean an iterable of client email dicts.
    Each dict must have a 'raw' key with the full RFC 2822 message string.
    Any additional keys (e.g. 'author') are passed through unchanged.
    """
    results = []
    for record in records_iter:
        raw = record.get("raw", "")
        cleaned = clean_email(raw, min_body_chars)
        if cleaned is None:
            continue
        extra = {k: v for k, v in record.items() if k != "raw"}
        cleaned.update(extra)
        cleaned["source"] = "client"
        results.append(cleaned)
    return pd.DataFrame(results)


def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_parquet(path, index=False, compression="snappy")
    print(f"Saved parquet → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def save_jsonl(df: pd.DataFrame, path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            subj = row.subject.strip() if row.subject else ""
            text = f"Subject: {subj}\n\n{row.body}" if subj else row.body
            fh.write(json.dumps({"text": text, "author": row.author}, ensure_ascii=False) + "\n")
    print(f"Saved JSONL  → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sent-only", action="store_true",
                        help="Only include emails from sent folders")
    parser.add_argument("--out-dir", default=".", type=pathlib.Path)
    args = parser.parse_args()

    print("Loading Enron dataset…")
    df = load_enron(sent_only=args.sent_only)
    print(f"  {len(df):,} emails after cleaning  ({df['author'].nunique()} authors)")

    out = args.out_dir
    save_parquet(df, out / "enron_cleaned.parquet")
    save_jsonl(df,   out / "enron_cleaned.jsonl")

    print("\nSample:")
    print(df[["author", "folder", "subject", "body"]].head(5).to_string())
