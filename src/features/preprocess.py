#!/usr/bin/env python3
"""
PhishGuard – English-Only Email Preprocessing Engine
Changes:
- Added `is_english_quality_check` to filter out non-English text.
- Modified pipeline to DROP rows that fail English detection.
- Fixed clean_for_embeddings to never return empty strings (prevents CSV misalignment).
"""

import os
import re
import time
import math
import hashlib
import logging
import email
from multiprocessing import Pool, cpu_count
from collections import Counter
from typing import Dict, List, Optional, Iterable
import sys
import csv

# Increase CSV field limit
_max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int / 10)

import pandas as pd
from bs4 import BeautifulSoup
from urlextract import URLExtract
from urllib.parse import urlparse
from email.utils import parseaddr

# =============================
# CONFIG
# =============================
RAW_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\raw"
PROCESSED_DATA_PATH = r"C:\Users\hassan\Desktop\phishing_detection_system\data\processed"
OUTPUT_FILE = "phishguard_features.csv"

MAX_WORKERS = max(1, cpu_count() - 1)
CSV_CHUNKSIZE = 5000

# Logging setup
logging.getLogger("urlextract").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LABEL_MAP = {
    "spam": 1, "phish": 1, "phishing": 1, "1": 1, "yes": 1, "true": 1, "malicious": 1,
    "ham": 0, "legit": 0, "legitimate": 0, "0": 0, "no": 0, "false": 0
}

URGENT_WORDS = {
    "urgent", "immediately", "asap", "now", "today", "within 24 hours",
    "limited time", "expires", "deadline", "final notice", "last chance",
    "verify", "verification required", "confirm", "validate",
    "suspended", "suspend", "locked", "blocked", "restricted",
    "unauthorized", "unusual activity", "compromised",
    "security alert", "account alert",
    "password", "login", "sign in", "sign-in", "reset password",
    "update credentials", "re-authenticate",
    "invoice", "payment", "paid", "overdue", "refund",
    "billing", "wire transfer", "gift card",
    "transaction", "purchase", "receipt",
    "legal action", "court", "lawsuit", "law enforcement",
    "irs", "tax", "penalty", "fine",
    "click below", "click here", "open attachment",
    "download attached file", "review document"
}

# =============================
# UTILITIES
# =============================
def normalize(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

# Regex Patterns
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
DIGIT_RE = re.compile(r"\d+")
HTML_TAG_RE = re.compile(r"<.*?>")
MULTI_SPACE_RE = re.compile(r"\s+")
REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")

# --- NEW: English Detection Heuristic ---
def is_english_quality_check(text: str) -> bool:
    """
    Returns True if text appears to be English.
    Logic: If > 40% of the alphabetic characters are ASCII (English), keep it.
    This handles cases where a footer might be non-English but the body is English.
    """
    if not text or len(text) < 3:
        return False # Too short to judge, likely garbage
    
    # Remove whitespace and numbers to check only letters
    text_only = re.sub(r"[\s\d\W]", "", text)
    if not text_only: 
        return True # It was likely just numbers/symbols (e.g. an invoice), keep it.
        
    ascii_chars = sum(1 for c in text_only if c.isascii())
    total_chars = len(text_only)
    
    if total_chars == 0: return False
    
    # If 90% of characters are non-ASCII, it's definitely not English
    ratio = ascii_chars / total_chars
    return ratio > 0.5

def clean_for_embeddings(text: str) -> str:
    """
    Clean text for FastText (ENGLISH ONLY).
    Safeguards against empty returns.
    """
    if not text:
        return "<EMPTY>"

    import html
    text = html.unescape(text)

    # Strip HTML
    if bool(re.search(r'<[^>]+>', text)):
        try:
            text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
        except Exception:
            text = HTML_TAG_RE.sub(" ", text)

    # Replace entities
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = DIGIT_RE.sub(" <NUM> ", text)

    # Strict English Filter: Remove non-alphanumeric (except placeholders)
    text = re.sub(r"[^a-zA-Z0-9\s<>]", " ", text)

    text = REPEATED_CHARS_RE.sub(r"\1", text)
    text = text.lower()
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    # Final check: If the cleaner stripped everything, return a token
    return text if text else "<EMPTY>"

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    counts = Counter(s)
    return -sum((c / len(s)) * math.log2(c / len(s)) for c in counts.values())

def compute_hash(subject: str, body: str, sender: str) -> str:
    base = f"{subject}|{body}|{sender}"
    return hashlib.sha256(base.encode("utf-8", errors="ignore")).hexdigest()

def normalize_label(value) -> int:
    if value is None: return -1
    raw = str(value).strip().lower()
    return LABEL_MAP.get(raw, -1)

# =============================
# URL & HEADER UTILS
# =============================
_URL_EXTRACTOR = URLExtract()

def safe_find_urls(text: str) -> List[str]:
    if not text: return []
    try:
        safe_text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        return _URL_EXTRACTOR.find_urls(safe_text) or []
    except Exception:
        return []

_SPFFLAGS = {"pass": 1, "fail": 0, "softfail": 0, "neutral": -1, "none": -1}
_DKIMFLAGS = {"pass": 1, "fail": 0, "neutral": -1, "none": -1}
_DMARCFLAGS = {"pass": 1, "fail": 0, "none": -1, "quarantine": 0, "reject": 0}
_SPF_RE = re.compile(r'\bspf=([a-zA-Z0-9_-]+)\b', flags=re.IGNORECASE)
_DKIM_RE = re.compile(r'\bdkim=([a-zA-Z0-9_-]+)\b', flags=re.IGNORECASE)
_DMARC_RE = re.compile(r'\bdmarc=([a-zA-Z0-9_-]+)\b', flags=re.IGNORECASE)

def _extract_flag(pattern, text, mapping, default=-1):
    if not text: return default
    m = pattern.search(text)
    return mapping.get(m.group(1).lower(), default) if m else default

def parse_auth_from_headers(headers_text: str) -> Dict[str, object]:
    result = {
        "auth_headers_present": False, "spf_result": -1, "dkim_result": -1, 
        "dmarc_result": -1, "return_path_domain": "", "received_count": 0
    }
    if not headers_text: return result
    lower = headers_text.lower()
    result["auth_headers_present"] = any(k in lower for k in ("authentication-results:", "received-spf", "dkim-signature", "dmarc="))
    result["spf_result"] = _extract_flag(_SPF_RE, headers_text, _SPFFLAGS)
    result["dkim_result"] = _extract_flag(_DKIM_RE, headers_text, _DKIMFLAGS)
    result["dmarc_result"] = _extract_flag(_DMARC_RE, headers_text, _DMARCFLAGS)
    
    m_rp = re.search(r'(?mi)^Return-Path:\s*<?([^>\r\n]+)>?', headers_text)
    if m_rp:
        rp_addr = parseaddr(m_rp.group(1).strip())[1]
        result["return_path_domain"] = rp_addr.split("@", 1)[1].lower() if "@" in rp_addr else ""
    
    result["received_count"] = len(re.findall(r'(?mi)^\s*Received:', headers_text))
    return result

# =============================
# FEATURE ENGINE (With Filtering)
# =============================
def build_features(subject: str, body: str, sender: str, urls: List[str], 
                   html_present: int, attachments: List[str], label: int, 
                   auth_info=None, header_fields=None) -> Optional[Dict]:
    """
    Returns Dict of features, OR Returns None if the text is not English.
    """
    
    # 1. English Filter Check
    combined_check = f"{subject} {body}"
    if not is_english_quality_check(combined_check):
        return None  # DROP THIS ROW

    auth_info = auth_info or {}
    header_fields = header_fields or {}
    domains = [urlparse(u).netloc for u in urls if urlparse(u).netloc]
    
    cleaned_text = clean_for_embeddings(combined_check)
    urgent_set = {w for w in URGENT_WORDS if w in combined_check.lower()}

    return {
        "subject": subject,
        "body": body,
        "clean_text": cleaned_text, # Guaranteed not to be empty string
        "sender": sender,
        "from_header": header_fields.get("from_header", ""),
        "recipient": header_fields.get("recipient", ""),
        "return_path": header_fields.get("return_path", ""),
        "to_header": header_fields.get("to_header", ""),
        "message_id": header_fields.get("message_id", ""),
        "x_mailer": header_fields.get("x_mailer", ""),
        "x_originating_ip": header_fields.get("x_originating_ip", ""),
        "content_type": header_fields.get("content_type", ""),
        "urls": ";".join(urls)[:32000],
        "domains": ";".join(dict.fromkeys(domains))[:32000],
        "ip_urls": list(u for u in urls if re.match(r"^https?://(?:\d{1,3}\.){3}\d{1,3}\b", u)),
        "urgent_words_count": len(urgent_set),
        "digit_ratio": sum(c.isdigit() for c in (body or "")) / max(len(body or ""), 1),
        "body_entropy": shannon_entropy(body or ""),
        "html_present": int(bool(html_present)),
        "attachment_names": ";".join(attachments) if attachments else "",
        "auth_headers_present": int(auth_info.get("auth_headers_present", 0)),
        "spf_result": int(auth_info.get("spf_result", -1)),
        "dkim_result": int(auth_info.get("dkim_result", -1)),
        "dmarc_result": int(auth_info.get("dmarc_result", -1)),
        "return_path_domain": auth_info.get("return_path_domain", ""),
        "received_count": int(auth_info.get("received_count", 0)),
        "label": int(label)
    }

# =============================
# EML PARSING
# =============================
def parse_eml(path: str) -> Optional[Dict]:
    try:
        msg = email.message_from_bytes(open(path, "rb").read())
    except Exception:
        return None

    subject = normalize(msg.get("Subject", ""))
    sender = normalize(msg.get("From", ""))
    
    body_parts = []
    html_present = 0
    attachments = []

    for part in msg.walk():
        try:
            ctype = part.get_content_type()
            disp = str(part.get_content_disposition() or "")
            payload = part.get_payload(decode=True)
            if not payload: continue
            
            # Try decoding as utf-8, fallback to latin-1 for western languages
            try:
                decoded = payload.decode("utf-8", errors="strict")
            except UnicodeError:
                decoded = payload.decode("latin-1", errors="ignore")
                
            if ctype == "text/plain" and "attachment" not in disp:
                body_parts.append(decoded)
            elif ctype == "text/html":
                html_present = 1
                body_parts.append(BeautifulSoup(decoded, "html.parser").get_text(" ", strip=False))
            if part.get_filename():
                attachments.append(part.get_filename())
        except Exception:
            continue

    body = normalize(" ".join(body_parts))
    urls = safe_find_urls(body)
    
    # Header Extraction
    header_fields = {
        "from_header": normalize(msg.get("From", "")),
        "to_header": normalize(msg.get("To", "")),
        "recipient": normalize(msg.get("Delivered-To", "") or msg.get("Envelope-To", "") or msg.get("To", "")),
        "return_path": normalize(msg.get("Return-Path", "")),
        "message_id": normalize(msg.get("Message-ID", "")),
        "x_mailer": normalize(msg.get("X-Mailer", "") or msg.get("X-Mailing", "")),
        "x_originating_ip": normalize(msg.get("X-Originating-IP", "")),
        "content_type": normalize(msg.get("Content-Type", ""))
    }
    
    headers_text = "\n".join(f"{k}: {v}" for k, v in msg.items())
    auth_info = parse_auth_from_headers(headers_text)

    return build_features(subject, body, sender, urls, html_present, attachments, -1, auth_info, header_fields)

# =============================
# CSV PARSING
# =============================
def parse_csv_row(row: Dict) -> Optional[Dict]:
    label = normalize_label(row.get("phish") or row.get("label") or row.get("class") or row.get("spam"))
    
    header_fields = {
        "from_header": normalize(row.get("from") or row.get("from_header")),
        "to_header": normalize(row.get("to") or row.get("to_header")),
        "recipient": normalize(row.get("delivered-to") or row.get("recipient") or row.get("to")),
        "return_path": normalize(row.get("return-path") or row.get("return_path")),
        "message_id": normalize(row.get("message-id") or row.get("message_id")),
        "x_mailer": normalize(row.get("x-mailer") or row.get("x_mailer")),
        "x_originating_ip": normalize(row.get("x-originating-ip") or row.get("x_originating_ip")),
        "content_type": normalize(row.get("content-type") or row.get("content_type"))
    }

    # Auth parsing
    headers_text = str(row.get("raw_headers") or row.get("headers") or "")
    auth_info = parse_auth_from_headers(headers_text) if headers_text else {}

    # Find Body/Subject
    text_fields = {k: normalize(v) for k, v in row.items() if isinstance(v, str) and normalize(v)}
    if not text_fields: return None
    
    sorted_fields = sorted(text_fields.items(), key=lambda x: len(x[1]), reverse=True)
    body = sorted_fields[0][1]
    subject = sorted_fields[1][1] if len(sorted_fields) > 1 else ""
    sender = normalize(row.get("from") or row.get("sender") or "")

    urls = safe_find_urls(body)

    # Note: build_features now returns None if text is not English
    return build_features(subject, body, sender, urls, 0, [], label, auth_info, header_fields)

# =============================
# MAIN LOGIC
# =============================
def iter_csv_rows(path: str) -> Iterable[Dict]:
    try:
        # Added encoding="utf-8" and encoding_errors="ignore" to skip bad bytes
        for chunk in pd.read_csv(path, dtype=str, engine="python", on_bad_lines="skip", 
                               chunksize=CSV_CHUNKSIZE, encoding="utf-8", encoding_errors="ignore"):
            chunk = chunk.fillna("")
            for record in chunk.to_dict(orient="records"):
                yield record
    except Exception as e:
        logging.error("Failed reading CSV %s: %s", path, e)

def process_emls(files: List[str]) -> List[Dict]:
    if not files: return []
    with Pool(MAX_WORKERS) as pool:
        results = pool.map(parse_eml, files)
    # Filter out None values (non-English emails are now None)
    return [r for r in results if r]

def load_raw_data(raw_dir: str) -> pd.DataFrame:
    records = []
    eml_files = []
    csv_files = []

    for root, _, files in os.walk(raw_dir):
        for f in files:
            full = os.path.join(root, f)
            if f.lower().endswith(".eml"):
                eml_files.append(full)
            elif f.lower().endswith(".csv"):
                csv_files.append(full)

    if eml_files:
        logging.info("Processing EML files...")
        records.extend(process_emls(eml_files))

    for csv_path in csv_files:
        logging.info("Reading CSV: %s", csv_path)
        for row in iter_csv_rows(csv_path):
            try:
                rec = parse_csv_row(row)
                if rec: # Only append if rec is not None (i.e. is English)
                    records.append(rec)
            except Exception:
                continue

    return pd.DataFrame(records)

def main():
    start = time.time()
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE)

    old_df = pd.read_csv(output_path, dtype=str).fillna("") if os.path.exists(output_path) else pd.DataFrame()
    new_df = load_raw_data(RAW_DATA_PATH)

    logging.info("New English records extracted: %d", len(new_df))

    if "label" not in new_df.columns: new_df["label"] = -1
    if not old_df.empty and "label" not in old_df.columns: old_df["label"] = -1

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)

    if combined.empty:
        logging.warning("No data loaded.")
        return

    combined["_hash"] = combined.apply(lambda r: compute_hash(str(r.get("subject", "")), str(r.get("body", "")), str(r.get("sender", ""))), axis=1)
    combined.drop_duplicates("_hash", inplace=True)
    combined.drop(columns=["_hash"], inplace=True)

    # Ensure clean_text is never empty before saving
    if "clean_text" in combined.columns:
        combined["clean_text"] = combined["clean_text"].replace("", "<EMPTY>")

    try:
        combined["label"] = combined["label"].astype(int)
    except Exception:
        combined["label"] = combined["label"].apply(lambda v: int(v) if str(v).isdigit() else -1)

    combined.to_csv(output_path, index=False)
    logging.info("Finished in %.2fs. Total rows: %d", time.time() - start, len(combined))

# =============================
# PRODUCTION REAL-TIME FUNCTION
# =============================
from bs4 import BeautifulSoup
import email

def production_preprocessing(raw_email: str) -> Optional[Dict]:
    """
    Takes raw email string → returns processed feature dict.
    Patched to retain raw metadata for Expert Overrides.
    """
    try:
        msg = email.message_from_string(raw_email)
    except Exception:
        return None

    subject = normalize(msg.get("Subject", ""))
    sender = normalize(msg.get("From", ""))

    body_parts = []
    html_present = 0
    attachments = []
    
    # NEW: Create a dedicated list to catch hidden HTML URLs
    extracted_html_urls = []

    for part in msg.walk():
        try:
            ctype = part.get_content_type()
            disp = str(part.get_content_disposition() or "")
            payload = part.get_payload(decode=True)

            if not payload:
                continue

            try:
                decoded = payload.decode("utf-8", errors="strict")
            except UnicodeError:
                decoded = payload.decode("latin-1", errors="ignore")

            if ctype == "text/plain" and "attachment" not in disp:
                body_parts.append(decoded)

            elif ctype == "text/html":
                html_present = 1
                soup = BeautifulSoup(decoded, "html.parser")
                
                # BUG 1 FIX: Extract href links before stripping HTML
                for a_tag in soup.find_all('a', href=True):
                    extracted_html_urls.append(a_tag['href'])
                
                body_parts.append(soup.get_text(" ", strip=False))

            if part.get_filename():
                attachments.append(part.get_filename())

        except Exception:
            continue

    body = normalize(" ".join(body_parts))
    
    # Combine plain text URLs with the hidden HTML URLs
    urls = safe_find_urls(body) + extracted_html_urls
    
    # Remove duplicates just in case
    urls = list(set(urls))

    header_fields = {
        "from_header": normalize(msg.get("From", "")),
        "to_header": normalize(msg.get("To", "")),
        "recipient": normalize(msg.get("Delivered-To", "") or msg.get("To", "")),
        "return_path": normalize(msg.get("Return-Path", "")),
        "message_id": normalize(msg.get("Message-ID", "")),
        "x_mailer": normalize(msg.get("X-Mailer", "")),
        "x_originating_ip": normalize(msg.get("X-Originating-IP", "")),
        "content_type": normalize(msg.get("Content-Type", "")),
    }

    headers_text = "\n".join(f"{k}: {v}" for k, v in msg.items())
    auth_info = parse_auth_from_headers(headers_text)

    features = build_features(
        subject, body, sender, urls,
        html_present, attachments,
        label=-1,
        auth_info=auth_info,
        header_fields=header_fields
    )

    # BUG 2 FIX: Force the raw metadata into the dictionary 
    # so predictor.py can use them for security heuristics
    if features is not None:
        features["urls"] = urls
        features["from_header"] = header_fields["from_header"]
        features["return_path"] = header_fields["return_path"] 
        features["raw_text"] = raw_email # Useful if you need to regex the raw headers later

    return features    
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        raw_email = sys.stdin.read()
        result = production_preprocessing(raw_email)
        print(result)
    else:
        main()

