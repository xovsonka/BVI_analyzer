from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from email.utils import parseaddr
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from classification_config import CONFIG, LABEL_TO_ID


RANDOM_STATE = 42
MIN_TEXT_LENGTH = 30

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
EMAIL_TOKEN_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
HEADER_VALUE_RE = re.compile(r"(?im)^(?P<name>[a-z0-9-]+):\s*(?P<value>.+)$")
ANCHOR_RE = re.compile(
    r"<a\b[^>]*href\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "cutt.ly",
}
SUSPICIOUS_TLDS = {"zip", "review", "country", "click", "work", "gq", "tk", "ml", "cf"}
KNOWN_BRANDS = {
    "paypal",
    "microsoft",
    "google",
    "apple",
    "amazon",
    "outlook",
    "facebook",
    "instagram",
    "linkedin",
}
DISPLAY_NAME_BRANDS = KNOWN_BRANDS | {"bank", "usaa"}

URGENT_WORDS = {"action", "required", "immediately", "suspended", "urgent", "deadline", "verify"}
CREDENTIAL_WORDS = {"account", "security", "signin", "password", "confirm", "login"}
THREAT_WORDS = {"terminate", "locked", "disable", "expired", "suspend", "blocked"}
FINANCIAL_WORDS = {"payment", "invoice", "bank", "billing", "fund", "card", "transfer"}

NEW_FEATURES = [
    "display_name_spoof_flag",
    "message_id_domain_mismatch",
    "external_domain_url_count",
    "sender_url_domain_mismatch_ratio",
    "obfuscated_url_count",
    "mismatched_anchor_count",
    "punycode_url_count",
]

BACKFILL_RAW_COLS = [
    "subject_raw",
    "body_raw",
    "text_raw",
    "sender",
    "receiver",
    "date",
]


def normalize_whitespace(text: str) -> str:
    out = str(text or "")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return normalize_whitespace(text.replace("\x00", " "))


def normalize_model_text(text: str) -> str:
    out = clean_text(text)
    out = URL_RE.sub(" <URL> ", out)
    out = EMAIL_TOKEN_RE.sub(" <EMAIL> ", out)
    out = NUMBER_TOKEN_RE.sub(" <NUM> ", out)
    return normalize_whitespace(out)


def combine_subject_body_raw(subject: str, body: str) -> str:
    s = clean_text(subject)
    b = clean_text(body)
    if s and b:
        return f"{s}\n\n{b}"
    return s or b


def combine_subject_body(subject: str, body: str) -> str:
    s = normalize_model_text(subject)
    b = normalize_model_text(body)
    if s and b:
        return f"{s}\n\n{b}"
    return s or b


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="ignore")).hexdigest()


def make_child_row_id(parent_row_id: str, method: str, salt: str = "") -> str:
    return stable_hash(f"{parent_row_id}::{method}::{salt}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sample_df(df: pd.DataFrame, n: int | None, random_state: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def extract_header_value(text: str, header_name: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    pattern = re.compile(rf"(?im)^{re.escape(header_name)}:\s*(.+)$")
    match = pattern.search(text)
    return clean_text(match.group(1)) if match else ""


def count_header_occurrences(text: str, header_name: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    pattern = re.compile(rf"(?im)^{re.escape(header_name)}:\s*")
    return len(pattern.findall(text))


def extract_email_domain(value: str) -> str:
    _, addr = parseaddr(str(value or ""))
    if "@" not in addr:
        return ""
    return addr.split("@")[-1].strip().lower()


def registered_domain(host: str) -> str:
    host = str(host or "").strip(".").lower()
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    return ".".join(parts[-2:])


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def keyword_count(text: str, words: set[str]) -> int:
    tokens = WORD_RE.findall(str(text or "").lower())
    return sum(1 for token in tokens if token in words)


def parse_auth_results(value: str) -> tuple[str, str, str]:
    lower = str(value or "").lower()

    def _pick(token: str) -> str:
        match = re.search(rf"\b{token}\s*=\s*([a-z_]+)", lower)
        if not match:
            return "none"
        candidate = match.group(1)
        if candidate in {"pass", "fail", "softfail", "neutral", "temperror", "permerror", "none"}:
            return candidate
        return "none"

    return _pick("spf"), _pick("dkim"), _pick("dmarc")


def _extract_display_name(value: str) -> str:
    name, _ = parseaddr(str(value or ""))
    return clean_text(name).lower()


def _extract_anchor_mismatch_count(text: str) -> int:
    count = 0
    html = str(text or "")
    for href, anchor_text in ANCHOR_RE.findall(html):
        href_host = (urlparse(href).hostname or "").lower()
        if not href_host:
            continue
        anchor_clean = TAG_RE.sub(" ", anchor_text or "")
        for text_url in URL_RE.findall(anchor_clean):
            text_host = (urlparse(text_url).hostname or "").lower()
            if text_host and text_host != href_host:
                count += 1
                break
    return count


def _count_obfuscated_urls(urls: list[str]) -> int:
    count = 0
    for raw_url in urls:
        try:
            parsed = urlparse(raw_url)
        except ValueError:
            count += 1
            continue
        host = (parsed.netloc or "").lower()
        query = parsed.query or ""
        if "@" in raw_url or "%" in raw_url or len(query) > 80 or host.count(".") >= 4:
            count += 1
    return count


def _compute_url_bundle(text: str, sender_domain: str) -> dict[str, float]:
    urls = URL_RE.findall(str(text or ""))
    hosts: list[str] = []
    ip_url_count = 0
    shortener_url_count = 0
    punycode_url_count = 0
    suspicious_tld_count = 0

    for url in urls:
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            continue
        if not host:
            continue
        hosts.append(host)
        if IP_HOST_RE.fullmatch(host):
            ip_url_count += 1
        if host in SHORTENER_DOMAINS:
            shortener_url_count += 1
        if "xn--" in host:
            punycode_url_count += 1
        if "." in host and host.split(".")[-1] in SUSPICIOUS_TLDS:
            suspicious_tld_count += 1

    reg_urls = {registered_domain(host) for host in hosts if host}
    reg_sender = registered_domain(sender_domain)
    external_domain_url_count = 0
    sender_url_domain_mismatch_ratio = 0.0
    if reg_sender and reg_urls:
        external_domain_url_count = sum(1 for d in reg_urls if d and d != reg_sender)
        sender_url_domain_mismatch_ratio = float(external_domain_url_count / max(1, len(reg_urls)))

    return {
        "url_count": float(len(urls)),
        "ip_url_count": float(ip_url_count),
        "shortener_url_count": float(shortener_url_count),
        "unique_domain_count": float(len(set(hosts))),
        "punycode_url_count": float(punycode_url_count),
        "suspicious_tld_count": float(suspicious_tld_count),
        "external_domain_url_count": float(external_domain_url_count),
        "sender_url_domain_mismatch_ratio": float(sender_url_domain_mismatch_ratio),
        "obfuscated_url_count": float(_count_obfuscated_urls(urls)),
        "mismatched_anchor_count": float(_extract_anchor_mismatch_count(text)),
    }


def _ensure_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "subject" not in out.columns and "subject_raw" in out.columns:
        out["subject"] = out["subject_raw"].astype(str).map(normalize_model_text)
    if "body" not in out.columns and "body_raw" in out.columns:
        out["body"] = out["body_raw"].astype(str).map(normalize_model_text)

    for col in ["subject", "body", "text", "subject_raw", "body_raw", "text_raw", "sender", "receiver", "date"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    if out["text_raw"].str.strip().eq("").all():
        out["text_raw"] = out.apply(
            lambda row: combine_subject_body_raw(row.get("subject_raw", ""), row.get("body_raw", "")),
            axis=1,
        )

    if out["text"].str.strip().eq("").all():
        out["text"] = out.apply(
            lambda row: combine_subject_body(row.get("subject", ""), row.get("body", "")),
            axis=1,
        )

    if out["subject_raw"].str.strip().eq("").all() and not out["subject"].str.strip().eq("").all():
        out["subject_raw"] = out["subject"].astype(str)
    if out["body_raw"].str.strip().eq("").all() and not out["body"].str.strip().eq("").all():
        out["body_raw"] = out["body"].astype(str)

    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_text_columns(df)

    signal_col = "text_raw" if "text_raw" in out.columns else "text"
    signal = out[signal_col].astype(str)

    from_header = signal.map(lambda text: extract_header_value(text, "From"))
    to_header = signal.map(lambda text: extract_header_value(text, "To"))
    reply_to_header = signal.map(lambda text: extract_header_value(text, "Reply-To"))
    return_path_header = signal.map(lambda text: extract_header_value(text, "Return-Path"))
    message_id_header = signal.map(lambda text: extract_header_value(text, "Message-ID"))
    auth_results_header = signal.map(lambda text: extract_header_value(text, "Authentication-Results"))
    received_hops_count = signal.map(lambda text: count_header_occurrences(text, "Received")).astype(int)

    sender_effective = out["sender"].astype(str)
    sender_missing = sender_effective.str.strip().eq("")
    sender_effective.loc[sender_missing] = from_header.loc[sender_missing]
    out["sender"] = sender_effective

    receiver_effective = out["receiver"].astype(str)
    receiver_missing = receiver_effective.str.strip().eq("")
    receiver_effective.loc[receiver_missing] = to_header.loc[receiver_missing]
    out["receiver"] = receiver_effective

    out["from_domain"] = out["sender"].map(extract_email_domain)
    out["reply_to_domain"] = reply_to_header.map(extract_email_domain)
    out["return_path_domain"] = return_path_header.map(extract_email_domain)
    out["receiver_domain"] = out["receiver"].map(extract_email_domain)
    out["message_id_domain"] = message_id_header.map(extract_email_domain)

    parsed_auth = auth_results_header.map(parse_auth_results)
    out["spf_result"] = parsed_auth.map(lambda item: item[0])
    out["dkim_result"] = parsed_auth.map(lambda item: item[1])
    out["dmarc_result"] = parsed_auth.map(lambda item: item[2])
    out["received_hops_count"] = received_hops_count

    url_bundles = [
        _compute_url_bundle(text=text, sender_domain=sender_domain)
        for text, sender_domain in zip(signal.tolist(), out["from_domain"].tolist())
    ]
    url_df = pd.DataFrame(url_bundles, index=out.index)
    for col in url_df.columns:
        out[col] = pd.to_numeric(url_df[col], errors="coerce").fillna(0.0)

    out["from_reply_mismatch"] = (
        (out["from_domain"] != "")
        & (out["reply_to_domain"] != "")
        & (out["from_domain"] != out["reply_to_domain"])
    ).astype(int)

    out["from_return_path_mismatch"] = (
        (out["from_domain"] != "")
        & (out["return_path_domain"] != "")
        & (out["from_domain"] != out["return_path_domain"])
    ).astype(int)

    out["sender_receiver_domain_mismatch"] = (
        (out["from_domain"] != "")
        & (out["receiver_domain"] != "")
        & (out["from_domain"] != out["receiver_domain"])
    ).astype(int)

    out["message_id_domain_mismatch"] = (
        (out["from_domain"] != "")
        & (out["message_id_domain"] != "")
        & (out["from_domain"] != out["message_id_domain"])
    ).astype(int)

    display_name = out["sender"].map(_extract_display_name)
    out["display_name_spoof_flag"] = [
        int(
            bool(name)
            and any(brand in name for brand in DISPLAY_NAME_BRANDS)
            and bool(domain)
            and not any(brand in domain for brand in DISPLAY_NAME_BRANDS)
        )
        for name, domain in zip(display_name.tolist(), out["from_domain"].tolist())
    ]

    brand_typosquat = []
    for sender_domain, text in zip(out["from_domain"].tolist(), signal.tolist()):
        candidates = set()
        if sender_domain and "." in sender_domain:
            candidates.add(sender_domain.split(".")[0])
        for url in URL_RE.findall(text):
            try:
                host = (urlparse(url).hostname or "").lower()
            except ValueError:
                continue
            if host and "." in host:
                candidates.add(host.split(".")[0])

        flag = 0
        for candidate in candidates:
            normalized = re.sub(r"[^a-z0-9]", "", candidate.lower())
            if not normalized:
                continue
            for brand in KNOWN_BRANDS:
                if normalized != brand and levenshtein(normalized, brand) == 1:
                    flag = 1
                    break
            if flag:
                break
        brand_typosquat.append(flag)
    out["brand_typosquat_flag"] = brand_typosquat

    text_for_keywords = out["subject_raw"].astype(str) + " " + out["body_raw"].astype(str)
    out["urgent_keyword_count"] = text_for_keywords.map(lambda text: keyword_count(text, URGENT_WORDS)).astype(int)
    out["credential_keyword_count"] = text_for_keywords.map(lambda text: keyword_count(text, CREDENTIAL_WORDS)).astype(int)
    out["threat_keyword_count"] = text_for_keywords.map(lambda text: keyword_count(text, THREAT_WORDS)).astype(int)
    out["financial_keyword_count"] = text_for_keywords.map(lambda text: keyword_count(text, FINANCIAL_WORDS)).astype(int)

    out["subject_len"] = out["subject"].astype(str).str.len().astype(int)
    out["text_len"] = out["text"].astype(str).str.len().astype(int)
    out["exclamation_count"] = out["text"].astype(str).str.count("!").astype(int)

    def _uppercase_ratio(value: str) -> float:
        letters = [ch for ch in str(value) if ch.isalpha()]
        if not letters:
            return 0.0
        return float(sum(1 for ch in letters if ch.isupper()) / len(letters))

    out["uppercase_ratio"] = out["text"].astype(str).map(_uppercase_ratio)
    out["digit_ratio"] = out["text"].astype(str).map(
        lambda value: float(sum(ch.isdigit() for ch in value) / len(value)) if value else 0.0
    )

    if "has_html_only" not in out.columns:
        html_flag = signal.str.contains(r"<html|<body|<a\s+href|</[a-z]+>", case=False, regex=True)
        plain_flag = signal.str.contains(r"\b[a-zA-Z]{3,}\b", regex=True)
        out["has_html_only"] = (html_flag & ~plain_flag).astype(int)

    if "attachment_count" not in out.columns:
        out["attachment_count"] = 0
    if "risky_attachment_ext_count" not in out.columns:
        out["risky_attachment_ext_count"] = 0

    out["attachment_count"] = pd.to_numeric(out["attachment_count"], errors="coerce").fillna(0).astype(int)
    out["risky_attachment_ext_count"] = (
        pd.to_numeric(out["risky_attachment_ext_count"], errors="coerce").fillna(0).astype(int)
    )

    out["trusted_domain_only"] = out["from_domain"].map(lambda value: int(registered_domain(value) in {"microsoft.com", "google.com", "apple.com", "amazon.com", "outlook.com", "office.com", "github.com", "bvi.local", "local.lab"}))

    for col in [
        "url_count",
        "ip_url_count",
        "shortener_url_count",
        "unique_domain_count",
        "from_reply_mismatch",
        "from_return_path_mismatch",
        "sender_receiver_domain_mismatch",
        "urgent_keyword_count",
        "credential_keyword_count",
        "threat_keyword_count",
        "financial_keyword_count",
        "subject_len",
        "text_len",
        "exclamation_count",
        "uppercase_ratio",
        "digit_ratio",
        "has_html_only",
        "attachment_count",
        "risky_attachment_ext_count",
        "received_hops_count",
        "suspicious_tld_count",
        "brand_typosquat_flag",
        "trusted_domain_only",
        "display_name_spoof_flag",
        "message_id_domain_mismatch",
        "external_domain_url_count",
        "sender_url_domain_mismatch_ratio",
        "obfuscated_url_count",
        "mismatched_anchor_count",
        "punycode_url_count",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def add_label_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_id"] = out["label"].map(LABEL_TO_ID)
    if out["label_id"].isna().any():
        invalid = sorted(out.loc[out["label_id"].isna(), "label"].dropna().unique())
        raise ValueError(f"Failed to map labels to label_id: {invalid}")
    out["label_id"] = out["label_id"].astype(int)
    return out


def make_mild_rebalanced_train(
    df: pd.DataFrame,
    majority_under_factor: float = 0.6,
    minority_over_factor: float = 1.5,
) -> pd.DataFrame:
    if not 0 < majority_under_factor <= 1.0:
        raise ValueError("majority_under_factor must be in (0, 1]")
    if minority_over_factor < 1.0:
        raise ValueError("minority_over_factor must be >= 1")

    counts = df["label"].value_counts().to_dict()
    if not counts:
        raise ValueError("Empty dataframe in make_mild_rebalanced_train")

    majority_label = max(counts, key=counts.get)
    median_count = int(pd.Series(list(counts.values())).median())
    parts: list[pd.DataFrame] = []

    for label in CONFIG.labels:
        part = df[df["label"] == label].copy()
        if part.empty:
            continue

        current = len(part)
        if label == majority_label:
            target = max(1, int(current * majority_under_factor))
        elif current < median_count:
            target = int(current * minority_over_factor)
        else:
            target = current

        if target <= current:
            sampled = part.sample(n=target, random_state=RANDOM_STATE).copy()
            sampled["is_synthetic"] = False
            sampled["synthetic_method"] = "original"
        else:
            extra = part.sample(n=target - current, replace=True, random_state=RANDOM_STATE).copy()
            extra["is_synthetic"] = True
            extra["synthetic_method"] = "oversample"
            extra["source"] = extra["source"].astype(str) + "_oversampled"
            base = part.copy()
            base["is_synthetic"] = False
            base["synthetic_method"] = "original"
            sampled = pd.concat([base, extra], ignore_index=True)

        parts.append(sampled)

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def _replace_terms(text: str, replacements: dict[str, str], rng: random.Random) -> str:
    out = str(text)
    for src, dst in replacements.items():
        if src in out.lower() and rng.random() < 0.5:
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return out


def _augment_subject(subject: str, label: str, rng: random.Random) -> str:
    subject_clean = clean_text(subject)
    if not subject_clean:
        defaults = {
            "spam": "Limited-time offer",
            "phishing": "Action required",
            "financial_fraud": "Confidential transfer notice",
            "legit": "General information",
        }
        subject_clean = defaults.get(label, "Important notice")
    prefixes = {
        "spam": ["Special", "Exclusive", "Offer"],
        "phishing": ["Security", "Urgent", "Verification"],
        "financial_fraud": ["Confidential", "Private", "Official"],
        "legit": ["Update", "Info", "Notice"],
    }
    if rng.random() < 0.6:
        return f"[{rng.choice(prefixes.get(label, ['Notice']))}] {subject_clean}"
    return subject_clean


def _augment_body(body: str, label: str, rng: random.Random) -> str:
    body_clean = clean_text(body)
    if not body_clean:
        return body_clean

    common = {
        "verify": "confirm",
        "account": "profile",
        "password": "access code",
        "urgent": "important",
        "immediately": "as soon as possible",
    }
    fraudish = {
        "transfer": "remit",
        "fund": "amount",
        "bank": "financial institution",
        "payment": "transaction",
    }

    out = _replace_terms(body_clean, common, rng)
    if label in {"financial_fraud", "spam", "phishing"}:
        out = _replace_terms(out, fraudish, rng)

    extra = {
        "spam": [
            "This offer is valid for a limited period.",
            "Please review the details and respond promptly.",
        ],
        "phishing": [
            "To avoid service interruption, complete verification now.",
            "Failure to act may temporarily limit your access.",
        ],
        "financial_fraud": [
            "Kindly handle this transfer discreetly and confidentially.",
            "Your cooperation is required to finalize this transaction.",
        ],
    }
    if label in extra and rng.random() < 0.7:
        out = f"{out}\n\n{rng.choice(extra[label])}"
    return clean_text(out)


def apply_llm_bonus_augmentation(
    df: pd.DataFrame,
    llm_labels: set[str],
    api_key: str,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    llm_timeout: int,
    llm_samples_per_label: int,
    llm_max_fraction_per_class: float,
    llm_sleep_ms: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    del api_key, llm_model, llm_temperature, llm_max_tokens, llm_timeout, llm_sleep_ms

    if llm_samples_per_label <= 0 or not llm_labels:
        return df.copy(), {label: 0 for label in llm_labels}
    if not 0 < llm_max_fraction_per_class <= 1.0:
        raise ValueError("llm_max_fraction_per_class must be in (0, 1]")

    rng = random.Random(RANDOM_STATE)
    generated_rows: list[pd.Series] = []
    generated_counts: dict[str, int] = {}

    for label in sorted(llm_labels):
        part = df[df["label"] == label].copy()
        if part.empty:
            generated_counts[label] = 0
            continue

        class_cap = max(1, int(len(part) * llm_max_fraction_per_class))
        n = min(llm_samples_per_label, class_cap)
        generated_counts[label] = n
        base = part.sample(n=n, replace=True, random_state=RANDOM_STATE)

        for idx, (_, row) in enumerate(base.iterrows()):
            new_subject = _augment_subject(str(row.get("subject_raw", row.get("subject", ""))), label, rng)
            new_body = _augment_body(str(row.get("body_raw", row.get("body", ""))), label, rng)

            new_row = row.copy()
            new_row["subject_raw"] = clean_text(new_subject)
            new_row["body_raw"] = clean_text(new_body)
            new_row["text_raw"] = combine_subject_body_raw(new_row["subject_raw"], new_row["body_raw"])
            new_row["subject"] = normalize_model_text(new_row["subject_raw"])
            new_row["body"] = normalize_model_text(new_row["body_raw"])
            new_row["text"] = combine_subject_body(new_row["subject"], new_row["body"])
            new_row["source"] = f"{row.get('source', 'unknown')}_llm"
            new_row["is_synthetic"] = True
            new_row["synthetic_method"] = "llm_rule"
            parent_id = str(row.get("row_id", ""))
            new_row["synthetic_parent_row_id"] = parent_id
            new_row["synthetic_seed"] = RANDOM_STATE
            new_row["row_id"] = make_child_row_id(parent_id, "llm_rule", str(idx))
            generated_rows.append(new_row)

    if not generated_rows:
        return df.copy(), generated_counts

    generated_df = pd.DataFrame(generated_rows)
    out = pd.concat([df.copy(), generated_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out, generated_counts


def add_hard_legit_samples(
    df: pd.DataFrame,
    ratio: float,
    strategy: str = "append",
    source_mode: str = "suffix",
) -> pd.DataFrame:
    if ratio <= 0:
        return df.copy()
    if strategy not in {"append", "replace"}:
        raise ValueError("strategy must be 'append' or 'replace'")
    if source_mode not in {"suffix", "keep", "shared"}:
        raise ValueError("source_mode must be 'suffix', 'keep', or 'shared'")

    legit_df = df[df["label"] == "legit"].copy()
    if legit_df.empty:
        return df.copy()

    n_variants = max(1, int(len(legit_df) * ratio))
    base = legit_df.sample(n=n_variants, replace=True, random_state=RANDOM_STATE)
    rng = random.Random(RANDOM_STATE)

    subject_prefixes = [
        "Monthly newsletter",
        "Product updates",
        "Community digest",
        "Service highlights",
        "Customer news",
    ]
    legit_snippets = [
        "This is an informational update. No account verification is required.",
        "You can manage email preferences or unsubscribe at any time.",
        "For full details, visit the official website and help center.",
    ]

    generated_rows: list[pd.Series] = []
    for _, row in base.iterrows():
        subject_raw = str(row.get("subject_raw", row.get("subject", "")))
        body_raw = str(row.get("body_raw", row.get("body", "")))
        new_subject_raw = f"{rng.choice(subject_prefixes)} - {clean_text(subject_raw or 'Latest updates')}"
        new_body_raw = (
            f"{clean_text(body_raw)}\n\n"
            f"{rng.choice(legit_snippets)}\n"
            "View in browser: https://example.com/newsletter\n"
            "Unsubscribe: https://example.com/unsubscribe"
        )

        new_row = row.copy()
        new_row["subject_raw"] = clean_text(new_subject_raw)
        new_row["body_raw"] = clean_text(new_body_raw)
        new_row["text_raw"] = combine_subject_body_raw(new_row["subject_raw"], new_row["body_raw"])
        new_row["subject"] = normalize_model_text(new_row["subject_raw"])
        new_row["body"] = normalize_model_text(new_row["body_raw"])
        new_row["text"] = combine_subject_body(new_row["subject"], new_row["body"])

        parent_source = str(row.get("source", "unknown"))
        if source_mode == "keep":
            new_row["source"] = parent_source
        elif source_mode == "shared":
            new_row["source"] = "hard_legit_shared"
        else:
            new_row["source"] = f"{parent_source}_hard_legit"

        new_row["is_synthetic"] = True
        new_row["synthetic_method"] = "hard_legit"
        generated_rows.append(new_row)

    generated_df = pd.DataFrame(generated_rows)
    if strategy == "append":
        out = pd.concat([df.copy(), generated_df], ignore_index=True)
    else:
        drop_n = min(len(generated_df), len(legit_df))
        legit_drop = legit_df.sample(n=drop_n, random_state=RANDOM_STATE).index
        kept = df.drop(index=legit_drop).copy()
        out = pd.concat([kept, generated_df.head(drop_n).copy()], ignore_index=True)

    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def _feature_nonzero_stats(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    total = max(1, len(df))
    for feature in features:
        if feature not in df.columns:
            stats[feature] = {"present": 0.0, "non_zero_rows": 0.0, "non_zero_ratio": 0.0, "mean": 0.0}
            continue
        values = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        non_zero = int((values != 0).sum())
        stats[feature] = {
            "present": 1.0,
            "non_zero_rows": float(non_zero),
            "non_zero_ratio": float(non_zero / total),
            "mean": float(values.mean()),
        }
    return stats


def save_scenario_outputs(
    output_dir: Path,
    scenario_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    extra_meta: dict | None = None,
    extra_splits: dict[str, pd.DataFrame] | None = None,
) -> None:
    scenario_dir = output_dir / scenario_name
    ensure_output_dir(scenario_dir)

    parquet_ok = True
    try:
        train_df.to_parquet(scenario_dir / "train.parquet", index=False)
        val_df.to_parquet(scenario_dir / "val.parquet", index=False)
        test_df.to_parquet(scenario_dir / "test.parquet", index=False)
    except ImportError:
        parquet_ok = False

    train_df.to_csv(scenario_dir / "train.csv", index=False)
    val_df.to_csv(scenario_dir / "val.csv", index=False)
    test_df.to_csv(scenario_dir / "test.csv", index=False)

    split_meta: dict[str, dict] = {
        "train": {
            "rows": int(len(train_df)),
            "label_distribution": train_df["label"].value_counts().to_dict(),
        },
        "val": {
            "rows": int(len(val_df)),
            "label_distribution": val_df["label"].value_counts().to_dict(),
        },
        "test": {
            "rows": int(len(test_df)),
            "label_distribution": test_df["label"].value_counts().to_dict(),
        },
    }

    if extra_splits:
        for split_name, split_df in extra_splits.items():
            split_df.to_csv(scenario_dir / f"{split_name}.csv", index=False)
            try:
                split_df.to_parquet(scenario_dir / f"{split_name}.parquet", index=False)
            except ImportError:
                parquet_ok = False
            split_meta[split_name] = {
                "rows": int(len(split_df)),
                "label_distribution": split_df["label"].value_counts().to_dict(),
            }

    meta = {
        "scenario": scenario_name,
        "splits": split_meta,
        "new_feature_nonzero": {
            "train": _feature_nonzero_stats(train_df, NEW_FEATURES),
            "val": _feature_nonzero_stats(val_df, NEW_FEATURES),
            "test": _feature_nonzero_stats(test_df, NEW_FEATURES),
        },
    }
    if extra_splits:
        for split_name, split_df in extra_splits.items():
            meta["new_feature_nonzero"][split_name] = _feature_nonzero_stats(split_df, NEW_FEATURES)

    if extra_meta:
        meta.update(extra_meta)

    with (scenario_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(f"[SAVE] Scenario outputs saved to: {scenario_dir}")
    if not parquet_ok:
        print(f"[SAVE] Scenario '{scenario_name}' parquet export skipped; CSV files were saved.")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def _clean_id_series(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.replace({"nan": "", "None": "", "<NA>": ""})


def _load_shared_backfill_source(path: Path) -> pd.DataFrame:
    wanted = {"row_id", *BACKFILL_RAW_COLS}
    return pd.read_csv(path, low_memory=False, usecols=lambda col: col in wanted)


def _backfill_train_raw_fields_from_shared(
    train_df: pd.DataFrame,
    shared_train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    out = train_df.copy()
    if "row_id" not in out.columns or "row_id" not in shared_train_df.columns:
        return out, {
            "enabled": False,
            "reason": "missing_row_id_column",
            "rows_total": int(len(out)),
        }

    shared = shared_train_df.copy()
    shared["row_id"] = _clean_id_series(shared["row_id"])
    shared = shared[shared["row_id"] != ""].drop_duplicates("row_id", keep="first")
    if shared.empty:
        return out, {
            "enabled": False,
            "reason": "empty_shared_lookup",
            "rows_total": int(len(out)),
        }

    row_id = _clean_id_series(out["row_id"])
    if "synthetic_parent_row_id" in out.columns:
        parent_id = _clean_id_series(out["synthetic_parent_row_id"])
    else:
        parent_id = pd.Series([""] * len(out), index=out.index, dtype=str)

    shared_keys = set(shared["row_id"].tolist())
    use_row = row_id.map(lambda value: value in shared_keys)
    use_parent = (~use_row) & parent_id.map(lambda value: value in shared_keys)
    lookup_key = row_id.where(use_row, "")
    lookup_key = lookup_key.where(~use_parent, parent_id)

    shared_idx = shared.set_index("row_id")
    for col in BACKFILL_RAW_COLS:
        if col not in out.columns:
            out[col] = ""
        source_map = shared_idx[col].to_dict() if col in shared_idx.columns else {}
        mapped = lookup_key.map(source_map)
        out[col] = mapped.fillna(out[col]).fillna("")

    out["subject_raw"] = out["subject_raw"].astype(str)
    out["body_raw"] = out["body_raw"].astype(str)
    out["text_raw"] = out["text_raw"].astype(str)

    return out, {
        "enabled": True,
        "rows_total": int(len(out)),
        "rows_backfilled_from_row_id": int(use_row.sum()),
        "rows_backfilled_from_parent_id": int(use_parent.sum()),
        "rows_backfilled_total": int((lookup_key != "").sum()),
        "rows_unmatched": int((lookup_key == "").sum()),
        "model_text_columns_preserved": True,
    }


def _prepare_split(df: pd.DataFrame, split_name: str, force_recompute_features: bool) -> pd.DataFrame:
    out = df.copy()
    missing_new_features = [col for col in NEW_FEATURES if col not in out.columns]
    if force_recompute_features or missing_new_features:
        out = add_engineered_features(out)
    out["split"] = split_name
    out = add_label_id(out)
    out = out[out["label"].isin(CONFIG.labels)].copy()
    out = out[out["text"].fillna("").astype(str).str.len() >= MIN_TEXT_LENGTH].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def prepare_from_existing_splits(
    processed_dir: Path,
    base_train_scenario: str,
    scenario_name: str,
    force_recompute_features: bool,
    include_hard_splits: bool,
    include_deployment_split: bool,
    train_from_shared: bool,
    enrich_train_raw_from_shared: bool,
) -> Path:
    shared_dir = processed_dir / "shared"
    if train_from_shared:
        train_path = shared_dir / "train_full.csv"
    else:
        train_path = processed_dir / base_train_scenario / "train.csv"

    val_path = shared_dir / "val.csv"
    if not val_path.exists():
        val_path = shared_dir / "val_iid.csv"

    test_path = shared_dir / "test.csv"
    if not test_path.exists():
        test_path = shared_dir / "test_iid.csv"

    train_raw = _load_csv(train_path)
    train_backfill_meta = {
        "enabled": False,
        "reason": "disabled_or_not_needed",
        "rows_total": int(len(train_raw)),
    }
    if enrich_train_raw_from_shared and not train_from_shared:
        shared_train_full_path = shared_dir / "train_full.csv"
        if shared_train_full_path.exists():
            shared_backfill_source = _load_shared_backfill_source(shared_train_full_path)
            train_raw, train_backfill_meta = _backfill_train_raw_fields_from_shared(
                train_raw,
                shared_backfill_source,
            )
        else:
            train_backfill_meta = {
                "enabled": False,
                "reason": "shared_train_full_missing",
                "rows_total": int(len(train_raw)),
            }

    train_df = _prepare_split(train_raw, "train", force_recompute_features)
    val_df = _prepare_split(_load_csv(val_path), "val_iid", force_recompute_features)
    test_df = _prepare_split(_load_csv(test_path), "test_iid", force_recompute_features)

    extra_splits: dict[str, pd.DataFrame] = {}
    if include_hard_splits:
        hard_source_path = shared_dir / "test_hard_source.csv"
        hard_cluster_path = shared_dir / "test_hard_cluster.csv"
        if hard_source_path.exists():
            extra_splits["test_hard_source"] = _prepare_split(
                _load_csv(hard_source_path),
                "test_hard_source",
                force_recompute_features,
            )
        if hard_cluster_path.exists():
            extra_splits["test_hard_cluster"] = _prepare_split(
                _load_csv(hard_cluster_path),
                "test_hard_cluster",
                force_recompute_features,
            )

    if include_deployment_split:
        deployment_path = shared_dir / "test_deployment.csv"
        if deployment_path.exists():
            extra_splits["test_deployment"] = _prepare_split(
                _load_csv(deployment_path),
                "test_deployment",
                force_recompute_features,
            )

    save_scenario_outputs(
        output_dir=processed_dir,
        scenario_name=scenario_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        extra_splits=extra_splits,
        extra_meta={
            "base_train_path": str(train_path),
            "base_train_scenario": base_train_scenario,
            "uses_existing_shared_splits": True,
            "force_recompute_features": bool(force_recompute_features),
            "added_features": NEW_FEATURES,
            "train_raw_backfill": train_backfill_meta,
        },
    )
    return processed_dir / scenario_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare multiclass scenario using already existing splits. "
            "This command does not create new random split boundaries."
        )
    )
    parser.add_argument("--processed-dir", default="dataset/processed")
    parser.add_argument(
        "--base-train-scenario",
        default="scenario_m_anti_template_feature_regularized",
        help="Scenario that provides train.csv when --train-from-shared is not used.",
    )
    parser.add_argument(
        "--scenario-name",
        default="scenario_m_feature_probe",
        help="Output scenario folder name.",
    )
    parser.add_argument(
        "--train-from-shared",
        action="store_true",
        help="Use shared/train_full.csv as training source instead of scenario train.csv.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Recompute engineered features even when columns already exist.",
    )
    parser.add_argument(
        "--no-hard-splits",
        action="store_true",
        help="Do not copy/process test_hard_source and test_hard_cluster.",
    )
    parser.add_argument(
        "--include-deployment-split",
        action="store_true",
        help="Also include shared/test_deployment.csv in the scenario folder.",
    )
    parser.add_argument(
        "--no-enrich-train-raw-from-shared",
        action="store_true",
        help=(
            "Do not backfill train raw fields from shared/train_full.csv "
            "when building from a scenario train split."
        ),
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    out_dir = prepare_from_existing_splits(
        processed_dir=processed_dir,
        base_train_scenario=args.base_train_scenario,
        scenario_name=args.scenario_name,
        force_recompute_features=bool(args.force_recompute_features),
        include_hard_splits=not bool(args.no_hard_splits),
        include_deployment_split=bool(args.include_deployment_split),
        train_from_shared=bool(args.train_from_shared),
        enrich_train_raw_from_shared=not bool(args.no_enrich_train_raw_from_shared),
    )

    metadata_path = out_dir / "metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("\nAdded feature coverage (train split):")
    for feature in NEW_FEATURES:
        info = meta.get("new_feature_nonzero", {}).get("train", {}).get(feature, {})
        ratio = float(info.get("non_zero_ratio", 0.0))
        non_zero = int(info.get("non_zero_rows", 0.0))
        print(f"- {feature}: non_zero={non_zero:,} ratio={ratio:.4f}")

    print(f"\nDone. Scenario ready at: {out_dir}")


if __name__ == "__main__":
    main()
