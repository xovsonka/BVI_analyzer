from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path
from urllib.parse import urlparse, urlunparse

try:
    import regex as regex_lib
except ImportError:
    regex_lib = None


URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
HREF_RE = re.compile(r"href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

ADVANCED_URL_PATTERN = r"""
(?(DEFINE)
  (?<HOSTLABEL>
    (?:
      [\p{L}\p{N}\p{M}](?![\p{L}\p{N}\p{M}-]{1,2}--)(?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?
      |xn--(?![A-Za-z0-9]{2}--)(?:[A-Za-z0-9-]{1,59}[A-Za-z0-9])?
      |[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?
    )
  )
  (?<HOSTNAME>
    (?:(?&HOSTLABEL)(?:\.(?&HOSTLABEL))+\.?)
  )
)
(?<![@\p{L}\p{N}\p{M}_])
(?:(?:https?|ftps?)://
  (?:[\p{L}\p{N}\p{M}\-._~!$&'()*+,;=%]+(?::[\p{L}\p{N}\p{M}\-._~!$&'()*+,;=%]*)?@)?
  (?:localhost|(?&HOSTNAME)|(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|\[[0-9A-Fa-f:]+\])
  |
  (?&HOSTNAME)
)
(?::\d{1,5})?
(?:[/?#](?:%[0-9A-Fa-f]{2}|[A-Za-z0-9\-._~]|[\p{L}\p{N}\p{M}\p{S}\p{P}])*)?
(?![@\p{L}\p{N}\p{M}_])
"""

if regex_lib is not None:
    ADVANCED_URL_RE = regex_lib.compile(
        ADVANCED_URL_PATTERN,
        flags=regex_lib.IGNORECASE | regex_lib.UNICODE | regex_lib.VERBOSE,
    )
else:
    ADVANCED_URL_RE = None


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
TRUSTED_DOMAINS = {
    "microsoft.com",
    "google.com",
    "apple.com",
    "amazon.com",
    "outlook.com",
    "office.com",
    "github.com",
    "bvi.local",
    "local.lab",
}
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
RISKY_ATTACHMENT_EXTS = {
    ".exe",
    ".js",
    ".scr",
    ".zip",
    ".rar",
    ".html",
    ".hta",
    ".bat",
    ".cmd",
}
URGENT_KEYWORDS = {
    "urgent",
    "immediately",
    "verify",
    "suspended",
    "action",
    "required",
    "deadline",
}
CREDENTIAL_KEYWORDS = {
    "password",
    "login",
    "signin",
    "account",
    "update",
    "security",
    "confirm",
}
THREAT_KEYWORDS = {"blocked", "locked", "disable", "suspend", "expired", "terminate"}
FINANCIAL_KEYWORDS = {"invoice", "payment", "transfer", "bank", "card", "billing"}
BRAND_KEYWORDS = {
    "paypal",
    "microsoft",
    "google",
    "apple",
    "amazon",
    "bank",
    "usaa",
    "outlook",
}
TRACKING_HOSTS = {"localhost", "127.0.0.1"}


MODEL_NUMERIC_FEATURES = [
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
    "external_domain_url_count",
    "sender_url_domain_mismatch_ratio",
    "credential_keyword_density",
    "urgent_keyword_density",
    "unicode_mixed_script_flag",
]


def _extract_email_domain(value: str) -> str:
    _, addr = parseaddr(value or "")
    if "@" not in addr:
        return ""
    return addr.split("@")[-1].strip().lower()


def _parse_auth_results(auth_header: str) -> dict:
    lower = (auth_header or "").lower()

    def pick(token: str) -> str:
        match = re.search(rf"\b{token}\s*=\s*([a-z_]+)", lower)
        if not match:
            return "none"
        value = match.group(1)
        if value in {
            "pass",
            "fail",
            "softfail",
            "neutral",
            "temperror",
            "permerror",
            "none",
        }:
            return value
        return "none"

    return {
        "spf_result": pick("spf"),
        "dkim_result": pick("dkim"),
        "dmarc_result": pick("dmarc"),
    }


def _registered_domain(host: str) -> str:
    host = (host or "").strip(".").lower()
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    return ".".join(parts[-2:])


def _levenshtein(a: str, b: str) -> int:
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
            curr.append(
                min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1))
            )
        prev = curr
    return prev[-1]


def _normalize_host_idn(host: str) -> str:
    host = (host or "").strip().strip(".").lower()
    if not host:
        return ""
    labels = host.split(".")
    decoded = []
    for label in labels:
        if label.startswith("xn--"):
            try:
                decoded.append(label.encode("ascii").decode("idna"))
            except Exception:
                decoded.append(label)
        else:
            decoded.append(label)
    return ".".join(decoded)


def _keyword_count(text: str, keywords: set[str]) -> int:
    words = WORD_RE.findall((text or "").lower())
    return sum(1 for w in words if w in keywords)


def _keyword_density(text: str, keyword_count: int) -> float:
    words = WORD_RE.findall((text or "").lower())
    if not words:
        return 0.0
    return float(keyword_count / len(words))


def _extract_subject_tags(subject: str) -> dict:
    text = (subject or "").strip()
    out = {
        "campaign_name": "",
        "label_hint": "",
        "dataset_id_hint": "",
    }
    if not text:
        return out

    bracket_tokens = re.findall(r"\[([^\]]+)\]", text)
    if bracket_tokens:
        first = bracket_tokens[0].strip()
        if first and ":" not in first:
            out["campaign_name"] = first

    m_label = re.search(r"\[label:(\d+)\]", text, flags=re.IGNORECASE)
    if m_label:
        out["label_hint"] = m_label.group(1)

    m_id = re.search(r"\[id:(\d+)\]", text, flags=re.IGNORECASE)
    if m_id:
        out["dataset_id_hint"] = m_id.group(1)

    return out


def _infer_mailhog_id_from_file(file_path: str) -> str:
    name = Path(file_path).name
    stem = Path(name).stem
    m = re.match(r"^\d{8}_\d{6}_(.+)$", stem)
    if m:
        candidate = m.group(1)
        if "__" in candidate:
            return candidate.split("__", 1)[0]
        return candidate
    if "_" not in stem:
        return ""
    after_first = stem.split("_", 1)[1]
    if "__" in after_first:
        return after_first.split("__", 1)[0]
    return after_first


def _is_unicode_mixed_script(host: str) -> int:
    host = _normalize_host_idn(host)
    if not host:
        return 0

    has_ascii_alpha = False
    non_ascii_scripts = set()
    for ch in host:
        if not ch.isalpha():
            continue
        if ord(ch) < 128:
            has_ascii_alpha = True
            continue
        script = "OTHER"
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:
            script = "CYRILLIC"
        elif "GREEK" in name:
            script = "GREEK"
        elif "LATIN" in name:
            script = "LATIN_NON_ASCII"
        non_ascii_scripts.add(script)

    if has_ascii_alpha and non_ascii_scripts:
        return 1
    if len(non_ascii_scripts) > 1:
        return 1
    return 0


def _find_urls(text: str) -> list[str]:
    if not text:
        return []
    if ADVANCED_URL_RE is not None:
        return [m.group(0) for m in ADVANCED_URL_RE.finditer(text)]
    return URL_RE.findall(text)


def _extract_display_name(value: str) -> str:
    name, _ = parseaddr(value or "")
    return (name or "").strip().lower()


def _extract_anchor_mismatch_count(html_parts: list[str]) -> int:
    count = 0
    anchor_pattern = re.compile(
        r"<a\b[^>]*href\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    for html in html_parts:
        for href, anchor_text in anchor_pattern.findall(html):
            href_host = (urlparse(href).hostname or "").lower()
            if not href_host:
                continue
            cleaned_anchor = TAG_RE.sub(" ", anchor_text or "")
            for text_url in URL_RE.findall(cleaned_anchor):
                text_host = (urlparse(text_url).hostname or "").lower()
                if text_host and text_host != href_host:
                    count += 1
                    break
    return count


def _count_obfuscated_urls(urls: list[str]) -> int:
    count = 0
    for url in urls:
        parsed = urlparse(url)
        host = parsed.netloc or ""
        query = parsed.query or ""
        if "@" in url or "%" in url or len(query) > 80 or host.count(".") >= 4:
            count += 1
    return count


def _compute_uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _normalize_url(url: str) -> str:
    try:
        raw = url.strip()
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        host = _normalize_host_idn(parsed.netloc)
        return urlunparse(
            (
                parsed.scheme.lower(),
                host,
                parsed.path,
                parsed.params,
                parsed.query,
                "",
            )
        )
    except Exception:
        return url.strip()


def _is_tracking_url(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    query = parsed.query or ""
    return host in TRACKING_HOSTS and "rid=" in query.lower()


def _strip_tracking_anchors(html: str) -> str:
    if not html:
        return ""

    def _replace(match: re.Match[str]) -> str:
        href = match.group(1)
        anchor_text = TAG_RE.sub(" ", match.group(2) or "")
        if _is_tracking_url(href) and anchor_text.strip().lower() == "open link":
            return " "
        return match.group(0)

    anchor_pattern = re.compile(
        r'<a\b[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    return anchor_pattern.sub(_replace, html)


def _normalize_text_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _merge_text_bodies(text_body: str, html_text: str) -> str:
    text_body = str(text_body or "").strip()
    html_text = str(html_text or "").strip()
    if not text_body:
        return html_text
    if not html_text:
        return text_body

    norm_text = _normalize_text_for_compare(text_body)
    norm_html = _normalize_text_for_compare(html_text)
    if not norm_text:
        return html_text
    if not norm_html:
        return text_body
    if norm_text == norm_html or norm_text in norm_html or norm_html in norm_text:
        return text_body if len(norm_text) >= len(norm_html) else html_text
    return "\n\n".join([text_body, html_text]).strip()


def _host_to_token(host: str) -> str:
    host = (host or "").strip().lower()
    if not host:
        return ""
    if IP_HOST_RE.fullmatch(host):
        return "url_ip"
    safe = re.sub(r"[^a-z0-9]+", "_", host).strip("_")
    if not safe:
        return ""
    return f"urlhost_{safe}"


def _build_url_host_tokens(urls: list[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for url in urls:
        try:
            host = _normalize_host_idn(urlparse(url).hostname or "")
        except Exception:
            continue
        token = _host_to_token(host)
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _build_text_input(subject: str, combined_text: str, urls: list[str]) -> str:
    parts = [str(subject or "").strip(), str(combined_text or "").strip()]
    base = "\n\n".join([part for part in parts if part]).strip()
    host_tokens = _build_url_host_tokens(urls)
    if host_tokens:
        base = f"{base}\n\n{' '.join(host_tokens)}".strip()
    return base


def parse_eml_file(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    headers = {k: msg.get_all(k, []) for k in msg.keys()}
    text_parts = []
    html_parts = []
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "")).lower()
            filename = part.get_filename()

            if filename:
                attachments.append(
                    {
                        "filename": filename,
                        "content_type": ctype,
                    }
                )
                continue

            if "attachment" in disp:
                continue

            if ctype == "text/plain":
                content = part.get_content()
                if content:
                    text_parts.append(str(content))
            elif ctype == "text/html":
                content = part.get_content()
                if content:
                    html_parts.append(str(content))
    else:
        ctype = msg.get_content_type()
        content = msg.get_content()
        if ctype == "text/html":
            html_parts.append(str(content))
        else:
            text_parts.append(str(content))

    urls = set()
    for text in text_parts:
        for u in _find_urls(text):
            normalized = _normalize_url(u)
            if not _is_tracking_url(normalized):
                urls.add(normalized)
    for html in html_parts:
        for href in HREF_RE.findall(html):
            if href.lower().startswith("http"):
                normalized = _normalize_url(href)
                if not _is_tracking_url(normalized):
                    urls.add(normalized)
        for u in _find_urls(html):
            normalized = _normalize_url(u)
            if not _is_tracking_url(normalized):
                urls.add(normalized)

    return {
        "file": str(file_path),
        "headers": headers,
        "text_parts": text_parts,
        "html_parts": html_parts,
        "attachments": attachments,
        "urls": sorted(urls),
    }


def parse_eml_folder(folder_path: str | Path) -> list[dict]:
    parts_list = []
    for path in Path(folder_path).glob("*.eml"):
        parts_list.append(parse_eml_file(path))
    return parts_list


def extract_features_from_parts(parts: dict) -> dict:
    headers = parts.get("headers", {})
    text_parts = parts.get("text_parts", [])
    html_parts = parts.get("html_parts", [])
    attachments = parts.get("attachments", [])
    urls = parts.get("urls", [])

    subject = (
        headers.get("Subject", [""])[0] if headers.get("Subject") else ""
    ).strip()
    from_header = (headers.get("From", [""])[0] if headers.get("From") else "").strip()
    to_header = (headers.get("To", [""])[0] if headers.get("To") else "").strip()
    reply_to = (
        headers.get("Reply-To", [""])[0] if headers.get("Reply-To") else ""
    ).strip()
    return_path = (
        headers.get("Return-Path", [""])[0] if headers.get("Return-Path") else ""
    ).strip()
    message_id = (
        headers.get("Message-ID", [""])[0] if headers.get("Message-ID") else ""
    ).strip()
    auth_results = (
        headers.get("Authentication-Results", [""])[0]
        if headers.get("Authentication-Results")
        else ""
    ).strip()
    received = headers.get("Received", [])

    cleaned_html_parts = [_strip_tracking_anchors(h) for h in html_parts]
    text_body = "\n".join(text_parts).strip()
    html_body = "\n".join(cleaned_html_parts).strip()
    html_text = " ".join(TAG_RE.sub(" ", h) for h in cleaned_html_parts).strip()
    combined_text = _merge_text_bodies(text_body, html_text)

    from_domain = _extract_email_domain(from_header)
    reply_to_domain = _extract_email_domain(reply_to)
    return_path_domain = _extract_email_domain(return_path)
    message_id_domain = _extract_email_domain(message_id)
    receiver_domain = _extract_email_domain(to_header)

    auth = _parse_auth_results(auth_results)

    ip_url_count = 0
    shortener_url_count = 0
    punycode_url_count = 0
    suspicious_tld_count = 0
    external_domain_url_count = 0
    unicode_mixed_script_flag = 0
    unique_domains = set()
    for url in urls:
        try:
            host = _normalize_host_idn(urlparse(url).hostname or "")
        except ValueError:
            continue
        if host:
            unique_domains.add(host)
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            ip_url_count += 1
        if host in SHORTENER_DOMAINS:
            shortener_url_count += 1
        if "xn--" in host:
            punycode_url_count += 1
        if _is_unicode_mixed_script(host):
            unicode_mixed_script_flag = 1
        tld = host.split(".")[-1] if "." in host else ""
        if tld in SUSPICIOUS_TLDS:
            suspicious_tld_count += 1

    reg_sender = _registered_domain(from_domain)
    reg_urls = {_registered_domain(h) for h in unique_domains if h}
    if reg_sender:
        external_domain_url_count = sum(1 for d in reg_urls if d and d != reg_sender)
    trusted_domain_only = int(
        bool(reg_sender)
        and reg_sender in TRUSTED_DOMAINS
        and all((not d) or (d in TRUSTED_DOMAINS) for d in reg_urls)
    )

    sender_url_domain_mismatch_ratio = (
        float(external_domain_url_count / max(1, len(reg_urls))) if reg_urls else 0.0
    )

    candidate_labels = {from_domain.split(".")[0]} if from_domain else set()
    candidate_labels.update({d.split(".")[0] for d in unique_domains if d and "." in d})
    brand_typosquat_flag = 0
    for cand in candidate_labels:
        c = re.sub(r"[^a-z0-9]", "", cand.lower())
        if not c:
            continue
        for brand in KNOWN_BRANDS:
            if c != brand and _levenshtein(c, brand) == 1:
                brand_typosquat_flag = 1
                break
        if brand_typosquat_flag:
            break

    risky_attachment_ext_count = 0
    for att in attachments:
        filename = str(att.get("filename", "")).lower()
        if any(filename.endswith(ext) for ext in RISKY_ATTACHMENT_EXTS):
            risky_attachment_ext_count += 1

    has_html_only = int(bool(html_body and not text_body))
    mismatched_anchor_count = _extract_anchor_mismatch_count(cleaned_html_parts)
    obfuscated_url_count = _count_obfuscated_urls(urls)
    display_name = _extract_display_name(from_header)
    display_name_spoof_flag = int(
        bool(display_name)
        and any(brand in display_name for brand in BRAND_KEYWORDS)
        and bool(from_domain)
        and not any(brand in from_domain for brand in BRAND_KEYWORDS)
    )
    received_anomaly_flag = int(len(received) <= 1 and bool(from_domain))

    style_text = f"{subject} {combined_text}"
    uppercase_ratio = _compute_uppercase_ratio(style_text)
    exclamation_count = style_text.count("!")
    digit_ratio = (
        (sum(ch.isdigit() for ch in style_text) / len(style_text))
        if style_text
        else 0.0
    )

    urgent_keyword_count = _keyword_count(style_text, URGENT_KEYWORDS)
    credential_keyword_count = _keyword_count(style_text, CREDENTIAL_KEYWORDS)
    threat_keyword_count = _keyword_count(style_text, THREAT_KEYWORDS)
    financial_keyword_count = _keyword_count(style_text, FINANCIAL_KEYWORDS)
    urgent_keyword_density = _keyword_density(style_text, urgent_keyword_count)
    credential_keyword_density = _keyword_density(style_text, credential_keyword_count)

    subject_tags = _extract_subject_tags(subject)
    message_id = message_id.strip("<>")
    mailhog_id = _infer_mailhog_id_from_file(parts.get("file", ""))

    return {
        "file": parts.get("file", ""),
        "subject": subject,
        "from": from_header,
        "to": to_header,
        "reply_to": reply_to,
        "return_path": return_path,
        "message_id": message_id,
        "mailhog_id": mailhog_id,
        "campaign_name": subject_tags["campaign_name"],
        "label_hint": subject_tags["label_hint"],
        "dataset_id_hint": subject_tags["dataset_id_hint"],
        "text": combined_text,
        "text_input": _build_text_input(subject, combined_text, urls),
        "urls": urls,
        "from_domain": from_domain,
        "reply_to_domain": reply_to_domain,
        "return_path_domain": return_path_domain,
        "message_id_domain": message_id_domain,
        "receiver_domain": receiver_domain,
        "received_hops_count": len(received),
        "spf_result": auth["spf_result"],
        "dkim_result": auth["dkim_result"],
        "dmarc_result": auth["dmarc_result"],
        "url_count": len(urls),
        "unique_domain_count": len(unique_domains),
        "ip_url_count": ip_url_count,
        "shortener_url_count": shortener_url_count,
        "punycode_url_count": punycode_url_count,
        "suspicious_tld_count": suspicious_tld_count,
        "brand_typosquat_flag": brand_typosquat_flag,
        "trusted_domain_only": trusted_domain_only,
        "external_domain_url_count": external_domain_url_count,
        "sender_url_domain_mismatch_ratio": sender_url_domain_mismatch_ratio,
        "unicode_mixed_script_flag": unicode_mixed_script_flag,
        "obfuscated_url_count": obfuscated_url_count,
        "mismatched_anchor_count": mismatched_anchor_count,
        "from_reply_mismatch": int(
            bool(from_domain and reply_to_domain and from_domain != reply_to_domain)
        ),
        "from_return_path_mismatch": int(
            bool(
                from_domain and return_path_domain and from_domain != return_path_domain
            )
        ),
        "sender_receiver_domain_mismatch": int(
            bool(from_domain and receiver_domain and from_domain != receiver_domain)
        ),
        "message_id_domain_mismatch": int(
            bool(from_domain and message_id_domain and from_domain != message_id_domain)
        ),
        "display_name_spoof_flag": display_name_spoof_flag,
        "received_anomaly_flag": received_anomaly_flag,
        "has_html_only": has_html_only,
        "attachment_count": len(attachments),
        "risky_attachment_ext_count": risky_attachment_ext_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "exclamation_count": exclamation_count,
        "urgent_keyword_count": urgent_keyword_count,
        "urgent_keyword_density": urgent_keyword_density,
        "credential_keyword_count": credential_keyword_count,
        "credential_keyword_density": credential_keyword_density,
        "threat_keyword_count": threat_keyword_count,
        "financial_keyword_count": financial_keyword_count,
        "subject_len": len(subject),
        "text_len": len(combined_text),
    }


def compute_heuristic_score(features: dict) -> dict:
    score = 0
    reasons = []

    if features.get("spf_result") == "fail":
        score += 20
        reasons.append("spf_fail")
    if features.get("dkim_result") == "fail":
        score += 15
        reasons.append("dkim_fail")
    if features.get("dmarc_result") == "fail":
        score += 20
        reasons.append("dmarc_fail")

    if features.get("from_reply_mismatch"):
        score += 15
        reasons.append("from_reply_mismatch")
    if features.get("from_return_path_mismatch"):
        score += 15
        reasons.append("from_return_path_mismatch")

    score += min(20, features.get("ip_url_count", 0) * 10)
    if features.get("ip_url_count", 0) > 0:
        reasons.append("ip_url")

    score += min(10, features.get("shortener_url_count", 0) * 5)
    if features.get("shortener_url_count", 0) > 0:
        reasons.append("shortener_url")

    score += min(10, features.get("obfuscated_url_count", 0) * 3)
    if features.get("obfuscated_url_count", 0) > 0:
        reasons.append("obfuscated_url")

    score += min(10, features.get("mismatched_anchor_count", 0) * 5)
    if features.get("mismatched_anchor_count", 0) > 0:
        reasons.append("anchor_mismatch")

    score += min(10, features.get("suspicious_tld_count", 0) * 5)
    if features.get("suspicious_tld_count", 0) > 0:
        reasons.append("suspicious_tld")

    score += min(12, features.get("external_domain_url_count", 0) * 3)
    if features.get("external_domain_url_count", 0) > 0:
        reasons.append("external_domain_urls")

    mismatch_ratio = float(features.get("sender_url_domain_mismatch_ratio", 0.0) or 0.0)
    if mismatch_ratio >= 0.75:
        score += 10
        reasons.append("sender_url_domain_mismatch_high")
    elif mismatch_ratio >= 0.4:
        score += 5
        reasons.append("sender_url_domain_mismatch_mid")

    if features.get("brand_typosquat_flag", 0):
        score += 15
        reasons.append("brand_typosquat")

    score += min(15, features.get("credential_keyword_count", 0) * 3)
    score += min(10, features.get("urgent_keyword_count", 0) * 2)
    if features.get("credential_keyword_count", 0) > 0:
        reasons.append("credential_keywords")
    if features.get("urgent_keyword_count", 0) > 0:
        reasons.append("urgent_keywords")

    if float(features.get("credential_keyword_density", 0.0) or 0.0) >= 0.04:
        score += 5
        reasons.append("credential_keyword_density")
    if float(features.get("urgent_keyword_density", 0.0) or 0.0) >= 0.03:
        score += 4
        reasons.append("urgent_keyword_density")

    score += min(10, features.get("risky_attachment_ext_count", 0) * 10)
    if features.get("risky_attachment_ext_count", 0) > 0:
        reasons.append("risky_attachment")

    if features.get("display_name_spoof_flag", 0):
        score += 10
        reasons.append("display_name_spoof")

    if features.get("unicode_mixed_script_flag", 0):
        score += 10
        reasons.append("unicode_mixed_script")

    if features.get("received_anomaly_flag", 0):
        score += 5
        reasons.append("received_anomaly")

    if features.get("uppercase_ratio", 0) > 0.35:
        score += 5
        reasons.append("high_uppercase_ratio")

    if features.get("exclamation_count", 0) >= 3:
        score += 5
        reasons.append("many_exclamations")

    # false-positive guardrails for trusted traffic
    if (
        features.get("trusted_domain_only", 0)
        and features.get("spf_result") in {"pass", "none"}
        and features.get("dkim_result") in {"pass", "none"}
        and features.get("dmarc_result") in {"pass", "none"}
        and features.get("ip_url_count", 0) == 0
        and features.get("credential_keyword_count", 0) == 0
    ):
        score = max(0, score - 20)
        reasons.append("trusted_domain_guardrail")

    score = min(100, int(score))
    risk_level = "low" if score < 30 else "medium" if score < 60 else "high"
    return {
        "heuristic_score": score,
        "risk_level": risk_level,
        "heuristic_reasons": ",".join(reasons),
    }


def build_model_input_row(parts: dict, features: dict, heuristic: dict) -> dict:
    row = {
        "file": features.get("file", ""),
        "mailhog_id": features.get("mailhog_id", ""),
        "message_id": features.get("message_id", ""),
        "campaign_name": features.get("campaign_name", ""),
        "label_hint": features.get("label_hint", ""),
        "dataset_id_hint": features.get("dataset_id_hint", ""),
        "subject": features.get("subject", ""),
        "body": features.get("text", ""),
        "text": features.get("text", ""),
        "text_input": features.get("text_input", ""),
        "sender": features.get("from", ""),
        "receiver": features.get("to", ""),
        "from_domain": features.get("from_domain", ""),
        "reply_to_domain": features.get("reply_to_domain", ""),
        "return_path_domain": features.get("return_path_domain", ""),
        "heuristic_score": heuristic.get("heuristic_score", 0),
        "risk_level": heuristic.get("risk_level", "low"),
        "heuristic_reasons": heuristic.get("heuristic_reasons", ""),
    }

    for col in MODEL_NUMERIC_FEATURES:
        row[col] = features.get(col, 0)

    row["spf_result"] = features.get("spf_result", "none")
    row["dkim_result"] = features.get("dkim_result", "none")
    row["dmarc_result"] = features.get("dmarc_result", "none")
    row["received_hops_count"] = features.get("received_hops_count", 0)
    row["punycode_url_count"] = features.get("punycode_url_count", 0)
    row["obfuscated_url_count"] = features.get("obfuscated_url_count", 0)
    row["mismatched_anchor_count"] = features.get("mismatched_anchor_count", 0)
    row["message_id_domain_mismatch"] = features.get("message_id_domain_mismatch", 0)
    row["display_name_spoof_flag"] = features.get("display_name_spoof_flag", 0)
    row["received_anomaly_flag"] = features.get("received_anomaly_flag", 0)
    row["url_list_json"] = json.dumps(parts.get("urls", []), ensure_ascii=False)
    row["attachment_count_total"] = len(parts.get("attachments", []))

    return row


def _load_metadata_map(metadata_csv: Path) -> dict[str, dict]:
    out = {}
    if not metadata_csv.exists():
        return out

    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eml_path = str(row.get("eml_path", "")).strip()
            if not eml_path:
                continue
            out[Path(eml_path).name] = row
    return out


def _enrich_row_with_metadata(row: dict, meta_map: dict[str, dict]) -> dict:
    if not meta_map:
        return row

    key = Path(str(row.get("file", ""))).name
    meta = meta_map.get(key)
    if not meta:
        return row

    if not row.get("mailhog_id"):
        row["mailhog_id"] = str(meta.get("mailhog_id", "") or "")
    if not row.get("message_id"):
        row["message_id"] = str(meta.get("message_id", "") or "")
    if not row.get("label_hint"):
        row["label_hint"] = str(meta.get("label_hint", "") or "")
    if not row.get("campaign_name"):
        row["campaign_name"] = str(meta.get("campaign_name", "") or "")
    return row


def analyze_eml_folder_to_csv(
    input_dir: str | Path,
    output_csv: str | Path,
    output_parts_jsonl: str | Path | None = None,
    metadata_csv: str | Path | None = None,
) -> None:
    parts_list = parse_eml_folder(input_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    meta_map = {}
    if metadata_csv is not None:
        meta_map = _load_metadata_map(Path(metadata_csv))

    rows = []
    for parts in parts_list:
        features = extract_features_from_parts(parts)
        heuristic = compute_heuristic_score(features)
        row = build_model_input_row(parts, features, heuristic)
        row = _enrich_row_with_metadata(row, meta_map)
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("")

    if output_parts_jsonl is not None:
        output_parts_jsonl = Path(output_parts_jsonl)
        output_parts_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_parts_jsonl.open("w", encoding="utf-8") as f:
            for parts in parts_list:
                f.write(json.dumps(parts, ensure_ascii=False) + "\n")

    print(f"Analyzed {len(parts_list)} .eml files")
    print(f"Saved model-ready rows to: {output_csv}")
    if output_parts_jsonl is not None:
        print(f"Saved parsed EML parts to: {output_parts_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse .eml files into parts, compute heuristics, and export model-ready dataset"
    )
    parser.add_argument(
        "--input-dir", default="dataset/campaign_eml", help="Folder with .eml files"
    )
    parser.add_argument(
        "--output-csv",
        default="results/eml_model_input.csv",
        help="Output CSV for model input",
    )
    parser.add_argument(
        "--output-parts-jsonl",
        default="results/eml_parts.jsonl",
        help="Optional JSONL with parsed parts",
    )
    parser.add_argument(
        "--metadata-csv",
        default="dataset/processed/mailhog_messages.csv",
        help="Optional metadata CSV to enrich rows (mailhog_id, label_hint, campaign_name)",
    )
    args = parser.parse_args()

    analyze_eml_folder_to_csv(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        output_parts_jsonl=args.output_parts_jsonl,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
