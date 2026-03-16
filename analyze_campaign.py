from pathlib import Path
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from urllib.parse import urlparse, urlunparse
import re

try:
    import regex as regex_lib
except ImportError:
    regex_lib = None


URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
HREF_RE = re.compile(r"href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

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
    (?:
      (?&HOSTLABEL)
      (?:\.(?&HOSTLABEL))+
      \.?
    )
  )
)

(?<![@\p{L}\p{N}\p{M}_])
(?:(?:https?|ftps?)://
  (?:[\p{L}\p{N}\p{M}\-._~!$&'()*+,;=%]+(?::[\p{L}\p{N}\p{M}\-._~!$&'()*+,;=%]*)?@)?
  (?:
    localhost
    |(?&HOSTNAME)
    |(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})
    |\[[0-9A-Fa-f:]+\]
  )
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


def _extract_email_domain(value: str) -> str:
    _, addr = parseaddr(value or "")
    if "@" not in addr:
        return ""
    return addr.split("@")[-1].strip().lower()


def _parse_auth_results(auth_header: str) -> dict:
    lower = (auth_header or "").lower()
    return {
        "spf_result": "pass"
        if "spf=pass" in lower
        else "fail"
        if "spf=fail" in lower
        else "none",
        "dkim_result": "pass"
        if "dkim=pass" in lower
        else "fail"
        if "dkim=fail" in lower
        else "none",
        "dmarc_result": "pass"
        if "dmarc=pass" in lower
        else "fail"
        if "dmarc=fail" in lower
        else "none",
    }


def _keyword_count(text: str, keywords: set[str]) -> int:
    words = WORD_RE.findall((text or "").lower())
    return sum(1 for w in words if w in keywords)


def _find_urls(text: str) -> list[str]:
    if not text:
        return []

    if ADVANCED_URL_RE is not None:
        return [match.group(0) for match in ADVANCED_URL_RE.finditer(text)]

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
    text_url_pattern = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

    for html in html_parts:
        for href, anchor_text in anchor_pattern.findall(html):
            href_host = (urlparse(href).hostname or "").lower()
            if not href_host:
                continue

            cleaned_anchor = TAG_RE.sub(" ", anchor_text or "")
            text_urls = text_url_pattern.findall(cleaned_anchor)
            for text_url in text_urls:
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
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                parsed.query,
                "",  # drop fragment
            )
        )
    except Exception:
        return url.strip()


def _extract_body_parts(msg):
    text_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "")).lower()
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
        content = msg.get_content()
        if msg.get_content_type() == "text/html":
            html_parts.append(str(content))
        else:
            text_parts.append(str(content))

    return text_parts, html_parts


def _extract_urls(text_parts, html_parts):
    urls = set()

    for text in text_parts:
        for url in _find_urls(text):
            urls.add(_normalize_url(url))

    for html in html_parts:
        for href in HREF_RE.findall(html):
            if href.lower().startswith("http"):
                urls.add(_normalize_url(href))
        for url in _find_urls(html):
            urls.add(_normalize_url(url))

    return sorted(urls)


def _html_to_text(html_parts):
    text = []
    for html in html_parts:
        text.append(TAG_RE.sub(" ", html))
    return " ".join(text)


def load_eml_folder(folder_path):
    emails = []

    for path in Path(folder_path).glob("*.eml"):
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        text_parts, html_parts = _extract_body_parts(msg)
        text_body = "\n".join(text_parts).strip() if text_parts else ""
        html_body = "\n".join(html_parts).strip() if html_parts else ""

        emails.append(
            {
                "file": str(path),
                "subject": msg.get("Subject", ""),
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "date": msg.get("Date", ""),
                "reply_to": msg.get("Reply-To", ""),
                "return_path": msg.get("Return-Path", ""),
                "message_id": msg.get("Message-ID", ""),
                "auth_results": msg.get("Authentication-Results", ""),
                "received": msg.get_all("Received", []),
                "text": text_body,
                "html": html_body,
                "raw_message": msg,
            }
        )

    return emails


def extract_features(email_obj):
    subject = (email_obj.get("subject") or "").strip()
    from_header = (email_obj.get("from") or "").strip()
    to_header = (email_obj.get("to") or "").strip()
    reply_to = (email_obj.get("reply_to") or "").strip()
    return_path = (email_obj.get("return_path") or "").strip()
    message_id = (email_obj.get("message_id") or "").strip()
    auth_results = (email_obj.get("auth_results") or "").strip()
    received = email_obj.get("received") or []
    text_body = (email_obj.get("text") or "").strip()
    html_body = (email_obj.get("html") or "").strip()

    text_parts = [text_body] if text_body else []
    html_parts = [html_body] if html_body else []

    urls = _extract_urls(text_parts, html_parts)
    html_text = _html_to_text(html_parts).strip()
    combined_text = "\n".join([p for p in [text_body, html_text] if p]).strip()

    from_domain = _extract_email_domain(from_header)
    reply_to_domain = _extract_email_domain(reply_to)
    return_path_domain = _extract_email_domain(return_path)
    message_id_domain = _extract_email_domain(message_id)

    auth = _parse_auth_results(auth_results)

    ip_url_count = 0
    shortener_url_count = 0
    punycode_url_count = 0
    unique_domains = set()
    for url in urls:
        host = urlparse(url).hostname or ""
        host = host.lower()
        if host:
            unique_domains.add(host)
        if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
            ip_url_count += 1
        if host in SHORTENER_DOMAINS:
            shortener_url_count += 1
        if "xn--" in host:
            punycode_url_count += 1

    raw_message = email_obj.get("raw_message")
    attachment_count = 0
    risky_attachment_ext_count = 0
    if raw_message is not None:
        for part in raw_message.walk():
            filename = part.get_filename()
            if not filename:
                continue
            attachment_count += 1
            lower_name = filename.lower()
            if any(lower_name.endswith(ext) for ext in RISKY_ATTACHMENT_EXTS):
                risky_attachment_ext_count += 1

    has_html_only = int(bool(html_body and not text_body))
    mismatched_anchor_count = _extract_anchor_mismatch_count(html_parts)
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

    urgent_keyword_count = _keyword_count(f"{subject} {combined_text}", URGENT_KEYWORDS)
    credential_keyword_count = _keyword_count(
        f"{subject} {combined_text}", CREDENTIAL_KEYWORDS
    )
    threat_keyword_count = _keyword_count(f"{subject} {combined_text}", THREAT_KEYWORDS)
    financial_keyword_count = _keyword_count(
        f"{subject} {combined_text}", FINANCIAL_KEYWORDS
    )

    return {
        "file": email_obj.get("file", ""),
        "subject": subject,
        "from": from_header,
        "to": to_header,
        "reply_to": reply_to,
        "return_path": return_path,
        "text": combined_text,
        "urls": urls,
        "from_domain": from_domain,
        "reply_to_domain": reply_to_domain,
        "return_path_domain": return_path_domain,
        "message_id_domain": message_id_domain,
        "received_hops_count": len(received),
        "spf_result": auth["spf_result"],
        "dkim_result": auth["dkim_result"],
        "dmarc_result": auth["dmarc_result"],
        "url_count": len(urls),
        "unique_domain_count": len(unique_domains),
        "ip_url_count": ip_url_count,
        "shortener_url_count": shortener_url_count,
        "punycode_url_count": punycode_url_count,
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
        "message_id_domain_mismatch": int(
            bool(from_domain and message_id_domain and from_domain != message_id_domain)
        ),
        "display_name_spoof_flag": display_name_spoof_flag,
        "received_anomaly_flag": received_anomaly_flag,
        "has_html_only": has_html_only,
        "attachment_count": attachment_count,
        "risky_attachment_ext_count": risky_attachment_ext_count,
        "uppercase_ratio": uppercase_ratio,
        "exclamation_count": exclamation_count,
        "urgent_keyword_count": urgent_keyword_count,
        "credential_keyword_count": credential_keyword_count,
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

    score += min(15, features.get("credential_keyword_count", 0) * 3)
    score += min(10, features.get("urgent_keyword_count", 0) * 2)
    if features.get("credential_keyword_count", 0) > 0:
        reasons.append("credential_keywords")
    if features.get("urgent_keyword_count", 0) > 0:
        reasons.append("urgent_keywords")

    score += min(10, features.get("risky_attachment_ext_count", 0) * 10)
    if features.get("risky_attachment_ext_count", 0) > 0:
        reasons.append("risky_attachment")

    if features.get("display_name_spoof_flag", 0):
        score += 10
        reasons.append("display_name_spoof")

    if features.get("received_anomaly_flag", 0):
        score += 5
        reasons.append("received_anomaly")

    if features.get("uppercase_ratio", 0) > 0.35:
        score += 5
        reasons.append("high_uppercase_ratio")

    if features.get("exclamation_count", 0) >= 3:
        score += 5
        reasons.append("many_exclamations")

    score = min(100, int(score))
    risk_level = "low" if score < 30 else "medium" if score < 60 else "high"
    return {
        "heuristic_score": score,
        "risk_level": risk_level,
        "heuristic_reasons": ",".join(reasons),
    }
