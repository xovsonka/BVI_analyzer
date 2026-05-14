from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


INPUT_MODES = ["text_only", "text_plus_features", "features_only"]

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
TRACKING_HOSTS = {"localhost", "127.0.0.1"}

SEMANTIC_FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "cta_login_flag": (
        r"\bsign[\s-]?in\b",
        r"\blog[\s-]?in\b",
        r"\blogin\b",
        r"\baccess (?:your )?account\b",
        r"\bconfirm sign[\s-]?in\b",
    ),
    "cta_verify_flag": (
        r"\bverify (?:your )?(?:account|mailbox|identity|details)\b",
        r"\bverification required\b",
        r"\baccount verification\b",
    ),
    "cta_recovery_flag": (
        r"\baccount recovery\b",
        r"\brecovery details\b",
        r"\brecover (?:your )?account\b",
        r"\breset recovery\b",
    ),
    "cta_payment_flag": (
        r"\bmake payment\b",
        r"\bpayment required\b",
        r"\bpay now\b",
        r"\bconfirm payment\b",
        r"\bpayment status\b",
    ),
    "cta_bank_transfer_flag": (
        r"\bbank transfer\b",
        r"\bwire transfer\b",
        r"\btransfer reference\b",
        r"\btransfer confirmation\b",
    ),
    "invoice_reference_flag": (
        r"\binvoice(?:\s+(?:number|no\.?|#))?\b",
        r"\binv[- ]?\d+\b",
        r"\bbilling reference\b",
    ),
    "remittance_reference_flag": (
        r"\bremittance\b",
        r"\bremittance advice\b",
        r"\bsettlement reference\b",
    ),
    "currency_amount_pattern_flag": (
        r"\b(?:EUR|USD|GBP)\s?\d[\d,.]*\b",
        r"[$€£]\s?\d[\d,.]*",
    ),
    "due_date_pattern_flag": (
        r"\bdue today\b",
        r"\bpayment due\b",
        r"\boverdue\b",
        r"\bbefore end of day\b",
        r"\bdue date\b",
    ),
}

ACCOUNT_SECURITY_PATTERNS: tuple[str, ...] = (
    r"\bsecurity alert\b",
    r"\bsuspicious activity\b",
    r"\bsign[\s-]?in attempt\b",
    r"\bmailbox review\b",
    r"\baccount protection\b",
)

SPAM_SEMANTIC_FLAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "spam_reward_flag": (
        r"\bmember reward\b",
        r"\bclaim (?:your )?reward\b",
        r"\bbonus access\b",
        r"\bmember perk\b",
        r"\bbonus package\b",
    ),
    "spam_unsubscribe_context_flag": (
        r"\bunsubscribe from future\b",
        r"\bunsubscribe if you\b",
        r"\bstop receiving promotional\b",
        r"\bno longer want future promotional\b",
    ),
}

SPAM_MARKETING_PATTERNS: tuple[str, ...] = (
    r"\bcampaign list\b",
    r"\bpromotional updates\b",
    r"\bpromotional notice\b",
    r"\bmarketing reward\b",
    r"\bselected recipients\b",
    r"\bselected members\b",
    r"\bfuture promotional updates\b",
    r"\bfuture promotional mailings\b",
    r"\breward campaign\b",
)

LEAKAGE_DROP_COLS = [
    "source",
    "source_profile",
    "sender",
    "receiver",
    "date",
    "from_domain",
    "reply_to_domain",
    "return_path_domain",
    "receiver_domain",
    "message_id_domain",
    "message_id",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "row_id",
    "cluster_id",
    "source_subgroup_id",
    "exact_template_id",
    "near_signature",
    "template_norm",
    "text_hash",
    "text_norm",
]

NUMERIC_FEATURES = [
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
    "cta_login_flag",
    "cta_verify_flag",
    "cta_recovery_flag",
    "account_security_phrase_count",
    "cta_payment_flag",
    "cta_bank_transfer_flag",
    "invoice_reference_flag",
    "remittance_reference_flag",
    "currency_amount_pattern_flag",
    "due_date_pattern_flag",
    "spam_reward_flag",
    "spam_unsubscribe_context_flag",
    "spam_marketing_phrase_count",
]


def _normalize_text_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _dedupe_text_blocks(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    parts = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        norm = _normalize_text_for_compare(part)
        if not norm or norm == "open link":
            continue
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(part)
    if kept:
        return "\n\n".join(kept).strip()
    return raw


def _is_tracking_url(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    query = parsed.query or ""
    return host in TRACKING_HOSTS and "rid=" in query.lower()


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


def _extract_urls_from_json(raw: object) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _extract_url_host_tokens(row: pd.Series) -> str:
    urls: list[str] = []
    if "url_list_json" in row.index:
        urls.extend(_extract_urls_from_json(row.get("url_list_json")))

    text_sources = []
    for col in ["body_raw", "text_raw", "body", "text", "text_input"]:
        if col in row.index:
            value = str(row.get(col, "") or "")
            if value:
                text_sources.append(value)
    for text in text_sources:
        urls.extend(URL_RE.findall(text))

    tokens: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if _is_tracking_url(url):
            continue
        try:
            host = (urlparse(url).hostname or "").lower()
        except Exception:
            continue
        token = _host_to_token(host)
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
        if len(tokens) >= 8:
            break
    return " ".join(tokens)


def _build_text_input_row(row: pd.Series) -> str:
    existing = str(row.get("text_input", "") or "").strip()
    if existing:
        base = existing
    else:
        subject = str(row.get("subject", "") or "").strip()
        body = str(row.get("body", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        base = f"{subject}\n\n{body}".strip() if (subject or body) else text
    base = _dedupe_text_blocks(base)
    tokens = _extract_url_host_tokens(row)
    if tokens and tokens not in base:
        base = f"{base}\n\n{tokens}".strip()
    return base


def _compute_semantic_cta_features(text: str) -> dict[str, float]:
    low = str(text or "").lower()
    out: dict[str, float] = {}
    for feature, patterns in SEMANTIC_FLAG_PATTERNS.items():
        out[feature] = float(any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in patterns))
    for feature, patterns in SPAM_SEMANTIC_FLAG_PATTERNS.items():
        out[feature] = float(any(re.search(pattern, low, flags=re.IGNORECASE) for pattern in patterns))
    out["account_security_phrase_count"] = float(
        sum(bool(re.search(pattern, low, flags=re.IGNORECASE)) for pattern in ACCOUNT_SECURITY_PATTERNS)
    )
    out["spam_marketing_phrase_count"] = float(
        sum(bool(re.search(pattern, low, flags=re.IGNORECASE)) for pattern in SPAM_MARKETING_PATTERNS)
    )
    return out


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in LEAKAGE_DROP_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])

    for col in ["subject", "body", "text"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    if "text_input" not in out.columns:
        out["text_input"] = ""
    out["text_input"] = out.apply(_build_text_input_row, axis=1)
    empty_text = out["text_input"].eq("")
    out.loc[empty_text, "text_input"] = out.loc[empty_text, "text"]

    semantic_cols = set(SEMANTIC_FLAG_PATTERNS.keys()) | set(SPAM_SEMANTIC_FLAG_PATTERNS.keys()) | {"account_security_phrase_count", "spam_marketing_phrase_count"}
    if any(col not in out.columns for col in semantic_cols):
        semantic_rows = out["text_input"].map(_compute_semantic_cta_features)
        semantic_df = pd.DataFrame(semantic_rows.tolist(), index=out.index)
        for col in semantic_df.columns:
            if col not in out.columns:
                out[col] = semantic_df[col]

    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "label" not in out.columns:
        raise ValueError("Dataset is missing required 'label' column")

    return out


def select_model_input(df: pd.DataFrame, input_mode: str) -> pd.DataFrame:
    if input_mode == "text_only":
        return df[["text_input"]]
    if input_mode == "features_only":
        return df[NUMERIC_FEATURES]
    if input_mode == "text_plus_features":
        return df[["text_input", *NUMERIC_FEATURES]]
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def build_model(model_name: str, input_mode: str = "text_plus_features") -> Pipeline:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    prep_hybrid = ColumnTransformer(
        [
            ("text", text_vectorizer, "text_input"),
            ("num", num_pipe, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    prep_text_only = ColumnTransformer(
        [
            ("text", text_vectorizer, "text_input"),
        ],
        remainder="drop",
    )

    prep_features_only = ColumnTransformer(
        [
            ("num", num_pipe, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    if input_mode == "text_only":
        prep = prep_text_only
    elif input_mode == "features_only":
        prep = prep_features_only
    else:
        prep = prep_hybrid

    if model_name == "hybrid_logreg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    elif model_name == "hybrid_linear_svc_cal":
        clf = CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=10000,
                tol=1e-4,
            ),
            method="sigmoid",
            cv=3,
        )
    elif model_name == "hybrid_sgd_log":
        clf = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=3000,
            tol=1e-3,
            random_state=42,
        )
    elif model_name == "hybrid_sgd_hinge":
        clf = SGDClassifier(
            loss="hinge",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=3000,
            tol=1e-3,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("prep", prep), ("clf", clf)])
