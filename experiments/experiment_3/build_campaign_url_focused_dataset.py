from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd


TARGET_BY_CLASS = {
    "legit": 20,
    "phishing": 35,
    "spam": 20,
    "financial_fraud": 25,
}
TOTAL_TARGET = sum(TARGET_BY_CLASS.values())
LLM_TARGET_TOTAL = 50

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

LEGIT_SENDERS = [
    "Operations Coordination <ops@bvi.local>",
    "Facilities Desk <facilities@bvi.local>",
    "Workplace Services <coordination@bvi.local>",
]

PHISHING_SENDERS = [
    "Microsoft Account Operations <alerts@accountprotection-msft-mail.com>",
    "Microsoft Security <notify@cheap-host.example>",
    "Google Workspace <admin@workspace-reset.example>",
]

SPAM_SENDERS = [
    "Member Rewards <rewards@promo-benefits.example>",
    "Offer Desk <offers@discount-hub.example>",
    "Bonus Access <access@member-perks.example>",
]

FRAUD_SENDERS = [
    "Northbridge Vendor Services <collections@northbridge-vendor.example>",
    "Accounts Receivable <billing@remittance-ops.example>",
    "Payment Desk <invoices@vendor-settlement.example>",
]

CLASS_SUBJECTS = {
    "legit": [
        "Tuesday Operations Sync - 10:30 AM",
        "Facilities Coordination Update",
        "Updated Visitor Access Schedule",
        "Internal Compliance Review Note",
    ],
    "phishing": [
        "Immediate Account Review Required",
        "Mailbox Verification Required Today",
        "Action Required: Verify Your Microsoft Account",
        "Recovery Details Review Required",
    ],
    "spam": [
        "Limited-time reward for selected members",
        "Claim your member perk before midnight",
        "Exclusive discount window closes today",
        "Final notice: your bonus access ends tonight",
    ],
    "financial_fraud": [
        "Urgent: Invoice INV-{invoice} Overdue",
        "Final Reminder: Invoice INV-{invoice} Due",
        "Invoice INV-{invoice} Payment Status",
        "Remittance Needed for Invoice INV-{invoice}",
    ],
}

CLASS_BODIES = {
    "legit": [
        "Please join the weekly internal operations sync on Tuesday at 10:30 AM in Conference Room 3B. We will review open office services items, confirm facilities coverage for the rest of the week, and align on workspace requests before month-end. If you cannot attend, reply with any updates that should be included in the meeting notes.",
        "This is a short internal coordination note for Workplace Services and Facilities. We will review visitor access requests, desk move timing, and room scheduling updates that need to be confirmed before the next reporting cycle. Please send agenda additions by the end of the day.",
        "Please keep time for a brief internal compliance review focused on visitor logs, workspace access approvals, and reception handoff notes that need confirmation this week. The goal is to close remaining action items before the reporting package is finalized.",
        "We are holding a short internal planning meeting to confirm support coverage, pending facilities requests, and equipment moves that affect the office operations schedule. If you are responsible for any open action item, please be ready to share a quick update.",
    ],
    "phishing": [
        "We are contacting you on behalf of Microsoft Account Operations regarding a sign-in protection event detected earlier today. To prevent interruption, verify account ownership, confirm your current recovery details, and complete the account review process so the temporary hold can be removed before the security case is escalated.",
        "A recent authentication event triggered an urgent security workflow for your mailbox. Please review the sign-in notice, verify your account details, and complete the recovery confirmation sequence so access restrictions are not applied to Outlook and related Microsoft services.",
        "Your account was added to a high-priority protection review after a new device check failed. Please confirm your sign-in details, verify your mailbox, and complete the recovery review so the session hold can be cleared by the operations team.",
        "We detected unusual account activity and need you to complete a mailbox verification review today. Please verify account ownership, confirm your current recovery contact details, and acknowledge the sign-in protection notice to avoid interruption.",
    ],
    "spam": [
        "You are still marked as eligible for the bonus access package announced earlier this week. This promotion closes tonight and includes a limited member perk for selected recipients. Confirm interest before the campaign window is retired, or unsubscribe if you no longer want future promotional updates.",
        "Our campaign list still shows your address as eligible for a limited-time member reward. The current offer includes a short-lived discount and a promotional perk that will expire after tonight's deadline. Confirm interest now or unsubscribe from future offers.",
        "This is the final reminder for the seasonal member promotion tied to your account. The current offer includes a reward, a short-lived discount, and a member benefit available only during the active campaign window. Reply if you want details, or unsubscribe from future promotional mailings.",
        "We are sending one last notice before this marketing reward is withdrawn from the active campaign list. The offer includes bonus access and a promotional discount available for a short period only. Confirm interest today or unsubscribe from future promotions.",
    ],
    "financial_fraud": [
        "Accounts Payable Team, our records show invoice INV-{invoice} remains unpaid past the agreed remittance window. Please review the balance, confirm whether payment has already been released, and send the transfer reference today so the account is not moved to restricted payment status before end-of-day reconciliation.",
        "This is a formal billing reminder regarding invoice INV-{invoice}, which is still marked open in the vendor ledger. Please confirm payment status, provide the remittance reference if settlement has already been initiated, or arrange bank transfer approval today so the balance can be cleared without escalation.",
        "Our finance team is reconciling open vendor balances and invoice INV-{invoice} is still listed as outstanding. Please review the amount due, confirm whether transfer approval has been completed, and send payment confirmation today to avoid late processing action on the account.",
        "Please treat this as an urgent remittance reminder for invoice INV-{invoice}. The balance remains unpaid and our accounts receivable team needs payment confirmation, transfer details, or a confirmed settlement date before the end-of-day payment window closes.",
    ],
}


def _pick(items: list[str], idx: int) -> str:
    return items[(idx - 1) % len(items)]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    def _col(name: str, fallback: str = "") -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([fallback] * len(df), index=df.index, dtype=str)

    out = pd.DataFrame()
    out["subject"] = _col("subject")
    out["text"] = _col("body") if "body" in df.columns else _col("text")
    out["sender"] = _col("from_email") if "from_email" in df.columns else _col("sender")
    out["attack_type"] = (
        _col("attack_type").str.strip().str.lower().replace({"spoofing": "phishing"})
    )
    return out


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) >= n:
        return df.sample(n=n, random_state=seed)
    return df.sample(n=n, random_state=seed, replace=True)


def _strip_urls_and_tags(text: str) -> str:
    cleaned = TAG_RE.sub(" ", text or "")
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_emails(text: str) -> str:
    cleaned = EMAIL_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _url_augments(
    cls: str, idx: int, include_anchor_mismatch: bool
) -> dict[str, object]:
    expected = {
        "expected_ip_url": 0,
        "expected_shortener_url": 0,
        "expected_anchor_mismatch": 0,
        "expected_suspicious_tld": 0,
        "expected_brand_typosquat": 0,
        "expected_obfuscated_url": 0,
        "expected_binary_label": "legit" if cls == "legit" else "suspicious",
    }

    if cls == "legit":
        if idx % 3 == 0:
            link = f"https://portal.bvi.local/announcements/{idx}"
        elif idx % 3 == 1:
            link = f"https://portal.bvi.local/meetings/ops-sync-{idx}"
        else:
            link = f"https://portal.bvi.local/workplace/coordination-{idx}"
        return {
            "url_payload": f"\n\nReference: {link}",
            "html_payload": "",
            **expected,
        }

    if cls == "phishing":
        expected.update(
            {
                "expected_brand_typosquat": 1,
            }
        )
        html = ""
        if include_anchor_mismatch:
            expected["expected_anchor_mismatch"] = 1
            expected["expected_suspicious_tld"] = 1
            html = "\n\n" + (
                '<a href="http://mailbox-check.click/session/{idx}">'
                "https://microsoft.com/security"
                "</a>"
            ).format(idx=idx)
        return {
            "url_payload": (
                f"\n\nSign in now: https://micros0ft.com/auth/{idx}"
                f"\nAccount review: https://account-review-micros0ft.com/session/{idx}"
            ),
            "html_payload": html,
            **expected,
        }

    if cls == "spam":
        expected.update({"expected_shortener_url": 1, "expected_suspicious_tld": 1})
        return {
            "url_payload": (
                "\n\nOffer links: https://bit.ly/3J4zM9k "
                f"and http://promo-{idx}.review/reward"
                f"\nTracking page: http://affiliate-{idx}.click/offer"
            ),
            "html_payload": "",
            **expected,
        }

    expected.update({"expected_obfuscated_url": 1})
    return {
        "url_payload": (
            f"\n\nInvoice portal: https://billing.vendor-payments.example/invoice/{idx}"
            f"\nPayment review: http://login.company.com@billing-gateway.com/invoice/{idx}?token=%7Babc%7D"
        ),
        "html_payload": "",
        **expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Exp3 URL-focused campaign dataset (100 rows, 50/50 source mix)"
    )
    parser.add_argument(
        "--llm-csv", default="results/experiment_1/llm_drafts/templates.csv"
    )
    parser.add_argument(
        "--mutated-csv", default="results/experiment_1/mutated/mutated_templates.csv"
    )
    parser.add_argument(
        "--real-edits-csv",
        default="results/experiment_1/multi_real_edits/multi_real_edited_templates.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="results/experiment_3/gophish_seed_input_url_focused_50_50.csv",
    )
    parser.add_argument(
        "--expected-csv",
        default="results/experiment_3/gophish_seed_input_url_focused_expected.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-strip-source-urls",
        action="store_true",
        help="Keep original URLs from source text (not recommended for clean Exp3 Track B tags)",
    )
    parser.add_argument(
        "--enable-anchor-mismatch",
        action="store_true",
        help="Inject anchor-mismatch HTML tag payload (requires real HTML template flow)",
    )
    parser.add_argument(
        "--strip-source-emails",
        action="store_true",
        help="Also remove email addresses from source text before URL injection",
    )
    args = parser.parse_args()

    llm_pool = _load(Path(args.llm_csv))
    llm_pool["generation_kind"] = "llm"
    llm_pool["generation_family"] = "llm"

    mut_pool = _load(Path(args.mutated_csv))
    mut_pool["generation_kind"] = "mutated"
    mut_pool["generation_family"] = "non_llm"

    real_pool = _load(Path(args.real_edits_csv))
    real_pool["generation_kind"] = "real_edit"
    real_pool["generation_family"] = "non_llm"

    rule_pool = pd.concat([mut_pool, real_pool], ignore_index=True)

    classes = list(TARGET_BY_CLASS.keys())
    llm_quota = {cls: TARGET_BY_CLASS[cls] // 2 for cls in classes}
    remainder = LLM_TARGET_TOTAL - sum(llm_quota.values())
    if remainder > 0:
        for cls in sorted(classes, key=lambda c: TARGET_BY_CLASS[c], reverse=True):
            if remainder <= 0:
                break
            if llm_quota[cls] < TARGET_BY_CLASS[cls]:
                llm_quota[cls] += 1
                remainder -= 1

    rows: list[dict] = []
    next_id = 1
    for cls, target in TARGET_BY_CLASS.items():
        llm_n = llm_quota[cls]
        rule_n = target - llm_n
        llm_cls = llm_pool[llm_pool["attack_type"] == cls]
        rule_cls = rule_pool[rule_pool["attack_type"] == cls]
        if llm_cls.empty or rule_cls.empty:
            raise RuntimeError(f"Missing rows for class '{cls}' in source pools")

        chosen = pd.concat(
            [
                _sample(llm_cls, llm_n, args.seed),
                _sample(rule_cls, rule_n, args.seed + 1),
            ],
            ignore_index=True,
        )

        for _, row in chosen.iterrows():
            aug = _url_augments(
                cls, next_id, include_anchor_mismatch=args.enable_anchor_mismatch
            )
            invoice = 48000 + next_id
            base_text = _pick(CLASS_BODIES[cls], next_id).format(invoice=invoice)
            if not args.no_strip_source_urls:
                base_text = _strip_urls_and_tags(base_text)
            if args.strip_source_emails:
                base_text = _strip_emails(base_text)
            if cls == "legit":
                base_text += f"\n\nAgenda reference: OPS-{next_id:03d}."
            elif cls == "phishing":
                base_text += f"\n\nSecurity case reference: ACCT-{next_id:03d}."
            elif cls == "spam":
                base_text += f"\n\nPromotion batch reference: PR-{next_id:03d}."
            else:
                base_text += f"\n\nBilling case reference: INV-{invoice}."
            body_text = f"{base_text}{aug['url_payload']}"
            body_html = ""
            if str(aug["html_payload"]).strip():
                escaped = html.escape(body_text).replace("\n", "<br>")
                body_html = (
                    "<html><body><p>"
                    f"{escaped}</p>{str(aug['html_payload']).strip()}"
                    "</body></html>"
                )
            subject = _pick(CLASS_SUBJECTS[cls], next_id).format(invoice=invoice)
            if cls == "legit":
                sender = _pick(LEGIT_SENDERS, next_id)
            elif cls == "phishing":
                sender = _pick(PHISHING_SENDERS, next_id)
            elif cls == "spam":
                sender = _pick(SPAM_SENDERS, next_id)
            else:
                sender = _pick(FRAUD_SENDERS, next_id)

            rows.append(
                {
                    "id": next_id,
                    "text": body_text,
                    "body": body_text,
                    "body_text": body_text,
                    "body_html": body_html,
                    "label": 0 if cls == "legit" else 1,
                    "source": f"exp3_{row['generation_kind']}",
                    "subject": subject,
                    "sender": sender,
                    "receiver": f"user{next_id}@local.lab",
                    "attack_type": cls,
                    "generation_kind": row["generation_kind"],
                    "generation_family": row.get("generation_family", "non_llm"),
                    "expected_ip_url": int(aug["expected_ip_url"]),
                    "expected_shortener_url": int(aug["expected_shortener_url"]),
                    "expected_anchor_mismatch": int(aug["expected_anchor_mismatch"]),
                    "expected_suspicious_tld": int(aug["expected_suspicious_tld"]),
                    "expected_brand_typosquat": int(aug["expected_brand_typosquat"]),
                    "expected_obfuscated_url": int(aug["expected_obfuscated_url"]),
                    "expected_binary_label": str(aug["expected_binary_label"]),
                    "injection_mode": "controlled_url_injection",
                }
            )
            next_id += 1

    out = pd.DataFrame(rows)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["id"] = range(1, len(out) + 1)
    out["receiver"] = out["id"].map(lambda i: f"user{i}@local.lab")

    if len(out) != TOTAL_TARGET:
        raise RuntimeError(f"Dataset has {len(out)} rows, expected {TOTAL_TARGET}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[
        [
            "id",
            "text",
            "body",
            "body_text",
            "body_html",
            "label",
            "source",
            "subject",
            "sender",
            "receiver",
            "attack_type",
            "generation_kind",
            "generation_family",
        ]
    ].to_csv(out_path, index=False)

    expected_path = Path(args.expected_csv)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    out[
        [
            "id",
            "expected_ip_url",
            "expected_shortener_url",
            "expected_anchor_mismatch",
            "expected_suspicious_tld",
            "expected_brand_typosquat",
            "expected_obfuscated_url",
            "expected_binary_label",
            "injection_mode",
        ]
    ].to_csv(expected_path, index=False)

    print(f"Saved campaign dataset: {out_path} ({len(out)} rows)")
    print("Class distribution:")
    print(out["attack_type"].value_counts().to_string())
    print("Generation family:")
    print(out["generation_family"].value_counts().to_string())
    print(f"Saved expected URL flags (injection tags): {expected_path}")


if __name__ == "__main__":
    main()
