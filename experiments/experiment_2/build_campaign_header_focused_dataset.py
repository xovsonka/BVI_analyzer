from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_BY_CLASS = {
    "legit": 20,
    "phishing": 35,
    "spam": 20,
    "financial_fraud": 25,
}

LLM_TARGET_TOTAL = 50
TOTAL_TARGET = sum(TARGET_BY_CLASS.values())

TRUSTED_SENDERS = [
    "Operations Coordination <ops@bvi.local>",
    "Facilities Desk <facilities@bvi.local>",
    "Workplace Services <coordination@bvi.local>",
]

PHISHING_SENDERS = [
    "Microsoft Security <notify@cheap-host.example>",
    "Microsoft Account Operations <alerts@accountprotection-msft-mail.com>",
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
        "Tuesday Operations Check-In",
        "Facilities Coordination Meeting",
        "Quarterly Compliance Review",
        "Updated Office Services Agenda",
    ],
    "phishing": [
        "Mailbox Verification Required Today",
        "Immediate Account Review Required",
        "Sign-In Confirmation Needed",
        "Recovery Details Review Required",
    ],
    "spam": [
        "Final notice: your bonus access ends tonight",
        "Limited-time reward for selected members",
        "Claim your member perk before midnight",
        "Exclusive discount window closes today",
    ],
    "financial_fraud": [
        "Invoice INV-{invoice} Payment Status",
        "Final Reminder: Invoice INV-{invoice} Due",
        "Urgent: Invoice INV-{invoice} Overdue",
        "Remittance Needed for Invoice INV-{invoice}",
    ],
}

CLASS_BODIES = {
    "legit": [
        "Please join the weekly internal operations check-in on Tuesday at 10:30 AM in Conference Room 3B. We will review open facilities tasks, staffing coverage for the rest of the week, and pending workspace requests before month-end. If you cannot attend, please reply with any updates that should be included in the agenda.",
        "This is a short internal coordination note for Workplace Services and Facilities. We will review visitor access requests, desk move timing, and room scheduling updates that need to be confirmed before the next reporting cycle. Please send any agenda items by end of day so they can be included in the meeting pack.",
        "We are holding a brief compliance review meeting for internal teams to confirm document retention checkpoints and access log sign-offs before the quarter closes. Attendance is requested from team coordinators who handle office support, contractor access, or shared workspace administration.",
        "Please keep time for a short office services planning meeting focused on reception coverage, equipment moves, and open support tickets that need follow-up this week. The goal is to align on ownership and close any remaining action items before Friday afternoon.",
    ],
    "phishing": [
        "Security Operations flagged a sign-in inconsistency tied to your mailbox and needs the account owner to complete a verification review today. To prevent interruption, confirm your mailbox status, review recent access details, and complete the account verification step so the temporary hold can be removed before the review window closes.",
        "We detected changes to recovery details and an unrecognized sign-in path associated with your account. Please verify account ownership, confirm your current recovery options, and complete the login review process today to avoid temporary restrictions on mailbox access and related services.",
        "Your account was added to an urgent security review after a recent authentication check failed. Please complete the mailbox verification sequence, confirm your sign-in details, and acknowledge the protection notice so the service desk can release the account from review.",
        "A new device check triggered an account protection workflow for your mailbox. Please review the sign-in notice, verify your account credentials, and confirm your recovery details so the security team can close the case without further interruption.",
    ],
    "spam": [
        "You are still marked as eligible for the bonus access package announced earlier this week. This promotion closes tonight and includes a limited member perk for selected recipients. To keep your reward active, confirm interest before the offer window is retired. If you no longer want updates like this, unsubscribe from future promotional notices.",
        "Our campaign list still shows your address as eligible for a limited-time member reward. The access window closes soon and the included promo discount will not be extended after the current cycle ends. Confirm interest before the reward page is retired, or unsubscribe if you no longer want future offers.",
        "This is the final reminder for the seasonal member promotion tied to your account. The offer includes a short-lived discount and bonus access that expires after tonight's campaign deadline. Reply if you want details, or unsubscribe if you prefer not to receive future promotional updates.",
        "We are sending one final notice before this marketing reward is withdrawn from the active campaign list. The current offer includes a member perk and promotional discount available only for a short period. Confirm interest today or unsubscribe from future promotional mailings.",
    ],
    "financial_fraud": [
        "Accounts Payable Team, our records show invoice INV-{invoice} remains unpaid past the agreed remittance window. Please review the outstanding balance, confirm whether payment has already been released, and send the remittance reference today so the account does not move to restricted payment status. If settlement is still pending, arrange bank transfer approval before close of business.",
        "This is a formal billing reminder regarding invoice INV-{invoice}, which is still marked open in the vendor ledger. Please confirm payment status, provide the transfer reference if remittance has already been initiated, or escalate internally so the balance can be cleared before today's payment deadline.",
        "Our finance team is reconciling open vendor balances and invoice INV-{invoice} is still listed as outstanding. Please review the amount due, confirm whether bank transfer approval has been completed, and send payment confirmation today to avoid late processing action on the account.",
        "Please treat this as an urgent remittance reminder for invoice INV-{invoice}. The balance remains unpaid and our accounts receivable team needs payment confirmation, transfer details, or a confirmed settlement date before the end-of-day reconciliation window closes.",
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
        _col("attack_type")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"spoofing": "phishing"})
    )
    out["source"] = _col("source", "exp2")
    return out


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) >= n:
        return df.sample(n=n, random_state=seed)
    return df.sample(n=n, random_state=seed, replace=True)


def _dedup_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dedup_key"] = (
        out["subject"].astype(str).str.strip().str.lower()
        + "||"
        + out["sender"].astype(str).str.strip().str.lower()
        + "||"
        + out["text"].astype(str).str.strip().str.lower()
    )
    return out.drop_duplicates(subset="dedup_key").drop(columns=["dedup_key"])


def _augment_row(row: pd.Series, cls: str, idx: int) -> dict:
    invoice = 48000 + idx
    subject = _pick(CLASS_SUBJECTS[cls], idx).format(invoice=invoice)
    body = _pick(CLASS_BODIES[cls], idx).format(invoice=invoice)
    sender = ""

    expected_display_name_spoof = 0
    expected_external_url = 0
    expected_sender_url_mismatch = 0
    expected_suspicious_tld = 0

    if cls == "legit":
        sender = _pick(TRUSTED_SENDERS, idx)
    elif cls == "phishing":
        sender = _pick(PHISHING_SENDERS, idx)
        expected_display_name_spoof = 1
    elif cls == "spam":
        sender = _pick(SPAM_SENDERS, idx)
    elif cls == "financial_fraud":
        sender = _pick(FRAUD_SENDERS, idx)

    if cls == "legit":
        body += f"\n\nAgenda reference: OPS-{idx:03d}."
    elif cls == "phishing":
        body += f"\n\nSecurity case reference: ACCT-{idx:03d}."
    elif cls == "spam":
        body += f"\n\nPromotion batch reference: PR-{idx:03d}."
    else:
        body += f"\n\nBilling case reference: INV-{invoice}."

    return {
        "subject": subject,
        "text": body,
        "sender": sender,
        "attack_type": cls,
        "expected_display_name_spoof_flag": expected_display_name_spoof,
        "expected_external_domain_url": expected_external_url,
        "expected_sender_url_mismatch": expected_sender_url_mismatch,
        "expected_suspicious_tld": expected_suspicious_tld,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Exp2 campaign dataset focused on header/auth-related indicators"
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
        default="results/experiment_2/gophish_seed_input_header_focused_50_50.csv",
    )
    parser.add_argument(
        "--expected-csv",
        default="results/experiment_2/gophish_seed_input_header_focused_expected.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    llm_pool = _dedup_pool(_load(Path(args.llm_csv)))
    llm_pool["generation_kind"] = "llm"
    llm_pool["generation_family"] = "llm"

    mut_pool = _dedup_pool(_load(Path(args.mutated_csv)))
    mut_pool["generation_kind"] = "mutated"
    mut_pool["generation_family"] = "non_llm"

    real_pool = _dedup_pool(_load(Path(args.real_edits_csv)))
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

    rows = []
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
            aug = _augment_row(row, cls, next_id)
            rows.append(
                {
                    "id": next_id,
                    "text": aug["text"],
                    "body": aug["text"],
                    "label": 0 if cls == "legit" else 1,
                    "source": f"exp2_{row['generation_kind']}",
                    "subject": aug["subject"],
                    "sender": aug["sender"],
                    "receiver": f"user{next_id}@local.lab",
                    "attack_type": cls,
                    "generation_kind": row["generation_kind"],
                    "generation_family": row.get("generation_family", "non_llm"),
                    "expected_display_name_spoof_flag": aug[
                        "expected_display_name_spoof_flag"
                    ],
                    "expected_external_domain_url": aug["expected_external_domain_url"],
                    "expected_sender_url_mismatch": aug["expected_sender_url_mismatch"],
                    "expected_suspicious_tld": aug["expected_suspicious_tld"],
                }
            )
            next_id += 1

    out = pd.DataFrame(rows)
    out = out[
        out["subject"].astype(str).str.strip().ne("")
        & out["text"].astype(str).str.strip().ne("")
        & out["sender"].astype(str).str.strip().ne("")
    ].copy()
    out["dedup_key"] = (
        out["subject"].astype(str).str.strip().str.lower()
        + "||"
        + out["sender"].astype(str).str.strip().str.lower()
        + "||"
        + out["text"].astype(str).str.strip().str.lower()
    )
    duplicate_rows = int(out["dedup_key"].duplicated().sum())
    out = out.drop(columns=["dedup_key"])
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["id"] = np.arange(1, len(out) + 1)
    out["receiver"] = out["id"].map(lambda i: f"user{i}@local.lab")

    if len(out) != TOTAL_TARGET:
        raise RuntimeError(
            f"Dataset has {len(out)} rows after dedup, expected {TOTAL_TARGET}. "
            "Adjust source pools or seed to keep exact campaign size."
        )

    class_counts = out["attack_type"].value_counts().to_dict()
    for cls, target in TARGET_BY_CLASS.items():
        got = int(class_counts.get(cls, 0))
        if got != target:
            raise RuntimeError(f"Class '{cls}' count is {got}, expected {target}")

    family_counts = out["generation_family"].value_counts().to_dict()
    if int(family_counts.get("llm", 0)) != LLM_TARGET_TOTAL:
        raise RuntimeError(
            f"LLM family count is {int(family_counts.get('llm', 0))}, expected {LLM_TARGET_TOTAL}"
        )
    if int(family_counts.get("non_llm", 0)) != TOTAL_TARGET - LLM_TARGET_TOTAL:
        raise RuntimeError(
            "non_llm family count is "
            f"{int(family_counts.get('non_llm', 0))}, expected {TOTAL_TARGET - LLM_TARGET_TOTAL}"
        )

    expected = out[
        [
            "id",
            "attack_type",
            "generation_kind",
            "generation_family",
            "expected_display_name_spoof_flag",
            "expected_external_domain_url",
            "expected_sender_url_mismatch",
            "expected_suspicious_tld",
        ]
    ].copy()
    expected["expected_binary_label"] = np.where(
        expected["attack_type"].astype(str).eq("legit"), "legit", "suspicious"
    )
    expected["injection_mode"] = "controlled_header_injection"

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[
        [
            "id",
            "text",
            "body",
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
    expected.to_csv(expected_path, index=False)

    print(f"Saved campaign dataset: {out_path} ({len(out)} rows)")
    print("Class distribution:")
    print(out["attack_type"].value_counts().to_string())
    print("Generation mix:")
    print(out["generation_kind"].value_counts().to_string())
    print("Generation family:")
    print(out["generation_family"].value_counts().to_string())
    print(f"Potential duplicate messages (subject+sender+text): {duplicate_rows}")
    print(f"Saved expected header-focused flags: {expected_path}")


if __name__ == "__main__":
    main()
