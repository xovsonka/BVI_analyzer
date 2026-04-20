from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_dataset_multiclass import add_engineered_features


STYLE_FAMILIES = ("header_auth_style", "url_style")
LABELS = ("legit", "phishing", "spam", "financial_fraud")

LEGIT_SENDERS = [
    "Operations Coordination <ops@bvi.local>",
    "Facilities Desk <facilities@bvi.local>",
    "Workplace Services <coordination@bvi.local>",
]

PHISHING_SENDERS = [
    "Microsoft Account Team <notice@identity-check.example>",
    "Google Workspace Security <alerts@workspace-protect.example>",
    "Outlook Security Desk <review@account-notice.example>",
]

SPAM_SENDERS = [
    "Member Deals <offers@member-perks.example>",
    "Rewards Desk <rewards@bonus-access.example>",
    "Promo Center <discounts@campaign-rewards.example>",
]

FRAUD_SENDERS = [
    "Accounts Receivable <remittance@vendor-settlement.example>",
    "Payment Operations <billing@invoice-clearance.example>",
    "Collections Desk <collections@supplier-remittance.example>",
]

SUBJECTS = {
    "header_auth_style": {
        "legit": [
            "Internal facilities coordination note",
            "Office services planning update",
            "Visitor access review meeting",
        ],
        "phishing": [
            "Account verification notice",
            "Recovery details confirmation required",
            "Sign-in protection review",
        ],
        "spam": [
            "Bonus access offer for selected members",
            "Promotional reward window",
            "Member discount update",
        ],
        "financial_fraud": [
            "Payment reminder for invoice INV-{invoice}",
            "Vendor remittance status for INV-{invoice}",
            "Accounts payable follow-up for INV-{invoice}",
        ],
    },
    "url_style": {
        "legit": [
            "Updated operations portal reference",
            "Facilities coordination reminder",
            "Workplace services follow-up",
        ],
        "phishing": [
            "Microsoft mailbox review required",
            "Google account verification notice",
            "Outlook access confirmation",
        ],
        "spam": [
            "Limited member reward offer",
            "Promotional discount window",
            "Exclusive reward campaign",
        ],
        "financial_fraud": [
            "Invoice INV-{invoice} settlement required",
            "Remittance update for INV-{invoice}",
            "Billing portal follow-up for INV-{invoice}",
        ],
    },
}

BODY_TEMPLATES = {
    "header_auth_style": {
        "legit": [
            "This is a short internal coordination note for Workplace Services and Facilities. Please review the open office items, confirm room allocation changes, and reply if your team has any scheduling conflicts before the next internal planning meeting.",
            "We are aligning internal support coverage for the upcoming week and need a quick confirmation from each team on workspace moves, visitor access approvals, and pending facilities requests. Please send updates by the end of the day.",
            "Please keep time for a brief internal review focused on visitor logs, front-desk coverage, and outstanding office services requests that need confirmation before the next reporting cycle closes.",
        ],
        "phishing": [
            "Our security operations queue flagged a sign-in inconsistency associated with your mailbox. Please confirm account ownership, review your recovery details, and complete the verification workflow so the temporary hold can be removed before the case is escalated.",
            "A new authentication check triggered a protection review on your account. Please verify your sign-in details, confirm your recovery contact information, and acknowledge the account notice to prevent interruption to mailbox services.",
            "We detected an account recovery update that requires user confirmation. Please complete the mailbox verification sequence and review the sign-in protection notice so the security desk can close the incident.",
        ],
        "spam": [
            "You are still listed as eligible for a short-lived member promotion. The current offer includes a bonus reward and limited discount window available only during this campaign cycle. Confirm interest today or unsubscribe from future promotional mailings.",
            "Our campaign list still shows your address as eligible for a promotional access package. The reward page will be retired shortly, so please confirm interest if you want to keep the current bonus and discount offer active.",
            "This is a final promotional reminder for the current member reward campaign. The offer includes a bonus perk, a discount, and optional future updates if you remain subscribed to promotional mailings.",
        ],
        "financial_fraud": [
            "Our records show invoice INV-{invoice} remains unpaid past the agreed remittance window. Please review the outstanding balance, confirm whether the transfer has already been initiated, and send payment confirmation today so the account is not escalated to restricted billing terms.",
            "Accounts Payable Team, invoice INV-{invoice} is still marked open in the vendor ledger. Please confirm remittance status, provide the transfer reference if settlement has already been released, or arrange payment approval before the reconciliation window closes.",
            "This is a formal payment follow-up for invoice INV-{invoice}. Please review the balance, confirm bank transfer timing, and send a remittance reference or confirmed settlement date so the vendor account can be cleared today.",
        ],
    },
    "url_style": {
        "legit": [
            "Please review the internal operations reference below before the next workplace services sync. The linked page contains the updated room schedule and facilities notes for this week's office coordination tasks.",
            "This is a short internal reminder to review the workplace services portal entry linked below. It contains the latest visitor access guidance and office coordination notes that will be discussed at the next internal check-in.",
            "Please use the internal reference below to review updated facilities planning notes before the next operations meeting. The link points to the standard internal portal used for office services announcements and scheduling updates.",
        ],
        "phishing": [
            "We detected a sign-in protection event tied to your mailbox and need you to complete an account verification review. Please confirm account ownership, verify your recovery options, and follow the sign-in review link below so temporary restrictions can be removed.",
            "Our account protection workflow flagged a login anomaly that requires immediate user confirmation. Please review the account notice, verify your current sign-in details, and complete the recovery confirmation flow using the link below.",
            "A mailbox protection case was opened after an unusual authentication event. Please complete the account verification steps, review the sign-in details, and confirm your recovery information so the security team can close the incident.",
        ],
        "spam": [
            "This is a promotional reminder that your address is still eligible for a limited-time reward. Please review the offer links below if you want to keep the current bonus package, or unsubscribe if you no longer want future marketing updates.",
            "Our member campaign is still showing your account as eligible for a promotional reward and discount package. Please review the offer links below before the campaign window closes, or unsubscribe from future promotional notices.",
            "This is a final marketing notice for the current reward campaign. The offer below includes a short-lived bonus and discount package available only during this campaign period.",
        ],
        "financial_fraud": [
            "Invoice INV-{invoice} remains outstanding and our remittance team needs payment confirmation today. Please review the billing links below, confirm whether a bank transfer has already been initiated, and send the remittance reference so the account can be cleared.",
            "This is an urgent remittance reminder for invoice INV-{invoice}. Please review the billing portal below, confirm payment status, and provide the transfer reference or settlement date before the payment window closes.",
            "Our vendor ledger still shows invoice INV-{invoice} as unpaid. Please use the billing references below to review the amount due, confirm payment timing, and send remittance details so further collection action can be avoided.",
        ],
    },
}


def _pick(items: list[str], idx: int) -> str:
    return items[(idx - 1) % len(items)]


def _pick_sender(label: str, idx: int) -> str:
    mapping = {
        "legit": LEGIT_SENDERS,
        "phishing": PHISHING_SENDERS,
        "spam": SPAM_SENDERS,
        "financial_fraud": FRAUD_SENDERS,
    }
    return _pick(mapping[label], idx)


def _auth_header(label: str, style_family: str) -> str:
    if label == "legit":
        return "spf=pass dkim=pass dmarc=pass"
    if label == "phishing":
        return "spf=fail dkim=fail dmarc=fail"
    if label == "spam":
        return "spf=pass dkim=none dmarc=none"
    if style_family == "header_auth_style":
        return "spf=neutral dkim=fail dmarc=none"
    return "spf=pass dkim=none dmarc=none"


def _build_url_payload(label: str, idx: int) -> str:
    if label == "legit":
        return f"\n\nInternal reference: https://portal.bvi.local/ops/briefing-{idx}"
    if label == "phishing":
        return (
            f"\n\nReview account: https://micros0ft-support.com/auth/{idx}"
            f"\nRecovery page: https://account-check-microsoft-security.com/session/{idx}"
        )
    if label == "spam":
        return (
            f"\n\nOffer links: https://bit.ly/promo{idx} "
            f"and http://bonus-offer-{idx}.review/reward"
        )
    return (
        f"\n\nBilling portal: https://billing.vendor-payments.example/invoice/{idx}"
        f"\nPayment review: http://login.company.com@billing-gateway.com/invoice/{idx}?token=%7Bpay%7D"
    )


def _build_header_values(label: str, idx: int, style_family: str) -> dict[str, str]:
    sender = _pick_sender(label, idx)
    _, sender_addr = sender.rsplit("<", 1) if "<" in sender else ("", sender)
    sender_addr = sender_addr.strip().strip(">")
    sender_domain = sender_addr.split("@", 1)[-1].lower() if "@" in sender_addr else "example.com"
    receiver = f"adapt{idx}@local.lab"
    reply_to = sender_addr
    return_path = sender_addr
    message_id_domain = sender_domain
    if label == "phishing":
        reply_to = f"verify{idx}@mailbox-verify.click"
        return_path = f"return{idx}@account-review.click"
        message_id_domain = "identity-review.click"
    elif label == "financial_fraud" and style_family == "header_auth_style":
        reply_to = f"accounts{idx}@remittance-desk.example"
        return_path = f"collections{idx}@settlement-review.example"
        message_id_domain = "invoice-gateway.example"

    return {
        "sender": sender,
        "receiver": receiver,
        "reply_to": reply_to,
        "return_path": return_path,
        "message_id": f"<adapt-{style_family}-{label}-{idx}@{message_id_domain}>",
    }


def _build_text_raw(subject: str, body: str, header_values: dict[str, str], auth_header: str) -> str:
    return "\n".join(
        [
            f"From: {header_values['sender']}",
            f"To: {header_values['receiver']}",
            f"Reply-To: {header_values['reply_to']}",
            f"Return-Path: <{header_values['return_path']}>",
            f"Message-ID: {header_values['message_id']}",
            f"Authentication-Results: mx.local; {auth_header}",
            "Received: from mx.local by local.lab with ESMTP id adapt001",
            "Received: from relay.local by mx.local with ESMTP id adapt002",
            f"Subject: {subject}",
            "",
            body,
        ]
    ).strip()


def build_rows(rows_per_class_per_style: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    next_id = 1
    for style_family in STYLE_FAMILIES:
        for label in LABELS:
            for _ in range(rows_per_class_per_style):
                invoice = 59000 + next_id
                subject = _pick(SUBJECTS[style_family][label], next_id).format(invoice=invoice)
                body = _pick(BODY_TEMPLATES[style_family][label], next_id).format(invoice=invoice)
                if style_family == "url_style":
                    body += _build_url_payload(label, next_id)
                else:
                    if label == "legit":
                        body += f"\n\nAgenda reference: ADAPT-OPS-{next_id:03d}."
                    elif label == "phishing":
                        body += f"\n\nSecurity case reference: ADAPT-ACCT-{next_id:03d}."
                    elif label == "spam":
                        body += f"\n\nPromotion batch reference: ADAPT-PR-{next_id:03d}."
                    else:
                        body += f"\n\nBilling case reference: ADAPT-INV-{invoice}."
                header_values = _build_header_values(label, next_id, style_family)
                auth_header = _auth_header(label, style_family)
                text_raw = _build_text_raw(subject, body, header_values, auth_header)
                rows.append(
                    {
                        "row_id": f"adapt-{style_family}-{label}-{next_id:04d}",
                        "subject": subject,
                        "body": body,
                        "text": body,
                        "subject_raw": subject,
                        "body_raw": body,
                        "text_raw": text_raw,
                        "sender": header_values["sender"],
                        "receiver": header_values["receiver"],
                        "date": "Mon, 01 Apr 2026 10:00:00 +0000",
                        "label": label,
                        "source": f"campaign_adapt_{style_family}",
                        "source_profile": "campaign_adaptation",
                        "generation_kind": style_family,
                        "generation_family": "campaign_adaptation",
                        "adaptation_family": style_family,
                    }
                )
                next_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-leaky campaign-style adaptation training set")
    parser.add_argument("--rows-per-class-per-style", type=int, default=12)
    parser.add_argument(
        "--output-csv",
        default="results/adaptation/campaign_style_train_adaptation.csv",
    )
    args = parser.parse_args()

    df = build_rows(rows_per_class_per_style=int(args.rows_per_class_per_style))
    df = add_engineered_features(df)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved adaptation training set: {out_path} ({len(df)} rows)")
    print(df["label"].value_counts().to_string())
    print(df["adaptation_family"].value_counts().to_string())


if __name__ == "__main__":
    main()
