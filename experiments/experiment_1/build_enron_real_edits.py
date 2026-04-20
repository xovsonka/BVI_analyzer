from __future__ import annotations

import argparse
import csv
import random
from email import policy
from email.parser import Parser
from pathlib import Path

import pandas as pd


def parse_enron_message(raw_msg: str) -> dict:
    msg = Parser(policy=policy.default).parsestr(raw_msg or "")
    subject = str(msg.get("Subject", "")).strip()
    from_h = str(msg.get("From", "")).strip()
    to_h = str(msg.get("To", "")).strip()

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = str(part.get_content()).strip()
                    if body:
                        break
                except Exception:
                    continue
    else:
        try:
            body = str(msg.get_content()).strip()
        except Exception:
            body = raw_msg or ""

    return {
        "subject": subject,
        "from_email": from_h,
        "to_email": to_h,
        "body": body,
    }


def create_edits(base: dict) -> list[dict]:
    s = base["subject"] or "Update"
    b = base["body"] or "Please review the details below."

    legit = {
        "attack_type": "legit",
        "label": 0,
        "subject": s,
        "body": b,
        "from_name": "Internal Communication",
        "from_email": "internal@company.local",
        "reply_to": "internal@company.local",
        "return_path": "internal@company.local",
    }
    phishing = {
        "attack_type": "phishing",
        "label": 1,
        "subject": f"Action required: {s}",
        "body": b
        + "\n\nYour account will be restricted unless you verify now: "
        + "https://company-portal-verify-help.com/login",
        "from_name": "IT Security",
        "from_email": "security@company-helpdesk.net",
        "reply_to": "verify@protonmail.com",
        "return_path": "bounce@company-helpdesk.net",
    }
    fraud = {
        "attack_type": "financial_fraud",
        "label": 1,
        "subject": "Invoice payment required - urgent",
        "body": b
        + "\n\nPlease process transfer today. Payment details: "
        + "https://vendor-billing-update.net/pay",
        "from_name": "Finance Desk",
        "from_email": "billing@vendor-billing-update.net",
        "reply_to": "accounts@vendor-billing-update.net",
        "return_path": "noreply@vendor-billing-update.net",
    }
    spam = {
        "attack_type": "spam",
        "label": 1,
        "subject": "Limited offer - claim premium access today",
        "body": b + "\n\nSpecial promo link: https://deals-fast-offer.biz/win",
        "from_name": "Promo Deals",
        "from_email": "offers@deals-fast-offer.biz",
        "reply_to": "offers@deals-fast-offer.biz",
        "return_path": "bounce@deals-fast-offer.biz",
    }
    return [legit, phishing, fraud, spam]


def format_txt(row: dict) -> str:
    return (
        f"Subject: {row['subject']}\n"
        f"From: {row['from_name']} <{row['from_email']}>\n"
        f"Reply-To: {row['reply_to']}\n"
        f"Return-Path: {row['return_path']}\n"
        f"Attack-Type: {row['attack_type']}\n"
        f"Label: {row['label']}\n"
        "\n"
        f"{row['body']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create realistic edited drafts from Enron emails"
    )
    parser.add_argument("--enron-csv", default="dataset/source/enron/emails.csv")
    parser.add_argument("--out-dir", default="results/experiment_1/enron_edits")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    enron_csv = Path(args.enron_csv)
    if not enron_csv.exists():
        raise FileNotFoundError(f"Missing Enron CSV: {enron_csv}")

    rnd = random.Random(args.seed)
    df = pd.read_csv(enron_csv, usecols=["file", "message"], low_memory=False)
    df = df.dropna(subset=["message"])
    if len(df) == 0:
        raise RuntimeError("Enron CSV has no usable rows")

    n = min(args.samples, len(df))
    sampled = df.sample(n=n, random_state=args.seed)

    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "emails_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    idx = 1
    for _, r in sampled.iterrows():
        base = parse_enron_message(str(r["message"]))
        edits = create_edits(base)
        rnd.shuffle(edits)
        for e in edits:
            row = dict(e)
            row["id"] = idx
            row["source"] = "exp1_enron_edit"
            row["template_id"] = f"enron_edit_{idx:05d}"
            row["text"] = f"{row['subject']}\n\n{row['body']}"
            txt_path = txt_dir / f"{row['template_id']}.txt"
            txt_path.write_text(format_txt(row), encoding="utf-8")
            row["txt_path"] = str(txt_path)
            rows.append(row)
            idx += 1

    out_csv = out_dir / "enron_edited_templates.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "template_id",
                "attack_type",
                "label",
                "source",
                "subject",
                "from_name",
                "from_email",
                "reply_to",
                "return_path",
                "body",
                "text",
                "txt_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    seed_csv = out_dir / "gophish_seed_input.csv"
    with seed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "source"])
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "label": r["label"],
                    "source": r["source"],
                }
            )

    print(f"Saved edited Enron CSV: {out_csv}")
    print(f"Saved editable TXT files: {txt_dir}")
    print(f"Saved GoPhish seed CSV: {seed_csv}")


if __name__ == "__main__":
    main()
