from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import pandas as pd
from email import policy
from email.parser import Parser


TEXT_COL_CANDIDATES = [
    "message",
    "text",
    "body",
    "content",
    "email",
    "mail",
]

SUBJECT_COL_CANDIDATES = ["subject", "title", "email_subject"]


def _parse_message_blob(message: str) -> tuple[str, str]:
    try:
        msg = Parser(policy=policy.default).parsestr(message or "")
        subject = str(msg.get("Subject", "")).strip()
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = str(part.get_content()).strip()
                    if body:
                        break
        else:
            body = str(msg.get_content()).strip()
        return subject, body
    except Exception:
        return "", str(message or "").strip()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_text_from_row(row: pd.Series) -> tuple[str, str]:
    lower_map = {str(c).lower(): c for c in row.index}

    subject = ""
    for c in SUBJECT_COL_CANDIDATES:
        if c in lower_map:
            v = row.get(lower_map[c], "")
            if pd.notna(v) and str(v).strip():
                subject = str(v).strip()
                break

    for c in TEXT_COL_CANDIDATES:
        if c in lower_map:
            v = row.get(lower_map[c], "")
            if pd.notna(v) and str(v).strip():
                text = str(v)
                if c == "message":
                    s2, b2 = _parse_message_blob(text)
                    if s2 and not subject:
                        subject = s2
                    text = b2 or text
                return subject, _normalize_space(text)

    # fallback: concatenate medium/long text-like columns
    chunks = []
    for col in row.index:
        v = row.get(col, "")
        if pd.isna(v):
            continue
        s = str(v).strip()
        if len(s) >= 40:
            chunks.append(s)
    return subject, _normalize_space("\n".join(chunks))


def collect_candidates(
    source_root: Path, max_rows_per_file: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for csv_path in source_root.rglob("*.csv"):
        rel = csv_path.relative_to(source_root)
        source_group = rel.parts[0] if rel.parts else "unknown"

        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=max_rows_per_file)
        except Exception:
            try:
                df = pd.read_csv(
                    csv_path,
                    low_memory=False,
                    nrows=max_rows_per_file,
                    encoding="latin1",
                )
            except Exception:
                continue

        if df.empty:
            continue

        # randomize local order so first rows don't dominate
        idxs = list(df.index)
        rng.shuffle(idxs)
        for i in idxs:
            row = df.loc[i]
            subject, body = _extract_text_from_row(row)
            if len(body) < 80:
                continue
            out.append(
                {
                    "source_group": source_group,
                    "source_file": str(rel),
                    "subject_base": subject,
                    "body_base": body,
                }
            )
    return out


def _clip_text(text: str, max_words: int = 130) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _format_txt(row: dict) -> str:
    return (
        f"Subject: {row['subject']}\n"
        f"From: {row['from_name']} <{row['from_email']}>\n"
        f"Reply-To: {row['reply_to']}\n"
        f"Return-Path: {row['return_path']}\n\n"
        f"{row['body']}\n"
    )


def _build_legit(base: dict) -> dict:
    body = _clip_text(base["body_base"], 120)
    return {
        "attack_type": "legit",
        "label": 0,
        "subject": base["subject_base"] or "Internal update",
        "body": body,
        "from_name": "Internal Operations",
        "from_email": "ops@company.local",
        "reply_to": "ops@company.local",
        "return_path": "ops@company.local",
    }


def _build_phishing(base: dict) -> dict:
    body = _clip_text(base["body_base"], 90)
    body += (
        "\n\nSecurity notice: your mailbox verification is pending. "
        "Please confirm access now: https://mailbox-verification-center.example/auth"
    )
    return {
        "attack_type": "phishing",
        "label": 1,
        "subject": "Urgent mailbox verification required",
        "body": body,
        "from_name": "IT Security",
        "from_email": "alerts@company-helpdesk.example",
        "reply_to": "verify-account@outlook-security.example",
        "return_path": "bounce@company-helpdesk.example",
    }


def _build_fraud(base: dict) -> dict:
    body = _clip_text(base["body_base"], 95)
    body += (
        "\n\nInvoice INV-94821 is awaiting payment confirmation. "
        "Review billing details here: https://billing-support-center.example/pay"
    )
    return {
        "attack_type": "financial_fraud",
        "label": 1,
        "subject": "Invoice payment pending - action today",
        "body": body,
        "from_name": "Vendor Billing Desk",
        "from_email": "billing@vendor-payments.example",
        "reply_to": "accounts@vendor-payments.example",
        "return_path": "noreply@vendor-payments.example",
    }


def _build_spoofing(base: dict) -> dict:
    body = _clip_text(base["body_base"], 90)
    body += (
        "\n\nMicrosoft security alert: unusual sign-in activity detected. "
        "Confirm identity here: https://microsoft-security-check.example/verify"
    )
    return {
        "attack_type": "spoofing",
        "label": 1,
        "subject": "Microsoft account security alert",
        "body": body,
        "from_name": "Microsoft Security",
        "from_email": "alerts@micr0soft-protect.example",
        "reply_to": "case-review@msverify-center.example",
        "return_path": "bounce@acct-defense.example",
    }


def _build_spam(base: dict) -> dict:
    body = _clip_text(base["body_base"], 85)
    body += (
        "\n\nLimited promotional offer available now. "
        "Claim your package: https://offers-clearance-lab.example/promo"
    )
    return {
        "attack_type": "spam",
        "label": 1,
        "subject": "Final reminder: promotional offer ends tonight",
        "body": body,
        "from_name": "Promo Rewards",
        "from_email": "offers@reward-center.example",
        "reply_to": "support@reward-center.example",
        "return_path": "bounce@mailer.reward-center.example",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real-data edited drafts from multiple source datasets"
    )
    parser.add_argument("--source-root", default="dataset/source")
    parser.add_argument("--out-dir", default="results/experiment_1/multi_real_edits")
    parser.add_argument("--emails-per-category", type=int, default=10)
    parser.add_argument("--max-rows-per-file", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    all_candidates = collect_candidates(source_root, args.max_rows_per_file, args.seed)
    if not all_candidates:
        raise RuntimeError("No usable text candidates found in source datasets")

    legit_pool = [c for c in all_candidates if c["source_group"] == "enron"]
    phish_pool = [c for c in all_candidates if c["source_group"] == "phishing"]
    fraud_pool = [c for c in all_candidates if c["source_group"] == "fraud_email"]

    if not legit_pool:
        raise RuntimeError("No legit pool found (expected dataset/source/enron)")
    if not phish_pool:
        raise RuntimeError("No phishing pool found (expected dataset/source/phishing)")
    if not fraud_pool:
        raise RuntimeError(
            "No financial fraud pool found (expected dataset/source/fraud_email)"
        )

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "emails_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        ("legit", _build_legit, legit_pool),
        ("phishing", _build_phishing, phish_pool),
        ("financial_fraud", _build_fraud, fraud_pool),
        ("spoofing", _build_spoofing, legit_pool),
        ("spam", _build_spam, phish_pool),
    ]

    rows: list[dict] = []
    row_id = 1
    for attack_type, fn, pool in builders:
        for _ in range(args.emails_per_category):
            base = rng.choice(pool)
            row = fn(base)
            row["id"] = row_id
            row["template_id"] = f"{attack_type}_{row_id:04d}"
            row["source"] = "exp1_multi_real_edit"
            row["source_group_origin"] = base["source_group"]
            row["source_file_origin"] = base["source_file"]
            row["text"] = f"{row['subject']}\n\n{row['body']}"

            txt_path = txt_dir / f"{row['template_id']}.txt"
            txt_path.write_text(_format_txt(row), encoding="utf-8")
            row["txt_path"] = str(txt_path)

            rows.append(row)
            row_id += 1

    out_csv = out_dir / "multi_real_edited_templates.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "template_id",
                "attack_type",
                "label",
                "source",
                "source_group_origin",
                "source_file_origin",
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

    print(f"Saved multi-source edited CSV: {out_csv}")
    print(f"Saved editable TXT files: {txt_dir}")
    print(f"Saved GoPhish seed CSV: {seed_csv}")


if __name__ == "__main__":
    main()
