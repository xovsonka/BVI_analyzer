from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


URL_MUTATIONS = [
    "http://192.168.1.1/login",
    "https://bit.ly/security-check",
    "http://paypa1.com/verify",
    "http://microsoft-account-check.help/auth",
]


def mutate_url(body: str, rnd: random.Random) -> str:
    mutated = re.sub(r"https?://\S+", rnd.choice(URL_MUTATIONS), body, count=1)
    if mutated == body:
        mutated += f"\n\nVerify here: {rnd.choice(URL_MUTATIONS)}"
    return mutated


def mutate_header_fields(
    from_email: str, reply_to: str, rnd: random.Random
) -> tuple[str, str]:
    from_candidates = [
        "alerts@secure-company-mail.com",
        "support@account-update-center.net",
        "billing@invoice-helpdesk.org",
    ]
    reply_candidates = [
        "verify-now@protonmail.com",
        "helpdesk-confirm@outlook.com",
        "urgent-check@fastmail.com",
    ]
    if rnd.random() < 0.7:
        return rnd.choice(from_candidates), rnd.choice(reply_candidates)
    return from_email, reply_to


def mutate_text_style(subject: str, body: str, rnd: random.Random) -> tuple[str, str]:
    mode = rnd.choice(["urgent", "upper", "typo"])
    if mode == "urgent":
        return f"URGENT: {subject}", body + "\n\nImmediate action required."
    if mode == "upper":
        return subject.upper(), body
    typo_body = body.replace("account", "acount").replace("verify", "verfy")
    return subject, typo_body


def format_txt(row: dict) -> str:
    return (
        f"Subject: {row['subject']}\n"
        f"From: {row.get('from_name', 'Security Team')} <{row['from_email']}>\n"
        f"Reply-To: {row['reply_to']}\n"
        f"Return-Path: {row['return_path']}\n"
        f"Attack-Type: {row['attack_type']}\n"
        f"Mutation-Type: {row['mutation_type']}\n"
        f"Label: {row['label']}\n"
        "\n"
        f"{row['body']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create URL/header/text mutations")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", default="results/experiment_1/mutated")
    parser.add_argument("--variants-per-template", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rnd = random.Random(args.seed)
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    rows = list(csv.DictReader(input_csv.open("r", encoding="utf-8", newline="")))
    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "emails_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    out_id = 1
    mutation_types = ["url", "header", "text", "combo"]

    for base in rows:
        for i in range(args.variants_per_template):
            mut = mutation_types[i % len(mutation_types)]
            row = dict(base)
            row["id"] = out_id
            row["source"] = "exp1_mutated"
            row["mutation_type"] = mut
            row["template_id"] = f"{base.get('template_id', 'tpl')}_m{out_id:04d}"

            subject = row.get("subject", "")
            body = row.get("body", "")
            from_email = row.get("from_email", "")
            reply_to = row.get("reply_to", "")

            if mut in {"url", "combo"}:
                body = mutate_url(body, rnd)
            if mut in {"header", "combo"}:
                from_email, reply_to = mutate_header_fields(from_email, reply_to, rnd)
            if mut in {"text", "combo"}:
                subject, body = mutate_text_style(subject, body, rnd)

            row["subject"] = subject
            row["body"] = body
            row["from_email"] = from_email
            row["reply_to"] = reply_to
            row["text"] = f"{subject}\n\n{body}"

            txt_path = txt_dir / f"{row['template_id']}.txt"
            txt_path.write_text(format_txt(row), encoding="utf-8")
            row["txt_path"] = str(txt_path)

            out_rows.append(row)
            out_id += 1

    out_csv = out_dir / "mutated_templates.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in out_rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    seed_csv = out_dir / "gophish_seed_input.csv"
    with seed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "source"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "label": r.get("label", 1),
                    "source": r["source"],
                }
            )

    print(f"Saved mutated CSV: {out_csv}")
    print(f"Saved editable TXT files: {txt_dir}")
    print(f"Saved GoPhish seed CSV: {seed_csv}")


if __name__ == "__main__":
    main()
