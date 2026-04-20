from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path


def _sanitize_filename(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text or ""))
    return safe.strip("._-") or "message"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_message(row: dict[str, str], domain: str) -> EmailMessage:
    row_id = str(row.get("id", "")).strip() or "0"
    subject = str(row.get("subject", "")).strip() or f"seed message {row_id}"
    sender = str(row.get("sender", "")).strip() or "security-lab@example.com"
    receiver = str(row.get("receiver", "")).strip() or f"user{row_id}@local.lab"
    body_text = (
        str(row.get("body_text", "")).strip()
        or str(row.get("body", "")).strip()
        or str(row.get("text", "")).strip()
        or "(empty body)"
    )
    body_html = str(row.get("body_html", "")).strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    msg["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    msg["Message-ID"] = f"<seed-{row_id}@{domain}>"
    msg.set_content(body_text)
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Export campaign seed CSV rows to local .eml files")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--message-id-domain", default="local.seed")
    parser.add_argument(
        "--mapping-csv",
        default="",
        help="Optional CSV with file -> y_true_multiclass mapping for local evaluation",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(input_csv)
    mapping_rows: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("id", "")).strip() or "0"
        subject = str(row.get("subject", "")).strip()
        filename = f"seed_{int(row_id):04d}_{_sanitize_filename(subject)[:80]}.eml"
        msg = _build_message(row, domain=str(args.message_id_domain))
        file_path = output_dir / filename
        file_path.write_bytes(msg.as_bytes())
        mapping_rows.append(
            {
                "file": str(file_path).replace("/", "\\"),
                "dataset_id": row_id,
                "y_true_multiclass": str(row.get("attack_type", "")).strip(),
            }
        )

    if args.mapping_csv:
        mapping_path = Path(args.mapping_csv)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "dataset_id", "y_true_multiclass"])
            writer.writeheader()
            writer.writerows(mapping_rows)
        print(f"Saved mapping CSV to: {mapping_path}")

    print(f"Exported {len(rows)} seed emails to: {output_dir}")


if __name__ == "__main__":
    main()
