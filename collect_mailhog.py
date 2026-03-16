import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "campaign_eml"
DEFAULT_METADATA = BASE_DIR / "dataset" / "processed" / "mailhog_messages.csv"


def api_get_json(base_url: str, endpoint: str):
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    request = Request(url=url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"MailHog API GET {endpoint} failed: {exc.code} {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach MailHog API: {exc}") from exc


def load_existing_ids(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()

    ids = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.add(row["mailhog_id"])
    return ids


def sanitize_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return safe[:160]


def header_value(headers: dict, key: str) -> str:
    values = headers.get(key, [])
    if not values:
        return ""
    return str(values[0])


def infer_label_from_subject(subject: str):
    match = re.search(r"\[label:(\d+)\]", subject)
    if not match:
        return ""
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="Collect messages from MailHog API to .eml files"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    base_url = os.getenv("MAILHOG_BASE_URL", "http://localhost:8025")

    payload = api_get_json(base_url, f"/api/v2/messages?limit={args.limit}")
    items = payload.get("items", [])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(args.metadata_csv)
    write_header = not args.metadata_csv.exists()

    new_count = 0
    with args.metadata_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "mailhog_id",
                    "message_id",
                    "subject",
                    "from",
                    "to",
                    "created",
                    "label_hint",
                    "eml_path",
                ]
            )

        for item in items:
            mailhog_id = item.get("ID", "")
            if not mailhog_id or mailhog_id in existing_ids:
                continue

            headers = item.get("Content", {}).get("Headers", {})
            raw_data = item.get("Raw", {}).get("Data", "")
            if not raw_data:
                continue

            created = item.get("Created", "")
            subject = header_value(headers, "Subject")
            from_header = header_value(headers, "From")
            to_header = header_value(headers, "To")
            message_id = header_value(headers, "Message-ID")
            label_hint = infer_label_from_subject(subject)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{sanitize_filename(mailhog_id)}.eml"
            eml_path = args.output_dir / filename
            eml_path.write_text(raw_data, encoding="utf-8", errors="ignore")

            writer.writerow(
                [
                    mailhog_id,
                    message_id,
                    subject,
                    from_header,
                    to_header,
                    created,
                    label_hint,
                    eml_path.as_posix(),
                ]
            )
            existing_ids.add(mailhog_id)
            new_count += 1

    print(f"Collected {new_count} new messages")
    print(f"Saved .eml files to: {args.output_dir}")
    print(f"Metadata CSV: {args.metadata_csv}")


if __name__ == "__main__":
    main()
