import argparse
import csv
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "dataset" / "processed" / "test.csv"
DEFAULT_MAP = BASE_DIR / "dataset" / "processed" / "gophish_seed_map.csv"


def api_request(
    base_url: str, api_key: str, method: str, endpoint: str, payload: dict | None = None
):
    url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    data = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url=url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"API {method} {endpoint} failed: {exc.code} {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach GoPhish API: {exc}") from exc


def load_rows(csv_path: Path, limit: int, only_label: int | None):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = int(row.get("label", "0"))
            if only_label is not None and label != only_label:
                continue
            rows.append(
                {
                    "id": int(row["id"]),
                    "text": row["text"].strip(),
                    "label": label,
                    "source": row.get("source", "unknown"),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def find_by_name(items: list[dict], name: str):
    for item in items:
        if item.get("name") == name:
            return item
    return None


def get_smtp_profile(base_url: str, api_key: str, profile_name: str):
    profiles = api_request(base_url, api_key, "GET", "/api/smtp/")
    profile = find_by_name(profiles, profile_name)
    if not profile:
        raise RuntimeError(f"SMTP profile '{profile_name}' not found in GoPhish")
    return profile


def ensure_page(base_url: str, api_key: str, page_name: str):
    pages = api_request(base_url, api_key, "GET", "/api/pages/")
    page = find_by_name(pages, page_name)
    if page:
        return page

    payload = {
        "name": page_name,
        "html": "<html><body><h2>BVI test page</h2><p>Thank you.</p></body></html>",
        "capture_credentials": False,
        "capture_passwords": False,
        "redirect_url": "",
    }
    return api_request(base_url, api_key, "POST", "/api/pages/", payload)


def create_group(base_url: str, api_key: str, name: str, email: str):
    payload = {
        "name": name,
        "targets": [
            {
                "first_name": "Test",
                "last_name": "User",
                "email": email,
                "position": "Student",
            }
        ],
    }
    return api_request(base_url, api_key, "POST", "/api/groups/", payload)


def create_template(
    base_url: str, api_key: str, name: str, subject: str, body_text: str
):
    safe_body = html.escape(body_text)
    html_body = (
        "<html><body>"
        f"<p>{safe_body.replace(chr(10), '<br>')}</p>"
        '<p><a href="{{.URL}}">Open link</a></p>'
        "</body></html>"
    )

    payload = {
        "name": name,
        "subject": subject,
        "html": html_body,
        "text": body_text,
    }
    return api_request(base_url, api_key, "POST", "/api/templates/", payload)


def create_campaign(
    base_url: str,
    api_key: str,
    name: str,
    url: str,
    smtp: dict,
    template: dict,
    page: dict,
    group: dict,
):
    payload = {
        "name": name,
        "url": url,
        "launch_date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "template": {"id": template["id"], "name": template["name"]},
        "page": {"id": page["id"], "name": page["name"]},
        "smtp": {"id": smtp["id"], "name": smtp["name"]},
        "groups": [{"id": group["id"], "name": group["name"]}],
    }
    return api_request(base_url, api_key, "POST", "/api/campaigns/", payload)


def main():
    parser = argparse.ArgumentParser(
        description="Seed campaigns in GoPhish from processed dataset"
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--only-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--sending-profile", default="MailHog Local")
    parser.add_argument("--landing-page", default="BVI Landing")
    parser.add_argument("--campaign-prefix", default="BVI")
    parser.add_argument("--recipient-domain", default="local.lab")
    parser.add_argument("--phish-url", default="http://localhost:8080")
    args = parser.parse_args()

    base_url = os.getenv("GOPHISH_BASE_URL", "http://localhost:3333")
    api_key = os.getenv(
        "GOPHISH_API_KEY",
        "feacfa768fb64055c71f3ab5712617b511194ccd952847fb4c8723eca9957731",
    )
    if not api_key:
        raise RuntimeError("Missing GOPHISH_API_KEY environment variable")

    if not args.input_csv.exists():
        raise RuntimeError(f"Input CSV not found: {args.input_csv}")

    rows = load_rows(args.input_csv, args.rows, args.only_label)
    if not rows:
        raise RuntimeError("No rows selected from input CSV")

    smtp = get_smtp_profile(base_url, api_key, args.sending_profile)
    page = ensure_page(base_url, api_key, args.landing_page)

    args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.mapping_csv.exists()

    with args.mapping_csv.open("a", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        if write_header:
            writer.writerow(
                [
                    "dataset_id",
                    "label",
                    "source",
                    "group_id",
                    "template_id",
                    "campaign_id",
                    "recipient_email",
                    "campaign_name",
                ]
            )

        for row in rows:
            row_id = row["id"]
            label = row["label"]
            source = row["source"]
            suffix = f"{row_id}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
            recipient = f"user{row_id}@{args.recipient_domain}"

            group = create_group(
                base_url, api_key, f"{args.campaign_prefix}-G-{suffix}", recipient
            )
            subject = (
                f"[{args.campaign_prefix}][label:{label}][id:{row_id}] test message"
            )
            template = create_template(
                base_url,
                api_key,
                f"{args.campaign_prefix}-T-{suffix}",
                subject,
                row["text"],
            )
            campaign = create_campaign(
                base_url,
                api_key,
                f"{args.campaign_prefix}-C-{suffix}",
                args.phish_url,
                smtp,
                template,
                page,
                group,
            )

            writer.writerow(
                [
                    row_id,
                    label,
                    source,
                    group["id"],
                    template["id"],
                    campaign["id"],
                    recipient,
                    campaign["name"],
                ]
            )
            print(f"Created campaign {campaign['name']} for dataset row {row_id}")

    print(f"\nDone. Mapping saved to: {args.mapping_csv}")


if __name__ == "__main__":
    main()
