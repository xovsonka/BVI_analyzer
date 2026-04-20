import argparse
import csv
import html
import json
import os
import re
from typing import Any
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "scenario_m_anti_template_feature_regularized"
    / "test.csv"
)
DEFAULT_MAP = PROJECT_ROOT / "dataset" / "processed" / "gophish_seed_map.csv"


def api_request(
    base_url: str, api_key: str, method: str, endpoint: str, payload: dict | None = None
) -> Any:
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
                    "body": row.get("body", "").strip(),
                    "body_text": row.get("body_text", "").strip(),
                    "body_html": row.get("body_html", "").strip(),
                    "label": label,
                    "source": row.get("source", "unknown"),
                    "subject": row.get("subject", "").strip(),
                    "sender": row.get(
                        "sender", row.get("from_email", row.get("from", ""))
                    ).strip(),
                    "receiver": row.get(
                        "receiver", row.get("to", row.get("to_email", ""))
                    ).strip(),
                    "attack_type": row.get("attack_type", "").strip(),
                }
            )
            if len(rows) >= limit:
                break
    return rows


HEADER_LINE_RE = re.compile(r"^(subject|from|reply-to|return-path)\s*:\s*", re.I)


def normalize_body_for_template(text: str, subject: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    if lines and subject and lines[0].strip().lower() == subject.strip().lower():
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]

    cleaned = []
    skip_headers = True
    for line in lines:
        if skip_headers and HEADER_LINE_RE.match(line.strip()):
            continue
        if skip_headers and line.strip() == "":
            if cleaned:
                cleaned.append("")
            continue
        skip_headers = False
        cleaned.append(line)

    out = "\n".join(cleaned).strip()
    return out


def sanitize_filename(text: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    clean = clean.strip("._-")
    return clean or "unknown"


def extract_email(value: str) -> str:
    name, addr = parseaddr(value or "")
    return (addr or "").strip().lower()


def is_valid_email(addr: str) -> bool:
    if not addr or "@" not in addr:
        return False
    local, domain = addr.rsplit("@", 1)
    return bool(local.strip()) and bool(domain.strip()) and "." in domain


def normalize_from_address(raw: str, default_from: str) -> str:
    name, addr = parseaddr(raw or "")
    addr = (addr or "").strip()
    if is_valid_email(addr):
        return f"{name} <{addr}>" if name else addr
    return default_from


def smtp_from_address(value: str, fallback: str = "security-lab@example.com") -> str:
    _, addr = parseaddr(value or "")
    addr = (addr or "").strip().lower()
    if is_valid_email(addr):
        return addr
    _, fb_addr = parseaddr(fallback or "")
    fb_addr = (fb_addr or "").strip().lower()
    if is_valid_email(fb_addr):
        return fb_addr
    return "security-lab@example.com"


def split_name(value: str) -> tuple[str, str]:
    name, addr = parseaddr(value or "")
    if name:
        parts = [p for p in name.strip().split() if p]
        if len(parts) >= 2:
            return parts[0], " ".join(parts[1:])
        if len(parts) == 1:
            return parts[0], "User"
    if addr and "@" in addr:
        local = addr.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
        parts = [p.capitalize() for p in local.split() if p]
        if len(parts) >= 2:
            return parts[0], " ".join(parts[1:])
        if len(parts) == 1:
            return parts[0], "User"
    return "Test", "User"


def ensure_list_response(value: Any, endpoint: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RuntimeError(f"Unexpected response for {endpoint}: expected list")
    return [item for item in value if isinstance(item, dict)]


def ensure_dict_response(value: Any, endpoint: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected response for {endpoint}: expected object")
    return value


def find_by_name(items: list[dict[str, Any]], name: str):
    for item in items:
        if item.get("name") == name:
            return item
    return None


def get_smtp_profile(base_url: str, api_key: str, profile_name: str):
    profiles = ensure_list_response(
        api_request(base_url, api_key, "GET", "/api/smtp/"), "/api/smtp/"
    )
    profile = find_by_name(profiles, profile_name)
    if not profile:
        raise RuntimeError(f"SMTP profile '{profile_name}' not found in GoPhish")
    return profile


def ensure_smtp_profile(
    base_url: str,
    api_key: str,
    profile_name: str,
    host: str,
    from_address: str,
    username: str = "",
    password: str = "",
    ignore_cert_errors: bool = False,
):
    profiles = ensure_list_response(
        api_request(base_url, api_key, "GET", "/api/smtp/"), "/api/smtp/"
    )
    profile = find_by_name(profiles, profile_name)
    if profile:
        return profile

    payload = {
        "name": profile_name,
        "interface_type": "SMTP",
        "host": host,
        "username": username,
        "password": password,
        "from_address": smtp_from_address(from_address),
        "ignore_cert_errors": ignore_cert_errors,
    }
    try:
        return ensure_dict_response(
            api_request(base_url, api_key, "POST", "/api/smtp/", payload), "/api/smtp/"
        )
    except RuntimeError as exc:
        if "Invalid SMTP From address" not in str(exc):
            raise
        payload["from_address"] = "security-lab@example.com"
        return ensure_dict_response(
            api_request(base_url, api_key, "POST", "/api/smtp/", payload), "/api/smtp/"
        )


def ensure_page(base_url: str, api_key: str, page_name: str):
    pages = ensure_list_response(
        api_request(base_url, api_key, "GET", "/api/pages/"), "/api/pages/"
    )
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
    return ensure_dict_response(
        api_request(base_url, api_key, "POST", "/api/pages/", payload), "/api/pages/"
    )


def create_group(base_url: str, api_key: str, name: str, email: str):
    first_name, last_name = split_name(email)
    payload = {
        "name": name,
        "targets": [
            {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "position": "Student",
            }
        ],
    }
    return ensure_dict_response(
        api_request(base_url, api_key, "POST", "/api/groups/", payload), "/api/groups/"
    )


def create_template(
    base_url: str,
    api_key: str,
    name: str,
    subject: str,
    body_text: str,
    body_html: str = "",
):
    if body_html.strip():
        html_body = body_html
    else:
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
    return ensure_dict_response(
        api_request(base_url, api_key, "POST", "/api/templates/", payload),
        "/api/templates/",
    )


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
    return ensure_dict_response(
        api_request(base_url, api_key, "POST", "/api/campaigns/", payload),
        "/api/campaigns/",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Seed campaigns in GoPhish from processed dataset"
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--only-label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--sending-profile", default="MailHog Local")
    parser.add_argument(
        "--profile-mode",
        choices=["fixed", "by-sender", "by-attack-type"],
        default="fixed",
        help="How SMTP profile is selected/created per row",
    )
    parser.add_argument(
        "--smtp-host",
        default="mailhog:1025",
        help="SMTP host for auto-created profiles",
    )
    parser.add_argument(
        "--default-from",
        default="Security Lab <security-lab@bvi.local>",
        help="Fallback From address for auto-created profiles",
    )
    parser.add_argument("--landing-page", default="BVI Landing")
    parser.add_argument("--campaign-prefix", default="BVI")
    parser.add_argument("--recipient-domain", default="local.lab")
    parser.add_argument("--phish-url", default="http://localhost:8080")
    args = parser.parse_args()

    base_url = os.getenv("GOPHISH_BASE_URL", "http://localhost:3333")
    api_key = os.getenv(
        "GOPHISH_API_KEY",
        "2eb983d31f579f9851b374dac1235cf66dfff9e3c1f9380a24c97caf62502785",
    )
    if not api_key:
        raise RuntimeError("Missing GOPHISH_API_KEY environment variable")

    if not args.input_csv.exists():
        raise RuntimeError(f"Input CSV not found: {args.input_csv}")

    rows = load_rows(args.input_csv, args.rows, args.only_label)
    if not rows:
        raise RuntimeError("No rows selected from input CSV")

    smtp_fixed = None
    if args.profile_mode == "fixed":
        smtp_fixed = get_smtp_profile(base_url, api_key, args.sending_profile)
        if smtp_fixed is None:
            raise RuntimeError(
                f"SMTP profile '{args.sending_profile}' not found in GoPhish"
            )
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
            receiver_hint = row.get("receiver", "")
            receiver_email = extract_email(receiver_hint)
            recipient = (
                receiver_email
                if receiver_email
                else f"user{row_id}@{args.recipient_domain}"
            )

            sender_raw = row.get("sender", "")
            sender_email = extract_email(sender_raw)
            from_address = normalize_from_address(sender_raw, args.default_from)
            smtp = None

            if args.profile_mode == "fixed":
                smtp = smtp_fixed
            elif args.profile_mode == "by-sender":
                profile_key = sender_email or "unknown_sender"
                profile_name = f"Auto Sender - {sanitize_filename(profile_key)}"
                smtp = ensure_smtp_profile(
                    base_url,
                    api_key,
                    profile_name=profile_name,
                    host=args.smtp_host,
                    from_address=from_address,
                )
            else:  # by-attack-type
                attack_type = row.get("attack_type", "")
                if not attack_type:
                    attack_type = "legit" if label == 0 else "malicious"
                profile_name = (
                    f"Auto Attack - {sanitize_filename(str(attack_type).lower())}"
                )
                smtp = ensure_smtp_profile(
                    base_url,
                    api_key,
                    profile_name=profile_name,
                    host=args.smtp_host,
                    from_address=args.default_from,
                )

            if smtp is None:
                raise RuntimeError("Failed to resolve SMTP profile for campaign row")

            group = create_group(
                base_url, api_key, f"{args.campaign_prefix}-G-{suffix}", recipient
            )
            subject = row.get("subject", "").strip() or (
                f"[{args.campaign_prefix}][label:{label}][id:{row_id}] test message"
            )
            body_text = (
                row.get("body_text", "").strip()
                or row.get("body", "").strip()
                or row.get("text", "")
            )
            body_text = normalize_body_for_template(body_text, subject)
            if not body_text:
                body_text = "(empty body)"
            body_html = row.get("body_html", "").strip()
            template = create_template(
                base_url,
                api_key,
                f"{args.campaign_prefix}-T-{suffix}",
                subject,
                body_text,
                body_html,
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
