from __future__ import annotations

import argparse
import csv
from email.message import EmailMessage
from pathlib import Path


def _make_message(
    *,
    subject: str,
    from_addr: str,
    to_addr: str,
    text_body: str,
    message_id: str,
    html_body: str | None = None,
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Reply-To"] = from_addr.split("<")[-1].strip(">")
    msg["Return-Path"] = from_addr.split("<")[-1].strip(">")
    msg["Message-ID"] = message_id
    msg["Date"] = "Sat, 28 Mar 2026 12:00:00 +0000"
    msg["Authentication-Results"] = (
        "mx.local; spf=pass smtp.mailfrom=bvi.local; "
        "dkim=pass header.d=bvi.local; dmarc=pass"
    )
    msg["Received"] = (
        "from relay.local (relay.local [10.0.0.2]) "
        "by mx.local with ESMTP id EXP3; Sat, 28 Mar 2026 12:00:00 +0000"
    )
    msg["Received"] = (
        "from app.local (app.local [10.0.0.3]) "
        "by relay.local with ESMTP id EXP3B; Sat, 28 Mar 2026 11:59:59 +0000"
    )

    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    return msg


def build_cases() -> list[dict]:
    return [
        {
            "case_id": "clean_url",
            "from_addr": "IT Team <alerts@bvi.local>",
            "text_body": (
                "Routine update. Review policy at "
                "https://portal.bvi.local/security/policy"
            ),
            "html_body": None,
            "expected_ip_url": 0,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "legit",
            "expected_reasons": "",
        },
        {
            "case_id": "ip_url",
            "from_addr": "Security Desk <security@bvi.local>",
            "text_body": "Verify mailbox access at http://185.10.10.5/session/check",
            "html_body": None,
            "expected_ip_url": 1,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "suspicious",
            "expected_reasons": "ip_url",
        },
        {
            "case_id": "shortener_url",
            "from_addr": "Service Desk <support@bvi.local>",
            "text_body": "Open verification link: https://bit.ly/3J4zM9k",
            "html_body": None,
            "expected_ip_url": 0,
            "expected_shortener_url": 1,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "suspicious",
            "expected_reasons": "shortener_url",
        },
        {
            "case_id": "anchor_mismatch",
            "from_addr": "Update Center <updates@bvi.local>",
            "text_body": "HTML version contains a login link.",
            "html_body": (
                "<html><body><p>Security notice:</p>"
                '<a href="https://account-recovery-help.example/login">'
                "https://microsoft.com/security"
                "</a></body></html>"
            ),
            "expected_ip_url": 0,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 1,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "suspicious",
            "expected_reasons": "anchor_mismatch",
        },
        {
            "case_id": "suspicious_tld",
            "from_addr": "Compliance <compliance@bvi.local>",
            "text_body": "Policy action required: http://records-archive.click/review",
            "html_body": None,
            "expected_ip_url": 0,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 1,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "suspicious",
            "expected_reasons": "suspicious_tld",
        },
        {
            "case_id": "brand_typosquat",
            "from_addr": "Account Team <notice@bvi.local>",
            "text_body": "Sign in at https://micros0ft.com/auth",
            "html_body": None,
            "expected_ip_url": 0,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 1,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "suspicious",
            "expected_reasons": "brand_typosquat",
        },
        {
            "case_id": "obfuscated_url",
            "from_addr": "Portal Team <portal@bvi.local>",
            "text_body": (
                "Continue with secure link "
                "http://login.company.com@safe-portal.com/verify?token=%7Babc%7D"
            ),
            "html_body": None,
            "expected_ip_url": 0,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 1,
            "expected_binary_label": "suspicious",
            "expected_reasons": "obfuscated_url",
        },
        {
            "case_id": "combined_url_signals",
            "from_addr": "Notice Team <notice@bvi.local>",
            "text_body": (
                "Verification links: http://172.16.10.25/signin and "
                "https://bit.ly/3J4zM9k and http://payroll-portal.click/login"
            ),
            "html_body": (
                "<html><body>"
                '<a href="http://safe-portal.com@verify-workflow.com/auth">'
                "https://google.com"
                "</a></body></html>"
            ),
            "expected_ip_url": 1,
            "expected_shortener_url": 1,
            "expected_anchor_mismatch": 1,
            "expected_suspicious_tld": 1,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 1,
            "expected_binary_label": "suspicious",
            "expected_reasons": (
                "ip_url|shortener_url|anchor_mismatch|suspicious_tld|obfuscated_url"
            ),
        },
        {
            "case_id": "borderline_legit_url",
            "from_addr": "Security Office <security@bvi.local>",
            "text_body": (
                "Read external advisory: "
                "https://github.com/advisories/GHSA-xxxx-xxxx?ref=security-bulletin"
            ),
            "html_body": None,
            "expected_ip_url": 0,
            "expected_shortener_url": 0,
            "expected_anchor_mismatch": 0,
            "expected_suspicious_tld": 0,
            "expected_brand_typosquat": 0,
            "expected_obfuscated_url": 0,
            "expected_binary_label": "legit",
            "expected_reasons": "",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Exp3 controlled URL indicator EML dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/experiment_3/url_eml",
        help="Directory for generated .eml files",
    )
    parser.add_argument(
        "--expected-csv",
        default="dataset/experiment_3/url_expected.csv",
        help="Expected URL indicator table",
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for eml_file in output_dir.glob("*.eml"):
            eml_file.unlink()

    expected_csv = Path(args.expected_csv)
    expected_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    counter = 1
    for case in build_cases():
        for replica in range(args.replicates):
            file_name = f"exp3_{counter:03d}_{case['case_id']}.eml"
            msg = _make_message(
                subject=f"Exp3 URL case {case['case_id']} #{replica + 1}",
                from_addr=case["from_addr"],
                to_addr=f"user{counter}@local.lab",
                text_body=case["text_body"],
                message_id=f"<exp3-{counter}-{case['case_id']}@bvi.local>",
                html_body=case.get("html_body"),
            )
            file_path = output_dir / file_name
            file_path.write_bytes(msg.as_bytes())

            rows.append(
                {
                    "file": str(file_path),
                    "case_id": case["case_id"],
                    "expected_ip_url": case["expected_ip_url"],
                    "expected_shortener_url": case["expected_shortener_url"],
                    "expected_anchor_mismatch": case["expected_anchor_mismatch"],
                    "expected_suspicious_tld": case["expected_suspicious_tld"],
                    "expected_brand_typosquat": case["expected_brand_typosquat"],
                    "expected_obfuscated_url": case["expected_obfuscated_url"],
                    "expected_binary_label": case["expected_binary_label"],
                    "expected_reasons": case["expected_reasons"],
                }
            )
            counter += 1

    with expected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} .eml files in: {output_dir}")
    print(f"Saved expected URL indicator table: {expected_csv}")


if __name__ == "__main__":
    main()
