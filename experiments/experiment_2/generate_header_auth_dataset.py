from __future__ import annotations

import argparse
import csv
from email.message import EmailMessage
from pathlib import Path


def make_message(
    *,
    subject: str,
    from_addr: str,
    to_addr: str,
    reply_to: str,
    return_path: str,
    spf: str,
    dkim: str,
    dmarc: str,
    received_hops: int,
    message_id_domain: str,
    body: str,
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Reply-To"] = reply_to
    msg["Return-Path"] = return_path
    msg["Message-ID"] = f"<exp2-case@{message_id_domain}>"
    msg["Date"] = "Fri, 27 Mar 2026 12:00:00 +0000"
    msg["Authentication-Results"] = (
        f"mx.local; spf={spf} smtp.mailfrom={return_path}; "
        f"dkim={dkim} header.d=example.com; dmarc={dmarc}"
    )
    for idx in range(received_hops):
        msg["Received"] = (
            "from relay"
            f"{idx}.local (relay{idx}.local [10.0.0.{idx + 2}]) "
            "by mx.local with ESMTP id AAA; Fri, 27 Mar 2026 12:00:00 +0000"
        )
    msg.set_content(body)
    return msg


def build_cases() -> list[dict]:
    base_from = "Security Team <security@company.local>"
    base_reply = "security@company.local"
    base_return = "security@company.local"

    return [
        {
            "case_id": "clean_pass",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": base_return,
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "from_reply_mismatch",
            "from": base_from,
            "reply_to": "helpdesk@external-support.example",
            "return_path": base_return,
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 1,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "from_return_path_mismatch",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": "bounce@different-domain.example",
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 1,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "spf_fail",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": base_return,
            "spf": "fail",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 1,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "dkim_fail",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": base_return,
            "spf": "pass",
            "dkim": "fail",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 1,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "dmarc_fail",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": base_return,
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "fail",
            "received_hops": 2,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 1,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "received_anomaly",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": base_return,
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 1,
            "message_id_domain": "company.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 1,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "message_id_domain_mismatch",
            "from": base_from,
            "reply_to": base_reply,
            "return_path": base_return,
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "foreign-id.example",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 1,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "display_name_spoof",
            "from": "Microsoft Security <notify@cheap-host.example>",
            "reply_to": "notify@cheap-host.example",
            "return_path": "notify@cheap-host.example",
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "cheap-host.example",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 1,
            "expected_trusted_domain_only": 0,
        },
        {
            "case_id": "trusted_domain_guardrail",
            "from": "IT Team <alerts@bvi.local>",
            "reply_to": "alerts@bvi.local",
            "return_path": "alerts@bvi.local",
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
            "received_hops": 2,
            "message_id_domain": "bvi.local",
            "expected_from_reply_mismatch": 0,
            "expected_from_return_path_mismatch": 0,
            "expected_spf_fail": 0,
            "expected_dkim_fail": 0,
            "expected_dmarc_fail": 0,
            "expected_received_anomaly": 0,
            "expected_message_id_domain_mismatch": 0,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 1,
        },
        {
            "case_id": "combined_all",
            "from": "Alerts <notify@alerts.example>",
            "reply_to": "support@another.example",
            "return_path": "bounce@third.example",
            "spf": "fail",
            "dkim": "fail",
            "dmarc": "fail",
            "received_hops": 1,
            "message_id_domain": "invalid-id.example",
            "expected_from_reply_mismatch": 1,
            "expected_from_return_path_mismatch": 1,
            "expected_spf_fail": 1,
            "expected_dkim_fail": 1,
            "expected_dmarc_fail": 1,
            "expected_received_anomaly": 1,
            "expected_message_id_domain_mismatch": 1,
            "expected_display_name_spoof_flag": 0,
            "expected_trusted_domain_only": 0,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Exp2 header/auth controlled EML dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/experiment_2/header_auth_eml",
        help="Directory for generated .eml files",
    )
    parser.add_argument(
        "--expected-csv",
        default="dataset/experiment_2/header_auth_expected.csv",
        help="Expected indicator CSV output",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="How many replicas per case",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing .eml files in output directory before generation",
    )
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
            file_name = f"exp2_{counter:03d}_{case['case_id']}.eml"
            to_addr = f"user{counter}@local.lab"
            msg = make_message(
                subject=f"Exp2 case {case['case_id']} #{replica + 1}",
                from_addr=case["from"],
                to_addr=to_addr,
                reply_to=case["reply_to"],
                return_path=case["return_path"],
                spf=case["spf"],
                dkim=case["dkim"],
                dmarc=case["dmarc"],
                received_hops=case["received_hops"],
                message_id_domain=case["message_id_domain"],
                body="Experiment 2 controlled message for header/auth indicator validation.",
            )
            (output_dir / file_name).write_bytes(msg.as_bytes())

            rows.append(
                {
                    "file": str(output_dir / file_name),
                    "case_id": case["case_id"],
                    "expected_from_reply_mismatch": case[
                        "expected_from_reply_mismatch"
                    ],
                    "expected_from_return_path_mismatch": case[
                        "expected_from_return_path_mismatch"
                    ],
                    "expected_spf_fail": case["expected_spf_fail"],
                    "expected_dkim_fail": case["expected_dkim_fail"],
                    "expected_dmarc_fail": case["expected_dmarc_fail"],
                    "expected_received_anomaly": case["expected_received_anomaly"],
                    "expected_message_id_domain_mismatch": case[
                        "expected_message_id_domain_mismatch"
                    ],
                    "expected_display_name_spoof_flag": case[
                        "expected_display_name_spoof_flag"
                    ],
                    "expected_trusted_domain_only": case[
                        "expected_trusted_domain_only"
                    ],
                }
            )
            counter += 1

    with expected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} .eml files in: {output_dir}")
    print(f"Saved expected indicator table: {expected_csv}")


if __name__ == "__main__":
    main()
