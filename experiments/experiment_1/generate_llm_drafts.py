from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from urllib import request as urlrequest


# User-provided instructions (kept as-is, now used in LLM call)
INSTRUCTIONS = """You generate realistic email drafts for a controlled cybersecurity research dataset.

Your task is to create emails for evaluation of phishing-detection systems.
The emails must look realistic and varied, but remain safe for offline laboratory use.

Rules:
- Output valid JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
- Keep emails realistic, concise, and believable.
- Preserve the requested class semantics exactly.
- Do not generate malware, attachments, payloads, code, or instructions for bypassing security.
- URLs must be non-functional placeholder-style domains suitable for research.
- Vary tone, structure, wording, and formatting across examples.
- Avoid repetitive sentence patterns.
- For legitimate emails, make them plausible internal or business communication.
- For phishing/spoofing/spam/fraud emails, include realistic social-engineering cues.
- Keep subject and body aligned.
- Return these keys exactly:
subject, body, from_name, from_email, reply_to, return_path, rationale, red_flags
"""

USERPROMPT = """Generate email draft with the following specification:

attack_type: phishing
template_goal: credential harvesting through urgent mailbox verification
target_audience: regular employee
language: English
tone: urgent, credible, slightly formal
brand_style: internal IT/security communication
must_include:
- urgency
- account verification request
- suspicious verification URL
- mild consequence if user does not act
avoid:
- absurd grammar
- too many exclamation marks
- obvious scam phrases
- references to malware or attachments
constraints:
- subject max 12 words
- body 70 to 140 words
- exactly 1 URL
- no attachment mention
- no emojis
- realistic sender and reply-to mismatch
"""

# Additional quality constraints layered on top of user instructions
QUALITY_EXTENSION = """Quality constraints:
- Keep body length substantial and realistic (roughly 90-180 words).
- Avoid boilerplate openings that repeat across outputs.
- Use concrete business context details (department, process, timing, reason).
- Keep each output lexically distinct from previous ones in the same attack_type.
- Keep grammar clean and professional.
"""

STYLE_VARIANTS = [
    "concise corporate",
    "support/helpdesk",
    "operations coordination",
    "vendor communication",
    "compliance reminder",
]

STRUCTURE_VARIANTS = [
    "direct opener + short rationale + CTA",
    "context paragraph + action request + closing",
    "bullet-like procedural flow in prose",
    "incident framing + consequence + CTA",
]


SCENARIOS = [
    {
        "attack_type": "legit",
        "label": 0,
        "template": "Internal meeting invite",
        "goal": "routine internal coordination email",
        "target_audience": "employee",
        "tone": "neutral, professional",
        "must_include": [
            "specific meeting time",
            "office context",
            "normal business phrasing",
        ],
        "avoid": [
            "urgency manipulation",
            "external links",
            "security warning language",
        ],
        "url_policy": "no_url",
        "sender_style": "internal_consistent",
    },
    {
        "attack_type": "phishing",
        "label": 1,
        "template": "Mailbox verification lure",
        "goal": "credential harvesting via urgent mailbox verification",
        "target_audience": "employee",
        "tone": "urgent, credible",
        "must_include": [
            "account verification request",
            "mild threat of suspension",
            "one suspicious login URL",
            "IT/security framing",
        ],
        "avoid": ["cartoonish scam tone", "too many grammar mistakes"],
        "url_policy": "one_suspicious_url",
        "sender_style": "sender_reply_mismatch",
    },
    {
        "attack_type": "financial_fraud",
        "label": 1,
        "template": "Invoice payment pressure",
        "goal": "pressure recipient into paying or reviewing a fake invoice",
        "target_audience": "accounts payable employee",
        "tone": "formal, pressuring",
        "must_include": [
            "invoice reference",
            "payment urgency",
            "financial consequence",
            "one billing URL or payment instruction",
        ],
        "avoid": ["obvious lottery scam wording", "romance scam style"],
        "url_policy": "one_billing_url",
        "sender_style": "vendor_like_external",
    },
    {
        "attack_type": "spoofing",
        "label": 1,
        "template": "Brand impersonation alert",
        "goal": "impersonate trusted brand or provider",
        "target_audience": "consumer or employee",
        "tone": "security alert, authoritative",
        "must_include": [
            "trusted brand reference",
            "account/security concern",
            "urgent confirmation request",
            "domain mismatch cues",
        ],
        "avoid": ["obvious fake slogans", "too much hype"],
        "url_policy": "one_brand_like_suspicious_url",
        "sender_style": "brand_impersonation",
    },
    {
        "attack_type": "spam",
        "label": 1,
        "template": "Promotional pressure",
        "goal": "push user to click promotional offer",
        "target_audience": "general recipient",
        "tone": "pushy marketing",
        "must_include": [
            "offer or reward",
            "time pressure",
            "promotional CTA",
            "one landing-page URL",
        ],
        "avoid": ["explicit phishing login framing", "financial invoice framing"],
        "url_policy": "one_marketing_url",
        "sender_style": "marketing_external",
    },
]


def format_txt_email(row: dict) -> str:
    return (
        f"Subject: {row['subject']}\n"
        f"From: {row['from_name']} <{row['from_email']}>\n"
        f"Reply-To: {row['reply_to']}\n"
        f"Return-Path: {row['return_path']}\n\n"
        f"{row['body']}\n"
    )


def _strip_json_wrappers(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _url_count(text: str) -> int:
    return len(re.findall(r"https?://\S+", text or "", flags=re.IGNORECASE))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{4,}\b", (text or "").lower()))


def _max_jaccard(candidate: str, previous: list[str]) -> float:
    a = _token_set(candidate)
    if not a or not previous:
        return 0.0
    best = 0.0
    for p in previous:
        b = _token_set(p)
        if not b:
            continue
        j = len(a & b) / max(1, len(a | b))
        if j > best:
            best = j
    return best


def _is_valid_for_scenario(
    row: dict,
    scenario: dict,
    min_body_words: int,
    max_body_words: int,
) -> tuple[bool, str]:
    for key in [
        "subject",
        "body",
        "from_name",
        "from_email",
        "reply_to",
        "return_path",
    ]:
        if not str(row.get(key, "")).strip():
            return False, f"missing_{key}"

    body_words = _word_count(row.get("body", ""))
    if body_words < min_body_words or body_words > max_body_words:
        return False, "body_length"

    urls = _url_count((row.get("subject", "") + "\n" + row.get("body", "")).strip())
    if scenario.get("url_policy") == "no_url" and urls != 0:
        return False, "url_policy_no_url"
    if scenario.get("url_policy") != "no_url" and urls != 1:
        return False, "url_policy_one_url"

    return True, "ok"


def llm_generate_email(
    scenario: dict,
    model: str,
    api_key: str,
    timeout: int,
    style_variant: str,
    structure_variant: str,
    min_body_words: int,
    max_body_words: int,
) -> dict:
    prompt = (
        "Generate one email draft with this scenario.\n"
        f"attack_type: {scenario['attack_type']}\n"
        f"template: {scenario['template']}\n"
        f"goal: {scenario['goal']}\n"
        f"target_audience: {scenario['target_audience']}\n"
        f"tone: {scenario['tone']}\n"
        f"must_include: {', '.join(scenario['must_include'])}\n"
        f"avoid: {', '.join(scenario['avoid'])}\n"
        f"url_policy: {scenario['url_policy']}\n"
        f"sender_style: {scenario['sender_style']}\n"
        f"style_variant: {style_variant}\n"
        f"structure_variant: {structure_variant}\n"
        f"constraints: subject <= 12 words, body {min_body_words}-{max_body_words} words.\n"
        "Return only valid JSON with keys: "
        "subject, body, from_name, from_email, reply_to, return_path, rationale, red_flags"
    )

    payload = {
        "model": model,
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "system", "content": QUALITY_EXTENSION},
            {"role": "user", "content": prompt},
        ],
    }

    req = urlrequest.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    content = raw["choices"][0]["message"]["content"]
    obj = json.loads(content)

    return {
        "subject": str(obj.get("subject", "")).strip(),
        "body": str(obj.get("body", "")).strip(),
        "from_name": str(obj.get("from_name", "")).strip(),
        "from_email": str(obj.get("from_email", "")).strip(),
        "reply_to": str(obj.get("reply_to", "")).strip(),
        "return_path": str(obj.get("return_path", "")).strip(),
        "rationale": str(obj.get("rationale", "")).strip(),
        "red_flags": ", ".join(obj.get("red_flags", []))
        if isinstance(obj.get("red_flags"), list)
        else str(obj.get("red_flags", "")).strip(),
    }


def fallback_generate_email(scenario: dict, rnd: random.Random) -> dict:
    if scenario["attack_type"] == "legit":
        openings = [
            "Hello team,",
            "Hi all,",
            "Good morning team,",
            "Dear colleagues,",
        ]
        closings = ["Best regards", "Kind regards", "Thanks", "Sincerely"]
        return {
            "subject": rnd.choice(
                [
                    "Team sync tomorrow at 09:30",
                    "Reminder: weekly coordination meeting tomorrow",
                    "Operations sync - tomorrow 09:30",
                    "Please join tomorrow's team status meeting",
                ]
            ),
            "body": (
                f"{rnd.choice(openings)}\n\n"
                "Reminder that tomorrow we have our team sync at 09:30 in Meeting Room A. "
                "Please prepare your weekly progress updates, open blockers, and dependencies. "
                "We will review current priorities first and then align on next-week actions. "
                "If you are unavailable, please send your notes by 08:45 so they can be included in the agenda.\n\n"
                f"{rnd.choice(closings)},\nOperations Team"
            ),
            "from_name": "Operations Team",
            "from_email": "ops@company.local",
            "reply_to": "ops@company.local",
            "return_path": "ops@company.local",
            "rationale": "baseline legit",
            "red_flags": "",
        }

    url_map = {
        "phishing": [
            "https://company-verify-helpdesk.com/login",
            "http://secure-portal-check.help/auth",
            "https://mailbox-session-restore.net/verify",
        ],
        "financial_fraud": [
            "https://vendor-billing-update.net/pay",
            "https://invoice-confirmation-center.com/payment",
            "http://billing-review-portal.help/invoice",
        ],
        "spoofing": [
            "https://microsoft-account-validate-help.com/security",
            "https://secure-brand-check-center.net/confirm",
            "http://brand-session-alert.help/verify",
        ],
        "spam": [
            "https://promo-offer-center.biz/claim",
            "https://exclusive-deals-hub.net/reward",
            "http://limited-discount-portal.help/offer",
        ],
    }
    subject_map = {
        "phishing": [
            "Action required: mailbox verification needed",
            "Security notice: verify your account today",
            "Urgent: confirm mailbox access",
        ],
        "financial_fraud": [
            "Invoice notice: payment pending",
            "Action required: overdue invoice confirmation",
            "Payment reminder: invoice requires review",
        ],
        "spoofing": [
            "Security alert: unusual sign-in detected",
            "Account warning: verify recent login activity",
            "Identity confirmation required",
        ],
        "spam": [
            "Limited offer: claim your reward",
            "Exclusive promotion expires today",
            "Special package waiting for you",
        ],
    }
    attack = scenario.get("attack_type", "phishing")
    url = rnd.choice(url_map.get(attack, url_map["phishing"]))
    subject = rnd.choice(subject_map.get(attack, subject_map["phishing"]))

    context = {
        "phishing": "We detected a mailbox policy mismatch and your account requires immediate verification to avoid temporary restriction.",
        "financial_fraud": "Our records show a pending invoice item that requires same-day confirmation to avoid late processing penalties.",
        "spoofing": "A security event was flagged for your account and confirmation is required to maintain uninterrupted access.",
        "spam": "You have been selected for a limited promotional package available for a short time window.",
    }

    return {
        "subject": subject,
        "body": (
            "Hello,\n\n"
            f"{context.get(attack, context['phishing'])} "
            "Please complete the required step using the secure portal below. "
            f"Verification link: {url}\n\n"
            "If no action is completed in time, access or processing may be temporarily limited.\n\n"
            "Regards,\nSecurity Desk"
        ),
        "from_name": "Security Desk",
        "from_email": "alerts@external-support.net",
        "reply_to": "verify@protonmail.com",
        "return_path": "bounce@external-support.net",
        "rationale": "fallback",
        "red_flags": "urgency, external link",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate editable draft templates for Experiment 1"
    )
    parser.add_argument("--out-dir", default="results/experiment_1/llm_drafts")
    parser.add_argument("--copies-per-scenario", type=int, default=None)
    parser.add_argument(
        "--emails-per-category",
        type=int,
        default=4,
        help="Number of distinct emails per attack category",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-timeout", type=int, default=60)
    parser.add_argument("--min-body-words", type=int, default=90)
    parser.add_argument("--max-body-words", type=int, default=180)
    parser.add_argument("--max-llm-retries", type=int, default=4)
    parser.add_argument("--max-similarity", type=float, default=0.72)
    args = parser.parse_args()

    per_category = (
        args.copies_per_scenario
        if args.copies_per_scenario is not None
        else args.emails_per_category
    )

    rnd = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "emails_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "templates.csv"
    existing_rows: list[dict] = []
    max_existing_id = 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        for r in existing_rows:
            try:
                max_existing_id = max(
                    max_existing_id, int(str(r.get("id", "0") or "0"))
                )
            except Exception:
                continue

    api_key = os.getenv("OPENAI_API_KEY", "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")
    use_llm = args.use_llm and bool(api_key)
    if args.use_llm and not api_key:
        raise RuntimeError("--use-llm requested but OPENAI_API_KEY is not set")

    rows: list[dict] = []
    previous_by_attack: dict[str, list[str]] = {}
    fallback_count = 0
    idx = max_existing_id + 1
    for sc in SCENARIOS:
        for _ in range(per_category):
            gen = None
            if use_llm:
                for _attempt in range(args.max_llm_retries):
                    style_variant = rnd.choice(STYLE_VARIANTS)
                    structure_variant = rnd.choice(STRUCTURE_VARIANTS)
                    try:
                        candidate = llm_generate_email(
                            sc,
                            args.llm_model,
                            api_key,
                            args.llm_timeout,
                            style_variant,
                            structure_variant,
                            args.min_body_words,
                            args.max_body_words,
                        )
                    except Exception:
                        continue

                    ok, _reason = _is_valid_for_scenario(
                        candidate,
                        sc,
                        args.min_body_words,
                        args.max_body_words,
                    )
                    if not ok:
                        continue

                    prev = previous_by_attack.get(sc["attack_type"], [])
                    sim = _max_jaccard(candidate.get("body", ""), prev)
                    if sim > args.max_similarity:
                        continue
                    gen = candidate
                    break

            if gen is None:
                gen = fallback_generate_email(sc, rnd)
                generation_mode = "fallback"
                fallback_count += 1
            else:
                generation_mode = "llm"

            row = {**sc, **gen}
            row["id"] = idx
            row["source"] = "exp1_llm_draft"
            row["template_id"] = f"{sc['attack_type']}_{idx:04d}"
            row["text"] = f"{row['subject']}\n\n{row['body']}"
            row["generation_mode"] = generation_mode

            previous_by_attack.setdefault(sc["attack_type"], []).append(row["body"])

            txt_path = txt_dir / f"{row['template_id']}.txt"
            txt_path.write_text(format_txt_email(row), encoding="utf-8")
            row["txt_path"] = str(txt_path)
            rows.append(row)
            idx += 1

    fieldnames = [
        "id",
        "template_id",
        "attack_type",
        "template",
        "goal",
        "target_audience",
        "tone",
        "must_include",
        "avoid",
        "url_policy",
        "sender_style",
        "label",
        "source",
        "subject",
        "from_name",
        "from_email",
        "reply_to",
        "return_path",
        "body",
        "text",
        "rationale",
        "red_flags",
        "txt_path",
        "generation_mode",
    ]
    merged_rows = existing_rows + rows
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    json_path = out_dir / "templates.json"
    existing_json: list[dict] = []
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                existing_json = parsed
        except Exception:
            existing_json = []
    json_path.write_text(
        json.dumps(existing_json + rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    seed_csv = out_dir / "gophish_seed_input.csv"
    existing_seed_rows: list[dict] = []
    if seed_csv.exists() and seed_csv.stat().st_size > 0:
        with seed_csv.open("r", encoding="utf-8", newline="") as f:
            existing_seed_rows = list(csv.DictReader(f))

    with seed_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "source"])
        writer.writeheader()
        for r in existing_seed_rows:
            writer.writerow(
                {
                    "id": r.get("id", ""),
                    "text": r.get("text", ""),
                    "label": r.get("label", ""),
                    "source": r.get("source", ""),
                }
            )
        for r in rows:
            writer.writerow(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "label": r["label"],
                    "source": r["source"],
                }
            )

    print(f"Saved templates CSV: {csv_path}")
    print(f"Saved editable TXT drafts: {txt_dir}")
    print(f"Saved GoPhish seed CSV: {seed_csv}")
    print(f"Appended new rows: {len(rows)} (total CSV rows: {len(merged_rows)})")
    print(
        f"LLM generations: {len(rows) - fallback_count}; fallback generations: {fallback_count}"
    )


if __name__ == "__main__":
    main()
