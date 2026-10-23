"""Evaluate configured signals that must be sent to the Transformer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>]+|(?<![@\w])(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>]*)?",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\w)\+?\d(?:[\d ().-]{3,}\d)(?!\w)|(?<!\w)\d{5,}(?!\w)")
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = ".,!?;:)\]}\"'"


@dataclass(frozen=True)
class RuleMatch:
    name: str
    reason: str


def extract_domains(text: str) -> list[str]:
    domains: list[str] = []
    for match in URL_RE.finditer(text):
        candidate = match.group(0).rstrip(TRAILING_URL_PUNCTUATION)
        if candidate.lower().startswith("www."):
            candidate = "https://" + candidate
        elif not candidate.lower().startswith(("http://", "https://")):
            candidate = "https://" + candidate
        hostname = urlsplit(candidate).hostname
        if hostname:
            domains.append(hostname.lower().rstrip("."))
    return domains


def contains_phone(text: str) -> bool:
    return PHONE_RE.search(text) is not None


def _domain_matches(domain: str, configured_domains: list[str]) -> bool:
    return any(domain == item.lower().lstrip(".") or domain.endswith("." + item.lower().lstrip(".")) for item in configured_domains)


def _has_keyword(text: str, keywords: list[str]) -> bool:
    words = {word.lower() for word in WORD_RE.findall(text)}
    return any(keyword.lower() in words for keyword in keywords)


def match_force_to_tier2(
    text: str,
    rules: list[dict],
    allowlisted_domains: list[str],
    tier1_label: str,
    sender_in_contacts: bool | None = None,
) -> RuleMatch | None:
    domains = extract_domains(text)
    has_phone = contains_phone(text)

    for rule in rules:
        rule_type = rule.get("type")
        matched = False
        if rule_type == "url_not_allowlisted":
            matched = tier1_label == "ham" and any(
                not _domain_matches(domain, allowlisted_domains) for domain in domains
            )
        elif rule_type == "keyword_with_contact":
            matched = _has_keyword(text, rule.get("keywords", [])) and (has_phone or bool(domains))
        elif rule_type == "shortlink":
            shortlink_domains = rule.get("domains", [])
            matched = any(_domain_matches(domain, shortlink_domains) for domain in domains)
        elif rule_type == "unknown_sender_phone":
            matched = has_phone and sender_in_contacts is False
        else:
            raise ValueError(f"Unsupported force_to_tier2 rule type: {rule_type!r}")

        if matched:
            return RuleMatch(name=rule["name"], reason=rule["reason"])
    return None
