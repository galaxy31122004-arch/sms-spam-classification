"""Route messages through the Tier-1 confidence filter and force rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tier1.rules import RuleMatch, match_force_to_tier2


@dataclass(frozen=True)
class RouteDecision:
    tier: str
    label: str | None
    score: float
    rule: str | None = None
    reason: str | None = None


def load_rules(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    required = {"thresholds", "force_to_tier2", "allowlist"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"Rule file must define: {', '.join(sorted(required))}.")
    return config


def route_message(
    text: str,
    score: float,
    config: dict,
    sender_in_contacts: bool | None = None,
) -> RouteDecision:
    thresholds = config["thresholds"]
    spam_threshold = float(thresholds["spam_confident"])
    ham_threshold = float(thresholds["ham_confident"])
    predicted_label = "spam" if score > 0 else "ham"
    match: RuleMatch | None = match_force_to_tier2(
        text,
        config["force_to_tier2"],
        config["allowlist"].get("domains", []),
        predicted_label,
        sender_in_contacts,
    )
    if match:
        return RouteDecision("tier2", None, score, match.name, match.reason)
    if score > spam_threshold:
        return RouteDecision("tier1", "spam", score, reason="confident_tier1_score")
    if score < ham_threshold:
        return RouteDecision("tier1", "ham", score, reason="confident_tier1_score")
    return RouteDecision("tier2", None, score, reason="tier1_score_uncertain")
