"""Candidate normalization: data/prepare_data.normalize_text plus a <CODE> token.

Digit strings that follow an identifier keyword (code, order, card ending, tracking, ...)
become <CODE> before the phone rule runs, so OTPs, order numbers and parcel IDs are no
longer mistaken for phone numbers.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.prepare_data import MONEY_RE, PHONE_RATE_RE, PHONE_RE, URL_RE, replace_phone, replace_url  # noqa: E402

CODE_KEYWORDS = (
    r"code|codes|otp|passcode|pin|confirmation|conf|booking|reservation|order|tracking|package|parcel|"
    r"shipment|ref|reference|ticket|invoice|receipt|account|acct|card|ending|policy|case|member|membership|"
    r"transaction|txn|id"
)
CODE_RE = re.compile(
    rf"(?<![\w<])((?:{CODE_KEYWORDS}))\b"
    r"((?:[\s:#.-]{1,3}(?:no|number|num|is|in|was|id|ending|#)\b){0,3}[\s:#.-]{0,3})"
    r"([a-z]{0,3}-?\d(?:[\d-]| (?=\d)){2,}\d|[a-z]{0,3}\d{4,})(?![\w])",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("å£", "£")
    text = URL_RE.sub(replace_url, text)
    text = PHONE_RATE_RE.sub(" <PHONE> <MONEY> ", text)

    def replace_money(match: re.Match[str]) -> str:
        symbol = match.group(1) or ""
        return f" {symbol}<MONEY> " if symbol else " <MONEY> "

    text = MONEY_RE.sub(replace_money, text)
    text = CODE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)} <CODE> ", text)
    text = PHONE_RE.sub(replace_phone, text)
    text = re.sub(r"<(url|phone|money|code)>", lambda match: match.group().upper(), text)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    for sample in (
        "Your FedEx package 7741 2093 is out for delivery today.",
        "Zelle: Use code 749201 to complete your transfer of $120.00 to David.",
        "Your confirmation number is 9847291.",
        "Wells Fargo: A purchase of $62.40 was made on your card ending 7781.",
        "Your Amazon order #112-4471 has shipped.",
        "Call 09061701461 to claim your prize. Claim code KL341.",
        "Txt WIN to 80085 now",
    ):
        print(normalize_text(sample))
