"""Score hand-written SMS against the saved checkpoints to probe out-of-distribution behaviour.

The benchmark splits only contain UCI-era messages, so these sets are written by hand to cover
message types the training data does not have: modern phishing, transactional notifications,
delivery and booking confirmations. They are a qualitative probe, not a metric: each set is tiny
and written by the report author, so results must not be reported as precision/recall.

Usage: python compare_demo_models.py [1|2|3|4|all] [--json <path>]
Sets 3 and 4 are the 15 spam / 15 ham messages tested on 2026-11-06; see
docs/kiem-thu-tin-nhan-thuc-te.md for the written procedure and findings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.prepare_data import normalize_text
from models.common import load_checkpoint, transform_texts

ROOT = Path(__file__).resolve().parent

# name -> (kind, directory). Missing directories are skipped with a note: all of these except
# the demo checkpoint and outputs/ are git-ignored and only exist after a local training run.
CHECKPOINTS = {
    "TRF-demo": ("transformer", ROOT / "models/transformer/demo_checkpoint"),
    "MLP": ("mlp", ROOT / "eval/results/baseline/mlp"),
    "LSTM": ("lstm", ROOT / "eval/results/baseline/lstm"),
    "TRF-aug": ("transformer", ROOT / "models/transformer/checkpoint"),
    "MLP-colab": ("mlp", ROOT / "outputs/mlp"),
}
DEFAULT_CHECKPOINTS = ("TRF-demo", "MLP", "LSTM", "TRF-aug")

# (expected label, short tag, message). No message is taken from any split.
SET_1 = [
    ("spam", "Chase khoa tai khoan", "Chase: Your account has been temporarily locked. Verify your identity at chase-secure-login.com/verify"),
    ("spam", "USPS giao hang", "USPS: Your package could not be delivered. Update your address: usps-redelivery.info/track"),
    ("spam", "Netflix thanh toan", "Your Netflix payment failed. Update billing details within 24h: bit.ly/3nfxupd"),
    ("spam", "HSBC dang nhap la", "HSBC ALERT: Unusual sign-in detected. If this was not you, secure your account: t.me/hsbcsupport"),
    ("spam", "IRS hoan thue", "IRS: You are eligible for a tax refund of $1,250. Claim now: irs-refund-gov.com/claim"),
    ("spam", "Amazon gift card", "Congratulations! You won a $500 Amazon gift card. Reply YES to claim your prize"),
    ("spam", "PayPal doi PIN", "Your PayPal account will be suspended. Confirm your PIN to avoid deactivation"),
    ("spam", "iPhone mien phi", "FREE entry to win a new iPhone! Text WIN to 80085 now. 150p/msg"),
    ("ham", "Giao dich the Chase", "Your Chase card ending 4821 was used for $23.50 at Starbucks. Not you? Call the number on your card."),
    ("ham", "OTP", "Your OTP is 448213. Do not share it with anyone."),
    ("ham", "Da giao hang", "Your package was delivered today at 2:14 PM. Thanks for shopping with us."),
    ("ham", "Hoi so tai khoan", "Hey, can you send me the bank details for the rent? I'll transfer tonight."),
    ("ham", "Lich hen nha si", "Reminder: your dentist appointment is tomorrow at 10am."),
    ("ham", "Me nhan tin", "Mum says dinner is at 7, don't be late!"),
    ("ham", "Thanh toan tien nuoc", "Payment of $45.00 to City Water received. Thank you."),
    ("ham", "Hen an trua", "Are we still meeting for lunch today?"),
]

SET_2 = [
    ("spam", "Wells Fargo thiet bi moi", "Wells Fargo: A new device was added to your account. If you did not authorize this, log in at wf-account-review.net"),
    ("spam", "Apple ID bi khoa", "Your Apple ID has been locked for security reasons. Unlock it here: apple-id-support.co/unlock"),
    ("spam", "DHL phi hai quan", "DHL: Customs fee of $2.99 required for your parcel. Pay now: dhl-parcel-fee.com/pay"),
    ("spam", "Gia danh nguoi than", "Hi mum, I dropped my phone and this is my new number. Can you send me $300 for rent? I'll pay you back"),
    ("spam", "Viec tai nha", "You have been selected for a work-from-home job paying $800/day. Reply JOB for details"),
    ("spam", "Bao hiem xe", "Final notice: your car warranty is about to expire. Call 1-888-555-0199 to renew today"),
    ("spam", "Venmo nhan tien", "Venmo: You received $950.00. Confirm the transfer to release funds: venmo-secure-pay.com/r/88x2"),
    ("spam", "Dau tu Bitcoin", "Crypto alert: Bitcoin is about to explode. Invest $250 today and earn $5000 by Friday. Reply INVEST"),
    ("spam", "Phi duong bo", "Your E-ZPass account has an unpaid toll of $4.15. Avoid a late fee: ezpass-tolls.info/pay"),
    ("spam", "Trung thuong tien mat", "WINNER! Your number was drawn for a cash prize of $10,000. Call now to claim before midnight"),
    ("ham", "Ma Uber", "Your Uber code is 7731. Never share this code."),
    ("ham", "Phong kham xac nhan", "Hi, it's Dr. Patel's office confirming your appointment on Thursday at 3:30pm. Reply C to confirm."),
    ("ham", "Amazon da gui hang", "Your Amazon order #112-4471 has shipped. Track it in the Amazon app."),
    ("ham", "Tra tien ve", "I paid you back $40 for the concert tickets on Venmo, check when you can"),
    ("ham", "Ket xe den muon", "Running 10 min late, traffic is crazy. Order me a coffee pls"),
    ("ham", "Sao ke ngan hang", "Bank of America: Your statement is ready. Sign in to the app to view it."),
    ("ham", "Chuc mung sinh nhat", "Happy birthday!! Hope you have an amazing day, drinks on me this weekend"),
    ("ham", "Mua do an", "Can you pick up milk and bread on your way home?"),
    ("ham", "Link bien ban hop", "Here's the link to the meeting notes: docs.google.com/document/d/1abc"),
    ("ham", "Chuyen bay United", "Your flight UA 482 to Denver departs at 6:45 AM from gate B12. Check in now in the United app."),
]

# Set 3: 15 modern spam types supplied by the user on 2026-11-06.
SET_3 = [
    ("spam", "Coca-Cola trung thuong", "Urgent! Your email ID has selected as the winner of £500,000 in the Coca-Cola National Promo. To claim your cash prize, email: coca-claims@xxx.com"),
    ("spam", "Dau tu / viec nhe luong cao", "Earn $500 a day passively by testing website security. No skills required, guaranteed payouts via PayPal every Friday. Register today: http://earn-easy-cash.xxx"),
    ("spam", "Apple iCloud canh bao", "Apple Support: Unauthorized login attempt detected on your iCloud account from Russia. If this wasn't you, reset your password immediately: http://icloud-secure-signon.xxx"),
    ("spam", "Thuoc giam can", "Get 80% off on premium vitamins and weight loss pills. No prescription needed. Worldwide shipping in discrete packaging. Order now: http://health-pharmacy.xxx"),
    ("spam", "Bat dong san", "Exclusive pre-launch offer! Luxury apartments in downtown starting from just $150k. Guaranteed 8% rental yield. Reply 'STOP' to opt out or click: http://city-condos.xxx"),
    ("spam", "Thu hoi no", "FINAL NOTICE: Your outstanding balance of $1,240.50 has been passed to a debt collection agency. Pay immediately to avoid legal action: http://legal-settlement.xxx"),
    ("spam", "Mao danh ban be", "Hey, it's been a while! I saw your photos on this site, are you serious? Look at what they posted about you: gossip-trends.xxx"),
    ("spam", "Costco gia han the", "Your Costco membership auto-renewal failed. Update your billing address within 24 hours to keep your rewards points active: http://costco-member-login.xxx"),
    ("spam", "Khao sat tang voucher", "Take our 2-minute survey about your recent shopping experience and get a free $50 Starbucks voucher! Start here: http://survey-star-rewards.xxx"),
    ("spam", "Dich vu SEO", "Hi there, we can rank your website on the first page of Google in less than 30 days! Reply for a free SEO audit report of your domain."),
    ("spam", "Walmart gift card", "CONGRATULATIONS! Your phone number won a $1,000 Walmart Gift Card! Click here to claim your prize now: http://walmart-winner.xxx"),
    ("spam", "Bitcoin x2", "Double your Bitcoin overnight! We are giving away 50 BTC to random users. Deposit 0.01 BTC to verify your wallet and get 0.1 BTC back instantly: bit.ly/crypto-gift"),
    ("spam", "Chase khoa tai khoan", "ALERT: Your Chase Bank account has been temporarily locked due to suspicious login attempts. Verify your identity immediately at: http://chase-security-update.xxx"),
    ("spam", "Vay tien nhanh", "Need quick cash? You are pre-approved for a personal loan up to $5,000 with no credit check! Reply 'YES' or call 1-800-XXX-XXXX to apply"),
    ("spam", "Amazon tuyen nhap lieu", "Amazon is hiring remote data entry clerks! Work 2 hours a day from home and earn $300/day. No experience needed. Contact us on WhatsApp: wa.me/xxxxxxx"),
]

# Set 4: 15 legitimate counterparts written for the same session, deliberately including
# transactional messages that carry links, money, codes and brand names.
SET_4 = [
    ("ham", "OTP ngan hang", "Your verification code is 582914. It expires in 10 minutes. Do not share it with anyone."),
    ("ham", "FedEx giao hang", "Your FedEx package 7741 2093 is out for delivery today between 2pm and 5pm."),
    ("ham", "Giao dich the that", "Wells Fargo: A purchase of $62.40 was made on your card ending 7781 at Trader Joe's."),
    ("ham", "Lich hen bac si", "Reminder: your appointment with Dr. Nguyen is on Thursday at 9:15 AM. Reply C to confirm."),
    ("ham", "Ban be den muon", "I'm running late, stuck at the office. Start without me, I'll be there by 8."),
    ("ham", "Link tai lieu cong viec", "Here's the slide deck for tomorrow: drive.google.com/file/d/1k2x9 - can you review slide 4?"),
    ("ham", "Chuyen tien ban be", "Sent you $25 for the pizza last night on Zelle, thanks again!"),
    ("ham", "Hoa don dien", "Your electricity bill of $84.20 is due on 2026-11-20. View it in the app."),
    ("ham", "Chuyen bay tre", "Delta flight DL287 to Seattle is delayed 40 minutes. New departure 7:25 PM, gate C14."),
    ("ham", "Truong nghi hoc", "School closed tomorrow due to heavy rain. Classes resume Monday."),
    ("ham", "Nguoi than nhan tin", "Grandma called, she says happy birthday and wants you to call her back tonight."),
    ("ham", "Dat ban nha hang", "Your table for 4 at Hoa Sen is confirmed for Saturday 7:00 PM. Reply CANCEL to cancel."),
    ("ham", "Canh bao dang nhap that", "Chase: We detected a sign-in from a new device. If this was you, no action is needed."),
    ("ham", "Dong nghiep hoi bai", "Can you send me the latest version of the report before the meeting?"),
    ("ham", "Amazon giao hang", "Your Amazon order of 2 items has shipped and will arrive Tuesday. Track it in the app."),
]

SETS = {"1": SET_1, "2": SET_2, "3": SET_3, "4": SET_4}


def load_models(names):
    models = {}
    for name in names:
        kind, directory = CHECKPOINTS[name]
        if not (directory / "config.json").is_file():
            print(f"[bo qua] {name}: khong co checkpoint tai {directory.relative_to(ROOT)}")
            continue
        models[name] = (kind, *load_checkpoint(kind, directory))
    if not models:
        raise SystemExit("Khong nap duoc checkpoint nao; train truoc (xem README).")
    return models


def score(models, samples):
    texts = [normalize_text(message) for _, _, message in samples]
    scores = {}
    for name, (kind, model, encoder, config) in models.items():
        with torch.inference_mode():
            features = transform_texts(kind, encoder, texts, config["max_len"])
            scores[name] = torch.sigmoid(model(features)).tolist()
    return texts, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", nargs="?", default="1", choices=[*SETS, "all"])
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS), choices=list(CHECKPOINTS))
    parser.add_argument("--json", type=Path, help="luu diem tung tin ra JSON")
    args = parser.parse_args()

    torch.set_num_threads(4)
    models = load_models(args.checkpoints)
    chosen = list(SETS) if args.set == "all" else [args.set]
    report = {}
    for key in chosen:
        samples = SETS[key]
        texts, scores = score(models, samples)
        print(f"\n=== Bo {key}: {len(samples)} tin "
              f"({sum(label == 'spam' for label, _, _ in samples)} spam / "
              f"{sum(label == 'ham' for label, _, _ in samples)} ham) ===")
        print(f"{'#':>2} {'nhan':<5} {'tin':<28} " + " ".join(f"{name:>9}" for name in models))
        right = {name: 0 for name in models}
        rows = []
        for index, (label, tag, message) in enumerate(samples):
            cells = []
            row = {"label": label, "tag": tag, "message": message, "normalized": texts[index], "scores": {}}
            for name in models:
                value = scores[name][index]
                predicted = "spam" if value >= 0.5 else "ham"
                right[name] += predicted == label
                row["scores"][name] = round(value, 4)
                cells.append(f"{value:8.2f}{' ' if predicted == label else 'X'}")
            rows.append(row)
            print(f"{index + 1:>2} {label:<5} {tag:<28} " + " ".join(cells))
        print("dung: " + ", ".join(f"{name} {count}/{len(samples)}" for name, count in right.items()))
        report[key] = {"correct": right, "total": len(samples), "rows": rows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDa luu {args.json}")


if __name__ == "__main__":
    main()
