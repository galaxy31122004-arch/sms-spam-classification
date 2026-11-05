"""Compare the old and new MLP checkpoints on hand-written sample messages."""

from pathlib import Path

import torch

from data.prepare_data import normalize_text
from models.common import load_checkpoint, transform_texts

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {"mlp_old": ("mlp", ROOT / "outputs/mlp"),
               "mlp_new": ("mlp", ROOT / "models/mlp/checkpoint"),
               "trf_demo": ("transformer", ROOT / "models/transformer/demo_checkpoint")}
# (expected label, message); not taken from any training or test split.
SAMPLES = [
    ("spam", "Chase: Your account has been temporarily locked. Verify your identity at chase-secure-login.com/verify"),
    ("spam", "USPS: Your package could not be delivered. Update your address: usps-redelivery.info/track"),
    ("spam", "Your Netflix payment failed. Update billing details within 24h: bit.ly/3nfxupd"),
    ("spam", "HSBC ALERT: Unusual sign-in detected. If this was not you, secure your account: t.me/hsbcsupport"),
    ("spam", "IRS: You are eligible for a tax refund of $1,250. Claim now: irs-refund-gov.com/claim"),
    ("spam", "Congratulations! You won a $500 Amazon gift card. Reply YES to claim your prize"),
    ("spam", "Your PayPal account will be suspended. Confirm your PIN to avoid deactivation"),
    ("spam", "FREE entry to win a new iPhone! Text WIN to 80085 now. 150p/msg"),
    ("ham", "Your Chase card ending 4821 was used for $23.50 at Starbucks. Not you? Call the number on your card."),
    ("ham", "Your OTP is 448213. Do not share it with anyone."),
    ("ham", "Your package was delivered today at 2:14 PM. Thanks for shopping with us."),
    ("ham", "Hey, can you send me the bank details for the rent? I'll transfer tonight."),
    ("ham", "Reminder: your dentist appointment is tomorrow at 10am."),
    ("ham", "Mum says dinner is at 7, don't be late!"),
    ("ham", "Payment of $45.00 to City Water received. Thank you."),
    ("ham", "Are we still meeting for lunch today?"),
]

# Second hand-written set: includes harder cases (ham with links/money/codes, spam without links).
SAMPLES_2 = [
    ("spam", "Wells Fargo: A new device was added to your account. If you did not authorize this, log in at wf-account-review.net"),
    ("spam", "Your Apple ID has been locked for security reasons. Unlock it here: apple-id-support.co/unlock"),
    ("spam", "DHL: Customs fee of $2.99 required for your parcel. Pay now: dhl-parcel-fee.com/pay"),
    ("spam", "Hi mum, I dropped my phone and this is my new number. Can you send me $300 for rent? I'll pay you back"),
    ("spam", "You have been selected for a work-from-home job paying $800/day. Reply JOB for details"),
    ("spam", "Final notice: your car warranty is about to expire. Call 1-888-555-0199 to renew today"),
    ("spam", "Venmo: You received $950.00. Confirm the transfer to release funds: venmo-secure-pay.com/r/88x2"),
    ("spam", "Crypto alert: Bitcoin is about to explode. Invest $250 today and earn $5000 by Friday. Reply INVEST"),
    ("spam", "Your E-ZPass account has an unpaid toll of $4.15. Avoid a late fee: ezpass-tolls.info/pay"),
    ("spam", "WINNER! Your number was drawn for a cash prize of $10,000. Call now to claim before midnight"),
    ("ham", "Your Uber code is 7731. Never share this code."),
    ("ham", "Hi, it's Dr. Patel's office confirming your appointment on Thursday at 3:30pm. Reply C to confirm."),
    ("ham", "Your Amazon order #112-4471 has shipped. Track it in the Amazon app."),
    ("ham", "I paid you back $40 for the concert tickets on Venmo, check when you can"),
    ("ham", "Running 10 min late, traffic is crazy. Order me a coffee pls"),
    ("ham", "Bank of America: Your statement is ready. Sign in to the app to view it."),
    ("ham", "Happy birthday!! Hope you have an amazing day, drinks on me this weekend"),
    ("ham", "Can you pick up milk and bread on your way home?"),
    ("ham", "Here's the link to the meeting notes: docs.google.com/document/d/1abc"),
    ("ham", "Your flight UA 482 to Denver departs at 6:45 AM from gate B12. Check in now in the United app."),
]
SETS = {"1": SAMPLES, "2": SAMPLES_2}


def main() -> None:
    import sys

    global SAMPLES
    SAMPLES = SETS[sys.argv[1] if len(sys.argv) > 1 else "1"]
    torch.set_num_threads(4)
    texts = [normalize_text(message) for _, message in SAMPLES]
    scores = {}
    for name, (kind, directory) in CHECKPOINTS.items():
        model, encoder, config = load_checkpoint(kind, directory)
        with torch.inference_mode():
            features = transform_texts(kind, encoder, texts, config["max_len"])
            scores[name] = torch.sigmoid(model(features)).tolist()
    correct = {name: 0 for name in CHECKPOINTS}
    for index, (expected, message) in enumerate(SAMPLES):
        cells = []
        for name in CHECKPOINTS:
            score = scores[name][index]
            predicted = "spam" if score >= 0.5 else "ham"
            correct[name] += predicted == expected
            cells.append(f"{name}={score:.2f}{'' if predicted == expected else ' X'}")
        print(f"[{expected:4}] {'  '.join(cells):24} {message[:70]}")
    print({name: f"{count}/{len(SAMPLES)}" for name, count in correct.items()})


if __name__ == "__main__":
    main()
