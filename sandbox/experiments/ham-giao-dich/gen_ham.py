"""Generate synthetic transactional HAM messages for training (never for evaluation).

Output: data/ham_extra.csv (label, raw_text, text). Messages near-duplicate (difflib >= 0.8)
to any evaluation message - val/test splits and the hand-written sets - are dropped.
"""

from __future__ import annotations

import csv
import difflib
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from normalize_v2 import normalize_text  # noqa: E402
from compare_demo_models import SETS  # noqa: E402
from manual_sets import SET_5  # noqa: E402

N_TARGET = 500
SEED = 7

BANKS = ["Chase", "Wells Fargo", "Bank of America", "Citi", "Capital One", "HSBC", "Barclays", "Monzo",
         "Revolut", "US Bank", "PNC", "TD Bank", "Lloyds", "NatWest", "Ally"]
SHOPS = ["Target", "Walmart", "Costco", "Starbucks", "Shell", "Whole Foods", "CVS", "Tesco", "Sainsbury's",
         "Home Depot", "IKEA", "Best Buy", "McDonald's", "Chipotle", "Walgreens"]
CARRIERS = ["UPS", "FedEx", "DHL", "USPS", "Royal Mail", "Evri", "DPD", "Amazon"]
STORES = ["Amazon", "eBay", "Etsy", "Zara", "Nike", "Apple", "ASOS", "Wayfair", "Uniqlo", "Argos"]
APPS = ["Google", "Microsoft", "Apple", "WhatsApp", "Instagram", "PayPal", "Venmo", "Zelle", "Coinbase",
        "Steam", "Discord", "Dropbox", "LinkedIn", "Uber", "Airbnb"]
AIRLINES = [("United", "UA"), ("Delta", "DL"), ("American", "AA"), ("British Airways", "BA"),
            ("Southwest", "WN"), ("easyJet", "U2"), ("Ryanair", "FR"), ("JetBlue", "B6")]
CITIES = ["Boston", "Chicago", "Denver", "London", "Seattle", "Austin", "Miami", "Dublin", "Atlanta", "Phoenix"]
HOTELS = ["Hilton", "Marriott", "Holiday Inn", "Hyatt", "Premier Inn", "Best Western"]
NAMES = ["Sam", "Priya", "Marco", "Aisha", "Ben", "Lena", "Omar", "Grace", "Tom", "Mia", "Kenji", "Rosa"]
CLINICS = ["Dr. Lee's office", "City Dental", "Bright Smile Dental", "Riverside Clinic", "Dr. Okafor",
           "Eye Care Center", "Northside Physio"]
UTILITIES = ["electricity", "gas", "water", "internet", "phone", "council tax"]
PROVIDERS = ["Comcast", "Verizon", "AT&T", "British Gas", "Octopus Energy", "T-Mobile", "Spectrum", "EE"]
SUBS = ["Netflix", "Spotify", "Disney+", "iCloud", "YouTube Premium", "Adobe", "Xbox Game Pass", "Hulu"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "tomorrow", "today"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CUR = ["$", "$", "$", "£", "€"]


def amount(r: random.Random, low=3, high=900) -> str:
    return f"{r.choice(CUR)}{r.uniform(low, high):,.2f}"


def digits(r: random.Random, n: int) -> str:
    return "".join(r.choice("0123456789") for _ in range(n))


def time_(r: random.Random) -> str:
    return f"{r.randint(1, 12)}:{r.choice(['00', '15', '30', '45', '05', '20', '40'])} {r.choice(['AM', 'PM'])}"


def date_(r: random.Random) -> str:
    return f"{r.choice(MONTHS)} {r.randint(1, 28)}"


TEMPLATES = [
    # one-time codes
    lambda r: f"{r.choice(APPS)}: {digits(r, 6)} is your verification code. Don't share it with anyone.",
    lambda r: f"Your {r.choice(APPS)} login code is {digits(r, 6)}. It expires in {r.choice([5, 10, 15])} minutes.",
    lambda r: f"{r.choice(BANKS)}: Use one-time passcode {digits(r, 6)} to approve your payment of {amount(r)}. We will never call to ask for it.",
    lambda r: f"G-{digits(r, 6)} is your Google verification code.",
    lambda r: f"Your security code for {r.choice(STORES)} is {digits(r, 6)}. If you didn't request it, you can ignore this message.",
    lambda r: f"{r.choice(BANKS)}: Code {digits(r, 6)} to confirm the new payee {r.choice(NAMES)}. Valid for 10 min.",
    # card / account alerts
    lambda r: f"{r.choice(BANKS)}: Card ending {digits(r, 4)} was charged {amount(r)} at {r.choice(SHOPS)} on {date_(r)}.",
    lambda r: f"{r.choice(BANKS)} alert: You spent {amount(r)} at {r.choice(SHOPS)}. Available balance {amount(r, 100, 5000)}.",
    lambda r: f"{r.choice(BANKS)}: A deposit of {amount(r, 50, 3000)} was credited to your account ending {digits(r, 4)}.",
    lambda r: f"{r.choice(BANKS)}: Your balance is below {amount(r, 50, 200)}. Current balance: {amount(r, 1, 50)}.",
    lambda r: f"{r.choice(BANKS)}: Transfer of {amount(r)} to {r.choice(NAMES)} completed. Ref {digits(r, 8)}.",
    lambda r: f"{r.choice(BANKS)}: Your new card has been activated. You can now use it for purchases.",
    lambda r: f"{r.choice(BANKS)}: Your monthly statement for account ending {digits(r, 4)} is now available in the app.",
    lambda r: f"{r.choice(BANKS)}: Autopay of {amount(r, 20, 600)} for your credit card is scheduled for {date_(r)}.",
    lambda r: f"Your paycheck of {amount(r, 800, 4000)} has been deposited. Have a great weekend!",
    lambda r: f"Refund of {amount(r)} from {r.choice(STORES)} has been issued to your card ending {digits(r, 4)}. Allow 3-5 days.",
    lambda r: f"{r.choice(['Venmo', 'Zelle', 'PayPal', 'Cash App'])}: {r.choice(NAMES)} sent you {amount(r, 5, 200)} for \"{r.choice(['dinner', 'rent', 'tickets', 'groceries', 'gas', 'coffee'])}\".",
    lambda r: f"You paid {r.choice(NAMES)} {amount(r, 5, 200)} with {r.choice(['Venmo', 'Zelle', 'PayPal', 'Cash App'])}.",
    # bills and subscriptions
    lambda r: f"{r.choice(PROVIDERS)}: Your {r.choice(UTILITIES)} bill of {amount(r, 20, 300)} is due on {date_(r)}. Autopay is on.",
    lambda r: f"Thanks! We received your payment of {amount(r, 20, 300)} to {r.choice(PROVIDERS)}. Confirmation {digits(r, 9)}.",
    lambda r: f"{r.choice(SUBS)}: Your subscription renewed for {amount(r, 2, 25)}. Receipt sent to your email.",
    lambda r: f"Your {r.choice(SUBS)} free trial ends on {date_(r)}. Manage your plan in account settings.",
    lambda r: f"Your {r.choice(UTILITIES)} meter reading is due this week. Submit it in the {r.choice(PROVIDERS)} app.",
    # delivery
    lambda r: f"{r.choice(CARRIERS)}: Your parcel {digits(r, 4)} {digits(r, 4)} {digits(r, 4)} will arrive {r.choice(DAYS)} between {r.randint(8, 12)}am and {r.randint(1, 6)}pm.",
    lambda r: f"{r.choice(CARRIERS)}: Package delivered to your front door at {time_(r)}.",
    lambda r: f"Your {r.choice(STORES)} order #{digits(r, 3)}-{digits(r, 7)} has shipped. Estimated arrival {r.choice(DAYS)}.",
    lambda r: f"Good news! Your {r.choice(STORES)} order has been dispatched. Tracking number {digits(r, 12)}.",
    lambda r: f"{r.choice(CARRIERS)}: We missed you today. Your package will be redelivered {r.choice(DAYS)}.",
    lambda r: f"Your order from {r.choice(STORES)} is ready for pickup at the store. Bring your order number {digits(r, 6)}.",
    lambda r: f"Your {r.choice(['DoorDash', 'Uber Eats', 'Deliveroo', 'Grubhub'])} order from {r.choice(['Pizza Palace', 'Thai Garden', 'Burger Barn', 'Sushi Go', 'Taco Town'])} is on the way. ETA {r.randint(10, 40)} min.",
    lambda r: f"Your {r.choice(['Uber', 'Lyft', 'Bolt'])} driver {r.choice(NAMES)} is arriving in a {r.choice(['grey', 'white', 'black', 'blue'])} {r.choice(['Toyota Camry', 'Honda Civic', 'Prius', 'Tesla Model 3'])}, plate {r.choice('ABCDEFGHJK')}{digits(r, 3)}{r.choice('XYZ')}.",
    lambda r: f"Thanks for riding with {r.choice(['Uber', 'Lyft', 'Bolt'])}. Your trip total was {amount(r, 5, 60)}.",
    # bookings and travel
    lambda r: f"{r.choice(HOTELS)}: Your stay on {date_(r)} is confirmed. Booking reference {digits(r, 8)}. Check-in from 3 PM.",
    lambda r: f"Your reservation at {r.choice(HOTELS)} {r.choice(CITIES)} for {r.randint(1, 5)} nights is confirmed. Confirmation no. {digits(r, 9)}.",
    lambda r: (lambda a: f"{a[0]}: Flight {a[1]} {r.randint(100, 2999)} to {r.choice(CITIES)} departs at {time_(r)} from gate {r.choice('ABCDE')}{r.randint(1, 40)}.")(r.choice(AIRLINES)),
    lambda r: (lambda a: f"{a[0]}: Your flight {a[1]}{r.randint(100, 2999)} is delayed. New departure time {time_(r)}. We apologize for the inconvenience.")(r.choice(AIRLINES)),
    lambda r: f"Check-in is now open for your flight to {r.choice(CITIES)} on {date_(r)}. Booking ref {r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{digits(r, 5)}.",
    lambda r: f"Your table for {r.randint(2, 8)} at {r.choice(['Luigi', 'The Oak Room', 'Saffron', 'Blue Door', 'Noodle Bar'])} is booked for {r.choice(DAYS)} at {time_(r)}. Reply C to confirm or X to cancel.",
    lambda r: f"Your train ticket to {r.choice(CITIES)} on {date_(r)} is confirmed. Ticket number {digits(r, 10)}.",
    lambda r: f"Airbnb: Your host {r.choice(NAMES)} confirmed your booking in {r.choice(CITIES)} for {date_(r)}.",
    lambda r: f"Your rental car pickup is {r.choice(DAYS)} at {time_(r)}. Reservation {digits(r, 8)}. Bring your driving licence.",
    # appointments
    lambda r: f"{r.choice(CLINICS)}: Reminder of your appointment on {r.choice(DAYS)} at {time_(r)}. Reply YES to confirm or call us to reschedule.",
    lambda r: f"Your prescription is ready to collect at {r.choice(['CVS', 'Walgreens', 'Boots', 'Rite Aid'])}. Order {digits(r, 7)}.",
    lambda r: f"Your haircut at {r.choice(['Style Studio', 'The Barber Shop', 'Cut & Co'])} is booked for {r.choice(DAYS)} at {time_(r)}. Reply CANCEL if you can't make it.",
    lambda r: f"Your car service at {r.choice(['Midas', 'Kwik Fit', 'Jiffy Lube', 'Halfords'])} is complete. Total {amount(r, 40, 500)}. You can pick it up now.",
    lambda r: f"Reminder: your vet appointment for {r.choice(['Max', 'Bella', 'Luna', 'Charlie', 'Milo'])} is {r.choice(DAYS)} at {time_(r)}.",
    # school / work / community
    lambda r: f"{r.choice(['Lincoln Elementary', 'Westfield High', 'Oak Park School', 'St Mary Primary'])}: School will close early at {r.choice([12, 1, 2])}:30 PM {r.choice(DAYS)} due to {r.choice(['weather', 'staff training', 'a power outage'])}.",
    lambda r: f"Library notice: the book you reserved is ready to collect. Your hold expires on {date_(r)}.",
    lambda r: f"IT Helpdesk: your ticket {digits(r, 6)} has been resolved. Reply if the issue continues.",
    lambda r: f"Reminder: the team meeting has moved to {time_(r)} in room {r.randint(100, 599)}.",
    lambda r: f"Your timesheet for the week ending {date_(r)} has been approved.",
    lambda r: f"Gym: your membership payment of {amount(r, 15, 80)} was received. See you soon!",
    lambda r: f"Your appointment at the DMV is confirmed for {date_(r)} at {time_(r)}. Case number {digits(r, 8)}.",
    lambda r: f"Your insurance policy {digits(r, 9)} renews on {date_(r)}. No action needed if your details are unchanged.",
]


def load_eval_texts() -> list[str]:
    texts = []
    for name in ("val", "test", "test_external", "test_smishing"):
        with (ROOT / "data" / "splits" / f"{name}.csv").open(encoding="utf-8-sig", newline="") as f:
            texts += [normalize_text(row["raw_text"]) for row in csv.DictReader(f)]
    manual = [m for s in SETS.values() for _, _, m in s] + [m for _, _, m in SET_5]
    return texts, [normalize_text(m) for m in manual]


def main() -> None:
    r = random.Random(SEED)
    split_texts, manual_texts = load_eval_texts()
    split_set = set(split_texts)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < N_TARGET and attempts < N_TARGET * 20:
        attempts += 1
        raw = r.choice(TEMPLATES)(r)
        text = normalize_text(raw)
        if text in seen or text in split_set:
            continue
        if any(difflib.SequenceMatcher(None, text, m).ratio() >= 0.8 for m in manual_texts):
            continue
        seen.add(text)
        rows.append({"label": "ham", "raw_text": raw, "text": text})
    out = HERE / "data" / "ham_extra.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("label", "raw_text", "text"), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"templates={len(TEMPLATES)} rows={len(rows)} attempts={attempts} -> {out}")


if __name__ == "__main__":
    main()
