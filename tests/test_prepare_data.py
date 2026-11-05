import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

from data.prepare_data import load_rows, main, normalize_text, stratified_split, summarize_splits


class PrepareDataTests(unittest.TestCase):
    def test_preserves_punctuation_after_urls(self):
        cases = {
            "Visit www.example.com! Save 20% $": "visit <URL> ! save 20% $",
            "See (https://example.com/path).": "see ( <URL> ).",
            "See https://example.com/a_(b)!": "see <URL> !",
            "www.example.com!$%": "<URL> !$%",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)

    def test_keeps_url_contents_out_of_money_and_phone_matching(self):
        raw = "Go https://example.com/a!b%20c?fee=$500&phone=08712778109!"
        self.assertEqual(normalize_text(raw), "go <URL> !")

    def test_handles_money_formats_without_eating_punctuation(self):
        cases = {"$1,000!": "$<MONEY> !", "£250k!": "£<MONEY> !", "€1.50,": "€<MONEY> ,",
                 "10p/min.": "<MONEY> /min.", "1.5p/min": "<MONEY> /min",
                 "10 pence": "<MONEY>", "100 GBP": "<MONEY>", "20 quid!": "<MONEY> !"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)

    def test_bare_domains_become_url_tokens(self):
        cases = {
            "Click: bit.ly/3xub3q2f": "click: <URL>",
            "Restore now: secure-mtn.net/86on40.": "restore now: <URL> .",
            "Claim: t.me/p2079p!": "claim: <URL> !",
            "Verify: update-account.kcb.co/qc6dr0": "verify: <URL>",
            "Visit ringtones.co.uk now": "visit <URL> now",
            "Apply: m-pesa-alert.com": "apply: <URL>",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)
        for raw in ("ok.me see you", "e.g. tomorrow", "mail me at a@b.com", "end.then"):
            with self.subTest(raw=raw):
                self.assertNotIn("<URL>", normalize_text(raw))

    def test_iso_currency_codes_become_money_tokens(self):
        cases = {
            "Refund of NGN25000 pending": "refund of <MONEY> pending",
            "Won USD250000!": "won <MONEY> !",
            "Earn ZAR25000/week": "earn <MONEY> /week",
            "Send KES 1,000 now": "send <MONEY> now",
            "Pay 500 GHS today": "pay <MONEY> today",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), expected)
        self.assertEqual(normalize_text("ngn25000abc"), "ngn25000abc")

    def test_separates_uk_phone_from_attached_pence_rate(self):
        self.assertEqual(normalize_text("Call 0871277810910p/min!"), "call <PHONE> <MONEY> /min!")
        self.assertEqual(normalize_text("08701417012150p per logo/pic"), "<PHONE> <MONEY> per logo/pic")
        # An ambiguous long number must not become a single money token.
        self.assertEqual(normalize_text("1234567890123456p/min"), "1234567890123456p/min")

    def test_normalizes_phones_but_preserves_dates_and_short_numbers(self):
        self.assertEqual(normalize_text("Call +44 (0)20 1234 5678!"), "call <PHONE> !")
        for raw in ("2026-10-25", "25.10.2026", "8007", "1234567890123456"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_text(raw), raw)

    def test_separate_phone_numbers_and_spaced_dates(self):
        self.assertEqual(normalize_text("Call 08712778109 08701417012!"), "call <PHONE> <PHONE> !")
        self.assertEqual(normalize_text("On 2026 - 10 - 25!"), "on 2026 - 10 - 25!")
        self.assertEqual(normalize_text("$1.50abc"), "$1.50abc")

    def test_unicode_and_normalization_are_stable(self):
        self.assertEqual(normalize_text("ＷＩＮ ＄５００！ ２０％"), "win $<MONEY> ! 20%")
        text = normalize_text("Call 0871277810910p/min! Visit www.example.com!")
        self.assertEqual(normalize_text(text), text)

    def test_normalizes_kaggle_mojibake_before_money_replacement(self):
        self.assertIn("£<MONEY>", normalize_text("Win å£500 now"))
        self.assertNotIn("å", normalize_text("Win å£500 now"))

    def test_loads_kaggle_csv_with_cp1252_currency_symbol(self):
        content = 'v1,v2,,,\r\nspam,"WIN a £1.50 prize!",,,\r\n'.encode("cp1252")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spam.csv"
            path.write_bytes(content)

            records = load_rows(path)

        self.assertEqual(records[0]["label"], "spam")
        self.assertIn("£<MONEY>", records[0]["text"])

    def test_reads_quoted_newline_in_kaggle_message(self):
        content = b'v1,v2,,,\r\nham,"before?\\""\rham""",,,\r\nham,next message,,,\r\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spam.csv"
            path.write_bytes(content)

            bad_sniff = SimpleNamespace(
                delimiter=",",
                quotechar='"',
                doublequote=False,
                escapechar=None,
                skipinitialspace=False,
                lineterminator="\r\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            with patch("data.prepare_data.csv.Sniffer.sniff", return_value=bad_sniff):
                records = load_rows(path)

        self.assertEqual([record["label"] for record in records], ["ham", "ham"])
        self.assertEqual(records[1]["raw_text"], "next message")


class GroupedSplitTests(unittest.TestCase):
    @staticmethod
    def records():
        rows = []
        for label in ("ham", "spam"):
            for index in range(30):
                text = f"{label} message {index}"
                rows.append({"label": label, "raw_text": text, "text": text})
            for amount in (10, 20, 30):
                raw = f"{label} offer ${amount}!"
                rows.append({"label": label, "raw_text": raw, "text": normalize_text(raw)})
        return rows

    def test_grouped_split_preserves_all_rows_and_prevents_overlap(self):
        records = self.records()
        splits = stratified_split(records, seed=42)
        original = Counter(tuple(sorted(record.items())) for record in records)
        actual = Counter(tuple(sorted(record.items())) for rows in splits.values() for record in rows)
        self.assertEqual(actual, original)
        for pairs in summarize_splits(splits)["overlap"].values():
            for overlap in pairs.values():
                self.assertEqual(overlap["rows_in_right"], 0)
        for rows in splits.values():
            self.assertEqual({row["label"] for row in rows}, {"ham", "spam"})
        for label in ("ham", "spam"):
            counts = [sum(row["label"] == label for row in splits[name]) for name in ("train", "val", "test")]
            self.assertEqual(counts, [23, 5, 5])

    def test_grouped_split_is_reproducible_without_mutating_input(self):
        records = self.records()
        before = [dict(record) for record in records]
        first = stratified_split(records, seed=42)
        self.assertEqual(first, stratified_split(records, seed=42))
        self.assertNotEqual(first, stratified_split(records, seed=43))
        self.assertEqual(records, before)

    def test_large_groups_populate_all_splits(self):
        rows = []
        for label in ("ham", "spam"):
            for index, size in enumerate((10, 5, 5)):
                text = f"{label} group {index}"
                rows.extend({"label": label, "raw_text": text, "text": text} for _ in range(size))
        splits = stratified_split(rows, seed=42)
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        for label in ("ham", "spam"):
            self.assertEqual([sum(row["label"] == label for row in splits[name])
                              for name in ("train", "val", "test")], [10, 5, 5])
        for pairs in summarize_splits(splits)["overlap"].values():
            self.assertTrue(all(pair["rows_in_right"] == 0 for pair in pairs.values()))

    def test_impossible_or_empty_grouped_split_is_rejected(self):
        row = {"label": "ham", "raw_text": "same", "text": "same"}
        with self.assertRaisesRegex(ValueError, "distinct ham groups"):
            stratified_split([dict(row) for _ in range(10)], seed=42)
        with self.assertRaisesRegex(ValueError, "empty"):
            stratified_split([], seed=42)

    def test_conflicting_duplicate_labels_are_rejected(self):
        rows = [{"label": label, "raw_text": "same", "text": "same"} for label in ("ham", "spam")]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            stratified_split(rows, seed=42)

    def test_duplicate_audit_counts_repeated_rows(self):
        row = {"label": "spam", "raw_text": "offer", "text": "offer"}
        report = summarize_splits({"train": [row], "val": [], "test": [row, row]})
        self.assertEqual(report["overlap"]["text"]["train:test"],
                         {"shared_groups": 1, "rows_in_right": 2, "spam_rows_in_right": 2})

    def test_dry_run_does_not_write_even_with_force(self):
        with patch("sys.argv", ["prepare_data.py", "--dry-run", "--force"]), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.open", mock_open(read_data="label,raw_text,text\nham,old,old\n")), \
             patch("data.prepare_data.load_rows", return_value=self.records()), \
             patch("data.prepare_data.write_csv") as write, redirect_stdout(io.StringIO()) as output:
            main()
        write.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["mode"], "dry_run")
        self.assertEqual(report["source_rows"], len(self.records()))

    def test_existing_splits_are_protected_without_force(self):
        with patch("sys.argv", ["prepare_data.py"]), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("data.prepare_data.write_csv") as write, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
