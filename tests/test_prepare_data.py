import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.prepare_data import load_rows, normalize_text


class PrepareDataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
