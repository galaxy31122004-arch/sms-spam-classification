import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))  # the data scripts import prepare_data as a top-level module

import augment_smishing  # noqa: E402
import build_external_test  # noqa: E402
from models.common import train  # noqa: E402


def row(label, text):
    return {"label": label, "raw_text": text, "text": text}


class AugmentSmishingTests(unittest.TestCase):
    def test_template_key_masks_digits_only(self):
        self.assertEqual(augment_smishing.template_key("ref txn626019 on 2026-08-14"),
                         "ref txn000000 on 0000-00-00")

    def test_group_split_keeps_templates_on_one_side_and_hits_target(self):
        groups = []
        for label in ("ham", "spam"):
            for index in range(20):
                groups.append([row(label, f"{label} template {chr(97 + index)} ref {n}") for n in range(3)])
        train_rows, test_rows = augment_smishing.group_split(groups, seed=42)
        train_keys = {augment_smishing.template_key(r["text"]) for r in train_rows}
        test_keys = {augment_smishing.template_key(r["text"]) for r in test_rows}
        self.assertFalse(train_keys & test_keys)
        self.assertEqual(len(train_rows) + len(test_rows), 120)
        for label in ("ham", "spam"):
            self.assertEqual(sum(r["label"] == label for r in test_rows), 9)  # 15% of 60 = 9, whole groups of 3
        self.assertEqual(augment_smishing.group_split(groups, seed=42), (train_rows, test_rows))

    def test_load_new_rows_maps_labels_and_rejects_bad_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps({"sms_text": "Pay NGN500 at bit.ly/x", "label": 1}) + "\n", encoding="utf-8")
            rows = augment_smishing.load_new_rows(path)
            self.assertEqual(rows[0]["label"], "spam")
            self.assertEqual(rows[0]["text"], "pay <MONEY> at <URL>")
            path.write_text(json.dumps({"sms_text": "", "label": 0}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid label or empty text"):
                augment_smishing.load_new_rows(path)


class ExternalTestTests(unittest.TestCase):
    def test_reads_mendeley_csv_from_zip(self):
        content = "LABEL,TEXT,URL,EMAIL,PHONE\nsmishing,Verify at bank.com/x,Yes,No,No\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("Dataset_5971.csv", content)
            rows = build_external_test.read_mendeley(path)
        self.assertEqual(rows, [{"LABEL": "smishing", "TEXT": "Verify at bank.com/x", "URL": "Yes",
                                 "EMAIL": "No", "PHONE": "No"}])

    def test_letters_key_ignores_case_digits_and_punctuation(self):
        self.assertEqual(build_external_test.letters_key("Hi! Call 0800 now."),
                         build_external_test.letters_key("hi call now"))


class TrainFileTests(unittest.TestCase):
    def test_train_reads_the_given_file_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "tiny.csv"
            with train_path.open("w", encoding="utf-8", newline="") as sink:
                writer = csv.DictWriter(sink, fieldnames=("label", "raw_text", "text"))
                writer.writeheader()
                for index in range(20):
                    writer.writerow(row("spam" if index % 4 == 0 else "ham", f"message {index} win prize"))
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                config = train("mlp", Path(directory) / "out", epochs=1, train_file=train_path)
        self.assertEqual(config["train_examples"], 20)
        self.assertEqual(config["train_file"], str(train_path.resolve()))


if __name__ == "__main__":
    unittest.main()
