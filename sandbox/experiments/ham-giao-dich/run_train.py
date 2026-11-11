"""Train MLP/LSTM/Transformer with the repo's train() on this experiment's re-normalized splits.

Usage: python run_train.py <config>  where config is base (train_base) or ham (train_ham).
Same hyperparameters, seed 42 and threshold 0.5 as the benchmark.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import models.common as common  # noqa: E402

common.ROOT = HERE  # val.csv is read from HERE/data/splits/val.csv (re-normalized copy)
TRAIN = {"base": "train_base.csv", "ham": "train_ham.csv", "mix": "train_mix.csv"}

config = sys.argv[1]
for kind in common.KINDS:
    result = common.train(kind, HERE / "checkpoints" / config / kind,
                          train_file=HERE / "data" / "splits" / TRAIN[config])
    print(json.dumps({k: result[k] for k in ("kind", "train_examples", "best_epoch", "val_spam_f1",
                                             "train_seconds")}), flush=True)
