# SMS Spam Evaluation

Test set: `data/splits/test.csv` (836 messages)
Transformer spam label: `1`

| Path | Evaluated | Tier-1 coverage | Spam precision | Spam recall | Spam F1 | Ham false-positive rate | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 836 | 100.0% | 97.7% | 76.8% | 86.0% | 0.3% | 0.40 ms |
| Tier-1 decided subset | 742 | 88.8% | 100.0% | 100.0% | 100.0% | 0.0% | 0.12 ms |
| Cascade | 836 | 100.0% | 99.0% | 90.2% | 94.4% | 0.1% | 0.16 ms |

Tier-2 rate: 11.2%
Tier-1 routes: 55 spam, 687 ham, 94 to Tier-2.

Tier-1 is a pre-filter. Messages routed to Tier-2 are classified by the Transformer.
