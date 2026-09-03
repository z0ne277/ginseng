# Ginseng retrieval benchmark

Canonical protocol: **271 组 / 1075 query**, 11,712 unlabeled pool images and
12787 Gallery images. Use the exact author-supplied splits, exclude the Query
from its own ranking, and do not replace Recall@K with Hit@K.

Start with the [reproduction guide](../docs/REPRODUCING.md). `MAIN_CODE_ROOT`
points to the parent repository containing `single_topo`, `simclr`, `moco`
and `moco_cbam`.

## Entry points

- `scripts/build_query_groups.py`: audit data and construct the query protocol.
- `scripts/stamp_feature_cache.py`: bind features to data/protocol metadata.
- `scripts/evaluate_features.py`: full-ranking metrics and instance bootstrap.
- `scripts/analyze_ssi.py`: SSI, equal-count bands and a diagnostic plot.
- [Existing models](docs/commands/existing-models.md): extraction and evaluation.
- [Frozen encoders](docs/commands/strong-baselines.md): separate modern environment.
- [Additional experiments](docs/commands/reviewer-experiments.md): training and ablations.

The retained Chinese guides describe execution mechanics. Historical completion
statements there are not a current result ledger; check the current manuscript
against actual result files separately.

## Execution boundary / 未运行耗时模型

No expensive models were trained/evaluated during release preparation. Unit tests
use synthetic inputs. Presence of a script does not prove reproduction of a
paper table. Data, exact splits and checkpoints must be supplied separately.

Legacy `tta` model IDs denote the manuscript's ITA. Preserve these IDs and do not
rename a cache to imply a different protocol.
