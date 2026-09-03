# Multi-Level Structure-Aware Contrastive Learning for Ginseng Re-Identification

Code for visual and morphological representation learning, inference-time
augmentation (ITA), full-Gallery retrieval evaluation, and the Same-instance
Similarity Index (SSI).

This revision targets the current journal manuscript, not an accepted/publication
claim. Earlier revisions used an older protocol; do not mix their result tables
with this revision.

## Terminology and protocol

- An **instance** is one physical ginseng root. A **sample/image** is one image.
  Legacy identifiers such as `identity` and `group_id` refer to physical-instance
  groups and are retained for file compatibility.
- The unlabeled pool contains **11,712 images**. The manuscript reports **6,425
  images** in the parameter-optimization subset. These are different quantities:
  use the exact authors' splits rather than randomly recreating them.
- The independent test set contains **271 instances and 1,075 images**.
- The unified Gallery has **12,787 images**. Each test image is a Query, which is
  excluded from its own ranking, leaving 12,786 candidates.
- MRR and mAP use the **complete ranking**. Recall@K measures the fraction of
  relevant images retrieved; Hit@K measures whether any relevant image is found.
- ITA uses `stretch224`, `contain224` and `contain256` with weights
  `[0.6, 0.25, 0.15]`. Existing CLI/configuration keys use the spelling `tta`.
- SSI is `0.5 * (1 + mean within-instance pairwise cosine similarity)`.
  It is post-hoc embedding cohesion, not a training objective or an independent
  physical-shape annotation.

The current manuscript reports MRR **0.9670** and mAP **0.8867** for the proposed
method with ITA. These are **manuscript-reported values**, not measurements from
source-release tests. Exact reproduction requires the matching data version,
splits, preprocessing and checkpoints.

## Layout

| Location | Purpose |
|---|---|
| `single_topo/` | Current structural model, training, extraction, ITA and Grad-CAM |
| `benchmark/` | Current evaluation protocol, baseline runners, ablations, SSI and tests |
| `simclr/`, `moco/` | Legacy baseline models and checkpoint feature extractors |
| `moco_cbam/` | MoCo V3 + CBAM model, training and extraction |
| `ginseng_extractor/` | Foreground extraction retained from the public repository |
| `hybrid_v2/`, `hybrid_v3/` | Historical variants, not the current journal-method entry points |
| `assets/` | Previously public illustrations, not a dataset or current results archive |

## Installation

Core verification uses Python 3.9.23, PyTorch 1.13.1, torchvision 0.14.1 and
NumPy 1.26.4. This is a local verification environment, not a claim about every
optional baseline's training environment.

The supplied core PowerShell runners expect a Conda environment named `gsam`.
Install a suitable PyTorch/torchvision build for your hardware, then run from
this repository's root:

```text
python -m pip install -r requirements-core.txt
python -m pip install -e ./benchmark
```

Modern frozen encoders use the separate `ginseng-baselines` environment and
`benchmark/requirements-modern.txt`; follow the benchmark guide instead of
upgrading the core environment in place. Foreground extraction additionally
needs `requirements.txt` and separately obtained GroundingDINO/SAM weights.

## Configure data and run

See [reproduction instructions](docs/REPRODUCING.md) and the
[benchmark guide](benchmark/README.md). Copy `benchmark/.env.example` to a local
`benchmark/.env` and set **absolute paths**. `MAIN_CODE_ROOT` must point to
**this repository root**. No real `.env` belongs in Git.

Images, exact author split CSVs, query manifests, trained weights and feature
caches are not bundled. Do not assume an older public dataset has the current
271-instance protocol. Dataset version/access is separate from code access.

From `benchmark/`, after installing the package and configuring the inputs:

```powershell
python scripts/build_query_groups.py --env .env --output artifacts/manifests/query_groups.json
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase all
python scripts/analyze_ssi.py --cache artifacts/features/validated/single_topo_tta_271_1075.npz --query-groups artifacts/manifests/query_groups.json --output-json artifacts/analysis/ssi.json --output-csv artifacts/analysis/ssi.csv --figure artifacts/analysis/ssi.png
```

Use `benchmark/scripts/evaluate_features.py` for manuscript metrics. Legacy
`batch_retrieval.py` utilities support visual inspection; do not use truncated
visualization rankings in place of complete-gallery evaluation.

## Tests and limitations

Run the data-independent tests from `benchmark/`:

```text
python -m unittest tests.test_metrics tests.test_cache tests.test_env tests.test_protocol tests.test_query_groups tests.test_evaluation tests.test_ssi_analysis tests.test_paired_bootstrap -v
```

These tests use synthetic inputs. Some other tests require PowerShell plus local
data/checkpoint configuration; passing the offline subset is not an end-to-end
reproduction. SimCLR and legacy MoCo V3 are checkpoint-extraction adapters, not
complete from-scratch training pipelines. Paper-specific plots tied to private
caches are not advertised as portable figure-generation tools.
See [release status](docs/RELEASE_STATUS.md).

## License

The existing MIT license is retained. Third-party packages, weights and datasets
remain subject to their own licenses. This repository grants no rights to files
it does not distribute.
