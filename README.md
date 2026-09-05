# Multi-Level Structure-Aware Contrastive Learning for Ginseng Re-Identification

A PyTorch implementation for retrieving different images of the same physical
ginseng root under changes in placement, orientation, unfolding and occlusion.
The framework combines a visual representation with a multi-level morphological
branch, uses inference-time augmentation (ITA) for retrieval, and introduces the
Same-instance Similarity Index (SSI) for post-hoc analysis.

**Quick links:** [Architecture](#architecture) · [Dataset](#dataset-and-evaluation-protocol) ·
[Installation](#installation) · [Training](#train-the-proposed-model) ·
[Evaluation](#extract-features-and-evaluate) · [SSI](#ssi-analysis) ·
[Reproduction guide](docs/REPRODUCING.md)

## Overview

Ginseng roots are non-rigid objects. Repeated placement can rearrange their
rootlets and obscure parts of the main root, while different roots may have
similar appearances. This project studies instance-level retrieval in this
setting through three components:

1. A multi-view ginseng evaluation dataset with repeated images of each instance.
2. A structure-aware contrastive framework that learns visual and morphological
   features from unlabeled images.
3. SSI, which measures the concentration of images from the same instance in the
   learned feature space and supports grouped retrieval analysis.

Throughout this README, an **instance** means one physical ginseng root, and an
**image/sample** means one image of that instance. Existing code identifiers such
as `identity` and `group_id` retain their original names for compatibility.

## Architecture

![Architecture of the structure-aware contrastive learning framework](assets/arch_diagram.png)

*Framework diagram from the current English manuscript. Open the image to view
the full figure. The upper panel shows training; the lower panel shows inference
with ITA.*

### Training: visual and morphological representations

Foreground processing is followed by two augmented image views. A ResNet-50
backbone with CBAM produces the feature maps used by the visual and morphological
branches. A query encoder learns through gradient updates, while a key encoder
is updated through momentum.

The morphological branch builds four structural levels, `E0–E3`: the original
feature map and three successive 3 × 3 erosion operations implemented through
min pooling. Each level is projected and pooled, and the resulting responses
are fused into a structural embedding. Separate visual and structural queues
support the contrastive objectives.

The combined training objective is:

```text
L = (1 - lambda) * L_visual + lambda * L_morphological
lambda = 0.35
```

### Inference: feature concatenation and ITA

The visual embedding has 256 dimensions and the structural embedding has
128 dimensions. Their concatenation is L2-normalized to form a 384-dimensional
retrieval feature.

ITA extracts a feature for each of three input transformations:

| Code name | Transformation | Fusion weight |
|---|---|---:|
| `stretch224` | Resize directly to 224 × 224 | 0.60 |
| `contain224` | Preserve aspect ratio within a 224 × 224 canvas | 0.25 |
| `contain256` | Preserve aspect ratio within a 256 × 256 canvas | 0.15 |

The weighted feature sum is normalized again. The same processing is applied to
Query and Gallery images, and cosine similarity determines the ranking.
Configuration keys and model IDs still use `tta`; these refer to the manuscript's
**ITA**. SSI is calculated afterwards and does not affect training or ranking.

## Dataset and evaluation protocol

| Component | Size | Role |
|---|---:|---|
| Unlabeled image pool | 11,712 images | Representation learning pool and Gallery distractors |
| Parameter-optimization subset | 6,425 images | Training subset within the unlabeled pool |
| Independent multi-view test set | 271 instances / 1,075 images | Query images and same-instance matches |
| Unified Gallery | 12,787 images | Unlabeled pool plus test images |
| Candidates per Query | 12,786 images | Unified Gallery after self-exclusion |

Each test image is used once as a Query. Other images of the same instance are
relevant results. The test set contains 265 instances with four images, three
with three images, and three with two images. The instance-labeled test set and
the unlabeled pool are instance-disjoint under the manuscript's protocol.

The 6,425-image optimization subset must not be confused with the entire
11,712-image unlabeled pool. Exact training, validation and development CSVs
are required to reproduce the manuscript protocol.

### Data availability

The updated dataset has been submitted to **Mendeley Data** and is undergoing
review, as reported by the authors on 5 September 2026. The public record is
[Ginseng Dataset For re-ID](https://data.mendeley.com/datasets/z57msctftm).
Until the updated version is released, this record may resolve to the earlier
version, whose description lists 292 instances and 1,182 images. That version
must not be assumed to match the current evaluation protocol.

Use the matching version and split files when available. Dataset images, split
CSVs, query manifests, trained weights and feature caches are not included in
this Git repository. Access to the 11,712-image pool and exact training splits
must be confirmed separately from access to the multi-view evaluation set.

### Metrics

- **MRR:** reciprocal rank of the first relevant image, averaged over Queries.
- **mAP:** average precision using the complete Gallery ranking, averaged over Queries.
- **Recall@K:** fraction of all relevant images retrieved in the first K results.
- **Hit@K:** whether at least one relevant image occurs in the first K results.

Recall@K and Hit@K are different. For an instance with four images, a correct
first result yields Recall@1 = 1/3 and Hit@1 = 1. Use the canonical evaluator
under `benchmark/` for manuscript metrics. The Top-K lists produced by older
visualization scripts do not replace complete-ranking evaluation.

## Manuscript result

| Method | MRR | mAP |
|---|---:|---:|
| Proposed method with ITA | 0.9670 | 0.8867 |

These values are reported in the current manuscript. Source-release verification
used synthetic tests and did not rerun the full experiments. Reproducing these
scores requires matching data, splits, weights and preprocessing. Older commits
contain results from an earlier protocol and should not be mixed with this table.

## Repository layout

```text
ginseng/
├── assets/                  # Architecture and illustrative images
├── single_topo/             # Current proposed model
│   ├── model.py             # Visual/morphological branches and contrastive loss
│   ├── train.py             # Training entry point
│   ├── extraction.py        # Gallery features and ITA
│   ├── gradcam_visualization.py
│   └── configs/             # Default/example model configurations
├── benchmark/
│   ├── src/ginseng_benchmark/ # Protocol, metrics, feature cache and SSI
│   ├── scripts/             # Training/extraction/evaluation runners
│   ├── configs/             # Baseline and controlled-ablation configurations
│   ├── tests/               # Synthetic unit and local integration tests
│   ├── docs/commands/       # Detailed experiment commands
│   └── .env.example         # Local path and optional token template
├── simclr/                  # SimCLR model and checkpoint extractor
├── moco/                    # MoCo V3 model and checkpoint extractor
├── moco_cbam/               # MoCo V3 + CBAM training and extraction
├── ginseng_extractor/       # Foreground extraction utilities
├── hybrid_v2/               # Historical model variant
├── hybrid_v3/               # Historical model variant
├── docs/REPRODUCING.md
├── requirements-core.txt
└── LICENSE
```

`single_topo/` is the recommended entry point for the current method.
Historical variants and retained illustrative assets are provided for context;
they do not define the current evaluation protocol.

## Installation

The core release checks were run with Python 3.9.23, PyTorch 1.13.1,
torchvision 0.14.1 and NumPy 1.26.4. Optional modern encoders use a separate
environment.

Clone the repository:

```text
git clone https://github.com/z0ne277/ginseng.git
cd ginseng
```

Create the core environment. The provided PowerShell runners expect it to be
named `gsam`:

```text
conda create -n gsam python=3.9
conda activate gsam
```

Install a compatible PyTorch/torchvision build using the
[official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/),
then install the project dependencies:

```text
python -m pip install -r requirements-core.txt
python -m pip install -e ./benchmark
```

Frozen modern encoders use `benchmark/environment-modern.yml` and
`benchmark/requirements-modern.txt` in the separate `ginseng-baselines`
environment. See the [strong-baseline guide](benchmark/docs/commands/strong-baselines.md).

Foreground extraction additionally requires dependencies listed in
`requirements.txt` and separately obtained GroundingDINO/SAM weights.
Training with pretrained backbone initialization can download torchvision weights
on first use.

## Configure local inputs

From the repository root, copy the example only if no local configuration exists:

```powershell
if (-not (Test-Path benchmark/.env)) {
    Copy-Item benchmark/.env.example benchmark/.env
}
```

Edit the following keys using **absolute paths**:

| Key | Required value |
|---|---|
| `MAIN_CODE_ROOT` | This repository's root, containing `single_topo/` |
| `LIBRARY_BINARY` | Preprocessed 11,712-image unlabeled pool |
| `TEST_BINARY_ROOT` | 271 folders containing the 1,075 test images |
| `MERGED_GALLERY` | Unified 12,787-image Gallery |
| `TRAIN_CSV` | Exact optimization split |
| `VAL_CSV` | Exact validation split |
| `DEV_CSV` | Image-only development split |
| `HF_TOKEN` | Optional; leave empty unless needed |

The dotenv reader does not expand expressions such as `${VARIABLE}`.
The legacy development CSV named `test.csv` is distinct from the independent
instance-labeled retrieval test set. Training CSVs require an `image` column;
absolute image paths are recommended. Keep the real `.env` local.

For direct model commands, paths in JSON files resolve relative to the working
directory. Run the following direct commands from the repository root, or
supply absolute paths through `--override`.

## Train the proposed model

After obtaining the exact split CSVs, replace the example paths:

```powershell
python single_topo/train.py `
    --override train_csv=/absolute/path/to/splits/train.csv `
    --override val_csv=/absolute/path/to/splits/val.csv `
    --override test_csv=/absolute/path/to/splits/test.csv `
    --override checkpoint_dir=/absolute/path/to/ginseng/single_topo/checkpoints/moco_v3_topo `
    --override seed=42
```

The current default configuration includes:

| Setting | Value |
|---|---:|
| Backbone | ResNet-50 with CBAM |
| Visual / structural dimensions | 256 / 128 |
| Structural levels | 4, including E0 |
| Erosion kernel | 3 × 3 |
| Negative queue size | 4,096 |
| Momentum coefficient | 0.999 |
| Contrastive temperature | 0.07 |
| Structural loss weight | 0.35 |
| Batch size | 128 |
| Learning rate | 0.0001 |
| Maximum epochs | 200 |
| Random seed | 42 |

Additional settings are in
[`single_topo/configs/default.json`](single_topo/configs/default.json).
For CPU execution, add `--override use_gpu=false`.

## Extract features and evaluate

The existing-model runner expects the following checkpoints relative to
`MAIN_CODE_ROOT`, unless its configuration is edited:

| Model ID | Checkpoint path |
|---|---|
| `single_topo_plain`, `single_topo_tta` | `single_topo/checkpoints/moco_v3_topo/best_model.pth` |
| `moco_v3_cbam` | `moco_cbam/checkpoints/moco_cbam/best_model.pth` |
| `simclr` | `simclr/model_epoch_200(1).pth` |
| `moco_v3` | `moco/model_epoch_200.pth` |

Weights are supplied separately. Load only trusted checkpoints because legacy
PyTorch loading can deserialize pickle.

Enter `benchmark/` and generate the canonical Query protocol:

```powershell
cd benchmark
python scripts/build_query_groups.py --env .env --output artifacts/manifests/query_groups.json
```

This verifies expected data counts and file consistency before writing the
protocol. Then inspect and run the proposed method with ITA:

```powershell
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -DryRun
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase extract
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase stamp
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase evaluate
```

The stages extract features, attach dataset/protocol metadata, and evaluate the
full ranking. `-DryRun` still checks that configured directories and checkpoint
paths exist. To disable ITA, use model ID `single_topo_plain`.

| Output directory, relative to benchmark/ | Contents |
|---|---|
| `artifacts/manifests/` | Canonical Query protocol |
| `artifacts/features/raw/` | Extracted raw features |
| `artifacts/features/validated/` | Validated NPZ caches and metadata |
| `artifacts/results/` | Aggregate and per-Query metrics |
| `artifacts/tables/` | Summarized tables |
| `artifacts/logs/` | Execution logs |

See [existing-model commands](benchmark/docs/commands/existing-models.md) for
all five legacy model configurations and same-checkpoint branch ablations.

## Controlled ablations and additional baselines

The benchmark includes separate configurations for:

- [Visual-only and morphological-only extraction](benchmark/configs/controlled_ablations.json)
  from the same checkpoint.
- [Erosion depth, identity operator, CBAM and backbone variants](benchmark/configs/topology_training_ablations.json).
- [In-domain self-supervised baselines](benchmark/configs/self_supervised_models.json).
- [Frozen pretrained encoders](benchmark/configs/strong_models.json).

From `benchmark/`, inspect a structural training variant:

```powershell
powershell -NoProfile -File scripts/run_topology_ablations.ps1 -Variants levels_2 -Phase train -DryRun
```

Remove `-DryRun` to execute after preparing inputs and compute. The ablation
runner writes checkpoints below `benchmark/artifacts/`; use its corresponding
extraction and evaluation stages. See
[additional experiment commands](benchmark/docs/commands/reviewer-experiments.md).

SimCLR and the legacy MoCo V3 release contain model definitions and extraction
adapters; their original from-scratch training drivers are not included.

## SSI analysis

For one instance with n images, SSI averages the cosine similarity of all
distinct within-instance pairs and maps the result to [0, 1]:

```text
SSI = 0.5 * (1 + mean cosine similarity over all distinct image pairs)
```

Higher SSI means greater feature concentration across images of the same
instance. Lower SSI means more dispersed features. SSI is computed from the
final retrieval representation and serves as a post-hoc analysis.

From `benchmark/`, after producing the final ITA cache:

```powershell
python scripts/analyze_ssi.py `
    --cache artifacts/features/validated/single_topo_tta_271_1075.npz `
    --query-groups artifacts/manifests/query_groups.json `
    --output-json artifacts/analysis/ssi.json `
    --output-csv artifacts/analysis/ssi.csv `
    --figure artifacts/analysis/ssi.png
```

This produces instance-level statistics and approximately equal-size Low/Mid/High
groups: 91, 90 and 90 instances. For cross-model comparisons, keep the grouping
defined by the proposed final ITA feature fixed. Regrouping each model separately
would answer a different question. These groups reflect the chosen model's
representation and are not independent physical difficulty labels.

The generated plot is diagnostic. Portable regeneration of every manuscript
figure is outside the current code release.

## Verification and troubleshooting

Run the data-independent tests from `benchmark/`:

```text
python -m unittest tests.test_metrics tests.test_cache tests.test_env tests.test_protocol tests.test_query_groups tests.test_evaluation tests.test_ssi_analysis tests.test_paired_bootstrap -v
```

The release preparation passed 143 offline, structure-configuration and
documentation checks, plus seven CLI help checks. These checks validate software
behavior on synthetic inputs; they do not reproduce the reported experimental
scores. Some other integration tests require Windows PowerShell and configured
data/checkpoints.

| Problem | What to check |
|---|---|
| `No module named ginseng_benchmark` | Install `./benchmark` into the Python environment used by the command |
| Missing `MAIN_CODE_ROOT` or dataset path | Edit `benchmark/.env` and use absolute paths |
| Missing checkpoint during `-DryRun` | Obtain the checkpoint and check its configured location |
| Dataset count/hash mismatch | Confirm the exact dataset version and merged Gallery; do not bypass the audit |
| Feature cache rejected | Regenerate/stamp features for the same dataset and Query protocol |
| Dependency conflicts with frozen encoders | Use the separate modern-baseline environment |
| Missing local files in historical variants | Follow the current `single_topo/` and `benchmark/` workflow |

See [release scope and remaining dependencies](docs/RELEASE_STATUS.md) for
data, checkpoint and figure-reproduction requirements.

## License and acknowledgements

This repository retains its [MIT license](LICENSE). External libraries,
pretrained weights and datasets retain their respective licenses.

The implementation uses PyTorch and torchvision, CBAM, momentum contrastive
learning and mathematical morphology. Optional components include pretrained
vision encoders, GroundingDINO and Segment Anything. Please acknowledge the
corresponding original works when using those components.
