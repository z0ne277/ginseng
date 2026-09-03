# Reproduction instructions

## Required inputs

Obtain the matching preprocessed images and exact image-only training,
validation and development split CSVs. The loaders expect an `image` column;
absolute image paths are recommended. The development CSV named `test.csv` in
legacy code is NOT the independent 271-instance retrieval test set.

The 11,712-image unlabeled pool and its 6,425-image optimization subset are
different quantities. Do not use independent test images for training, or
create new splits and claim exact reproduction. The audit checks files,
counts and hashes; physical-instance disjointness also requires correct
annotations and collection records. No matching dataset/split download is
promised by this code-only release. Confirm access with the authors.

## Install and configure

From the repository root, install a compatible PyTorch/torchvision pair, then
`python -m pip install -r requirements-core.txt` and
`python -m pip install -e ./benchmark`. PowerShell runners expect the core
Conda environment to be named `gsam`.

Copy `benchmark/.env.example` to `benchmark/.env` only if no local `.env` already
exists. Set every path to an absolute path; `MAIN_CODE_ROOT` is this repository
root. The dotenv reader does not expand `${VARIABLE}`. Never commit real tokens.

The direct core-model commands below run from the repository root and use
`single_topo/configs/default.json` as a safe example. Use an external JSON with
`--config` or set paths through `--override`. Relative paths resolve from the
working directory, not automatically from the JSON file.

## Train the proposed model

```text
python single_topo/train.py --override train_csv=/absolute/path/to/splits/train.csv --override val_csv=/absolute/path/to/splits/val.csv --override test_csv=/absolute/path/to/splits/test.csv --override checkpoint_dir=/absolute/path/to/ginseng/single_topo/checkpoints/moco_v3_topo --override seed=42
```

The copied source keeps its current algorithm and hyperparameters. No training
was performed during release preparation. Pretrained-backbone initialization
may download torchvision weights. For CPU execution set `--override use_gpu=false`.

For controlled erosion/attention/backbone training, configure the `.env`, enter
`benchmark/`, and inspect the command first:

```powershell
powershell -NoProfile -File scripts/run_topology_ablations.ps1 -Variants reference -Phase train -DryRun
```

Remove `-DryRun` only when inputs and compute are ready. This runner writes its
own weights under `benchmark/artifacts/`; use its later extraction/evaluation
stages or explicitly point an extractor to that checkpoint. This path differs
from the existing-checkpoint convention below.

## Evaluate existing checkpoints

`benchmark/configs/existing_models.json` expects these paths relative to
`MAIN_CODE_ROOT`; edit your local configuration if you store weights elsewhere:

| Model ID | Checkpoint |
|---|---|
| `single_topo_plain`, `single_topo_tta` | `single_topo/checkpoints/moco_v3_topo/best_model.pth` |
| `moco_v3_cbam` | `moco_cbam/checkpoints/moco_cbam/best_model.pth` |
| `simclr` | `simclr/model_epoch_200(1).pth` |
| `moco_v3` | `moco/model_epoch_200.pth` |

No trained weights are bundled. Only load trusted local checkpoints: legacy
PyTorch loading can deserialize pickle. From `benchmark/`:

```powershell
python scripts/build_query_groups.py --env .env --output artifacts/manifests/query_groups.json
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -DryRun
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase extract
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase stamp
powershell -NoProfile -File scripts/run_existing_models.ps1 -Models single_topo_tta -Phase evaluate
```

The runner's `-DryRun` checks that data/checkpoint paths exist. It is not a
dataset-free unit test. SimCLR and legacy MoCo V3 have checkpoint extractors
and model definitions, but their original training drivers are not included.

## SSI

From `benchmark/`, after producing the final ITA feature cache:

```text
python scripts/analyze_ssi.py --cache artifacts/features/validated/single_topo_tta_271_1075.npz --query-groups artifacts/manifests/query_groups.json --output-json artifacts/analysis/ssi.json --output-csv artifacts/analysis/ssi.csv --figure artifacts/analysis/ssi.png
```

For 271 instances, this produces 91 Low, 90 Mid and 90 High instances. For
cross-model comparisons keep the bands from the proposed final ITA feature
fixed. Independently running this command on each model redefines the groups
and is NOT the manuscript's cross-model grouping protocol. The supplied figure
is diagnostic, not a claim to regenerate every final paper panel.

## Offline verification

Run the data-independent unittest command in the root README from `benchmark/`.
Other original tests can invoke PowerShell integration runners requiring real
data/checkpoint configuration. No real training images are shipped as tests.
