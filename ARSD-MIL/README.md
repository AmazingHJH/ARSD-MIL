# train-main.py — Training and Evaluation Guide

This README documents the data requirements, workflow, command-line options, and outputs of `train-main.py` in this directory.

## 1. Overview

The script performs multiple instance learning (MIL) classification using pre-extracted whole-slide image (WSI) patch features. Prepare HDF5 feature files before running it; slide tiling and feature extraction are outside the script's workflow.

The pipeline consists of the following steps:

1. Extract cluster prototypes from benign slide features to construct a benign reference memory bank.
2. Compute each patch's Euclidean distance to its nearest memory-bank prototype as an AbRank abnormality score. Larger distances indicate greater dissimilarity to the benign reference features.
3. Normalize scores within each slide, rank patches in descending order, and group consecutive ranked patches into MiniBags.
4. Aggregate features within each MiniBag using score-guided gated attention, then incorporate coordinate and score information.
5. Use a Transformer with a CLS token for slide classification, optionally adding an auxiliary slide-to-MiniBag distillation loss.
6. Run patient-level five-fold outer cross-validation with rotating held-out test sets and an inner validation split, save selected checkpoints, and summarize test performance.

The model class is `AMB_ViT`. The main processing modules are `AbRankScorer`, `RobustMiniBagAggregator`, and `GuidedBlock`.

## 2. Environment and Dependencies

The script requires Python and these third-party packages:

```text
torch
numpy
pandas
h5py
scikit-learn
tqdm
matplotlib
seaborn
```

After installing a suitable PyTorch build, install the remaining dependencies:

```powershell
python -m pip install numpy pandas h5py scikit-learn tqdm matplotlib seaborn
```

Your PyTorch installation must support `torch.amp.GradScaler('cuda')`, `torch.amp.autocast('cuda')`, and the `weights_only` argument to `torch.load`. This directory has no dependency lockfile, so no unverified version combination is specified here.

The command-line default is CUDA preprocessing, which requires CUDA to be available in PyTorch. Use `--preprocess-device cpu` for CPU preprocessing. The training device is selected independently through `torch.cuda.is_available()`. Training AMP calls explicitly target CUDA, so CPU training may produce messages indicating that CUDA AMP is disabled.

## 3. Data Preparation

### 3.1 Dataset Configuration

Configure each dataset in `DATASET_SPECS` near the top of `train-main.py`:

- `h5_dir`: directory containing the target slides' `.h5` feature files.
- `csv_path`: path to the slide label table.
- `benign_h5_dir`: directory containing `.h5` files for the benign reference bank.
- `label_map`: mapping from CSV labels to integer class indices.
- `num_classes`: number of classes.

Built-in datasets are listed below. Label spelling and capitalization must match the configuration.

| Dataset argument | Class mapping | Default CSV path |
| --- | --- | --- |
| `YNC_OC` | `J: 0, N: 1, T: 2, Z: 3` | `data\YNC-OC.csv` |
| `UBC_OC` | `HGSC: 0, LGSC: 1, MUC: 2, CCOC: 3, ENOC: 4` | `data\UBC_OC_A.csv` |
| `TCGA_STAD` | `INT: 0, DIF: 1` | `data\TCGA_STAD.csv` |
| `TCGA_LUNG` | `LUSC: 0, LUAD: 1` | `data\TCGA_LUNG.csv` |
| `TCGA_BRCA` | `IDC: 0, ILC: 1` | `data\TCGA-BRCA.csv` |

The default dataset is `TCGA_BRCA`. Feature directories currently use machine-specific absolute paths on the `D:`, `E:`, and `F:` drives. Update them to your actual locations before running. Data paths and most training hyperparameters must be edited in the source configuration; the current CLI does not expose general overrides for them.

### 3.2 CSV Label Table

The training and testing CSV must contain the following columns, with one slide per row:

| Column | Meaning | Requirement |
| --- | --- | --- |
| `case_id` | Patient identifier | All slides from the same patient share this identifier |
| `slide_id` | Slide identifier | Unique; matches the HDF5 filename without the `.h5` extension |
| `label` | Slide class | A key in the selected dataset's `label_map` |

Example for `TCGA_BRCA`:

```csv
case_id,slide_id,label
patient_001,slide_001,IDC
patient_001,slide_002,IDC
patient_002,slide_003,ILC
```

This example illustrates the format only and is insufficient for five-fold training. Provide enough patients and samples per class. Prefer identifiers with a letter prefix to avoid losing leading zeros through automatic CSV type inference.

Split preparation checks for missing values, empty identifiers, and duplicate `slide_id` values. Preprocessing runs before these checks, so ensure the input table is complete, unique by slide, and free of unintended whitespace before execution.

If a patient has slides with different labels, the majority label is used only for patient-level stratification; the original slide labels remain the training targets. A tie for the majority label raises an error.

### 3.3 HDF5 Feature Files

Each target slide file must be stored in `h5_dir`, named `<slide_id>.h5`, and contain:

| HDF5 key | Shape | Description |
| --- | --- | --- |
| `features` | `[P, 1024]` | Features for P patches; the default feature dimension is 1024 |
| `coords` | `[P, 2]` | Two-dimensional patch coordinates, aligned row by row with the features |

Place benign reference files in `benign_h5_dir`. Bank construction uses only their `features`; these files do not require a label CSV or `coords`. Reference and target features should use the same extraction method and dimensionality.

Implementation details:

- Coordinates are min-max normalized independently within each slide.
- Patch scores are normalized within each slide and clipped to `[1e-4, 1.0]`.
- Each MiniBag contains 50 patches by default, with at most 256 MiniBags per slide.
- Slides with fewer than 50 patches are skipped. Remaining patches that cannot form a complete MiniBag are discarded.
- Grouping follows score order. Each MiniBag uses the normalized coordinates of its first patch.
- Missing HDF5 files, unknown labels, or unusable features may cause slides to be skipped. Preprocessing exceptions generate warnings; CUDA out-of-memory errors are re-raised.
- Training and testing include only slides with an existing cache file. Check preprocessing logs and the final number of eligible samples.

## 4. Usage

Run commands from the script directory to keep relative data paths and output locations consistent:

```powershell
Set-Location E:\1\ARSD-MIL
```

### 4.1 Preprocessing Only

Build or load the benign bank and generate MiniBag caches without training or preparing splits:

```powershell
python train-main.py --dataset TCGA_BRCA --mode preprocess --preprocess-device cuda
```

For CPU preprocessing:

```powershell
python train-main.py --dataset TCGA_BRCA --mode preprocess --preprocess-device cpu
```

### 4.2 Training

```powershell
python train-main.py --dataset TCGA_BRCA --mode train --seed 42
```

Training mode automatically performs preprocessing, patient-level splitting, five-fold training, and test evaluation. The default auxiliary distillation scheme is D, with weight `0.1`.

Specify the distillation scheme and weight:

```powershell
python train-main.py --dataset TCGA_BRCA --mode train --distill_type D --aux_weight 0.1
```

Use only the main classification loss:

```powershell
python train-main.py --dataset TCGA_BRCA --mode train --distill_type A
```

**Starting a new training run deletes `best_model_fold_*.pth`, `best_model_global.pth`, and `cv_fold_records.json` from the selected dataset's output directory. Back up experiment results you want to retain before rerunning training.** The memory bank, MiniBag caches, and split manifest are reused.

### 4.3 Testing Saved Models

```powershell
python train-main.py --dataset TCGA_BRCA --mode test --seed 42
```

Use the same data, seed, preprocessing version, and model architecture configuration as training. Fold-specific checkpoints and `cv_fold_records.json` are required. A standalone `best_model_global.pth` cannot support the current rotating held-out test protocol.

Test mode first checks the bank and caches, then reads the saved fold records. If complete test metrics are already present, it reuses those metrics and reevaluates the representative fold for plotting. If any available fold lacks test metrics, it reevaluates all available folds. Consequently, test mode does not recompute every fold's metrics on every invocation.

### 4.4 Creating a Separate Preprocessing Version

```powershell
python train-main.py --dataset TCGA_BRCA --mode preprocess --preprocess-device cuda --preprocess-tag gpu_v1
python train-main.py --dataset TCGA_BRCA --mode train --preprocess-device cuda --preprocess-tag gpu_v1
```

`--preprocess-tag` appends a suffix to the bank filename and cache directory, preserving existing versions. Model output directories and split files are shared across tags, so back up model results separately for different experiments. Tags accept letters, numbers, underscores, and hyphens only.

## 5. Command-Line Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `TCGA_BRCA` | Select a built-in dataset |
| `--mode` | `train` | `train`, `test`, or `preprocess` |
| `--preprocess-device` | `cuda` | `auto`, `cpu`, or `cuda`; controls preprocessing only |
| `--gpu-query-batch-size` | `2048` | Patches per GPU nearest-neighbor query batch |
| `--gpu-center-chunk-size` | `8192` | Prototypes per GPU distance-computation chunk |
| `--gpu-kmeans-steps` | `100` | GPU MiniBatch K-means update steps per initialization |
| `--preprocess-tag` | None | Version suffix for the bank and caches |
| `--seed` | `42` | Random seed and seed identifier in the split filename |
| `--distill_type` | `D` | Auxiliary distillation scheme: `A`, `B`, `C`, or `D` |
| `--aux_weight` | Configured value `0.1` when omitted | Auxiliary loss weight; values ≤ 0 disable the auxiliary loss |

`--distill_type` and `--aux_weight` use underscores; the other compound arguments use hyphens. GPU batch size, center chunk size, and K-means steps must be positive integers.

### Distillation Schemes

Every scheme includes slide-level cross-entropy. The auxiliary teacher target is the slide prediction probability, detached from the computation graph and used to supervise the MiniBag instance head.

| Scheme | Auxiliary loss |
| --- | --- |
| `A` | None; main classification cross-entropy only |
| `B` | Mean KL divergence over all sorted MiniBag logits |
| `C` | Mean KL divergence over the first `top_k_loss` MiniBag logits |
| `D` | Importance-weighted KL divergence over the first `top_k_loss` MiniBag logits |

The default is `top_k_loss = 20`. Scheme D uses each MiniBag's mean abnormality score as its weight and takes the mean of the weighted KL terms, without dividing by the sum of weights. MiniBags are zero-padded to equalize batch lengths; the auxiliary KL branch does not explicitly apply the padding mask.

## 6. Main Source Configuration

Edit these settings in `DATASET_SPECS` or `get_base_config()`; they are not exposed as current CLI arguments:

| Setting | Default | Purpose |
| --- | --- | --- |
| `feature_dim` | `1024` | Input patch feature dimension |
| `minibag_size` | `50` | Patches per MiniBag |
| `max_minibags` | `256` | Maximum MiniBags retained per slide |
| `max_memory_patches` | `50000` | Target bank size limit; all prototypes are used when fewer are available |
| `top_k_ratio` | `1.0` | Fraction of MiniBags selected by importance for the Transformer |
| `top_k_loss` | `20` | First K MiniBags used in schemes C/D |
| `dropout` | `0.3` | Training model input and Transformer dropout |
| `weight_decay` | `1e-3` | AdamW weight decay |
| `k_folds` | `5` | Number of outer cross-validation folds |
| `test_size` | `0.2` | Actually controls the validation fraction within each outer training partition |
| `batch_size` | `32` | WSIs per batch |
| `epochs` | `100` | Maximum epochs per fold |
| `lr` | `1e-4` | Initial learning rate |
| `patience` | `15` | Early-stopping threshold for epochs without strictly improved validation accuracy |
| `ablation_settings.use_agg` | `True` | Gated attention aggregation; disabling it uses mean aggregation |
| `ablation_settings.use_info` | `True` | Fuse coordinate and score information |
| `ablation_settings.depth` | `2` | Number of Transformer blocks |

The model constructor defaults to a hidden dimension of 256, four attention heads, and `min_keep = 16`. When proportional selection is enabled, the number retained is constrained by the maximum MiniBag count in the batch and `min_keep`.

The configuration dictionary's `head_dropout = 0.5` is currently not passed to the model constructor. The actual head dropout is the constructor default, `0.2`; changing that dictionary entry alone has no effect. Similarly, `n_neighbor = 9` initializes the CPU nearest-neighbor object, but scoring explicitly queries `n_neighbors=1`.

CPU preprocessing uses scikit-learn clustering and nearest-neighbor queries. GPU preprocessing uses custom chunked distance calculations and online MiniBatch K-means. GPU clustering initialization and updates differ from scikit-learn, so the two backends may produce different results.

## 7. Splitting, Model Selection, and Metrics

- Splits include only slides with existing caches and group slides by `case_id`, ensuring that training, validation, and testing patients are disjoint within each fold.
- The default outer five-fold scheme assigns every patient to exactly one held-out test fold. Approximately 20% of each fold's remaining patients form the validation set, giving overall patient proportions of roughly 64% training, 16% validation, and 20% testing. Actual proportions depend on sample counts.
- Multiple stratified split candidates are searched and selected using patient class proportions and subset sizes. Model test performance is not used to select splits.
- Each fold's checkpoint is selected by validation accuracy. Ties are resolved using the lower mean total training loss for that epoch. The recorded `val_loss` field actually stores this training loss.
- The early-stopping counter increases whenever validation accuracy does not strictly improve, including epochs that update the checkpoint solely through the loss tie-breaker.
- The optimizer is AdamW. The learning rate uses warmup for the first 10% of planned training steps, followed by cosine decay.
- Test reports include accuracy, Macro-F1, and Macro-AUC using one-vs-rest (OvR), with means and standard deviations across available folds. Metrics are computed per WSI; predictions are not aggregated by patient first.
- Confusion matrix and ROC plots come from a representative fold selected by validation accuracy, with recorded loss and fold number used as tie-breakers. They do not combine predictions from all five folds.
- Returned fields such as `cv_mean` summarize selected validation results across folds. Fields such as `acc`, `f1`, and `auc` summarize test results. Accuracy is stored as a percentage; F1 and AUC are stored on a 0–1 scale.
- Failed AUC calculations generate a warning and are recorded as `0.0`; inspect the logs when interpreting results.

The benign bank is built from the entire `benign_h5_dir` before cross-validation splitting and is shared across folds. For research use, verify how the reference cohort relates to the evaluation cohort. The code does not check whether benign reference patients overlap with evaluation patients.

## 8. Output Files

Example using `TCGA_BRCA`, default settings, and seed 42:

```text
ARSD-MIL/
├── train-main.py
├── README.md
├── ubiquity_aware_bank_benign_only_TCGA_BRCA.npy
├── data/
│   ├── TCGA-BRCA.csv
│   └── split/
│       └── TCGA_BRCA_outercv5_inner20_fullprop_seed42.csv
└── TCGA_BRCA/
    ├── processed_minibag_cache_v5_n50_len256/
    │   └── <slide_id>.pt
    ├── best_model_fold_1.pth
    ├── ...
    ├── best_model_fold_5.pth
    ├── best_model_global.pth
    ├── cv_fold_records.json
    ├── confusion_matrix_TCGA_BRCA.png
    └── roc_curve_TCGA_BRCA.png
```

This layout includes the CSV you must supply and files generated during execution. External HDF5 directories are omitted.

| File | Contents |
| --- | --- |
| Bank `.npy` | Benign reference prototype matrix |
| Slide cache `.pt` | `mb_raw_feats`, `mb_raw_scores`, `mb_coords`, and `label` |
| Split CSV | `fold, case_id, slide_id, label, role`, where role is `train`, `val`, or `test` |
| `best_model_fold_*.pth` | Each fold's validation-selected model `state_dict` |
| `best_model_global.pth` | Copy of the model selected across folds using validation results; not retrained on all data |
| `cv_fold_records.json` | Fold numbers, validation metrics, absolute checkpoint paths, split protocol, and test metrics |
| Two `.png` files | Representative fold's test confusion matrix and ROC curve at 300 dpi |

Bank, cache, and model output paths are relative to the current working directory. Split manifests are stored in `data/split` relative to the script's location. Checkpoints contain model weights only; optimizer and scheduler states are not saved, and the script has no resume-training entry point.

## 9. Cache Reuse and Troubleshooting

### Stale caches after changing data or settings

The script reuses banks and slide caches based only on file existence. It does not verify changes to source features, labels, seeds, or preprocessing parameters. After such changes, use a new `--preprocess-tag`, or back up and remove the corresponding old bank and caches. Keep preprocessing versions consistent between training and testing.

### Saved splits do not match the current data

The split manifest for the same dataset and seed is reused, with checks for slide/label coverage, patient separation, and rotating test assignments. Changes to eligible samples, labels, or split configuration may make it incompatible. Back up and regenerate the affected manifest after confirming the intended change. The `inner20` filename component is fixed text and does not automatically change when `test_size` is modified.

### CUDA is unavailable or preprocessing runs out of memory

The default `--preprocess-device cuda` raises an error when CUDA is unavailable. Use `cpu` or `auto` instead. For GPU preprocessing memory issues, reduce `--gpu-query-batch-size` and `--gpu-center-chunk-size`. For training memory issues, reduce `batch_size` or `max_minibags` in the source configuration. Regenerate caches when changing grouping settings.

### Slides are skipped or training produces no results

Check HDF5 filenames, the `features` and `coords` keys, feature dimensions, patch counts, and CSV labels. The pipeline returns immediately if no slides have valid caches. Too few patients in a class may cause stratification warnings or split failures.

### Testing cannot find checkpoints

`cv_fold_records.json` stores absolute checkpoint paths. Check them after moving the project. Missing models are skipped; if only some checkpoints are available, the summary covers only those folds. Inspect the reported fold list.

### Adding a dataset

Add its configuration to `DATASET_SPECS` and update the `--dataset` argument's `choices` in `main()`. Use consecutive class indices starting at 0, match `num_classes` to the mapping, and adjust `feature_dim` to the actual input features.

## 10. Verification Scope

This documentation is based on static inspection of the current `train-main.py`. Actual label tables, HDF5 features, and dependency environment specifications were not supplied in this directory, so training and dataset performance have not been validated.
