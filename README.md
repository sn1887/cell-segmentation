# TUM AI E-Lab Cell Segmentation

PyTorch Lightning + Hydra training scaffold for the Kaggle semantic segmentation challenge in `data/tum-ai-e-lab-cell-segmentation`.

The default model is `segmentation_models_pytorch` FPN with a ResNet-34 encoder, two semantic classes, weighted cross entropy, and an optional boundary-aware term based on the signed distance map formulation from [`LIVIAETS/boundary-loss`](https://github.com/LIVIAETS/boundary-loss).

## Layout

```text
configs/                 Hydra configs
notebooks/               Kaggle-friendly notebook entrypoint
src/cellseg_challenge/   Dataset, losses, model, runner, submission utilities
train.py                 Hydra CLI training entrypoint
predict.py               Hydra CLI checkpoint-to-submission entrypoint
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

On Kaggle, the notebook installs the same requirements and then calls the Python runner directly.

## Train

```bash
python train.py
```

Useful overrides:

```bash
python train.py model.arch=FPN model.encoder_name=efficientnet-b3 data.batch_size=16
python train.py model.encoder_weights=null
python train.py loss.class_weights=auto loss.boundary_weight=0.02
```

The dataset foreground ratio averages about `0.1603`, so `loss.class_weights=auto` creates inverse-frequency CE weights close to `[0.595, 3.120]` for background and foreground.

## Predict

```bash
python predict.py predict.ckpt_path=/path/to/best.ckpt
```

By default this writes `outputs/submission.csv` using `Archive/test/sample_submission.csv`. To use the active `new_sample_submission.csv` order:

```bash
python predict.py \
  predict.ckpt_path=/path/to/best.ckpt \
  data.sample_submission=../new_sample_submission.csv
```

The inference dataset resolves each submission ID as a zero-padded JPG in the configured image folders. The default folders are `test/images` first and `train/images` as a fallback, which handles both sample CSVs present in this workspace.

## Notebook

Open `notebooks/train_kaggle.ipynb`. It composes the Hydra config inside Python, trains a model, and writes a submission without needing shell-only Hydra commands.

