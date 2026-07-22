# Running `02_train.ipynb` as a Script

`run_train_round.py` is the script version of the training part in
`notebooks/02_train.ipynb`. It loads the initialization checkpoint from
`01_Initialize`, runs the selected training stages, optionally performs
in-batch merge and cross-batch align, and writes a checkpoint that can be used
as the next round's `init_path`.

## Inputs

Configure inputs in `configs/train_round.yaml` or override them from the
command line.

- `input.init_checkpoint`: checkpoint from initialization or a previous
  merge/align round. It must contain `P_align` or `P_init`, `align_masks`,
  `mapping_init`, and `gene_names`.
- `input.training_data_dir`: directory of ordered `*.h5ad` files created by
  `01_Initialize`. This is the preferred input for multimodal training.
- `input.init_adata` and `input.atac_da_dir`: fallback inputs used only when
  `training_data_dir` has no `*.h5ad` files.

## Outputs

By default outputs go to `output.output_dir`.

- `train_P_<round>.pt`: checkpoint after `learn_P`.
- `train_mapping_<round>.pt`: checkpoint after `learn_mapping`.
- `train_M_<round>.pt`: checkpoint after `learn_M`.
- `merge_round_<round>.pt`: merged/aligned checkpoint for the next round.

Each stage checkpoint also carries the initialization context needed for
inspection or resume, including `align_masks`, `mapping_init`, `gene_names`,
batch metadata, and `training_data_dir`.
- `training_round_<round>_annotated.h5ad`: combined AnnData with `pred`,
  `pred_merge`, and `pred_align` when available.
- `training_round_<round>_summary.json`: paths and batch metadata for the run.

## Common Commands

Run the full notebook training workflow:

```bash
python ResonanSC/run_train_round.py \
  --config ResonanSC/configs/train_round.yaml
```

Run in `screen`:

```bash
screen -S resonansc_train
python ResonanSC/run_train_round.py \
  --config ResonanSC/configs/train_round.yaml \
  --round 1
```

Detach with `Ctrl-a d`, reattach with:

```bash
screen -r resonansc_train
```

Start a second round from the previous merge/align checkpoint:

```bash
python ResonanSC/run_train_round.py \
  --config ResonanSC/configs/train_round.yaml \
  --init-checkpoint ResonanSC/outputs/result1/1/merge_round_1.pt \
  --round 2
```

Run only selected stages:

```bash
python ResonanSC/run_train_round.py \
  --config ResonanSC/configs/train_round.yaml \
  --stages learn_P learn_mapping \
  --epochs-P 100 \
  --epochs-mapping 200 \
  --no-merge-align
```

## Notebook Handoff

To continue in `02_train.ipynb`, set the notebook's `init_path` to the script
output:

```python
init_path = "outputs/result1/1/merge_round_1.pt"
```

The script writes the same fields used by the notebook training cell:
`P_align`, `M_align`, `align_masks`, `mapping_init`, `gene_names`, `batch_key`,
and `training_data_dir`.
