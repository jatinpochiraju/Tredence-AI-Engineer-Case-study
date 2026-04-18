# Tredence AI Engineering Case Study

This repository contains a PyTorch implementation of **The Self-Pruning Neural Network** case study.

## Files

- `self_pruning_cifar10.py` - complete source code for `PrunableLinear`, the CIFAR-10 model, training, evaluation, sparsity reporting, and gate histogram generation.
- `REPORT.md` - short explanation of the method and result table format.
- `requirements.txt` - Python dependencies.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 self_pruning_cifar10.py --smoke-test
python3 self_pruning_cifar10.py --epochs 10 --lambdas 0 5e-5 5e-4 --device auto --num-workers 0 --output-dir outputs_tuned
```

The experiment writes outputs to the selected output directory, including:

- `results.csv`
- `results.json`
- `results_table.md`
- `model_lambda_*.pt`
- `gate_histogram_lambda_*.png`

## Notes

The script uses a feed-forward MLP with custom `PrunableLinear` layers, as requested in the assignment. For quick iteration, use `--train-subset` and `--test-subset`; for final reporting, run on the full CIFAR-10 train and test sets.

The submitted run artifacts are in `results/`. Large checkpoints are left in the generated output directory and are not required for reviewing the source code and report.
