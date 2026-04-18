# Self-Pruning Neural Network Report

## Approach

The model replaces every dense layer with `PrunableLinear`, a custom PyTorch module that owns three trainable tensors: `weight`, `bias`, and `gate_scores`. During the forward pass, each `gate_score` is transformed with a sigmoid to produce a gate in `(0, 1)`. The effective layer weights are:

```text
pruned_weights = weight * sigmoid(gate_scores)
```

The optimizer updates both the ordinary weights and the gate scores through normal backpropagation.

## Why L1 on Gates Encourages Sparsity

Cross-entropy alone rewards predictive accuracy, so it has no reason to deactivate connections. The training objective therefore adds an L1 penalty over all gate values:

```text
Total Loss = CrossEntropyLoss + lambda * sum(sigmoid(gate_scores))
```

Because the gates are non-negative, the L1 term is simply the sum of gate values. This creates constant pressure for each gate to shrink. Connections that are useful for classification can stay open because they reduce cross-entropy enough to justify their penalty; weak or redundant connections are pushed toward zero. Since sigmoid gates rarely become mathematically equal to zero, sparsity is reported with a small threshold of `1e-2`, which treats near-closed gates as pruned.

## How to Run

Quick gradient-flow check:

```bash
python3 self_pruning_cifar10.py --smoke-test
```

Full experiment used for the submitted results:

```bash
python3 self_pruning_cifar10.py --epochs 10 --lambdas 0 5e-5 5e-4 --device mps --num-workers 0 --output-dir outputs_tuned
```

Faster local trial:

```bash
python3 self_pruning_cifar10.py --epochs 2 --train-subset 5000 --test-subset 1000 --lambdas 0 1e-6 5e-6
```

The script writes checkpoints, gate histograms, and result tables to the selected output directory. The lightweight submission artifacts are copied into `results/`.

## Results

| Lambda | Test Accuracy | Sparsity Level (%) |
|---:|---:|---:|
| 0 | 34.61% | 0.00% |
| 5e-05 | 46.85% | 91.32% |
| 0.0005 | 31.88% | 87.28% |

## Expected Trade-off

The `lambda=0` model keeps all gates active, giving the dense baseline behavior. The moderate `5e-5` run produced the best accuracy and still pruned more than 91% of the gated weights. The stronger `5e-4` run also pruned aggressively, but the accuracy dropped, which shows the expected sparsity-vs-accuracy trade-off.

## Gate Distribution Plot

For reported accuracy, gates below the sparsity threshold are hard-masked to zero during evaluation. The best gate distribution plot is included here:

```text
results/gate_histogram_best_lambda_5e-05.png
```

A successful self-pruning run should show many gates accumulated near zero and a smaller group of gates away from zero, representing retained connections.
