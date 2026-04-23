````markdown
# 🚀 Tredence AI Engineering Case Study  
## Self-Pruning Neural Network (PyTorch)

This project implements a **self-pruning neural network** that learns to remove unnecessary weights during training using learnable gates and L1 regularization.

---

## 🧠 Overview

- Each weight is paired with a learnable **gate**
- Gates control whether a connection is active
- The model learns to **suppress weak connections automatically**

---

## ⚙️ Method

```python
pruned_weights = weight * sigmoid(gate_scores)
````

**Loss Function:**

[
\text{Loss} = \text{CrossEntropy} + \lambda \cdot \sum \text{sigmoid}(gate_scores)
]

* CrossEntropy → accuracy
* L1 on gates → sparsity
* λ → controls pruning strength

---

## 🏗️ Architecture

* Dataset: CIFAR-10
* Model: MLP with custom `PrunableLinear` layers
* Activations: ReLU + Dropout

---

## 📁 Structure

```
self_pruning_cifar10.py   # Model + training + evaluation
REPORT.md                 # Explanation + results
requirements.txt
results/                  # Outputs
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python self_pruning_cifar10.py --smoke-test
python self_pruning_cifar10.py --epochs 10 --lambdas 0 5e-5 5e-4
```

---

## 📊 Results

| Lambda | Accuracy   | Sparsity   |
| ------ | ---------- | ---------- |
| 0      | 34.61%     | 0.00%      |
| 5e-5   | **46.85%** | **91.32%** |
| 5e-4   | 31.88%     | 87.28%     |

---

## 📈 Key Insights

* λ = 0 → no pruning (baseline)
* Moderate λ → best balance (high sparsity + better accuracy)
* High λ → over-pruning → accuracy drop

👉 Higher λ increases pruning pressure; too large removes useful connections early.

---

## 📊 Gate Distribution

![Gate Distribution](results/gate_histogram_lambda_5e-05.png)

* Spike near 0 → pruned weights
* Remaining values → important connections

---

## ⚡ Highlights

* Custom `PrunableLinear` layer
* Proper gradient flow (weights + gates)
* Separate LR for gate parameters
* Checkpointing + reproducibility
* Auto-generated results (CSV, JSON, plots)

---

## 🔮 Future Work

* Use CNN instead of MLP
* Structured pruning
* Adaptive thresholds

---

## 🏁 Summary

The model **learns sparsity during training**, achieving ~90% pruning while maintaining reasonable accuracy.

```
```
