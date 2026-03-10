# TAS-EGNN: Task-Aware Spectral Ego-Graph Neural Network

TAS-EGNN is a **fast coreset selection** framework for Graph Neural Networks (GNNs).  
It scores nodes inside lightweight **ego-graphs** using three complementary signals:

1) **Local structural complexity** (structural diversity)  
2) **Predictive uncertainty** (task difficulty)  
3) **Supervised error** (misclassification feedback)

A **greedy coverage** step then removes redundancy across selected ego-graphs.  
TAS-EGNN avoids heavy global operations and expensive optimization routines, delivering strong performance at small coreset ratios with **low runtime and memory**.

---

## ✨ Highlights
- **Task-aware** coreset selection combining structure + prediction signals  
- **Local computations only** (ego-graph based; scalable to large graphs)  
- Works across **citation**, **social**, **product**, and **transaction-fraud** graphs  
- Reports **Accuracy** (benchmark graphs) and **PR-AUC** (fraud graphs), plus runtime & peak GPU memory  
- **Backbone-agnostic**: compatible with common GNNs such as GCN and GraphSAGE

---

## 📦 Requirements

Main dependencies:

- Python ≥ 3.9  
- PyTorch (GPU optional)  
- PyTorch Geometric (PyG)  
- numpy  
- scipy  
- pandas  
- scikit-learn  
- tqdm

## 🔧 Installation

Clone the repository and install the environment:

```bash
git clone https://github.com/M-Allaoui/TAS-EGNN.git
cd TAS-EGNN

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

(Alternatively, you can use conda)

```bash
conda create -n tas-egnn python=3.9
conda activate tas-egnn
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
TAS-EGNN/
├── TAS-EGNN.py
├── datasets.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

### Benchmark Graph Datasets
Automatically loaded:
- Cora
- Citeseer

These tabular datasets are automatically converted into graphs using a k-nearest-neighbor graph construction, but if you use datasets as CSV files, please place them in a local folder, update the path inside `datasets.py` if needed.

---

## 🚀 Running TAS-EGNN

### Example: Cora / Citeseer
```bash
python TAS-EGNN.py --dataset Cora --model GraphSAGE --ratio 0.25
```
---

## 🔁 Reproducibility

For stable results:

- Run multiple seeds (recommended: 5 runs)
- Report mean ± standard deviation
- Keep dataset splits unchanged
- Use the same dependency versions listed in `requirements.txt`

---

## 📌 Output

The program reports:

- Accuracy
- F1 score
- PR-AUC (fraud datasets)
- Runtime
- Peak GPU memory (if CUDA is available)

---

## 📄 Citation

If you use this repository in your research, please cite:

```@inproceedings{
allaoui2026tasegnn,
title={{TAS}-{EGNN}: Task-Aware Spectral Ego-Graphs for Efficient {GNN}s-Based Classification},
author={Mebarka Allaoui and Rachid Hedjam and Sonia Gupta},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=C6GP4NM1FP}
}```

---
