# CyberSentinel-LLM

**A Large Language Model-Driven Autonomous Framework for Intelligent Cyber Threat Detection and Response**

> Published in: *Computers, Materials & Continua (CMC)*, 2025  
> DOI: 10.32604/cmc.2025.0xxxxx

---

## Abstract

CyberSentinel-LLM is an autonomous cyber threat detection and response framework that integrates a LoRA-adapted LLaMA-3-8B backbone with a four-agent multi-agent reinforcement learning system. The framework achieves **96.8% accuracy, 96.5% F1-score, and 98.7% AUC-ROC** on the HDFS benchmark at **45 ms inference latency**, outperforming ten state-of-the-art baselines.

---

## Architecture

```
Raw Logs / Network Traffic
        │
        ▼
┌─────────────────────────────────┐
│  1. Multi-Modal Log Parser       │  Drain3 + Semantic Embedding
│     Feature Extraction           │  Eq.(1)–(5)
└──────────────┬──────────────────┘
               │ z_i ∈ ℝ^{d_s + d_t}
               ▼
┌─────────────────────────────────┐
│  2. Temporal Transformer Encoder │  L=6, H=8, d=512, w=256
│     Causal + Windowed Attention  │  Eq.(6)–(11)
└──────────────┬──────────────────┘
               │ Z^L
               ▼
┌─────────────────────────────────┐
│  3. LLM Threat Intelligence      │  LLaMA-3-8B + LoRA (r=16)
│     Engine                       │  Eq.(12)–(17)
└──────────────┬──────────────────┘
               │ h^LLM, logits
        ┌──────┴───────┐
        ▼              ▼
┌──────────────┐  ┌──────────────────────────┐
│  4. Agents   │  │  5. DRL Response          │
│  Detection   │  │     Orchestrator (PPO)    │
│  Classify    │  │     6 Response Actions    │
│  Forensic    │  │     Eq.(22)–(25)          │
└──────────────┘  └──────────────────────────┘
```

---

## Results

### Table 3 — Performance Comparison on HDFS and BGL Datasets

| Method | HDFS Acc | HDFS F1 | BGL Acc | BGL F1 | Latency |
|--------|----------|---------|---------|--------|---------|
| LogFiT [1] | 93.2 | 92.8 | 91.8 | 91.8 | 78 ms |
| DeepLog [7] | 88.7 | 87.5 | 86.4 | 86.4 | 120 ms |
| LogBERT [2] | 90.1 | 90.1 | 88.5 | 88.5 | 95 ms |
| TLA-Net [4] | 91.2 | 91.1 | 89.8 | 89.8 | — |
| MADMM [13] | 91.5 | 91.4 | 90.2 | 90.2 | 135 ms |
| LogRESP [12] | 91.8 | 91.8 | 89.8 | 89.8 | 88 ms |
| **CyberSentinel-LLM (Ours)** | **96.8** | **96.5** | **95.5** | **95.2** | **45 ms** |

### Table 4 — Per-Class Performance (HDFS)

| Threat Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 98.2 | 99.4 | 98.8 | 4,880 |
| Malware | 97.5 | 98.0 | 97.7 | 1,000 |
| DDoS | 96.8 | 97.9 | 97.3 | 985 |
| Phishing | 96.0 | 97.4 | 96.7 | 967 |
| Insider Threat | 95.5 | 97.2 | 96.4 | 911 |
| APT | 94.8 | 98.9 | 96.8 | 887 |
| **Macro Avg** | **96.5** | **98.1** | **97.3** | 9,630 |

---

## Dataset

| | HDFS | BGL |
|---|---|---|
| **Name** | Hadoop Distributed File System Logs | BlueGene/L Supercomputer Logs |
| **Download** | [LogHub on GitHub](https://github.com/logpai/loghub/tree/master/HDFS) | [LogHub on GitHub](https://github.com/logpai/loghub/tree/master/BGL) |
| **Zenodo** | [DOI 10.5281/zenodo.3227177](https://doi.org/10.5281/zenodo.3227177) | Same |
| **Total Logs** | 11,175,629 | 4,747,963 |
| **Anomaly %** | 2.93% | 7.34% |
| **Split** | 80/10/10 | 80/10/10 |

**Place downloaded data in:**
```
data/
├── HDFS/
│   ├── HDFS.log
│   └── anomaly_label.csv
└── BGL/
    └── BGL.log
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-org/CyberSentinel-LLM.git
cd CyberSentinel-LLM

# 2. Create environment
conda create -n cybersentinel python=3.10
conda activate cybersentinel

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install CUDA-enabled PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Train
```bash
# Quick test with synthetic data (no download needed)
python train.py --synthetic --epochs 10

# Train on real HDFS data
python train.py --real_data --dataset HDFS --epochs 50

# Train on BGL
python train.py --real_data --dataset BGL --epochs 50
```

### Evaluate
```bash
python evaluate.py --dataset HDFS
python evaluate.py --dataset BGL --checkpoint checkpoints/cybersentinel_best.pt
```

### Inference
```bash
# Demo mode (synthetic)
python inference.py --demo

# Single log file
python inference.py --input data/HDFS/HDFS.log
```

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Folder Structure

```
CyberSentinel_LLM_Implementation/
│
├── figures/                  ← All figures extracted from paper
│   ├── fig01_ml_vs_proposed_comparison.png
│   ├── fig02_system_architecture.png
│   └── ...
│
├── data/                     ← Place downloaded datasets here
├── checkpoints/              ← Saved model weights
├── results/                  ← Evaluation outputs
├── logs/                     ← TensorBoard logs
│
├── notebooks/
│   └── demo.ipynb
│
├── config.py                 ← All hyperparameters
├── dataset.py                ← Data loading & preprocessing
├── model.py                  ← Full model architecture
├── train.py                  ← Training loop
├── evaluate.py               ← Evaluation & metrics
├── inference.py              ← Single-sample inference
├── utils.py                  ← Helper functions
└── requirements.txt
```

---

## Key Hyperparameters (Section 4.1.5)

| Parameter | Value |
|---|---|
| LLM Backbone | LLaMA-3-8B |
| LoRA Rank r | 16 |
| LoRA Alpha α | 32 |
| Transformer Layers L | 6 |
| Attention Heads H | 8 |
| Hidden Dim d_model | 512 |
| Window Size w | 256 |
| Optimizer | AdamW |
| Peak LR | 5e-5 |
| Batch Size | 32 (×4 accum = 128 eff.) |
| Epochs | 50 |
| Mixed Precision | FP16 |
| DRL Algorithm | PPO |
| PPO ε_clip | 0.2 |
| Discount γ | 0.99 |

---

## Citation

```bibtex
@article{cybersentinel2025,
  title   = {A Large Language Model-Driven Autonomous Framework for
             Intelligent Cyber Threat Detection and Response},
  journal = {Computers, Materials \& Continua},
  year    = {2025},
  doi     = {10.32604/cmc.2025.0xxxxx},
}
```

---

## License

This implementation is released under the MIT License.  
The HDFS and BGL datasets are distributed under the Apache 2.0 License  
via the [LogHub repository](https://github.com/logpai/loghub).
