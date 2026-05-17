"""
config.py — CyberSentinel-LLM Configuration
============================================
All hyperparameters and settings are taken directly from the paper:
  "A Large Language Model-Driven Autonomous Framework for
   Intelligent Cyber Threat Detection and Response"
  Published in: Computers, Materials & Continua (CMC), 2025

Section references point to the paper sections where values are defined.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
RESULTS_DIR     = BASE_DIR / "results"
LOG_DIR         = BASE_DIR / "logs"

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS  — Section 4.1
# ─────────────────────────────────────────────────────────────────────────────
DATASET = {
    "HDFS": {
        "name":         "HDFS Log Dataset",
        "url":          "https://github.com/logpai/loghub/tree/master/HDFS",
        "download_url": "https://zenodo.org/record/3227177/files/HDFS_1.tar.gz",
        "local_path":   DATA_DIR / "HDFS",
        "total_logs":   11_175_629,
        "anomaly_ratio": 0.0293,            # 2.93%
        "num_templates": 30,
        "window_type":  "session",          # block-level sessions
        "split":        (0.80, 0.10, 0.10), # train/val/test
    },
    "BGL": {
        "name":         "BlueGene/L Log Dataset",
        "url":          "https://github.com/logpai/loghub/tree/master/BGL",
        "download_url": "https://zenodo.org/record/3227177/files/BGL.tar.gz",
        "local_path":   DATA_DIR / "BGL",
        "total_logs":   4_747_963,
        "anomaly_ratio": 0.0734,            # 7.34%
        "num_templates": 376,
        "window_type":  "sliding",          # 1h window, 50% overlap
        "split":        (0.80, 0.10, 0.10),
    },
    # Supplementary datasets — Section 4.6
    "CICIDS2017": {
        "name":         "CICIDS2017 Network Intrusion Dataset",
        "url":          "https://www.unb.ca/cic/datasets/ids-2017.html",
        "download_url": "https://www.kaggle.com/api/v1/datasets/download/cicdataset/cicids2017",
        "local_path":   DATA_DIR / "CICIDS2017",
        "total_flows":  2_830_743,
        "num_features": 78,
    },
    "NSL_KDD": {
        "name":         "NSL-KDD Intrusion Detection Dataset",
        "url":          "https://www.unb.ca/cic/datasets/nsl.html",
        "download_url": "https://www.kaggle.com/api/v1/datasets/download/hassan06/nslkdd",
        "local_path":   DATA_DIR / "NSL_KDD",
        "train_size":   125_973,
        "test_size":    22_544,
        "num_features": 41,
    },
}

ACTIVE_DATASET = "HDFS"   # change to "BGL" for BGL experiments

# ─────────────────────────────────────────────────────────────────────────────
# LOG PARSING — Section 3.2
# ─────────────────────────────────────────────────────────────────────────────
LOG_PARSING = {
    "parser":            "Drain3",
    "similarity_thresh": 0.4,          # Section 4.1.6 — Data Preprocessing
    "depth":             4,
    "max_seq_len":       256,          # max log entries per session (padded/truncated)
    "sliding_window_h":  1,            # BGL: 1-hour window
    "sliding_overlap":   0.5,          # BGL: 50% overlap
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM BACKBONE — Section 3.3
# ─────────────────────────────────────────────────────────────────────────────
LLM = {
    "model_name":    "meta-llama/Meta-Llama-3-8B",
    "num_params":    8_000_000_000,     # 8B parameters
    "hidden_dim":    4096,
    "num_layers":    32,
    "num_heads":     32,
    "vocab_size":    128_256,
}

# ─────────────────────────────────────────────────────────────────────────────
# LoRA CONFIGURATION — Section 3.3 & 4.1.5
# ─────────────────────────────────────────────────────────────────────────────
LORA = {
    "rank":          16,               # r = 16 (HDFS optimal)
    "alpha":         32,               # α = 32
    "dropout":       0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias":          "none",
}

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRANSFORMER ENCODER — Section 3.2 & 4.1.5
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMER = {
    "num_layers":    6,                # L = 6
    "num_heads":     8,                # H = 8
    "hidden_dim":    512,              # d_model = 512
    "window_size":   256,              # w = 256 (windowed attention)
    "ffn_dim":       2048,             # 4× hidden_dim
    "dropout":       0.1,
    "activation":    "gelu",
}

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-AGENT SYSTEM — Section 3.4
# ─────────────────────────────────────────────────────────────────────────────
AGENTS = {
    "num_agents":    4,
    "names":         ["Detection", "Classification", "Response", "Forensic"],
    "threshold":     0.85,             # detection confidence threshold
    "memory_size":   1000,             # shared memory buffer size
}

# ─────────────────────────────────────────────────────────────────────────────
# DRL RESPONSE ORCHESTRATOR — Section 3.5
# ─────────────────────────────────────────────────────────────────────────────
DRL = {
    "algorithm":     "PPO",
    "gamma":         0.99,             # discount factor γ
    "epsilon_clip":  0.2,              # ε_clip for PPO
    "lambda_gae":    0.95,             # λ_GAE
    "ppo_epochs":    3,                # K PPO epochs per update
    "action_space":  6,                # 6 response actions
    "reward_weights": {
        "alpha1":    1.0,              # R_detect weight
        "alpha2":    0.8,              # R_response weight
        "alpha3":    0.3,              # C_penalty weight
    },
    "reward_tp":     1.0,
    "reward_fn":    -0.5,
    "reward_contain": 0.8,
    "penalty_fp":   -0.3,
    "penalty_action": -0.1,
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — Section 4.1.5
# ─────────────────────────────────────────────────────────────────────────────
TRAINING = {
    "num_epochs":        50,
    "batch_size":        32,
    "grad_accum_steps":  4,            # effective batch = 128
    "optimizer":         "AdamW",
    "betas":             (0.9, 0.999),
    "weight_decay":      0.01,
    "lr_schedule":       "cosine_annealing",
    "peak_lr":           5e-5,
    "warmup_steps":      500,
    "total_steps":       10_000,
    "mixed_precision":   "fp16",
    "seed":              42,
    "num_workers":       4,
    "pin_memory":        True,
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTRASTIVE LOSS — Section 3.3, Eq.(17)
# ─────────────────────────────────────────────────────────────────────────────
LOSS = {
    "temperature":   0.1,              # τ
    "lambda_contrastive": 0.1,        # λ weight for contrastive term
}

# ─────────────────────────────────────────────────────────────────────────────
# THREAT CLASSES — Section 4.1.1
# ─────────────────────────────────────────────────────────────────────────────
THREAT_CLASSES = ["Normal", "Malware", "DDoS", "Phishing", "Insider Threat", "APT"]
NUM_CLASSES     = len(THREAT_CLASSES)

# ─────────────────────────────────────────────────────────────────────────────
# SEMI-SUPERVISED / FEW-SHOT — Section 4.8
# ─────────────────────────────────────────────────────────────────────────────
SEMI_SUP = {
    "pseudo_label_threshold": 0.9,     # τ = 0.9
    "few_shot_per_class":     10,
    "synthetic_per_class":    100,
}

# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE & SOFTWARE
# ─────────────────────────────────────────────────────────────────────────────
ENV = {
    "gpu":      "NVIDIA A100 80GB",
    "num_gpus": 4,
    "cuda":     "12.1",
    "python":   "3.10.12",
    "torch":    "2.1.0",
    "transformers": "4.36.0",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTS & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT = {
    "best_model":  CHECKPOINT_DIR / "cybersentinel_best.pt",
    "last_model":  CHECKPOINT_DIR / "cybersentinel_last.pt",
    "drl_policy":  CHECKPOINT_DIR / "drl_policy.pt",
}

RESULTS = {
    "confusion_matrix": RESULTS_DIR / "confusion_matrix.png",
    "roc_curve":        RESULTS_DIR / "roc_curve.png",
    "metrics_table":    RESULTS_DIR / "metrics_table.csv",
    "forensic_report":  RESULTS_DIR / "forensic_report.txt",
}
