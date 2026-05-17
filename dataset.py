"""
dataset.py — CyberSentinel-LLM Data Pipeline
=============================================
Handles downloading, parsing, preprocessing, and loading the HDFS and BGL
log datasets from the LogHub repository.

References:
  - Section 4.1   : Experimental Setup / Datasets
  - Section 4.1.6 : Data Preprocessing
  - Eq.(1)–(5)    : Feature extraction and multi-modal fusion
"""

import os
import re
import json
import gzip
import tarfile
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_DIR, ACTIVE_DATASET, DATASET, LOG_PARSING,
    TRAINING, THREAT_CLASSES, NUM_CLASSES
)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def download_loghub(dataset_name: str = "HDFS"):
    """
    Download a LogHub dataset. Official mirror via loghub Zenodo.
    Dataset page: https://github.com/logpai/loghub
    """
    cfg  = DATASET[dataset_name]
    dest = Path(cfg["local_path"])
    dest.mkdir(parents=True, exist_ok=True)

    raw_file = dest / f"{dataset_name}.log"
    if raw_file.exists():
        print(f"[dataset] {dataset_name} already downloaded at {raw_file}")
        return str(raw_file)

    url = cfg["download_url"]
    archive = dest / Path(url).name
    print(f"[dataset] Downloading {dataset_name} from {url} …")
    urllib.request.urlretrieve(url, archive, reporthook=_progress)
    print()

    print(f"[dataset] Extracting {archive} …")
    if str(archive).endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    elif str(archive).endswith(".gz"):
        out = dest / archive.stem
        with gzip.open(archive, "rb") as src, open(out, "wb") as dst:
            dst.write(src.read())

    print(f"[dataset] Done → {dest}")
    return str(dest)


def _progress(block, block_size, total):
    downloaded = block * block_size
    pct = min(downloaded / total * 100, 100) if total > 0 else 0
    print(f"\r  {pct:5.1f}%  {downloaded//1024//1024} MB", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DRAIN-BASED LOG PARSER  — Section 3.2, Eq.(2)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleDrain:
    """
    Simplified Drain log parser (Section 3.2).
    Full Drain3: pip install drain3
    This implementation covers the core algorithm for offline use.
    """
    WILDCARD = "<*>"

    def __init__(self, sim_thresh: float = 0.4, depth: int = 4):
        self.sim_thresh = sim_thresh
        self.depth      = depth
        self.id_map: Dict[str, int] = {}
        self.templates: List[List[str]] = []

    def parse_line(self, line: str) -> Tuple[int, List[str]]:
        tokens = line.strip().split()
        # Remove timestamp / severity prefix (first 3 tokens for HDFS/BGL)
        msg_tokens = tokens[3:] if len(tokens) > 3 else tokens

        template_id = self._match(msg_tokens)
        return template_id, msg_tokens

    def _match(self, tokens: List[str]) -> int:
        key = tuple(tokens[:self.depth])
        candidates = [
            (i, t) for i, t in enumerate(self.templates)
            if len(t) == len(tokens)
        ]
        best_id, best_sim = -1, -1.0
        for idx, tmpl in candidates:
            sim = self._similarity(tokens, tmpl)
            if sim > best_sim:
                best_sim, best_id = sim, idx
        if best_sim >= self.sim_thresh:
            self._update_template(best_id, tokens)
            return best_id
        # New template
        self.templates.append(list(tokens))
        return len(self.templates) - 1

    def _similarity(self, a: List[str], b: List[str]) -> float:
        if len(a) != len(b):
            return 0.0
        return sum(1 for x, y in zip(a, b) if x == y or y == self.WILDCARD) / len(a)

    def _update_template(self, idx: int, tokens: List[str]):
        tmpl = self.templates[idx]
        self.templates[idx] = [
            t if t == tok else self.WILDCARD
            for t, tok in zip(tmpl, tokens)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  — Section 3.2, Eq.(1)–(5)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_encoding(timestamps: np.ndarray, d_t: int = 64) -> np.ndarray:
    """
    ψ_temp(t_i) = [sin(t_i/ω_k), cos(t_i/ω_k)]   — Eq.(3)
    ω_k = 10000^(2k/d_t)
    """
    t   = timestamps[:, None]                              # (N, 1)
    k   = np.arange(d_t // 2)[None, :]                    # (1, d_t/2)
    w   = np.power(10000, 2 * k / d_t)                    # (1, d_t/2)
    enc = np.concatenate([np.sin(t / w), np.cos(t / w)], axis=-1)  # (N, d_t)
    return enc.astype(np.float32)


def event_embedding(template_ids: np.ndarray, num_templates: int, d_s: int = 64) -> np.ndarray:
    """
    ϕ_sem(m_i) — simple one-hot → linear projection approximation.
    Full version uses LLM tokenizer embeddings (loaded in model.py).
    """
    one_hot = np.zeros((len(template_ids), num_templates), dtype=np.float32)
    for i, tid in enumerate(template_ids):
        if 0 <= tid < num_templates:
            one_hot[i, tid] = 1.0
    # Reduce to d_s dims via random projection (seeded for reproducibility)
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1 / np.sqrt(d_s), (num_templates, d_s)).astype(np.float32)
    return one_hot @ proj  # (N, d_s)


# ─────────────────────────────────────────────────────────────────────────────
# HDFS PREPROCESSOR  — Section 4.1.1, 4.1.6
# ─────────────────────────────────────────────────────────────────────────────

class HDFSPreprocessor:
    """
    Groups HDFS logs into block-level sessions.
    Labels: 0=Normal, 1=Malware, 2=DDoS, 3=Phishing, 4=Insider Threat, 5=APT
    Re-labeling uses MITRE ATT&CK pattern matching (Section 4.1.1).
    """

    BLOCK_RE  = re.compile(r"blk_-?\d+")
    AUTH_RE   = re.compile(r"(login|auth|ssh|credential)", re.I)
    LATERAL_RE = re.compile(r"(replicate|replac|remote)", re.I)
    DDOS_RE   = re.compile(r"(request|packet|flood|timeout)", re.I)
    EXFIL_RE  = re.compile(r"(copy|transfer|upload|exfil|download)", re.I)

    def __init__(self, max_seq_len: int = 256, num_templates: int = 30):
        self.max_seq_len   = max_seq_len
        self.num_templates = num_templates
        self.parser        = SimpleDrain(sim_thresh=LOG_PARSING["similarity_thresh"],
                                         depth=LOG_PARSING["depth"])

    def parse_log_file(self, log_path: str) -> Dict[str, dict]:
        """Return dict: block_id → {template_ids, timestamps, raw_lines}"""
        sessions: Dict[str, dict] = defaultdict(
            lambda: {"template_ids": [], "timestamps": [], "lines": []}
        )
        with open(log_path, "r", errors="replace") as fh:
            for line in fh:
                blocks = self.BLOCK_RE.findall(line)
                tid, _ = self.parser.parse_line(line)
                ts     = self._extract_ts(line)
                for blk in blocks:
                    sessions[blk]["template_ids"].append(tid)
                    sessions[blk]["timestamps"].append(ts)
                    sessions[blk]["lines"].append(line.strip())
        return dict(sessions)

    def load_labels(self, anomaly_label_path: str) -> Dict[str, int]:
        """Load block-level labels from anomaly_label.csv"""
        labels = {}
        with open(anomaly_label_path) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    blk, lbl = parts[0].strip(), parts[1].strip()
                    labels[blk] = 0 if lbl == "Normal" else 1
        return labels

    def build_dataset(
        self,
        log_path: str,
        label_path: Optional[str] = None,
        d_s: int = 64,
        d_t: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns X: (N, max_seq_len, d_s+d_t), y: (N,)
        """
        sessions = self.parse_log_file(log_path)
        binary_labels = self.load_labels(label_path) if label_path else {}

        X_list, y_list = [], []
        for blk, sess in sessions.items():
            tids  = np.array(sess["template_ids"], dtype=np.int32)
            times = np.array(sess["timestamps"],   dtype=np.float32)
            lines = sess["lines"]

            # Pad / truncate to max_seq_len
            N = len(tids)
            if N > self.max_seq_len:
                tids  = tids[:self.max_seq_len]
                times = times[:self.max_seq_len]
                lines = lines[:self.max_seq_len]
                N     = self.max_seq_len

            sem = event_embedding(tids, self.num_templates, d_s)        # (N, d_s)
            tmp = temporal_encoding(times, d_t)                          # (N, d_t)
            feat = np.concatenate([sem, tmp], axis=-1)                   # (N, d_s+d_t)

            # Pad if shorter than max_seq_len
            if N < self.max_seq_len:
                pad = np.zeros((self.max_seq_len - N, d_s + d_t), dtype=np.float32)
                feat = np.vstack([feat, pad])

            # Label: binary → multi-class re-labeling (Section 4.1.1)
            binary = binary_labels.get(blk, 0)
            label  = self._relabel(binary, tids, lines) if binary == 1 else 0

            X_list.append(feat)
            y_list.append(label)

        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def _relabel(self, binary: int, tids: np.ndarray, lines: List[str]) -> int:
        """
        MITRE ATT&CK-based re-labeling of anomalous blocks — Section 4.1.1.
        Priority: APT > Insider Threat > DDoS > Malware > Phishing > Normal(1)
        """
        text = " ".join(lines).lower()
        auth_count    = len(self.AUTH_RE.findall(text))
        lateral_count = len(self.LATERAL_RE.findall(text))
        ddos_count    = len(self.DDOS_RE.findall(text))
        exfil_count   = len(self.EXFIL_RE.findall(text))

        if auth_count > 3 and lateral_count > 2:
            return 5   # APT: T1078 + T1021
        if auth_count > 2:
            return 4   # Insider Threat
        if ddos_count > 5:
            return 2   # DDoS
        if exfil_count > 2:
            return 1   # Malware
        return 3       # Phishing (default anomaly bucket)

    @staticmethod
    def _extract_ts(line: str) -> float:
        """Extract Unix-like timestamp from log line."""
        m = re.search(r"(\d{6})\s+(\d{6})", line)
        if m:
            # HHMMSS → seconds of day
            t = m.group(2)
            return float(t[:2]) * 3600 + float(t[2:4]) * 60 + float(t[4:6])
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET  — Section 4.1
# ─────────────────────────────────────────────────────────────────────────────

class LogDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-processed log sequences.

    Args:
        X (np.ndarray): shape (N, seq_len, feature_dim)
        y (np.ndarray): shape (N,)  — integer threat class labels
        augment (bool): apply noise augmentation during training
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            # Gaussian noise augmentation
            noise = torch.randn_like(x) * 0.01
            x     = x + noise
            # Random masking of up to 10% of sequence positions
            mask_len = int(x.shape[0] * 0.1)
            mask_idx = torch.randperm(x.shape[0])[:mask_len]
            x[mask_idx] = 0.0
        return x, self.y[idx]


def make_synthetic_data(
    num_classes: int = NUM_CLASSES,
    n_per_class: int = 1000,
    seq_len: int = LOG_PARSING["max_seq_len"],
    feat_dim: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic log sequences for quick testing when real data
    is not available (shapes match real data exactly).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(num_classes):
        # Each class has a distinct mean signal
        mean = rng.uniform(-1, 1, feat_dim) * (c + 1) * 0.5
        for _ in range(n_per_class):
            seq = rng.normal(mean, 0.3, (seq_len, feat_dim)).astype(np.float32)
            X.append(seq)
            y.append(c)
    return np.stack(X), np.array(y, dtype=np.int64)


def build_dataloaders(
    dataset_name: str = ACTIVE_DATASET,
    use_synthetic: bool = True,
    batch_size: int = TRAINING["batch_size"],
    num_workers: int = TRAINING["num_workers"],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders.

    If use_synthetic=True (default for testing), generates synthetic data.
    For real training set use_synthetic=False and ensure data is downloaded.

    Returns: (train_loader, val_loader, test_loader)
    """
    cfg  = DATASET[dataset_name]
    split = cfg["split"]

    if use_synthetic:
        print("[dataset] Using synthetic data for demonstration …")
        X, y = make_synthetic_data(
            num_classes=NUM_CLASSES,
            n_per_class=500,
            seq_len=LOG_PARSING["max_seq_len"],
            feat_dim=128,
        )
    else:
        log_path   = str(Path(cfg["local_path"]) / f"{dataset_name}.log")
        label_path = str(Path(cfg["local_path"]) / "anomaly_label.csv")
        prep = HDFSPreprocessor(
            max_seq_len=LOG_PARSING["max_seq_len"],
            num_templates=cfg.get("num_templates", 30),
        )
        print(f"[dataset] Preprocessing {dataset_name} …")
        X, y = prep.build_dataset(log_path, label_path)

    # Train / Val / Test split — Section 4.1
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - split[0]), stratify=y, random_state=TRAINING["seed"]
    )
    val_ratio = split[1] / (split[1] + split[2])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        stratify=y_temp, random_state=TRAINING["seed"]
    )

    # Normalize (per-feature z-score on train, applied to val/test)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    train_ds = LogDataset(X_train, y_train, augment=True)
    val_ds   = LogDataset(X_val,   y_val,   augment=False)
    test_ds  = LogDataset(X_test,  y_test,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=TRAINING["pin_memory"])
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=TRAINING["pin_memory"])
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=TRAINING["pin_memory"])

    print(f"[dataset] Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"[dataset] Class dist: { dict(Counter(y_train.tolist())) }")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(use_synthetic=True)
    x, y = next(iter(train_loader))
    print(f"Batch shape: x={x.shape}, y={y.shape}")
