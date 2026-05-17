"""
inference.py — CyberSentinel-LLM Inference
===========================================
Run anomaly detection on a single log sequence or file.
Produces: threat classification, detection score, response action,
          and natural-language forensic report.

Usage:
    python inference.py --input logs/sample.log
    python inference.py --demo       # runs with synthetic sequence
"""

import argparse
import datetime
import uuid
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import CHECKPOINT, RESULTS, THREAT_CLASSES, NUM_CLASSES, TRANSFORMER
from model import build_model, ResponseOrchestrator, ForensicAgent
from dataset import HDFSPreprocessor, event_embedding, temporal_encoding
from utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CyberSentinelInference:
    """
    Wraps model loading and single-sequence inference.
    Implements Algorithm 1 from the paper (Section 3.6).
    """

    def __init__(self, checkpoint_path: str = None, device: str = "auto",
                 input_dim: int = 128):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = torch.device(device)
        self.model     = build_model(input_dim=input_dim, device=device)
        self.forensic  = ForensicAgent()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            print("[inference] No checkpoint provided — using random weights (demo mode)")

        self.model.eval()

    def _load_checkpoint(self, path: str):
        try:
            ckpt  = torch.load(path, map_location=self.device)
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state, strict=False)
            print(f"[inference] Loaded: {path}")
        except FileNotFoundError:
            print(f"[inference] Checkpoint not found: {path} — using random weights")

    @torch.no_grad()
    def predict(self, log_sequence: np.ndarray) -> dict:
        """
        Run full CyberSentinel-LLM pipeline on a single log sequence.

        Args:
            log_sequence: np.ndarray of shape (seq_len, feature_dim)

        Returns:
            dict with keys:
                threat_class, class_name, detection_score,
                confidence, action_id, action_name,
                response_names, forensic_report
        """
        # Add batch dim — Algorithm 1, Line 2
        x = torch.tensor(log_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        out        = self.model(x)
        logits     = out["logits"]          # (1, C)
        det_score  = out["det_score"]       # (1,)
        pred_class = out["pred_class"]      # (1,)
        action     = out["action"]          # (1,)

        probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        cls_id     = int(pred_class.item())
        det_val    = float(det_score.item())
        act_id     = int(action.item())
        confidence = float(probs[cls_id])

        class_name  = THREAT_CLASSES[cls_id]
        action_name = ResponseOrchestrator.ACTION_NAMES[act_id]

        # Generate forensic report — Eq.(20)
        report_id = f"CSL-{datetime.date.today()}-{str(uuid.uuid4())[:8].upper()}"
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        forensic_report = self.forensic.generate(
            detection_score=det_val,
            pred_class=cls_id,
            report_id=report_id,
            timestamp=timestamp,
        )

        return {
            "threat_class":   cls_id,
            "class_name":     class_name,
            "detection_score": det_val,
            "confidence":     confidence,
            "class_probs":    probs.tolist(),
            "action_id":      act_id,
            "action_name":    action_name,
            "forensic_report": forensic_report,
            "is_anomaly":     det_val > 0.85,
        }

    def predict_from_file(self, log_path: str) -> dict:
        """
        Load a raw log file, parse it, and run prediction.
        Uses HDFSPreprocessor for feature extraction.
        """
        prep = HDFSPreprocessor(max_seq_len=256, num_templates=30)
        sessions = prep.parse_log_file(log_path)

        results = []
        for blk, sess in sessions.items():
            tids  = np.array(sess["template_ids"], dtype=np.int32)
            times = np.array(sess["timestamps"],   dtype=np.float32)
            N = min(len(tids), 256)
            tids, times = tids[:N], times[:N]

            sem  = event_embedding(tids, 30, 64)
            tmp  = temporal_encoding(times, 64)
            feat = np.concatenate([sem, tmp], axis=-1)
            if N < 256:
                pad  = np.zeros((256 - N, 128), dtype=np.float32)
                feat = np.vstack([feat, pad])

            result = self.predict(feat)
            result["block_id"] = blk
            results.append(result)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_prediction(result: dict, save_path: str = None):
    """
    Visualize detection scores, class probabilities, and response action.
    Saves a publication-style figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")

    # ── Left: Class probability bar chart ────────────────────────────────────
    ax = axes[0]
    probs     = result["class_probs"]
    colors    = ["#1A73E8" if i == result["threat_class"] else "#BBBBBB"
                 for i in range(NUM_CLASSES)]
    ax.barh(THREAT_CLASSES, probs, color=colors)
    ax.axvline(x=0.85, color="red", linestyle="--", linewidth=1.0,
               label="Detection threshold (0.85)")
    ax.set_xlabel("Probability")
    ax.set_title("Threat Class Probabilities")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)

    # ── Right: Summary panel ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    summary_text = (
        f"CYBERSENTINEL-LLM DETECTION RESULT\n"
        f"{'─'*38}\n\n"
        f"  Threat Class  :  {result['class_name']}\n"
        f"  Confidence    :  {result['confidence']*100:.1f}%\n"
        f"  Anomaly Score :  {result['detection_score']:.4f}\n"
        f"  Is Anomaly    :  {'⚠ YES' if result['is_anomaly'] else '✓  NO'}\n\n"
        f"  Response Action:\n"
        f"    {result['action_name']}\n\n"
        f"{'─'*38}\n"
        f"  Latency Target: ≤ 100 ms (paper)\n"
        f"  Reported Best :  45 ms (Table 5)\n"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    plt.tight_layout(pad=1.0)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[inference] Visualization saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """Run inference on a synthetic log sequence for demonstration."""
    set_seed(42)
    print("\n" + "=" * 60)
    print("  CyberSentinel-LLM — Demo Inference")
    print("=" * 60)

    engine = CyberSentinelInference(
        checkpoint_path=str(CHECKPOINT["best_model"])
        if CHECKPOINT["best_model"].exists() else None,
        input_dim=128,
    )

    # Generate a synthetic APT-like log sequence
    rng          = np.random.default_rng(42)
    apt_feat     = rng.normal(2.5, 0.4, (256, 128)).astype(np.float32)
    result       = engine.predict(apt_feat)

    # Print results
    print(f"\nDetection Score  : {result['detection_score']:.4f}")
    print(f"Predicted Class  : {result['class_name']}")
    print(f"Confidence       : {result['confidence']*100:.1f}%")
    print(f"Is Anomaly       : {'YES ⚠' if result['is_anomaly'] else 'NO ✓'}")
    print(f"Response Action  : {result['action_name']}")
    print("\n" + "─" * 60)
    print(result["forensic_report"])

    # Visualize
    vis_path = str(RESULTS["confusion_matrix"]).replace("confusion_matrix", "inference_result")
    visualize_prediction(result, save_path=vis_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CyberSentinel-LLM Inference")
    p.add_argument("--input",      type=str, default=None,
                   help="Path to raw log file (.log)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint .pt file")
    p.add_argument("--demo",       action="store_true", default=True,
                   help="Run demo with synthetic data")
    p.add_argument("--save",       type=str, default=None,
                   help="Path to save visualization output")
    p.add_argument("--cpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input:
        engine  = CyberSentinelInference(
            checkpoint_path=args.checkpoint,
            input_dim=128,
            device="cpu" if args.cpu else "auto",
        )
        results = engine.predict_from_file(args.input)
        for r in results[:5]:
            print(f"\nBlock {r['block_id']}: {r['class_name']} "
                  f"(score={r['detection_score']:.3f})")
    else:
        run_demo()
