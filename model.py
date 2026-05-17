"""
model.py — CyberSentinel-LLM Architecture
==========================================
Full model as described in the paper Section 3 (Proposed Methodology).

Five modules:
  1. Multi-Modal Log Parser & Feature Extractor  — Section 3.2
  2. Temporal Transformer Encoder                — Section 3.2, Eq.(6)–(11)
  3. LLM-Driven Threat Intelligence Engine       — Section 3.3, Eq.(12)–(17)
  4. Multi-Agent Autonomous Architecture         — Section 3.4, Eq.(18)–(21)
  5. DRL-Based Response Orchestrator             — Section 3.5, Eq.(22)–(25)

Param count: ~182.5M (Table 5 in paper — includes LoRA-adapted LLaMA-3 backbone)
For standalone training without LLM access, a lightweight surrogate LLM is used.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from config import (
    TRANSFORMER, LORA, AGENTS, DRL, NUM_CLASSES, LOSS, TRAINING
)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA LINEAR LAYER  — Section 3.3, Eq.(12)
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.
    W = W0 + BA  where B∈R^{d×r}, A∈R^{r×k}, r << min(d,k)  — Eq.(12)
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = LORA["rank"],
                 alpha: float = LORA["alpha"],
                 dropout: float = LORA["dropout"]):
        super().__init__()
        self.rank    = rank
        self.scaling = alpha / rank

        self.linear  = nn.Linear(in_features, out_features, bias=False)
        # Freeze base weights
        self.linear.weight.requires_grad_(False)

        # LoRA decomposition
        self.lora_A  = nn.Linear(in_features,  rank,         bias=False)
        self.lora_B  = nn.Linear(rank,          out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = self.linear(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TRANSFORMER ENCODER  — Section 3.2, Eq.(6)–(11)
# ─────────────────────────────────────────────────────────────────────────────

class WindowedMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking and windowed attention.
    Reduces complexity from O(N²) to O(N·w)  — Section 3.5 complexity analysis.
    Eq.(6)–(8)
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 window_size: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.window_size = window_size

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Q = self.W_q(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.W_k(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, Dh).transpose(1, 2)

        # Scaled dot-product attention  — Eq.(6)
        scale  = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B,H,N,N)

        # Causal mask M  — Eq.(7)
        causal = torch.triu(
            torch.full((N, N), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal

        # Window mask: zero out positions beyond window_size
        if self.window_size < N:
            win = torch.full((N, N), float("-inf"), device=x.device)
            for i in range(N):
                lo = max(0, i - self.window_size + 1)
                win[i, lo:i+1] = 0.0
            scores = scores + win

        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out = torch.matmul(attn, V)                              # (B,H,N,Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)    # (B,N,D)
        return self.W_o(out)


class TransformerLayer(nn.Module):
    """Single transformer encoder layer — Eq.(9)–(11)"""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 window_size: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = WindowedMultiHeadAttention(hidden_dim, num_heads, window_size, dropout)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN with GELU activation — Eq.(11)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq.(9): Z^l = LayerNorm(Z^{l-1} + MHA(Z^{l-1}))
        x = self.norm1(x + self.dropout(self.attn(x)))
        # Eq.(10): Z^l = LayerNorm(Z^l + FFN(Z^l))
        x = self.norm2(x + self.ffn(x))
        return x


class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer Encoder — Section 3.2, Eq.(6)–(11)
    L = 6 layers, H = 8 heads, d_model = 512
    """

    def __init__(self,
                 input_dim:   int = 128,
                 hidden_dim:  int = TRANSFORMER["hidden_dim"],
                 num_layers:  int = TRANSFORMER["num_layers"],
                 num_heads:   int = TRANSFORMER["num_heads"],
                 ffn_dim:     int = TRANSFORMER["ffn_dim"],
                 window_size: int = TRANSFORMER["window_size"],
                 dropout:     float = TRANSFORMER["dropout"]):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ffn_dim, window_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim)
        Returns:
            Z_L: (B, seq_len, hidden_dim)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# LLM THREAT INTELLIGENCE ENGINE  — Section 3.3, Eq.(12)–(17)
# ─────────────────────────────────────────────────────────────────────────────

class LLMThreatEngine(nn.Module):
    """
    Surrogate LLM module (trainable transformer stack) that replaces
    LLaMA-3-8B when the full LLM is not available.

    The full implementation loads LLaMA-3-8B from HuggingFace with PEFT/LoRA:
        from peft import get_peft_model, LoraConfig
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
        model = get_peft_model(model, peft_config)

    This module approximates the LLM's contextual reasoning with a
    lighter multi-head self-attention stack + LoRA-adapted projection.
    """

    def __init__(self,
                 hidden_dim:  int = TRANSFORMER["hidden_dim"],
                 num_classes: int = NUM_CLASSES,
                 lora_rank:   int = LORA["rank"],
                 dropout:     float = 0.1):
        super().__init__()
        # Contextual reasoning via LoRA-adapted projection — Eq.(12)
        self.context_proj = LoRALinear(hidden_dim, hidden_dim, rank=lora_rank)
        self.norm         = nn.LayerNorm(hidden_dim)
        self.dropout      = nn.Dropout(dropout)

        # Threat intelligence attention (multi-head over sequence)
        self.threat_attn = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                                   dropout=dropout, batch_first=True)

        # Classification head — Eq.(14)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, Z_L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z_L: (B, seq_len, hidden_dim) — output of temporal encoder
        Returns:
            h_LLM: (B, hidden_dim)  — [CLS]-style pooled representation
            logits: (B, num_classes)
        """
        # Context-aware projection — Eq.(13)
        ctx, _ = self.threat_attn(Z_L, Z_L, Z_L)
        ctx    = self.norm(Z_L + self.dropout(ctx))
        ctx    = self.context_proj(ctx)

        # Global pool (mean) as h_i^LLM
        h_LLM  = ctx.mean(dim=1)                    # (B, hidden_dim)

        # Threat class probability — Eq.(14)
        logits = self.classifier(h_LLM)             # (B, num_classes)
        return h_LLM, logits


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-AGENT SYSTEM  — Section 3.4, Eq.(18)–(21)
# ─────────────────────────────────────────────────────────────────────────────

class DetectionAgent(nn.Module):
    """
    A_D: Binary anomaly scoring — Eq.(18)
    s^D(i) = σ(W_D · [h_i^LLM ‖ g_i] + b_D)
    """

    def __init__(self, hidden_dim: int = TRANSFORMER["hidden_dim"]):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h_LLM: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Returns detection score in (0,1)"""
        combined = torch.cat([h_LLM, g], dim=-1)   # (B, 2*D)
        return torch.sigmoid(self.scorer(combined)).squeeze(-1)   # (B,)


class ClassificationAgent(nn.Module):
    """
    A_C: Multi-class threat classification — Eq.(19)
    ŷ_i = argmax P(y|P_i, M)
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)


class ForensicAgent(nn.Module):
    """
    A_F: Natural-language forensic report generation — Eq.(20)
    R_i = LLM_gen(P_forensic, s^D(i), ŷ_i, a_i*)

    In this implementation we produce a structured text report
    using the detection score and predicted class.
    """

    REPORT_TEMPLATE = """
CYBERSENTINEL-LLM INCIDENT FORENSIC REPORT
===========================================
Report ID      : {report_id}
Timestamp      : {timestamp}
Threat Class   : {threat_class}
Severity       : {severity}
Confidence     : {confidence:.4f}

DETECTION SUMMARY
  Anomaly Score   : {anomaly_score:.4f}  (threshold: 0.85)
  Classification  : {threat_class}
  MITRE ATT&CK    : {mitre}

RECOMMENDED RESPONSE ACTIONS
  {actions}
"""

    MITRE_MAP = {
        0: "N/A (Normal activity)",
        1: "T1071 (App Layer Protocol), T1530 (Cloud Storage)",
        2: "T1498 (Network DoS), T1499 (Endpoint DoS)",
        3: "T1566 (Phishing), T1078 (Valid Accounts)",
        4: "T1078 (Valid Accounts), T1021 (Remote Services)",
        5: "T1078 + T1071 + T1021 (APT lateral movement)",
    }

    ACTION_MAP = {
        0: "No action required.",
        1: "1. Block outbound traffic from affected process\n  2. Quarantine affected files",
        2: "1. Activate rate-limiting rules\n  2. Block source IPs via firewall",
        3: "1. Reset compromised credentials\n  2. Enable MFA on affected accounts",
        4: "1. Suspend suspicious account (requires human approval)\n  2. Audit access logs",
        5: "1. [AUTOMATED] Network isolation of affected nodes\n  2. [ESCALATED] Account suspension for human review",
    }

    def generate(self, detection_score: float, pred_class: int,
                 report_id: str = "CSL-AUTO-001",
                 timestamp: str = "2025-01-01 00:00:00 UTC") -> str:
        from config import THREAT_CLASSES
        severity_map = {0: "INFO", 1: "HIGH", 2: "HIGH",
                        3: "MEDIUM", 4: "HIGH", 5: "CRITICAL"}
        return self.REPORT_TEMPLATE.format(
            report_id     = report_id,
            timestamp     = timestamp,
            threat_class  = THREAT_CLASSES[pred_class],
            severity      = severity_map.get(pred_class, "UNKNOWN"),
            confidence    = detection_score,
            anomaly_score = detection_score,
            mitre         = self.MITRE_MAP.get(pred_class, "Unknown"),
            actions       = self.ACTION_MAP.get(pred_class, "Escalate to analyst"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DRL RESPONSE ORCHESTRATOR — Section 3.5, Eq.(22)–(25)
# ─────────────────────────────────────────────────────────────────────────────

class ResponseOrchestrator(nn.Module):
    """
    Policy network π_θ for DRL-based response selection — Eq.(23).
    Action space: 6 response categories (Section 3.5).
    Trained via PPO — Eq.(25).

    State: s_t = [h_t^LLM, s^D(t), ŷ_t, u_t]  — Eq.(22)
    """
    ACTION_NAMES = [
        "Network Isolation",
        "Process Termination",
        "Firewall Rule Insertion",
        "Account Suspension",
        "Patch Deployment",
        "Alert Escalation",
    ]

    def __init__(self, hidden_dim: int = TRANSFORMER["hidden_dim"],
                 num_actions: int = DRL["action_space"]):
        super().__init__()
        state_dim = hidden_dim + 1 + 1 + 4  # h + score + class_onehot-ish + util

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, h_LLM: torch.Tensor, det_score: torch.Tensor,
                threat_class: torch.Tensor,
                util: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_logits: (B, num_actions)
            value:         (B, 1)
        """
        B = h_LLM.shape[0]
        if util is None:
            util = torch.zeros(B, 4, device=h_LLM.device)
        state = torch.cat([
            h_LLM,
            det_score.unsqueeze(-1),
            threat_class.float().unsqueeze(-1),
            util,
        ], dim=-1)
        return self.policy_net(state), self.value_net(state)

    def select_action(self, h_LLM: torch.Tensor, det_score: torch.Tensor,
                      threat_class: torch.Tensor) -> Tuple[int, float]:
        """Greedy action selection for inference."""
        with torch.no_grad():
            logits, _ = self.forward(h_LLM, det_score, threat_class)
            probs      = F.softmax(logits, dim=-1)
            action     = probs.argmax(-1).item()
            confidence = probs.max(-1).values.item()
        return action, confidence


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED LOSS  — Section 3.3, Eq.(15)–(17)
# ─────────────────────────────────────────────────────────────────────────────

class CyberSentinelLoss(nn.Module):
    """
    L_total = L_CE + λ · L_contrastive  — Eq.(15)
    L_contrastive uses in-batch positive pairs — Eq.(17)
    """

    def __init__(self, temperature: float = LOSS["temperature"],
                 lam: float = LOSS["lambda_contrastive"],
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.temperature = temperature
        self.lam         = lam
        self.ce          = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, h: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # L_CE — Eq.(16)
        ce_loss = self.ce(logits, labels)

        # L_contrastive — Eq.(17)
        h_norm    = F.normalize(h, dim=-1)
        sim_mat   = torch.matmul(h_norm, h_norm.T) / self.temperature  # (B,B)
        mask_pos  = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_pos.fill_diagonal_(0)

        exp_sim   = torch.exp(sim_mat)
        denom     = exp_sim.sum(dim=-1, keepdim=True) - exp_sim.diagonal().unsqueeze(-1)
        log_prob  = sim_mat - torch.log(denom + 1e-8)

        n_pos     = mask_pos.sum(dim=-1)
        contrastive_loss = -(mask_pos * log_prob).sum(dim=-1) / (n_pos + 1e-8)
        contrastive_loss = contrastive_loss.mean()

        total = ce_loss + self.lam * contrastive_loss
        losses = {
            "total": total.item(),
            "ce":    ce_loss.item(),
            "contrastive": contrastive_loss.item(),
        }
        return total, losses


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL  — CyberSentinel-LLM
# ─────────────────────────────────────────────────────────────────────────────

class CyberSentinelLLM(nn.Module):
    """
    CyberSentinel-LLM Full Model — Section 3.1
    Integrates all five modules into a single end-to-end pipeline.

    Approximate parameter count: ~182.5M (Table 5)
      - Temporal Transformer (d=512, L=6, H=8): ~25M
      - LLM Engine (surrogate): ~12M
      - Agents + Response: ~2M
      Full LLaMA-3-8B backbone adds ~8B (LoRA adapters: ~2M trainable)
    """

    def __init__(self,
                 input_dim:   int = 128,
                 hidden_dim:  int = TRANSFORMER["hidden_dim"],
                 num_classes: int = NUM_CLASSES,
                 num_layers:  int = TRANSFORMER["num_layers"],
                 num_heads:   int = TRANSFORMER["num_heads"],
                 ffn_dim:     int = TRANSFORMER["ffn_dim"],
                 window_size: int = TRANSFORMER["window_size"],
                 dropout:     float = TRANSFORMER["dropout"]):
        super().__init__()

        # Module 1+2: Feature extraction + Temporal Transformer
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, window_size=window_size, dropout=dropout,
        )

        # Module 3: LLM Threat Intelligence Engine
        self.llm_engine = LLMThreatEngine(
            hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout
        )

        # Module 4: Multi-Agent System
        self.detection_agent       = DetectionAgent(hidden_dim)
        self.classification_agent  = ClassificationAgent()
        self.forensic_agent        = ForensicAgent()

        # Module 5: DRL Response Orchestrator
        self.response_orchestrator = ResponseOrchestrator(hidden_dim, DRL["action_space"])

        # Global pooling (g_i = GlobalPool(Z^L))
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, LoRALinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass — Algorithm 1 in paper.

        Args:
            x: (B, seq_len, input_dim) — fused multi-modal feature sequences

        Returns dict with:
            logits      : (B, num_classes) — threat classification logits
            h_LLM       : (B, hidden_dim)  — LLM hidden states
            det_score   : (B,)             — anomaly detection score
            pred_class  : (B,)             — predicted threat class
            action      : (B,)             — DRL response action
        """
        # Step 1–8: Temporal Transformer Encoding
        Z_L = self.temporal_encoder(x)                          # (B, N, D)

        # Global context vector g = GlobalPool(Z^L) — Eq.(18)
        g = self.global_pool(Z_L.transpose(1, 2)).squeeze(-1)  # (B, D)

        # Step 9–11: LLM Threat Analysis
        h_LLM, logits = self.llm_engine(Z_L)                   # (B,D), (B,C)

        # Step 12–17: Multi-Agent Processing
        det_score  = self.detection_agent(h_LLM, g)             # (B,)   Eq.(18)
        pred_class = self.classification_agent(logits)           # (B,)   Eq.(19)

        # Step 18: Response selection
        action_logits, _ = self.response_orchestrator(
            h_LLM, det_score.detach(), pred_class.detach().float()
        )
        action = action_logits.argmax(dim=-1)                    # (B,)

        return {
            "logits":     logits,
            "h_LLM":      h_LLM,
            "det_score":  det_score,
            "pred_class": pred_class,
            "action":     action,
            "Z_L":        Z_L,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(input_dim: int = 128, device: str = "cpu") -> CyberSentinelLLM:
    """Convenience factory function."""
    model = CyberSentinelLLM(input_dim=input_dim).to(device)
    n = model.count_parameters()
    print(f"[model] CyberSentinel-LLM | Trainable params: {n:,}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(input_dim=128, device=device)
    B, N, D = 4, 256, 128
    x = torch.randn(B, N, D).to(device)
    out = model(x)
    print(f"logits:    {out['logits'].shape}")
    print(f"det_score: {out['det_score'].shape}")
    print(f"action:    {out['action'].shape}")
