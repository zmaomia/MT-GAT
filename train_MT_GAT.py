import os
import copy
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, JumpingKnowledge
from torch_geometric.utils import dropout_edge




# 1. Paths and hyper-parameters

DATA_PATH = "./data_multi_block.pt"
SAVE_DIR = "./MT_GAT"

os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training (identical to the shared-bottom baseline for a fair comparison)
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20000
PATIENCE = 500
WARMUP_EPOCHS = 20
GRAD_CLIP = 1.0

# Backbone (identical to the shared-bottom baseline)
HIDDEN_DIM = 128
NUM_GAT_LAYERS = 3
HEADS = 4
FEAT_DROPOUT = 0.3
ATTN_DROPOUT = 0.2
EDGE_DROP = 0.1
JK_MODE = "cat"

# MMoE-specific
NUM_EXPERTS = 4          # K experts; 3-6 is a typical range for 2 tasks
EXPERT_HIDDEN = 128      # each expert outputs HIDDEN_DIM-sized features
GATE_DROPOUT = 0.1       # mild dropout on gating logits (regularization)

STANDARDIZE_X = True


# 2. Utilities
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_std(x, dim=0, keepdim=False):
    s = x.std(dim=dim, keepdim=keepdim, unbiased=False)
    return torch.where(s < 1e-8, torch.ones_like(s), s)

def standardize_by_train_mask(x, train_mask):
    mean = x[train_mask].mean(dim=0, keepdim=True)
    std = safe_std(x[train_mask], dim=0, keepdim=True)
    return (x - mean) / std, mean, std

def standardize_y_by_train_mask(y, train_mask):
    mean = y[train_mask].mean(dim=0, keepdim=True)
    std = safe_std(y[train_mask], dim=0, keepdim=True)
    return (y - mean) / std, mean, std

def compute_metrics_1d(y_true, y_pred):
    y_true = y_true.view(-1); y_pred = y_pred.view(-1)
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    mae = torch.mean(torch.abs(y_pred - y_true))
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # lin_ccc
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(),  y_pred.var()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    LCC = (2 * cov) / (vt + vp + (mt - mp) ** 2 + 1e-12)

    return {"RMSE": rmse.item(), "MAE": mae.item(), "R2": r2.item(), "LCC": LCC.item()}


# 3. Building blocks
class GATv2Block(nn.Module):
    """Pre-LN GATv2 + residual. UNCHANGED from the shared-bottom baseline."""
    def __init__(self, hidden_dim, heads=4, feat_dropout=0.3, attn_dropout=0.2):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.norm = nn.LayerNorm(hidden_dim)
        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=attn_dropout,
        )
        self.feat_dropout = feat_dropout

    def forward(self, x, edge_index):
        h = self.norm(x)
        h = self.gat(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.feat_dropout, training=self.training)
        return x + h


class Expert(nn.Module):
    """One expert = a small MLP, structurally aligned with the original
    `shared` block so MMoE's expert capacity is comparable to the baseline."""
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    """Per-task gating: maps JK output to a softmax distribution over experts.
    Standard MMoE design: linear + softmax. We add light dropout on the logits
    for regularization, which empirically stabilizes gate learning on small
    datasets without affecting the soft routing semantics."""
    def __init__(self, in_dim, num_experts, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        logits = self.dropout(self.fc(x))
        return F.softmax(logits, dim=-1)   # [N, K]


class TaskTower(nn.Module):
    """Per-task 2-layer MLP head. UNCHANGED."""
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x)


class MMoEGATv2(nn.Module):
    """
    Encoder: input_proj + 3 × GATv2Block + JK   (identical to baseline)
    MMoE:    K experts + 1 gate per task
    Heads:   one TaskTower per task             (identical to baseline)

    Forward shape trace (N = num nodes, K = num experts):
        x        : [N, in_dim]
        → input_proj                            → [N, 128]
        → 3 × GATv2 blocks (collect each layer) → [N, 128] × 3
        → JK "cat"                              → [N, 384]
        ──── MMoE starts here ────
        → K experts, each Linear(384→128)+...   → [N, 128] × K
        → stack experts                         → [N, K, 128]
        → gate_olt(h_jk) softmax over K         → [N, K]
        → einsum α · experts                    → [N, 128]   (h_olt)
        → gate_om(h_jk) softmax over K          → [N, K]
        → einsum α · experts                    → [N, 128]   (h_om)
        ──── MMoE ends here ────
        → tower_olt(h_olt), tower_om(h_om)      → [N, 1] × 2
        → concat                                → [N, 2]
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        feat_dropout=0.3,
        attn_dropout=0.2,
        jk_mode="cat",
        num_experts=4,
        expert_hidden=128,
        gate_dropout=0.1,
    ):
        super().__init__()
        self.num_experts = num_experts

        # --- Encoder (unchanged) ---
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList([
            GATv2Block(hidden_dim, heads=heads,
                       feat_dropout=feat_dropout,
                       attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.jk = JumpingKnowledge(mode=jk_mode,
                                   channels=hidden_dim,
                                   num_layers=num_layers)
        jk_out_dim = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim

        # --- MMoE layer (replaces the original `shared` MLP) ---
        self.experts = nn.ModuleList([
            Expert(jk_out_dim, expert_hidden, dropout=feat_dropout)
            for _ in range(num_experts)
        ])
        # One gate per task; gates take the JK output as input (richest signal).
        self.gate_olt = Gate(jk_out_dim, num_experts, dropout=gate_dropout)
        self.gate_om  = Gate(jk_out_dim, num_experts, dropout=gate_dropout)

        # --- Task towers (unchanged) ---
        self.tower_olt = TaskTower(expert_hidden, expert_hidden // 2, dropout=feat_dropout)
        self.tower_om  = TaskTower(expert_hidden, expert_hidden // 2, dropout=feat_dropout)

    def forward(self, x, edge_index, return_gates=False):
        x = self.input_proj(x)
        xs = []
        for block in self.blocks:
            x = block(x, edge_index)
            xs.append(x)
        h_jk = self.jk(xs)                       # [N, jk_out_dim]

        # Run all experts on the same JK representation.
        # Stack to [N, K, expert_hidden].
        expert_outs = torch.stack(
            [expert(h_jk) for expert in self.experts], dim=1
        )

        # Per-task soft routing.
        alpha_olt = self.gate_olt(h_jk)          # [N, K]
        alpha_om  = self.gate_om(h_jk)           # [N, K]

        # Weighted sum of experts per task.
        # einsum: [N,K] x [N,K,D] -> [N,D]
        h_olt = torch.einsum("nk,nkd->nd", alpha_olt, expert_outs)
        h_om  = torch.einsum("nk,nkd->nd", alpha_om,  expert_outs)

        pred = torch.cat([self.tower_olt(h_olt), self.tower_om(h_om)], dim=1)

        if return_gates:
            return pred, alpha_olt, alpha_om
        return pred


class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall, Gal & Cipolla (2018). Kept identical to the shared-bottom
    baseline so MMoE's contribution is isolated.
        L = sum_i  exp(-s_i) * L_i + s_i
    """
    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total = 0.0
        for i, l in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * l + self.log_vars[i]
        return total

    def weights(self):
        return torch.exp(-self.log_vars).detach().cpu().tolist()

# 4. Data
set_seed(SEED)
data = torch.load(DATA_PATH, map_location="cpu")

for attr in ["x", "edge_index", "y", "train_mask", "val_mask", "test_mask"]:
    if not hasattr(data, attr):
        raise ValueError(f"Loaded data is missing required attribute: {attr}")
if data.y.dim() != 2 or data.y.size(1) != 2:
    raise ValueError(f"Expected y shape [N, 2], got {tuple(data.y.shape)}")

data.x = data.x.float()
data.y = data.y.float()
y_orig = data.y.clone()

if STANDARDIZE_X:
    data.x, x_mean, x_std = standardize_by_train_mask(data.x, data.train_mask)

y_scaled, y_mean, y_std = standardize_y_by_train_mask(data.y, data.train_mask)
data.y = y_scaled

data = data.to(DEVICE)
y_orig = y_orig.to(DEVICE)
y_mean = y_mean.to(DEVICE)
y_std  = y_std.to(DEVICE)

print("Data loaded.")
print(data)

# 5. Model / Loss / Optimizer / Scheduler
model = MMoEGATv2(
    in_dim=data.x.size(1),
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_GAT_LAYERS,
    heads=HEADS,
    feat_dropout=FEAT_DROPOUT,
    attn_dropout=ATTN_DROPOUT,
    jk_mode=JK_MODE,
    num_experts=NUM_EXPERTS,
    expert_hidden=EXPERT_HIDDEN,
    gate_dropout=GATE_DROPOUT,
).to(DEVICE)


mt_loss = UncertaintyWeightedLoss(num_tasks=2).to(DEVICE)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(mt_loss.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

mse = nn.MSELoss()

# Quick parameter accounting — useful when reporting in the paper.
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params:,}")


# 6. Train / Eval
@torch.no_grad()
def predict_original_scale(model, data, return_gates=False):
    model.eval()
    if return_gates:
        pred_scaled, a_olt, a_om = model(data.x, data.edge_index, return_gates=True)
        return pred_scaled * y_std + y_mean, a_olt, a_om
    pred_scaled = model(data.x, data.edge_index)
    return pred_scaled * y_std + y_mean

@torch.no_grad()
def evaluate_split(model, data, y_true_orig, mask):
    pred = predict_original_scale(model, data)
    m_olt = compute_metrics_1d(y_true_orig[mask, 0], pred[mask, 0])
    m_om  = compute_metrics_1d(y_true_orig[mask, 1], pred[mask, 1])
    avg_rmse = (m_olt["RMSE"] + m_om["RMSE"]) / 2.0

    return {"OLT": m_olt, "OM": m_om, "AVG_RMSE": avg_rmse}

def train_one_epoch():
    model.train()
    mt_loss.train()
    optimizer.zero_grad()

    if EDGE_DROP > 0:
        edge_index, _ = dropout_edge(data.edge_index, p=EDGE_DROP,
                                     force_undirected=True, training=True)
    else:
        edge_index = data.edge_index

    out = model(data.x, edge_index)
    l_olt = mse(out[data.train_mask, 0], data.y[data.train_mask, 0])
    l_om  = mse(out[data.train_mask, 1], data.y[data.train_mask, 1])
    loss  = mt_loss([l_olt, l_om])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(mt_loss.parameters()),
        max_norm=GRAD_CLIP,
    )
    optimizer.step()
    scheduler.step()
    return loss.item(), l_olt.item(), l_om.item()

# 7. Training loop with early stopping on val AVG_RMSE
best_val_score = float("inf")
best_epoch = -1
best_state = None
best_mt_state = None
wait = 0
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss, l_olt, l_om = train_one_epoch()
    train_metrics = evaluate_split(model, data, y_orig, data.train_mask)
    val_metrics   = evaluate_split(model, data, y_orig, data.val_mask)

    w_olt, w_om = mt_loss.weights()
    history.append({
        "epoch": epoch,
        "lr": scheduler.get_last_lr()[0],
        "train_loss": train_loss,
        "train_loss_olt": l_olt,
        "train_loss_om": l_om,
        "task_w_olt": w_olt,
        "task_w_om": w_om,
        "train_rmse_olt": train_metrics["OLT"]["RMSE"],
        "train_rmse_om":  train_metrics["OM"]["RMSE"],
        "train_mae_olt":  train_metrics["OLT"]["MAE"],
        "train_mae_om":  train_metrics["OM"]["MAE"],
        "train_lcc_olt":  train_metrics["OLT"]["LCC"],
        "train_lcc_om":  train_metrics["OM"]["LCC"],
        "val_rmse_olt":   val_metrics["OLT"]["RMSE"],
        "val_rmse_om":    val_metrics["OM"]["RMSE"],
        "val_mae_olt":  val_metrics["OLT"]["MAE"],
        "val_mae_om":  val_metrics["OM"]["MAE"],
        "val_lcc_olt":  val_metrics["OLT"]["LCC"],
        "val_lcc_om":  val_metrics["OM"]["LCC"],
        "val_avg_rmse":   val_metrics["AVG_RMSE"],
    })

    if val_metrics["AVG_RMSE"] < best_val_score:
        best_val_score = val_metrics["AVG_RMSE"]
        best_epoch = epoch
        best_state    = copy.deepcopy(model.state_dict())
        best_mt_state = copy.deepcopy(mt_loss.state_dict())
        wait = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": best_state,
                "mt_loss_state_dict": best_mt_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_avg_rmse": best_val_score,
                "y_mean": y_mean.detach().cpu(),
                "y_std":  y_std.detach().cpu(),
                "arch": f"MMoEGATv2(K={NUM_EXPERTS})",
            },
            os.path.join(SAVE_DIR, "best_model.pt"),
        )
    else:
        wait += 1

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"Epoch {epoch:04d} | lr {scheduler.get_last_lr()[0]:.2e} | "
            f"Loss {train_loss:.4f} | "
            f"Val AVG {val_metrics['AVG_RMSE']:.4f} "
            f"(OLT {val_metrics['OLT']['RMSE']:.4f}, OM {val_metrics['OM']['RMSE']:.4f}) | "
            f"task_w OLT={w_olt:.3f} OM={w_om:.3f}"
        )

    if wait >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
        break

# 8. Final evaluation with best weights
model.load_state_dict(best_state)

splits = {
    "Train":      data.train_mask,
    "Validation": data.val_mask,
    "Test":       data.test_mask,
}
print("\n===== Final Results (Best Model, MMoE) =====")
print(f"Best epoch: {best_epoch} | Best val AVG_RMSE: {best_val_score:.4f}")
for name, mask in splits.items():
    m = evaluate_split(model, data, y_orig, mask)
    for tgt in ["OLT", "OM"]:
        print(f"\n[{name} - {tgt}] "
              f"RMSE={m[tgt]['RMSE']:.4f}  MAE={m[tgt]['MAE']:.4f}  R2={m[tgt]['R2']:.4f} LCC={m[tgt]['LCC']:.4f}")


# 9. Save history + gate statistics for analysis
history_path = os.path.join(SAVE_DIR, "training_history.csv")
pd.DataFrame(history).to_csv(history_path, index=False)

# Dump per-task gate distributions on the test set. Use this to check whether
# the two tasks really learned different routings — if alpha_olt and alpha_om
# are basically identical, MMoE collapsed back to shared-bottom and you should
# either reduce NUM_EXPERTS or add a load-balancing penalty.
with torch.no_grad():
    _, a_olt, a_om = predict_original_scale(model, data, return_gates=True)
    test_mask = data.test_mask
    gate_stats = {
        "expert_id": list(range(NUM_EXPERTS)),
        "alpha_olt_mean_test": a_olt[test_mask].mean(dim=0).cpu().tolist(),
        "alpha_om_mean_test":  a_om[test_mask].mean(dim=0).cpu().tolist(),
    }
gate_path = os.path.join(SAVE_DIR, "gate_stats_test.csv")
pd.DataFrame(gate_stats).to_csv(gate_path, index=False)

print(f"\nSaved best model to: {os.path.join(SAVE_DIR, 'best_model.pt')}")
print(f"Saved training history to: {history_path}")
print(f"Saved gate distributions to: {gate_path}")
print("\nGate means on test set:")
print(pd.DataFrame(gate_stats).to_string(index=False))