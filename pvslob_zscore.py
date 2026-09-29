"""
pvslob_zscore.py — PVSLOB on FI-2010 NoAuction ZScore (CF_7 only)
===================================================================
PVSLOB ZScore: Price/Volume stream decomposition via two StreamEncoders
               + CrossAttnBlock (p2v + v2p) + LSTM(256,128)
               Price = cols[0,2,...,38] (20 levels), Volume = cols[1,3,...,39] (20 levels)
Protocol : Train CF_7 (80%) | Val CF_7 (20%, embargo=99)
           Test CF_8+CF_9 (ConcatDataset, CF_7 excluded)
Features : 144 loaded (40 LOB raw + 104 derived); model uses first 40 (price/vol split)
K_INDEX  : 4  = k=100 (FI-2010 ascending label order, paper convention)
LR       : 5e-4  (best from zscore_completed_runs.json, test acc=0.8479)
"""
import argparse, copy, json, os, random, zipfile
from datetime import datetime
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data",       default="BenchmarkDatasets.zip")
parser.add_argument("--out",        default="pvslob_zscore_results.json")
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--batch-size", type=int,   default=64)
parser.add_argument("--lr",         type=float, default=5e-4)
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_SIZE = 100; K_INDEX = 4; N_FEATS = 144; N_CLASSES = 3
EMBARGO  = SEQ_SIZE - 1
NORM     = "ZScore"
PREFIX   = "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/"
TRAIN_F  = PREFIX + "NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt"
TEST_FS  = [PREFIX + f"NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{i}.txt"
            for i in [8, 9]]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(args.seed)

def load_zip(member):
    with zipfile.ZipFile(args.data) as zf:
        with zf.open(member) as f: return np.loadtxt(f).astype(np.float32)

def load_data():
    cf7   = load_zip(TRAIN_F)
    sp    = int(0.8 * cf7.shape[1])
    train = cf7[:, :sp]; val = cf7[:, sp + EMBARGO:]
    tests = [load_zip(f) for f in TEST_FS]
    print(f"train={train.shape[1]} val={val.shape[1]} test={[t.shape[1] for t in tests]}")
    return train, val, tests

class LOBDataset(Dataset):
    def __init__(self, data):
        x = data[:N_FEATS].T.astype(np.float32)
        y = (data[-5:].T[:, K_INDEX] - 1).astype(np.int64)
        assert y.min() >= 0 and y.max() < N_CLASSES
        self.x = torch.from_numpy(x); self.y = torch.from_numpy(y)
        self._n = len(self.x) - SEQ_SIZE + 1
    def __len__(self): return self._n
    def __getitem__(self, i):
        # Returns (T, 144) — model internally extracts price/vol cols
        return self.x[i:i+SEQ_SIZE], self.y[i+SEQ_SIZE-1]

def make_loaders(tr, vl, tests):
    nw = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=(nw>0))
    tr_dl = DataLoader(LOBDataset(tr),  args.batch_size, shuffle=True,  **kw)
    vl_dl = DataLoader(LOBDataset(vl),  args.batch_size, shuffle=False, **kw)
    te_dl = DataLoader(ConcatDataset([LOBDataset(t) for t in tests]), args.batch_size, False, **kw)
    print(f"windows: train={len(tr_dl.dataset):,} val={len(vl_dl.dataset):,} test={len(te_dl.dataset):,}")
    return tr_dl, vl_dl, te_dl

# ── Model ─────────────────────────────────────────────────────────────────────
class StreamEncoder(nn.Module):
    """
    CNN encoder for a single price or volume stream.
    Input: (B, 1, T, 20) — 20 price or 20 volume levels over T timesteps.
    Output: (B, T', d) after Conv2d reduction + mean pooling over feature dim.
    """
    def __init__(self, d=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (1,2), stride=(1,2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4,1)),              nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, d,  (4,1)),              nn.LeakyReLU(0.01), nn.BatchNorm2d(d),
        )
    def forward(self, x):
        return self.conv(x).mean(dim=-1).permute(0, 2, 1)  # (B, T', d)

class CrossAttnBlock(nn.Module):
    """Cross-attention: query attends to key/value context."""
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(nn.Linear(d, d*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d*2, d))
        self.norm2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(0.1)
    def forward(self, q, k, v):
        a, _ = self.attn(q, k, v)
        x = self.norm1(q + self.drop(a))
        return self.norm2(x + self.drop(self.ffn(x)))

class PVSLOB(nn.Module):
    """
    Price-Volume separated LOB model:
    - StreamEncoder for price (every-other-col from first 40) and volume
    - Bidirectional cross-attention: price-queries-volume and volume-queries-price
    - Fusion [p, v, p-v, p⊙v] → LSTM(256,128) → FC(128,3)
    """
    def __init__(self):
        super().__init__()
        self.pe  = StreamEncoder(64); self.ve = StreamEncoder(64)
        self.p2v = CrossAttnBlock(64); self.v2p = CrossAttnBlock(64)
        self.lstm = nn.LSTM(256, 128, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(128, N_CLASSES)

    def forward(self, x):  # x: (B, T, 144)
        # Extract price (even indices 0,2,...,38) and volume (odd 1,3,...,39) from raw LOB
        p = self.pe(x[:, :, 0:40:2].unsqueeze(1))  # (B, T', 64)  — price stream
        v = self.ve(x[:, :, 1:40:2].unsqueeze(1))  # (B, T', 64)  — volume stream
        po = self.p2v(p, v, v)                       # price queries volume context
        vo = self.v2p(v, p, p)                       # volume queries price context
        cat = torch.cat([po, vo, po-vo, po*vo], dim=-1)  # (B, T', 256)
        h0 = torch.zeros(1, x.size(0), 128, device=x.device)
        c0 = torch.zeros(1, x.size(0), 128, device=x.device)
        out, _ = self.lstm(cat, (h0, c0))
        return self.fc(out[:, -1, :])

def evaluate(model, loader, criterion):
    model.eval(); losses, preds, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x); losses.append(criterion(logits, y).item())
            preds.extend(logits.argmax(1).cpu().tolist()); targets.extend(y.cpu().tolist())
    return {"loss": float(np.mean(losses)),
            "accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "report": classification_report(targets, preds, labels=[0,1,2],
                target_names=["Down","Stationary","Up"], digits=4, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(targets, preds, labels=[0,1,2]).tolist()}

def save_plots(history, test_metrics, prefix):
    eps = [h["epoch"] for h in history]
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(eps, [h["train_loss"] for h in history], label="Train loss")
    plt.plot(eps, [h["val_loss"]   for h in history], label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss curve"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(eps, [h["val_f1"] for h in history], label="Val macro-F1", color="orange")
    plt.xlabel("Epoch"); plt.ylabel("Macro F1"); plt.title("Val F1"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(prefix + "_loss_curve.png", dpi=150); plt.close()
    rep = test_metrics["report"]; classes = ["Down","Stationary","Up"]
    f1s = [rep[c]["f1-score"] for c in classes]
    plt.figure(figsize=(6,4)); bars = plt.bar(classes, f1s, color=["#e74c3c","#3498db","#2ecc71"])
    for b, v in zip(bars, f1s): plt.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=10)
    plt.ylim(0,1.05); plt.ylabel("F1"); plt.title(f"Per-class F1 ({NORM} PVSLOB)"); plt.grid(axis="y")
    plt.tight_layout(); plt.savefig(prefix + "_f1_bar.png", dpi=150); plt.close()
    cm = np.array(test_metrics["confusion_matrix"])
    plt.figure(figsize=(5,4)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
    for i in range(3):
        for j in range(3): plt.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=11)
    plt.xticks([0,1,2], classes); plt.yticks([0,1,2], classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(prefix + "_confusion.png", dpi=150); plt.close()

def train(model, tr_dl, vl_dl, te_dl):
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=1e-4)
    best_f1, best_state, best_ep, history = -1.0, None, -1, []
    print(f"\n{'='*65}\nPVSLOB | {NORM} | K_INDEX={K_INDEX} (k=100) | lr={args.lr} | N_FEATS={N_FEATS}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n{'='*65}")
    for ep in range(1, args.epochs+1):
        t0 = datetime.now(); model.train(); tl = tc = tn = 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True); logits = model(x); loss = crit(logits, y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            bs = y.size(0); tl += loss.item()*bs; tn += bs; tc += (logits.argmax(1)==y).sum().item()
        vm = evaluate(model, vl_dl, crit)
        is_best = vm["macro_f1"] > best_f1
        if is_best: best_f1 = vm["macro_f1"]; best_ep = ep; best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch":ep,"train_loss":tl/tn,"train_acc":tc/tn,
                         "val_loss":vm["loss"],"val_acc":vm["accuracy"],"val_f1":vm["macro_f1"]})
        print(f"Ep {ep:03d} | loss={tl/tn:.4f} acc={tc/tn:.4f} | val_acc={vm['accuracy']:.4f} "
              f"val_f1={vm['macro_f1']:.4f} {'*' if is_best else ''} | {datetime.now()-t0}", flush=True)
    model.load_state_dict(best_state)
    tm = evaluate(model, te_dl, crit)
    print(f"\nBEST ep={best_ep} val_f1={best_f1:.4f} | TEST acc={tm['accuracy']:.4f} F1={tm['macro_f1']:.4f}")
    return history, best_ep, best_f1, tm

def main():
    set_seed(args.seed)
    tr, vl, tests = load_data()
    tr_dl, vl_dl, te_dl = make_loaders(tr, vl, tests)
    model = PVSLOB().to(DEVICE)
    history, best_ep, best_f1, tm = train(model, tr_dl, vl_dl, te_dl)
    prefix = args.out.replace(".json","")
    results = {"model":"PVSLOB","norm":NORM,"k_index":K_INDEX,"n_feats":N_FEATS,
               "protocol":"CF_7 train/val (embargo=99), Test CF_8+CF_9",
               "note":"Model uses price cols[0::2] + vol cols[1::2] from first 40 raw LOB features",
               "hyperparams":{"epochs":args.epochs,"batch_size":args.batch_size,
                              "lr":args.lr,"seq_size":SEQ_SIZE,"seed":args.seed,"embargo":EMBARGO},
               "best_epoch":best_ep,"best_val_f1":best_f1,"test":tm,"history":history}
    with open(args.out,"w") as f: json.dump(results, f, indent=2)
    save_plots(history, tm, prefix)
    print(f"\nSaved → {args.out}  |  acc={tm['accuracy']:.4f}  macro-F1={tm['macro_f1']:.4f}")

if __name__ == "__main__": main()
