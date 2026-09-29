"""
sal_decpre.py — SAL on FI-2010 NoAuction DecPre (CF_7 only)
============================================================
SAL: DeepLOB backbone + Sparse Attention (top-k=10) + LSTM
Protocol : Train CF_7 (80%) | Val CF_7 (20%, embargo=99)
           Test CF_8+CF_9 (ConcatDataset)
Features : 144 | K_INDEX=4 (k=100) | LR=1e-4 (SALPlus-A best from exp4_metrics.json)
"""
import argparse, copy, json, os, random, zipfile
from datetime import datetime
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data",       default="BenchmarkDatasets.zip")
parser.add_argument("--out",        default="sal_decpre_results.json")
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--batch-size", type=int,   default=64)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_SIZE=100;K_INDEX=4;N_FEATS=144;N_CLASSES=3;EMBARGO=SEQ_SIZE-1
NORM="DecPre"
PREFIX="BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/"
TRAIN_F=PREFIX+"NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt"
TEST_FS=[PREFIX+f"NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_{i}.txt" for i in [8,9]]

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
set_seed(args.seed)

def load_zip(m):
    with zipfile.ZipFile(args.data) as z:
        with z.open(m) as f: return np.loadtxt(f).astype(np.float32)

def load_data():
    cf7=load_zip(TRAIN_F);sp=int(0.8*cf7.shape[1])
    tr=cf7[:,:sp];vl=cf7[:,sp+EMBARGO:];tests=[load_zip(f) for f in TEST_FS]
    print(f"train={tr.shape[1]} val={vl.shape[1]} test={[t.shape[1] for t in tests]}")
    return tr,vl,tests

class LOBDataset(Dataset):
    def __init__(self,data):
        x=data[:N_FEATS].T.astype(np.float32);y=(data[-5:].T[:,K_INDEX]-1).astype(np.int64)
        self.x=torch.from_numpy(x);self.y=torch.from_numpy(y);self._n=len(self.x)-SEQ_SIZE+1
    def __len__(self): return self._n
    def __getitem__(self,i): return self.x[i:i+SEQ_SIZE].unsqueeze(0),self.y[i+SEQ_SIZE-1]

def make_loaders(tr,vl,tests):
    nw=min(4,os.cpu_count() or 1)
    kw=dict(num_workers=nw,pin_memory=torch.cuda.is_available(),persistent_workers=(nw>0))
    return (DataLoader(LOBDataset(tr),args.batch_size,shuffle=True,**kw),
            DataLoader(LOBDataset(vl),args.batch_size,shuffle=False,**kw),
            DataLoader(ConcatDataset([LOBDataset(t) for t in tests]),args.batch_size,False,**kw))

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,32,(1,2),stride=(1,2)),nn.LeakyReLU(0.01),nn.BatchNorm2d(32),
                                  nn.Conv2d(32,32,(4,1)),nn.LeakyReLU(0.01),nn.BatchNorm2d(32),
                                  nn.Conv2d(32,32,(4,1)),nn.LeakyReLU(0.01),nn.BatchNorm2d(32))
        self.conv2=nn.Sequential(nn.Conv2d(32,32,(1,2),stride=(1,2)),nn.Tanh(),nn.BatchNorm2d(32),
                                  nn.Conv2d(32,32,(4,1)),nn.Tanh(),nn.BatchNorm2d(32),
                                  nn.Conv2d(32,32,(4,1)),nn.Tanh(),nn.BatchNorm2d(32))
        self.conv3=nn.Sequential(nn.Conv2d(32,32,(1,36)),nn.LeakyReLU(0.01),nn.BatchNorm2d(32),
                                  nn.Conv2d(32,32,(4,1)),nn.LeakyReLU(0.01),nn.BatchNorm2d(32),
                                  nn.Conv2d(32,32,(4,1)),nn.LeakyReLU(0.01),nn.BatchNorm2d(32))
        self.b1=nn.Sequential(nn.Conv2d(32,64,(1,1),padding="same"),nn.LeakyReLU(0.01),nn.BatchNorm2d(64),
                               nn.Conv2d(64,64,(3,1),padding="same"),nn.LeakyReLU(0.01),nn.BatchNorm2d(64))
        self.b2=nn.Sequential(nn.Conv2d(32,64,(1,1),padding="same"),nn.LeakyReLU(0.01),nn.BatchNorm2d(64),
                               nn.Conv2d(64,64,(5,1),padding="same"),nn.LeakyReLU(0.01),nn.BatchNorm2d(64))
        self.b3=nn.Sequential(nn.MaxPool2d((3,1),stride=(1,1),padding=(1,0)),
                               nn.Conv2d(32,64,(1,1),padding="same"),nn.LeakyReLU(0.01),nn.BatchNorm2d(64))
    def forward(self,x):
        x=self.conv1(x);x=self.conv2(x);x=self.conv3(x)
        return torch.cat([self.b1(x),self.b2(x),self.b3(x)],dim=1).squeeze(-1).permute(0,2,1)

class SAL(nn.Module):
    def __init__(self,top_k=10):
        super().__init__()
        self.backbone=Backbone(); self.top_k=top_k
        self.q_proj=nn.Linear(192,64); self.k_proj=nn.Linear(192,64); self.v_proj=nn.Linear(192,192)
        self.lstm=nn.LSTM(192,64,batch_first=True); self.fc=nn.Linear(64,N_CLASSES)
    def forward(self,x):
        h=self.backbone(x)  # (B,82,192)
        Q=self.q_proj(h);K=self.k_proj(h);V=self.v_proj(h)
        scores=torch.bmm(Q,K.transpose(1,2))/(64**0.5)
        topk_vals,_=scores.topk(self.top_k,dim=-1)
        scores=scores.masked_fill(scores<topk_vals[...,-1:],float('-inf'))
        attn=torch.softmax(scores,dim=-1); h=torch.bmm(attn,V)
        h,_=self.lstm(h); return self.fc(h[:,-1])

def evaluate(model,loader,criterion):
    model.eval();losses,preds,targets=[],[],[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE);logits=model(x)
            losses.append(criterion(logits,y).item())
            preds.extend(logits.argmax(1).cpu().tolist());targets.extend(y.cpu().tolist())
    return {"loss":float(np.mean(losses)),"accuracy":float(accuracy_score(targets,preds)),
            "macro_f1":float(f1_score(targets,preds,average="macro",zero_division=0)),
            "report":classification_report(targets,preds,labels=[0,1,2],
                target_names=["Down","Stationary","Up"],digits=4,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(targets,preds,labels=[0,1,2]).tolist()}

def save_plots(history,tm,prefix):
    eps=[h["epoch"] for h in history]
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(eps,[h["train_loss"] for h in history],label="Train loss")
    plt.plot(eps,[h["val_loss"] for h in history],label="Val loss")
    plt.xlabel("Epoch");plt.ylabel("Loss");plt.title("Loss curve");plt.legend();plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(eps,[h["val_f1"] for h in history],label="Val macro-F1",color="orange")
    plt.xlabel("Epoch");plt.ylabel("Macro F1");plt.title("Val F1");plt.legend();plt.grid(True)
    plt.tight_layout();plt.savefig(prefix+"_loss_curve.png",dpi=150);plt.close()
    rep=tm["report"];classes=["Down","Stationary","Up"];f1s=[rep[c]["f1-score"] for c in classes]
    plt.figure(figsize=(6,4));bars=plt.bar(classes,f1s,color=["#e74c3c","#3498db","#2ecc71"])
    for b,v in zip(bars,f1s): plt.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.3f}",ha="center",fontsize=10)
    plt.ylim(0,1.05);plt.ylabel("F1");plt.title(f"Per-class F1 ({NORM} SAL)");plt.grid(axis="y")
    plt.tight_layout();plt.savefig(prefix+"_f1_bar.png",dpi=150);plt.close()
    cm=np.array(tm["confusion_matrix"])
    plt.figure(figsize=(5,4));plt.imshow(cm,cmap="Blues");plt.colorbar()
    for i in range(3):
        for j in range(3): plt.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=11)
    plt.xticks([0,1,2],classes);plt.yticks([0,1,2],classes)
    plt.xlabel("Predicted");plt.ylabel("True");plt.title("Confusion Matrix")
    plt.tight_layout();plt.savefig(prefix+"_confusion.png",dpi=150);plt.close()

def train_loop(model,tr_dl,vl_dl,te_dl):
    crit=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-8,weight_decay=1e-4)
    best_f1,best_state,best_ep,history=-1.0,None,-1,[]
    print(f"\n{'='*65}\nSAL | {NORM} | K_INDEX={K_INDEX} | lr={args.lr} | N_FEATS={N_FEATS}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n{'='*65}")
    for ep in range(1,args.epochs+1):
        t0=datetime.now();model.train();tl=tc=tn=0
        for x,y in tr_dl:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(set_to_none=True);logits=model(x);loss=crit(logits,y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            bs=y.size(0);tl+=loss.item()*bs;tn+=bs;tc+=(logits.argmax(1)==y).sum().item()
        vm=evaluate(model,vl_dl,crit)
        is_best=vm["macro_f1"]>best_f1
        if is_best: best_f1=vm["macro_f1"];best_ep=ep;best_state=copy.deepcopy(model.state_dict())
        history.append({"epoch":ep,"train_loss":tl/tn,"train_acc":tc/tn,
                         "val_loss":vm["loss"],"val_acc":vm["accuracy"],"val_f1":vm["macro_f1"]})
        print(f"Ep {ep:03d} | loss={tl/tn:.4f} | val_f1={vm['macro_f1']:.4f} {'*' if is_best else ''} | {datetime.now()-t0}",flush=True)
    model.load_state_dict(best_state)
    tm=evaluate(model,te_dl,crit)
    print(f"\nBEST ep={best_ep} val_f1={best_f1:.4f} | TEST acc={tm['accuracy']:.4f} F1={tm['macro_f1']:.4f}")
    return history,best_ep,best_f1,tm

def main():
    set_seed(args.seed);tr,vl,tests=load_data()
    tr_dl,vl_dl,te_dl=make_loaders(tr,vl,tests);model=SAL().to(DEVICE)
    history,best_ep,best_f1,tm=train_loop(model,tr_dl,vl_dl,te_dl)
    prefix=args.out.replace(".json","")
    results={"model":"SAL","norm":NORM,"k_index":K_INDEX,"n_feats":N_FEATS,
             "protocol":"CF_7 train/val (embargo=99), Test CF_8+CF_9",
             "hyperparams":{"epochs":args.epochs,"batch_size":args.batch_size,
                            "lr":args.lr,"seq_size":SEQ_SIZE,"seed":args.seed},
             "best_epoch":best_ep,"best_val_f1":best_f1,"test":tm,"history":history}
    with open(args.out,"w") as f: json.dump(results,f,indent=2)
    save_plots(history,tm,prefix)
    print(f"\nSaved → {args.out}  |  acc={tm['accuracy']:.4f}  macro-F1={tm['macro_f1']:.4f}")

if __name__=="__main__": main()
