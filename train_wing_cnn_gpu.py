"""
Wing lineage classifier (C / M / O) with a convolutional neural network.

- Input: normalized wing images (wing on white background, 1000x450, base left).
- Split: 80% train / 20% test, stratified by lineage and grouped by colony
  (the number before the first '-' in the filename), so the same colony never
  appears in both train and test.
- Model: a small CNN trained from scratch (default), or a pretrained ResNet18
  with --pretrained (needs internet to download ImageNet weights).

GPU version: runs on an NVIDIA GPU (CUDA) when one is available, otherwise on the CPU.

Usage (Windows example):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn pillow
    python train_wing_cnn_gpu.py --data "E:\\ML\\subspecies\\subspecies Train\\Normalized_PNG" --out results
"""
import argparse, json, os, random, re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["C Lineage", "M Lineage", "O Lineage"]
EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def colony_id(name):
    return re.split(r"[-_ ]", name.strip())[0]


def list_images(root):
    items = []
    for ci, c in enumerate(CLASSES):
        for p in sorted((Path(root) / c).iterdir()):
            if p.suffix.lower() in EXTS:
                items.append((p, ci, f"{ci}_{colony_id(p.name)}"))
    return items


def load(path, size):
    im = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def augment(x):
    """x: (B,1,H,W) tensor. Small random affine + brightness/contrast jitter."""
    B, dev = x.shape[0], x.device
    ang = (torch.rand(B, device=dev) - 0.5) * 2 * np.deg2rad(4)
    sc = 1 + (torch.rand(B, device=dev) - 0.5) * 0.10
    tx = (torch.rand(B, device=dev) - 0.5) * 0.08
    ty = (torch.rand(B, device=dev) - 0.5) * 0.08
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # pad with white (1.0) background
    x = 1 - F.grid_sample(1 - x, grid, align_corners=False, padding_mode="zeros")
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.3
    b = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.1
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


class SmallCNN(nn.Module):
    def __init__(self, n_classes=3, w=32, p_drop=0.4):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w), block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x - 0.5))


def make_resnet(n_classes, weights_path=None):
    import torchvision
    if weights_path:  # local ImageNet checkpoint, e.g. timm resnet18_a1_0-d63eafa0.pth
        m = torchvision.models.resnet18()
        m.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    else:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__(); self.m = m
            self.register_buffer("mean", mean); self.register_buffer("std", std)
        def forward(self, x):
            return self.m((x.repeat(1, 3, 1, 1) - self.mean) / self.std)
    return Wrap()


def run_epoch(model, X, y, opt=None, bs=16, aug=False, weights=None):
    train = opt is not None
    model.train(train)
    idx = torch.randperm(len(X), device=X.device) if train else torch.arange(len(X), device=X.device)
    tot, correct, n = 0.0, 0, 0
    preds = []
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), bs):
            j = idx[i:i + bs]
            xb, yb = X[j], y[j]
            if aug: xb = augment(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb, weight=weights, label_smoothing=0.05 if train else 0.0)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j); n += len(j)
            p = out.argmax(1); correct += (p == yb).sum().item(); preds.append(p)
    return tot / n, correct / n, torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with C Lineage / M Lineage / O Lineage subfolders")
    ap.add_argument("--out", default="results")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet ResNet18 instead of the small CNN")
    ap.add_argument("--weights", default=None, help="local ResNet18 ImageNet weights file (used with --pretrained)")
    ap.add_argument("--fold", type=int, default=0, help="which of the 5 grouped folds is the test set (0-4)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache", default=None, help="optional .npy file to cache the resized images")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    args = ap.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads or os.cpu_count() or 4)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (no CUDA GPU found; install the CUDA build of PyTorch to use the GPU)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    items = list_images(args.data)
    paths = [p for p, _, _ in items]
    y_all = np.array([c for _, c, _ in items])
    g_all = np.array([g for _, _, g in items])
    print(f"{len(items)} images, per class: {np.bincount(y_all).tolist()}, colonies: {len(set(g_all))}")

    # 80/20 split: one fold of a 5-fold stratified group split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    tr_idx, te_idx = list(sgkf.split(np.zeros(len(y_all)), y_all, g_all))[args.fold]
    # validation (for picking the best epoch) taken from the training part, also grouped
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 1)
    a, b = next(sgkf2.split(np.zeros(len(tr_idx)), y_all[tr_idx], g_all[tr_idx]))
    fit_idx, val_idx = tr_idx[a], tr_idx[b]
    assert not set(g_all[tr_idx]) & set(g_all[te_idx])

    split = {"train": [str(paths[i]) for i in fit_idx], "val": [str(paths[i]) for i in val_idx],
             "test": [str(paths[i]) for i in te_idx]}
    (out / "split.json").write_text(json.dumps(split, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, ix in [("train", fit_idx), ("val", val_idx), ("test", te_idx)]:
        print(f"{k:5s}: {len(ix):4d} images, per class {np.bincount(y_all[ix], minlength=3).tolist()}, colonies {len(set(g_all[ix]))}")

    print("loading images ...")
    if args.cache and os.path.exists(args.cache):
        X = torch.from_numpy(np.load(args.cache))
    else:
        X = torch.from_numpy(np.stack([load(p, (args.width, args.height)) for p in paths]))[:, None]
        if args.cache: np.save(args.cache, X.numpy())
    Y = torch.from_numpy(y_all).long()
    X, Y = X.to(dev), Y.to(dev)
    fit_t, val_t, te_t = (torch.from_numpy(i).to(dev) for i in (fit_idx, val_idx, te_idx))

    model = (make_resnet(3, args.weights) if args.pretrained else SmallCNN(3)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr if not args.pretrained else 3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=args.epochs * ((len(fit_idx) + 15) // 16))
    cnt = np.bincount(y_all[fit_idx], minlength=3)
    w = torch.tensor(len(fit_idx) / (3 * cnt), dtype=torch.float32, device=dev)

    class StepOpt:  # step scheduler together with optimizer
        def zero_grad(self): opt.zero_grad()
        def step(self): opt.step(); sched.step()

    best, hist = (-1, 1e9), []
    for ep in range(1, args.epochs + 1):
        tl, ta, _ = run_epoch(model, X[fit_t], Y[fit_t], StepOpt(), aug=True, weights=w)
        vl, va, _ = run_epoch(model, X[val_t], Y[val_t])
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"epoch {ep:3d}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}", flush=True)
        if (va, -vl) > (best[0], -best[1]):
            best = (va, vl); torch.save(model.state_dict(), out / "best_model.pt")

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev))
    _, te_acc, pred = run_epoch(model, X[te_t], Y[te_t])
    yt, yp = y_all[te_idx], pred.cpu().numpy()
    rep = classification_report(yt, yp, target_names=CLASSES, digits=3)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    # colony-level vote: majority of wing predictions per colony
    col = {}
    for g, t, p in zip(g_all[te_idx], yt, yp): col.setdefault(g, [t, []])[1].append(p)
    col_acc = np.mean([t == np.bincount(ps, minlength=3).argmax() for t, ps in col.values()])

    txt = (f"Test accuracy (per wing): {te_acc:.3f}\nTest accuracy (per colony, majority vote): {col_acc:.3f} "
           f"({len(col)} colonies)\nBest val acc: {best[0]:.3f}\n\n{rep}\nConfusion matrix (rows=true, cols=pred, order C,M,O):\n{cm}\n")
    print(txt)
    (out / "test_report.txt").write_text(txt, encoding="utf-8")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    with torch.no_grad():
        model.eval()
        prob = torch.cat([F.softmax(model(X[te_t][i:i + 32]), 1) for i in range(0, len(te_idx), 32)]).cpu().numpy()
    np.save(out / "test_probs.npy", prob); np.save(out / "test_idx.npy", te_idx)
    with open(out / "test_predictions.csv", "w", encoding="utf-8") as f:
        f.write("file,true,pred\n")
        for i, p in zip(te_idx, yp): f.write(f"\"{paths[i]}\",{CLASSES[y_all[i]]},{CLASSES[p]}\n")


if __name__ == "__main__":
    main()
