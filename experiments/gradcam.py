"""Grad-CAM for the small CNN (fold-0 model): class-average maps over correctly classified test wings."""
import sys, json
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/project-files/wing_classifier")
import train_wing_cnn as t
import os
DATA = os.environ.get("WING_DATA", "/mnt/project-files/subspecies Train/Normalized_PNG")

X = torch.from_numpy(np.load(sys.argv[1])); model_path, out_png = sys.argv[2], sys.argv[3]
te = np.load(sys.argv[4]) if len(sys.argv) > 4 else None
items = t.list_images(DATA)
y = np.array([c for _, c, _ in items])
m = t.SmallCNN(3); m.load_state_dict(torch.load(model_path, map_location="cpu")); m.eval()
feats = {}
m.features[4].register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))

def cam(x, cls):
    x = x[None].clone(); out = m(x); a = feats["a"]
    g = torch.autograd.grad(out[0, cls], a)[0]
    w = g.mean((2, 3), keepdim=True)
    c = F.relu((w * a).sum(1, keepdim=True))
    c = F.interpolate(c, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (c / (c.max() + 1e-8)).detach().numpy(), out.argmax(1).item()

idx = te if te is not None else np.arange(len(X))
names = ["C", "M", "O"]
fig, ax = plt.subplots(3, 1, figsize=(8, 8.4))
summary = {}
for k in range(3):
    ids = [i for i in idx if y[i] == k]
    maps, imgs = [], []
    for i in ids:
        c, p = cam(X[i], k)
        if p == k: maps.append(c); imgs.append(X[i, 0].numpy())
    M, I = np.mean(maps, 0), np.mean(imgs, 0)
    ax[k].imshow(I, cmap="gray"); ax[k].imshow(M, cmap="jet", alpha=0.45, vmin=0, vmax=M.max())
    ax[k].set_title(f"Lineage {names[k]}  (mean Grad-CAM, {len(maps)} correctly classified test wings)", fontsize=10)
    ax[k].axis("off")
    # where is the attention: share of map mass in thirds along the wing (base / middle / tip)
    cols = M.sum(0); th = np.array_split(cols, 3)
    summary[names[k]] = [round(float(s.sum() / cols.sum()), 3) for s in th]
plt.tight_layout(); plt.savefig(out_png, dpi=150)
print(json.dumps(summary))
