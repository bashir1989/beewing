"""Pool the 5 grouped folds of every method: per-fold wing accuracy, macro-F1, colony accuracy, pooled confusion matrix."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
sys.path.insert(0, "/mnt/project-files/wing_classifier")
import train_wing_cnn as t
import os
DATA = os.environ.get("WING_DATA", "/mnt/project-files/subspecies Train/Normalized_PNG")
C = Path(sys.argv[1])
items = t.list_images(DATA)
y = np.array([c for _, c, _ in items]); g = np.array([gg for _, _, gg in items])
names = [str(p.relative_to(DATA)) for p, _, _ in items]

def fold_metrics(idx, pred):
    col = {}
    for i, p in zip(idx, pred): col.setdefault(g[i], [y[i], []])[1].append(p)
    cacc = np.mean([tt == np.bincount(ps, minlength=3).argmax() for tt, ps in col.values()])
    return dict(acc=float(np.mean(pred == y[idx])), f1=float(f1_score(y[idx], pred, average="macro")), col=float(cacc),
                ncol=len(col), wrong_col=[k for k, (tt, ps) in col.items() if tt != np.bincount(ps, minlength=3).argmax()])

methods = {}
# small CNN: all 5 folds from cv/
cnn = [(np.load(C / f"cnn_f{f}/test_idx.npy"), np.load(C / f"cnn_f{f}/test_probs.npy").argmax(1))
       for f in range(5) if (C / f"cnn_f{f}/test_probs.npy").exists()]
methods["CNN (from scratch)"] = cnn
methods["ResNet18 (ImageNet, fine-tuned)"] = [(np.load(C / f"res_f{f}/test_idx.npy"), np.load(C / f"res_f{f}/test_probs.npy").argmax(1))
                                              for f in range(5) if (C / f"res_f{f}/test_probs.npy").exists()]
b = json.load(open(C / "baseline.json"))
methods["PCA + LDA (pixels)"] = [(np.array(r["idx"]), np.array(r["pred"])) for r in b["pca_lda"]]
methods["HOG + SVM"] = [(np.array(r["idx"]), np.array(r["pred"])) for r in b["hog_svm"]]

out = {}
for m, folds in methods.items():
    fm = [fold_metrics(i, p) for i, p in folds]
    I = np.concatenate([i for i, _ in folds]); P = np.concatenate([p for _, p in folds])
    cm = confusion_matrix(y[I], P, labels=[0, 1, 2])
    out[m] = dict(n_folds=len(fm), folds=fm,
                  acc_mean=float(np.mean([f["acc"] for f in fm])), acc_sd=float(np.std([f["acc"] for f in fm], ddof=1)) if len(fm) > 1 else 0,
                  f1_mean=float(np.mean([f["f1"] for f in fm])), f1_sd=float(np.std([f["f1"] for f in fm], ddof=1)) if len(fm) > 1 else 0,
                  col_correct=int(sum(round(f["col"] * f["ncol"]) for f in fm)), col_total=int(sum(f["ncol"] for f in fm)),
                  recall=(cm.diagonal() / cm.sum(1)).round(3).tolist(), cm=cm.tolist(), pooled_acc=float(np.mean(P == y[I])))
    o = out[m]
    print(f"{m:34s} folds={o['n_folds']} acc={o['acc_mean']:.3f}±{o['acc_sd']:.3f} F1={o['f1_mean']:.3f}±{o['f1_sd']:.3f} "
          f"colonies {o['col_correct']}/{o['col_total']} recall C/M/O={o['recall']} per-fold acc={[round(f['acc'],3) for f in o['folds']]} wrong={[f['wrong_col'] for f in o['folds']]}")
json.dump(out, open(C / "cv_summary.json", "w"), indent=1)

# McNemar (exact binomial) on pooled per-wing correctness, all pairs
from scipy.stats import binomtest
corr = {}
for m, folds in methods.items():
    c = np.zeros(len(y), bool)
    for i, p in folds: c[i] = p == y[i]
    corr[m] = c
ms = list(methods); mc = {}
for a in range(len(ms)):
    for b2 in range(a + 1, len(ms)):
        A, B = corr[ms[a]], corr[ms[b2]]
        n01, n10 = int((A & ~B).sum()), int((~A & B).sum())
        pval = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        mc[f"{ms[a]} vs {ms[b2]}"] = dict(a_only=n01, b_only=n10, p=float(pval))
        print(f"McNemar {ms[a]} vs {ms[b2]}: only-first {n01}, only-second {n10}, p={pval:.4f}")
json.dump(dict(methods=out, mcnemar=mc), open(C / "cv_summary.json", "w"), indent=1)
