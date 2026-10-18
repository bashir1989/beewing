"""Classical baselines on the same 5 grouped folds as the CNN.
A) PCA on pixels (160x72) + LDA     B) HOG + PCA + RBF-SVM
Hyper-parameters (PCA size / SVM C) are tuned by grouped 4-fold CV inside each training part."""
import json, sys
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
sys.path.insert(0, "/mnt/project-files/wing_classifier")
import train_wing_cnn as t
import os
DATA = os.environ.get("WING_DATA", "/mnt/project-files/subspecies Train/Normalized_PNG")

X = np.load(sys.argv[1])[:, 0]          # (N,144,320) cached grayscale images
out = sys.argv[2]
items = t.list_images(DATA)
y = np.array([c for _, c, _ in items]); g = np.array([gg for _, _, gg in items])

pix = X[:, ::2, ::2].reshape(len(X), -1)
H = np.stack([hog(im, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)) for im in X])
print("pixel feat", pix.shape, "hog feat", H.shape, flush=True)

models = {
    "pca_lda": (pix, make_pipeline(StandardScaler(), PCA(random_state=0), LinearDiscriminantAnalysis()),
                {"pca__n_components": [40, 80, 160, 320, 0.95, 0.99]}),
    "hog_svm": (H, make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0), SVC(class_weight="balanced")),
                {"svc__C": [1, 10, 100]}),
}
outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(pix, y, g))
res = {}
for name, (F, pipe, grid) in models.items():
    rows = []
    for k, (tr, te) in enumerate(outer):
        gs = GridSearchCV(pipe, grid, cv=StratifiedGroupKFold(4, shuffle=True, random_state=1), n_jobs=1)
        gs.fit(F[tr], y[tr], groups=g[tr])
        p = gs.predict(F[te])
        col = {}
        for gg, tt, pp in zip(g[te], y[te], p): col.setdefault(gg, [tt, []])[1].append(pp)
        cacc = float(np.mean([tt == np.bincount(ps, minlength=3).argmax() for tt, ps in col.values()]))
        rows.append(dict(fold=k, acc=float((p == y[te]).mean()), macro_f1=float(f1_score(y[te], p, average="macro")),
                         colony_acc=cacc, n_col=len(col), best=gs.best_params_, pred=p.tolist(), idx=te.tolist()))
        print(name, k, rows[-1]["acc"], cacc, gs.best_params_, flush=True)
    res[name] = rows
json.dump(res, open(out, "w"))
