"""
HW5 分類器效能比較與模型選擇 — Part 2 ~ Part 5
執行：python hw5.py
輸出：figures/*.png、results.json（報告使用的所有數字）
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RS = 42
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
R = {}


def split(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)


def scores(model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    dt = time.perf_counter() - t0
    return {"acc": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, average="macro"),
            "time": dt}, pred


def base_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian NB": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=1.0),
    }


# ---------------- Part 2: Dataset A baseline ----------------
A = load_breast_cancer()
XA, yA = A.data, A.target
XA_tr, XA_te, yA_tr, yA_te = split(XA, yA)
R["A_info"] = {
    "n": int(XA.shape[0]), "d": int(XA.shape[1]),
    "classes": {str(A.target_names[c]): int((yA == c).sum()) for c in np.unique(yA)},
    "train": int(len(yA_tr)), "test": int(len(yA_te)),
}
rng = XA.max(0) - XA.min(0)
R["A_ranges"] = {
    "feature_min_of_max": float(XA.max(0).min()),
    "feature_max_of_max": float(XA.max(0).max()),
    "largest": [str(A.feature_names[i]) for i in np.argsort(-XA.max(0))[:2]],
    "largest_vals": [float(XA[:, i].max()) for i in np.argsort(-XA.max(0))[:2]],
    "smallest": [str(A.feature_names[i]) for i in np.argsort(XA.max(0))[:2]],
    "smallest_vals": [float(XA[:, i].max()) for i in np.argsort(XA.max(0))[:2]],
    "range_ratio": float(rng.max() / rng.min()),
}
R["A_raw"] = {k: scores(m, XA_tr, yA_tr, XA_te, yA_te)[0] for k, m in base_models().items()}

# ---------------- Part 3: Standardization ----------------
sc = StandardScaler().fit(XA_tr)               # 只用 training set fit
XA_tr_s, XA_te_s = sc.transform(XA_tr), sc.transform(XA_te)
R["A_scaled"] = {}
for k in ["kNN", "SVM", "Decision Tree", "Random Forest"]:
    R["A_scaled"][k] = scores(base_models()[k], XA_tr_s, yA_tr, XA_te_s, yA_te)[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
names = ["kNN", "SVM", "Decision Tree", "Random Forest"]
x = np.arange(len(names))
for ax, met, lab in zip(axes, ["acc", "f1"], ["Accuracy", "Macro-F1"]):
    raw = [R["A_raw"][n][met] for n in names]
    scl = [R["A_scaled"][n][met] for n in names]
    b1 = ax.bar(x - 0.2, raw, 0.4, label="Raw", color="#9aa5b1")
    b2 = ax.bar(x + 0.2, scl, 0.4, label="Standardized", color="#2f6db5")
    ax.bar_label(b1, fmt="%.3f", fontsize=7)
    ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylim(0.85, 1.025)
    ax.set_title(f"Dataset A test {lab}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/p3_standardization.png", dpi=200)
plt.close(fig)

# ---------------- Part 4: CV + hyperparameter selection ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
grids = {
    "kNN": ("n_neighbors", [1, 3, 5, 9, 15],
            lambda v: Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=v))])),
    "Decision Tree": ("max_depth", [2, 3, 5, 10, None],
                      lambda v: DecisionTreeClassifier(max_depth=v, random_state=RS)),
    "SVM": ("C", [0.1, 1, 10],
            lambda v: Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=v))])),
}
R["cv"] = {}
for name, (p, vals, make) in grids.items():
    rows = []
    for v in vals:
        r = cross_validate(make(v), XA_tr, yA_tr, cv=cv, scoring="f1_macro",
                           return_train_score=True)
        rows.append({"value": "None" if v is None else v,
                     "mean": float(r["test_score"].mean()), "std": float(r["test_score"].std()),
                     "train_mean": float(r["train_score"].mean())})
    R["cv"][name] = {"param": p, "rows": rows}
# 基準（不搜尋）
for name, m in [("Random Forest", RandomForestClassifier(n_estimators=200, random_state=RS)),
                ("Gaussian NB", GaussianNB())]:
    r = cross_validate(m, XA_tr, yA_tr, cv=cv, scoring="f1_macro", return_train_score=True)
    R["cv"][name] = {"param": None, "rows": [{"value": "default", "mean": float(r["test_score"].mean()),
                                              "std": float(r["test_score"].std()),
                                              "train_mean": float(r["train_score"].mean())}]}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, name in zip(axes, ["kNN", "Decision Tree", "SVM"]):
    rows = R["cv"][name]["rows"]
    labels = [str(r["value"]) for r in rows]
    m = np.array([r["mean"] for r in rows]); s = np.array([r["std"] for r in rows])
    tr = np.array([r["train_mean"] for r in rows])
    xi = np.arange(len(rows))
    ax.errorbar(xi, m, yerr=s, marker="o", capsize=4, color="#2f6db5", label="CV Macro-F1 (mean ± std)")
    ax.plot(xi, tr, marker="s", ls="--", color="#d9822b", label="Training Macro-F1")
    ax.set_xticks(xi, labels)
    ax.set_xlabel(R["cv"][name]["param"])
    ax.set_title(name)
    ax.set_ylim(0.85, 1.005)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Macro-F1")
axes[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/p4_cv_hyperparameter.png", dpi=200)
plt.close(fig)


TOL = 0.005  # 平均 Macro-F1 差距小於 0.5 個百分點視為「很接近」


def pick(name, prefer):
    """取 mean 最高；與最高差距 <= TOL 的設定視為平手，依 prefer 規則選較簡單 / 較不易 overfit 者。"""
    rows = R["cv"][name]["rows"]
    best = max(rows, key=lambda r: r["mean"])
    close = [r for r in rows if best["mean"] - r["mean"] <= TOL]
    return best, prefer(close), close


best_k, sel_k, close_k = pick("kNN", lambda c: max(c, key=lambda r: r["value"]))          # k 越大越平滑
best_d, sel_d, close_d = pick("Decision Tree",
                              lambda c: min(c, key=lambda r: 99 if r["value"] == "None" else r["value"]))  # 越淺越簡單
best_c, sel_c, close_c = pick("SVM", lambda c: min(c, key=lambda r: r["value"]))          # C 小 = margin 寬
R["selection"] = {
    "kNN": {"best": best_k["value"], "selected": sel_k["value"], "close": [r["value"] for r in close_k]},
    "Decision Tree": {"best": best_d["value"], "selected": sel_d["value"], "close": [r["value"] for r in close_d]},
    "SVM": {"best": best_c["value"], "selected": sel_c["value"], "close": [r["value"] for r in close_c]},
}
dval = None if sel_d["value"] == "None" else sel_d["value"]
final = {
    "kNN": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=sel_k["value"]))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=dval, random_state=RS),
    "SVM": Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=sel_c["value"]))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS),
    "Gaussian NB": GaussianNB(),
}
R["A_final"] = {k: scores(m, XA_tr, yA_tr, XA_te, yA_te)[0] for k, m in final.items()}

# ---------------- Part 5: Dataset B (Digits) ----------------
B = load_digits()
XB, yB = B.data, B.target
XB_tr, XB_te, yB_tr, yB_te = split(XB, yB)
R["B_info"] = {"n": int(XB.shape[0]), "d": int(XB.shape[1]), "k": int(len(np.unique(yB))),
               "counts": {int(c): int((yB == c).sum()) for c in np.unique(yB)},
               "min": float(XB.min()), "max": float(XB.max()),
               "zero_var_features": int((XB.std(0) == 0).sum()),
               "zero_frac": float((XB == 0).mean())}
B_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RS),
    "kNN": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Gaussian NB": GaussianNB(),
    "SVM": Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0))]),
}
R["B"] = {}
preds = {}
for k, m in B_models.items():
    R["B"][k], preds[k] = scores(m, XB_tr, yB_tr, XB_te, yB_te)
bestB = max(R["B"], key=lambda k: (R["B"][k]["f1"], R["B"][k]["acc"]))
R["B_best"] = bestB
cm = confusion_matrix(yB_te, preds[bestB])
R["B_cm"] = cm.tolist()
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ConfusionMatrixDisplay(cm, display_labels=B.target_names).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Digits test set — {bestB}")
fig.tight_layout()
fig.savefig(f"{OUT}/p5_digits_confusion.png", dpi=200)
plt.close(fig)
# 補充：GaussianNB 在 Digits 的 confusion matrix（解釋名次變化用）
R["B_cm_gnb"] = confusion_matrix(yB_te, preds["Gaussian NB"]).tolist()

# 補充：tree 的複雜度（可解釋性 / model complexity 討論用）
dtA = DecisionTreeClassifier(random_state=RS).fit(XA_tr, yA_tr)
dtB = DecisionTreeClassifier(random_state=RS).fit(XB_tr, yB_tr)
R["tree_size"] = {"A": {"depth": int(dtA.get_depth()), "leaves": int(dtA.get_n_leaves())},
                  "B": {"depth": int(dtB.get_depth()), "leaves": int(dtB.get_n_leaves())}}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)


def show(title, d):
    print(f"\n== {title} ==")
    for k, v in d.items():
        print(f"  {k:14s} acc={v['acc']:.4f}  f1={v['f1']:.4f}  time={v['time']*1000:.1f}ms")


print("Dataset A:", R["A_info"])
print("ranges:", R["A_ranges"])
show("Part 2 raw", R["A_raw"])
show("Part 3 scaled", R["A_scaled"])
print("\n== Part 4 CV ==")
for k, v in R["cv"].items():
    for r in v["rows"]:
        print(f"  {k:14s} {v['param']}={r['value']!s:7s} mean={r['mean']:.4f} std={r['std']:.4f} train={r['train_mean']:.4f}")
print("selection:", R["selection"])
show("Part 4 final test", R["A_final"])
print("\nDataset B:", R["B_info"])
show("Part 5 digits", R["B"])
print("best B:", bestB)
print("tree size:", R["tree_size"])
