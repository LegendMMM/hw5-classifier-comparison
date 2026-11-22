"""
HW5 Part 1 — ScikitLearn 操作記錄單的練習程式
每個 function 用兩組不同設定各執行一次，輸出存到 part1_results.json。
分類用 Breast Cancer（80/20 stratified, random_state=42）；
迴歸用 Diabetes；MultinomialNB 用 Digits（非負的計數型 features）。
"""
import json
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve, r2_score)
from sklearn.model_selection import (KFold, ShuffleSplit, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RS = 42
out = {}

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=RS)

Xd, yd = load_digits(return_X_y=True)
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=RS)


def clf(m, a=Xtr, b=Xte):
    m.fit(a, ytr)
    p = m.predict(b)
    return {"acc": round(accuracy_score(yte, p), 4), "f1": round(f1_score(yte, p, average="macro"), 4),
            "train_acc": round(accuracy_score(ytr, m.predict(a)), 4)}


def reg(m):
    m.fit(Xr_tr, yr_tr)
    return {"r2": round(r2_score(yr_te, m.predict(Xr_te)), 4),
            "train_r2": round(r2_score(yr_tr, m.predict(Xr_tr)), 4)}


out["KNeighborsClassifier"] = {
    "n_neighbors=1 (scaled)": clf(KNeighborsClassifier(n_neighbors=1), Xtr_s, Xte_s),
    "n_neighbors=7 (scaled)": clf(KNeighborsClassifier(n_neighbors=7), Xtr_s, Xte_s),
    "n_neighbors=7, weights=distance (scaled)": clf(KNeighborsClassifier(n_neighbors=7, weights="distance"), Xtr_s, Xte_s),
}
out["KNeighborsRegressor"] = {
    "n_neighbors=1": reg(KNeighborsRegressor(n_neighbors=1)),
    "n_neighbors=15": reg(KNeighborsRegressor(n_neighbors=15)),
}
out["GaussianNB"] = {
    "var_smoothing=1e-9 (default)": clf(GaussianNB()),
    "var_smoothing=1e-2": clf(GaussianNB(var_smoothing=1e-2)),
}
m1 = MultinomialNB(alpha=1.0).fit(Xd_tr, yd_tr)
m2 = MultinomialNB(alpha=10.0).fit(Xd_tr, yd_tr)
out["MultinomialNB (digits)"] = {
    "alpha=1.0": {"acc": round(m1.score(Xd_te, yd_te), 4)},
    "alpha=10.0": {"acc": round(m2.score(Xd_te, yd_te), 4)},
}
t1 = DecisionTreeClassifier(random_state=RS)
t2 = DecisionTreeClassifier(max_depth=3, random_state=RS)
r1, r2 = clf(t1), clf(t2)
r1["depth"], r1["leaves"] = int(t1.get_depth()), int(t1.get_n_leaves())
r2["depth"], r2["leaves"] = int(t2.get_depth()), int(t2.get_n_leaves())
out["DecisionTreeClassifier"] = {"max_depth=None": r1, "max_depth=3": r2,
                                 "criterion=entropy, max_depth=3": clf(DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=RS))}
out["DecisionTreeRegressor"] = {
    "max_depth=None": reg(DecisionTreeRegressor(random_state=RS)),
    "max_depth=3": reg(DecisionTreeRegressor(max_depth=3, random_state=RS)),
}
out["LinearSVC"] = {
    "C=0.01 (scaled)": clf(LinearSVC(C=0.01, random_state=RS), Xtr_s, Xte_s),
    "C=10 (scaled)": clf(LinearSVC(C=10, random_state=RS, max_iter=5000), Xtr_s, Xte_s),
}
s1, s2, s3 = SVC(kernel="rbf", C=1.0), SVC(kernel="rbf", C=1.0), SVC(kernel="linear", C=1.0)
out["SVC"] = {
    "rbf, C=1, raw features": clf(s1),
    "rbf, C=1, scaled": clf(s2, Xtr_s, Xte_s),
    "linear, C=1, scaled": clf(s3, Xtr_s, Xte_s),
}
out["SVC"]["rbf, C=1, scaled"]["n_support"] = int(s2.n_support_.sum())
mlp1 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RS)
mlp2 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RS)
mlp3 = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=RS)
out["MLPClassifier"] = {
    "hidden=(100,), raw": clf(mlp1),
    "hidden=(100,), scaled": clf(mlp2, Xtr_s, Xte_s),
    "hidden=(5,), scaled": clf(mlp3, Xtr_s, Xte_s),
}
out["MLPRegressor"] = {
    "hidden=(10,), scaled, max_iter=200": reg(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(10,), max_iter=200, random_state=RS))),
    "hidden=(100,50), scaled, max_iter=2000": reg(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=RS))),
}
out["RandomForestClassifier"] = {
    "n_estimators=5": clf(RandomForestClassifier(n_estimators=5, random_state=RS)),
    "n_estimators=200": clf(RandomForestClassifier(n_estimators=200, random_state=RS)),
    "n_estimators=200, max_depth=2": clf(RandomForestClassifier(n_estimators=200, max_depth=2, random_state=RS)),
}
out["GradientBoosting"] = {
    "Clf lr=0.1, n=100": clf(GradientBoostingClassifier(random_state=RS)),
    "Clf lr=0.1, n=5": clf(GradientBoostingClassifier(n_estimators=5, random_state=RS)),
    "Reg lr=0.1, n=100": reg(GradientBoostingRegressor(random_state=RS)),
    "Reg lr=0.05, n=300, max_depth=2": reg(GradientBoostingRegressor(learning_rate=0.05, n_estimators=300, max_depth=2, random_state=RS)),
}

# ---- Evaluation ----
pipe = make_pipeline(StandardScaler(), SVC())
kf_plain = cross_val_score(pipe, Xtr, ytr, cv=KFold(5), scoring="f1_macro")
kf_shuf = cross_val_score(pipe, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=RS), scoring="f1_macro")
skf = cross_val_score(pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), scoring="f1_macro")
out["KFold"] = {
    "KFold(5) no shuffle": {"folds": [round(v, 4) for v in kf_plain], "mean": round(kf_plain.mean(), 4), "std": round(kf_plain.std(), 4)},
    "KFold(5, shuffle=True)": {"folds": [round(v, 4) for v in kf_shuf], "mean": round(kf_shuf.mean(), 4), "std": round(kf_shuf.std(), 4)},
    "StratifiedKFold(5, shuffle=True)": {"folds": [round(v, 4) for v in skf], "mean": round(skf.mean(), 4), "std": round(skf.std(), 4)},
}
ss1 = cross_val_score(pipe, Xtr, ytr, cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=RS), scoring="f1_macro")
ss2 = cross_val_score(pipe, Xtr, ytr, cv=ShuffleSplit(n_splits=20, test_size=0.1, random_state=RS), scoring="f1_macro")
out["ShuffleSplit"] = {
    "n_splits=5, test_size=0.2": {"mean": round(ss1.mean(), 4), "std": round(ss1.std(), 4)},
    "n_splits=20, test_size=0.1": {"mean": round(ss2.mean(), 4), "std": round(ss2.std(), 4)},
}
pipe.fit(Xtr, ytr)
pred = pipe.predict(Xte)
cm = confusion_matrix(yte, pred)
rep = classification_report(yte, pred, target_names=["malignant", "benign"], digits=4)
svc_prob = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False)).fit(Xtr, ytr)
prec, rec, thr = precision_recall_curve(yte, svc_prob.predict_proba(Xte)[:, 1])
# 找 recall >= 0.99 時最高的 precision
mask = rec[:-1] >= 0.99
i = np.argmax(prec[:-1] * mask)
out["metrics"] = {
    "confusion_matrix": cm.tolist(),
    "classification_report": rep,
    "f1_macro": round(f1_score(yte, pred, average="macro"), 4),
    "f1_binary_pos=benign": round(f1_score(yte, pred), 4),
    "f1_binary_pos=malignant": round(f1_score(yte, pred, pos_label=0), 4),
    "pr_curve_points": int(len(thr)),
    "pr_at_recall>=0.99": {"threshold": round(float(thr[i]), 4), "precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4)},
}

with open("part1_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
for k, v in out.items():
    print(f"\n## {k}")
    if k == "metrics":
        print("confusion_matrix:", v["confusion_matrix"])
        print(v["classification_report"])
        print({kk: vv for kk, vv in v.items() if kk not in ("confusion_matrix", "classification_report")})
    else:
        for kk, vv in v.items():
            print(f"  {kk:42s} {vv}")
