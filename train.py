"""Train + evaluate phishing classifiers. Run: python train.py"""
import json, joblib, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)

SEED = 42
raw = pd.read_csv("data/urldata_raw.csv")
print("Raw:", raw.shape, "| missing:", raw.isna().sum().sum(), "| duplicate rows:", raw.duplicated().sum())

# 1) Clean: remove exact duplicates (5,626 rows are repeats)
df = raw.drop_duplicates()
print("After de-dup:", df.shape, df.Label.value_counts().to_dict())

# 2) Keep only the rows we need: balanced sample (all legit rows + equal number of phishing)
n = df.Label.value_counts().min()
df = pd.concat([df[df.Label == k].sample(n, random_state=SEED) for k in (0, 1)]).sample(frac=1, random_state=SEED)
df.to_csv("data/urldata_clean.csv", index=False)
print("Balanced sample used:", df.shape)

FEATURES = [c for c in df.columns if c not in ("Domain", "Label")]
X, y = df[FEATURES], df.Label
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# --- EDA figures ---
sns.countplot(x=y); plt.title("Class balance (0=legit, 1=phishing)"); plt.savefig("eda_class_balance.png", dpi=120); plt.clf()
plt.figure(figsize=(9, 7)); sns.heatmap(df[FEATURES + ["Label"]].corr(), cmap="coolwarm", center=0)
plt.title("Correlation heatmap"); plt.tight_layout(); plt.savefig("eda_correlation.png", dpi=120); plt.clf()
df[FEATURES + ["Label"]].groupby("Label").mean().T.plot.barh(figsize=(8, 6), title="Mean feature value by class")
plt.tight_layout(); plt.savefig("eda_feature_means.png", dpi=120); plt.close()

# --- Models ---
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
}
results = {}
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5, scoring="f1").mean()
    m.fit(Xtr, ytr)
    p, pr = m.predict(Xte), m.predict_proba(Xte)[:, 1]
    results[name] = dict(cv_f1=round(cv, 3), accuracy=round(accuracy_score(yte, p), 3),
                         precision=round(precision_score(yte, p), 3), recall=round(recall_score(yte, p), 3),
                         f1=round(f1_score(yte, p), 3), roc_auc=round(roc_auc_score(yte, pr), 3),
                         confusion_matrix=confusion_matrix(yte, p).tolist())
    print(name, results[name])

best = "Logistic Regression"   # chosen for simplicity, interpretability and calibrated probabilities
model = models[best]
ConfusionMatrixDisplay.from_estimator(model, Xte, yte, display_labels=["Legit", "Phishing"]); plt.savefig("confusion_matrix.png", dpi=120); plt.close()
RocCurveDisplay.from_estimator(model, Xte, yte); plt.savefig("roc_curve.png", dpi=120); plt.close()

coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
coefs.plot.barh(figsize=(8, 6), title="Logistic Regression coefficients (+ = phishing)"); plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120); plt.close()
print(coefs.round(2))

joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
json.dump({"chosen": best, "results": results}, open("metrics.json", "w"), indent=2)
