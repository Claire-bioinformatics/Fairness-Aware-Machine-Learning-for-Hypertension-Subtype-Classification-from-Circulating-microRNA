import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn import set_config
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# -----------------------------
# Settings
# -----------------------------
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# 1) Load data
# -----------------------------
X = pd.read_csv("expression.csv", index_col=0)
metadata = pd.read_csv("info.csv", index_col=0)
assert all(X.index == metadata.index), 

# Encode labels
le = LabelEncoder()
y = le.fit_transform(metadata['DiseaseSubtype.1'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 2) Class/sample weights (imbalance)
# -----------------------------
classes = np.unique(y_train)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_map = {c: w for c, w in zip(classes, cw)}
sample_weight_train = np.vectorize(class_weight_map.get)(y_train)

# -----------------------------
# 3) Tuned base models
# -----------------------------
best_rf_params = {
    "n_estimators": 300, "max_depth": None, "min_samples_split": 10,
    "criterion": "gini", "min_samples_leaf": 1
}
best_xgb_params = {
    "learning_rate": 0.15, "max_depth": 7, "n_estimators": 80,
    "subsample": 0.8, "colsample_bytree": 1.0
}
best_cat_params = {
    "iterations": 300, "depth": 4, "learning_rate": 0.01,
    "l2_leaf_reg": 1.0, "rsm": 1.0, "bootstrap_type": "Bernoulli"
}
best_svm_params = {
    "C": 10.0, "kernel": "rbf", "gamma": "scale", "degree": 3
}

rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    **best_rf_params
)

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(classes),
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False,
    **best_xgb_params
)

cat = CatBoostClassifier(
    loss_function='MultiClass',
    random_seed=42,
    verbose=0,
    class_weights=cw.tolist(),
    **best_cat_params
)

svm = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42,
    **best_svm_params
)

base_models = [
    ('rf', rf),
    ('xgb', xgb),
    ('cat', cat),
    ('svm', svm),
]

# -----------------------------
# 4) Enable metadata routing + request sample_weight
# -----------------------------
set_config(enable_metadata_routing=True)

final_lr = LogisticRegression(max_iter=1000, random_state=42)
for est in [rf, xgb, cat, svm, final_lr]:
    if hasattr(est, "set_fit_request"):
        est.set_fit_request(sample_weight=True)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=final_lr,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=False
)

# -----------------------------
# 5) Meta-learner grid only
# -----------------------------
param_grid = {
    'final_estimator__C': [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
    'final_estimator__class_weight': [None, 'balanced'],
    'final_estimator__solver': ['lbfgs', 'saga'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=stack,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=0   
)

# -----------------------------
# 6) Fit
# -----------------------------
grid.fit(X_train, y_train, sample_weight=sample_weight_train)
best_stack = grid.best_estimator_

# -----------------------------
# 7) Evaluate: base only (save PNG)
# -----------------------------
y_pred = best_stack.predict(X_test)
labels = np.arange(len(le.classes_))
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=ax, values_format='d', colorbar=False
)
plt.title("Confusion Matrix (base)")
plt.tight_layout()
outpath = os.path.join(OUTDIR, "cm_base.png")
fig.savefig(outpath, dpi=180)
plt.close(fig)

print(f"Saved: {outpath}")

