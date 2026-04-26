import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier

import optuna
from optuna.samplers import TPESampler

# -----------------------------
# 1) Load data
# -----------------------------
X = pd.read_csv("expression.csv", index_col=0)
metadata = pd.read_csv("info.csv", index_col=0)
assert all(X.index == metadata.index)

le = LabelEncoder()
y = le.fit_transform(metadata['DiseaseSubtype.1'])

labels_order = np.unique(y)
class_names = ['CS', 'HV', 'PA', 'PHT', 'PPGL']  

# -----------------------------
# 2) CV setup
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# 3) Evaluation helper
# -----------------------------
def evaluate_params(params):
    """Return mean balanced accuracy over CV for given RF params."""
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            **params
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        scores.append(balanced_accuracy_score(y_te, y_pred))
    return float(np.mean(scores))

# -----------------------------
# 4) Optuna objective with SAME BOUNDARY as GridSearch
# -----------------------------
def objective(trial: optuna.Trial) -> float:
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [300, 400, 500, 600]),
        'max_depth': trial.suggest_categorical('max_depth', [None, 30, 40, 50]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),  
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
    }
    return evaluate_params(params)

# -----------------------------
# 5) Run Optuna 
# -----------------------------
storage_url = 'sqlite:///rf_optuna_study.db'
study_name = 'rf_optuna_same_boundary'

sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction='maximize',
    study_name=study_name,
    storage=storage_url,
    load_if_exists=True,
    sampler=sampler,
)

# ---- Early stop if no recent improvement ----
def stop_when_no_improvement(study, trial):
    window = 30            # check last 30 trials
    min_improvement = 0.002  # require ≥0.002 improvement

    if len(study.trials) < window:
        return

    vals = [t.value for t in study.trials if t.value is not None]
    if not vals:
        return
    best_val = max(vals)

    recent_vals = [t.value for t in study.trials[-window:] if t.value is not None]
    if not recent_vals:
        return

    if max(recent_vals) <= best_val - min_improvement:
        print(f"Stopping early: no improvement ≥ {min_improvement} in last {window} trials.")
        study.stop()

n_trials = int(os.getenv('OPTUNA_N_TRIALS', '60'))  # default 60

start_time = time.time()
study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[stop_when_no_improvement])
end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# 6) Build CSVs with SAME FILENAMES as GridSearch version
# -----------------------------
trials_data = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    trials_data.append({
        'params': t.params,
        'mean_test_score': t.value,
        'std_test_score': np.nan,
        'rank_test_score': None,  
        'number': t.number,
        'datetime_start': t.datetime_start,
        'datetime_complete': t.datetime_complete,
    })

results_df = pd.DataFrame(trials_data)
results_df['rank_test_score'] = results_df['mean_test_score'].rank(ascending=False, method='min').astype(int)
results_df = results_df.sort_values('rank_test_score')
results_df.to_csv("RF_Optuna_full_results_2.csv", index=False)

summary_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].rename(
    columns={'mean_test_score': 'mean_balanced_accuracy', 'std_test_score': 'std_balanced_accuracy'}
).sort_values('rank_test_score')
summary_df.to_csv("RF_Optuna_balanced_accuracy_summary_2.csv", index=False)

print("\nBalanced Accuracy Summary (RF, Optuna):")
print(summary_df.head())

# -----------------------------
# 7) Extended metrics per parameter set (sens/spec/acc/F1)
# -----------------------------
extended_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nComputing full metrics (sensitivity, specificity, accuracy, F1) for RF — Optuna trials...\n")
for _, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
    params = dict(row['params'])

    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=labels_order)

        TN_FP_FN_TP = []
        for class_idx in range(len(labels_order)):
            TP = cm[class_idx, class_idx]
            FN = cm[class_idx, :].sum() - TP
            FP = cm[:, class_idx].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            TN_FP_FN_TP.append((TN, FP, FN, TP))

        class_sens = [TP / (TP + FN) if (TP + FN) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        class_spec = [TN / (TN + FP) if (TN + FP) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]

        all_class_sensitivities.append(class_sens)
        all_class_specificities.append(class_spec)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='macro'))

    mean_sens = np.mean(all_class_sensitivities, axis=0)
    mean_spec = np.mean(all_class_specificities, axis=0)
    mean_accuracy = float(np.mean(fold_accuracies))
    mean_f1_macro = float(np.mean(fold_f1s))

    result_row = {
        'mean_balanced_accuracy': float(row['mean_balanced_accuracy']),
        'rank_balanced_accuracy': int(row['rank_test_score']),
        'mean_accuracy': mean_accuracy,
        'mean_f1_macro': mean_f1_macro,
    }
    result_row.update(params)

    for idx, val in enumerate(mean_sens):
        result_row[f'sensitivity_class_{idx}'] = float(val)
    for idx, val in enumerate(mean_spec):
        result_row[f'specificity_class_{idx}'] = float(val)

    extended_results.append(result_row)

full_metrics_df = pd.DataFrame(extended_results).sort_values('rank_balanced_accuracy')

# Rename class metrics 
n_classes = len(labels_order)
if len(class_names) == n_classes:
    rename_dict = {f'sensitivity_class_{i}': f'sensitivity_{name}' for i, name in enumerate(class_names)}
    rename_dict.update({f'specificity_class_{i}': f'specificity_{name}' for i, name in enumerate(class_names)})
    full_metrics_df = full_metrics_df.rename(columns=rename_dict)

full_metrics_df.to_csv("RF_Optuna_sens_spec_f1_acc_report_2.csv", index=False)

# -----------------------------
# 8) Print best
# -----------------------------
print(f"\nOptuna took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print("Best RF parameters:", study.best_trial.params)
print("Best RF Balanced Accuracy:", study.best_value)
print("\nTop Rows from Extended Metrics (RF, Optuna):")
print(full_metrics_df.head())

