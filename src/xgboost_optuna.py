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
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
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
n_classes = len(np.unique(y))

# -----------------------------
# 2) CV + weighting setup
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


sample_weights_global = compute_sample_weight(class_weight='balanced', y=y)

def evaluate_params(params):
   
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        
        w_tr = sample_weights_global[tr_idx]

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=n_classes,
            random_state=42,
            n_jobs=-1,
            
            tree_method='hist',
            predictor='auto',
            **params
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        y_pred = model.predict(X_te)

        # sklearn BA (macro recall over classes present in test fold)
        scores.append(balanced_accuracy_score(y_te, y_pred))

    return float(np.mean(scores))

# -----------------------------
# 3) Optuna objective with SAME BOUNDARY as GridSearch
# -----------------------------
def objective(trial: optuna.Trial) -> float:
    params = {
        'learning_rate': trial.suggest_categorical('learning_rate', [0.15, 0.2, 0.25]),
        'max_depth': trial.suggest_categorical('max_depth', [4, 5, 6, 7]),
        'n_estimators': trial.suggest_categorical('n_estimators', [80, 100, 120, 150]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9]),
    }
    return evaluate_params(params)

# -----------------------------
# 4) Run Optuna 
# -----------------------------
storage_url = 'sqlite:///xgb_optuna_study.db'
study_name = 'xgb_optuna_same_boundary'

sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction='maximize',
    study_name=study_name,
    storage=storage_url,
    load_if_exists=True,
    sampler=sampler,
)

def stop_when_no_improvement(study, trial):
    window = 30
    min_improvement = 0.002
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

n_trials = int(os.getenv('OPTUNA_N_TRIALS', '60'))  

start_time = time.time()
study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[stop_when_no_improvement])
end_time = time.time()
elapsed = end_time - start_time

# -----------------------------
# 5) Save results with SAME FILENAMES as GridSearch version
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

results_xgb = pd.DataFrame(trials_data)
results_xgb['rank_test_score'] = results_xgb['mean_test_score'].rank(ascending=False, method='min').astype(int)
results_xgb = results_xgb.sort_values('rank_test_score')
results_xgb.to_csv("XGB_optuna_full_results.csv", index=False)

summary_xgb = results_xgb[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].rename(
    columns={'mean_test_score': 'mean_balanced_accuracy', 'std_test_score': 'std_balanced_accuracy'}
).sort_values('rank_test_score')
summary_xgb.to_csv("XGB_optuna_balanced_accuracy_summary.csv", index=False)

print("\nBalanced Accuracy Summary (XGB, Optuna):")
print(summary_xgb.head())

# -----------------------------
# 6) Extended metrics per parameter set (sens/spec/acc/F1)

# -----------------------------
extended_results_xgb = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nComputing full metrics (sensitivity, specificity, accuracy, F1) for XGB — Optuna trials...\n")
for _, row in tqdm(summary_xgb.iterrows(), total=len(summary_xgb)):
    params = dict(row['params'])

    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []
    fold_bal_accs = []  

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        
        sample_weights_fold = sample_weights_global[train_idx]

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=n_classes,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            predictor='auto',
            **params
        )
        model.fit(X_train, y_train, sample_weight=sample_weights_fold)
        y_pred = model.predict(X_test)

        
        fold_bal_accs.append(balanced_accuracy_score(y_test, y_pred))

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

    
    mean_bal_acc_from_preds = float(np.mean(fold_bal_accs))

    result_row = {
        'mean_balanced_accuracy': float(row['mean_balanced_accuracy']),
        'mean_balanced_accuracy_from_preds': mean_bal_acc_from_preds,  
        'delta_BA': float(mean_bal_acc_from_preds - float(row['mean_balanced_accuracy'])),
        'rank_balanced_accuracy': int(row['rank_test_score']),
        'mean_accuracy': mean_accuracy,
        'mean_f1_macro': mean_f1_macro,
    }
    result_row.update(params)

    for idx, val in enumerate(mean_sens):
        result_row[f'sensitivity_class_{idx}'] = float(val)
    for idx, val in enumerate(mean_spec):
        result_row[f'specificity_class_{idx}'] = float(val)

    extended_results_xgb.append(result_row)

full_metrics_xgb_df = pd.DataFrame(extended_results_xgb).sort_values('rank_balanced_accuracy')

# Rename class metrics if counts match
if len(class_names) == len(labels_order):
    rename_dict = {f'sensitivity_class_{i}': f'sensitivity_{name}' for i, name in enumerate(class_names)}
    rename_dict.update({f'specificity_class_{i}': f'specificity_{name}' for i, name in enumerate(class_names)})
    full_metrics_xgb_df = full_metrics_xgb_df.rename(columns=rename_dict)

full_metrics_xgb_df.to_csv("XGB_optuna_sens_spec_f1_acc_report.csv", index=False)

# -----------------------------
# 7) Print best
# -----------------------------
print(f"\nOptuna took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print("Best XGB parameters:", study.best_trial.params)
print("Best XGB Balanced Accuracy:", study.best_value)
print("\nTop Rows from Extended Metrics (XGB, Optuna):")
print(full_metrics_xgb_df.head())

