import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
)
!pip install optuna
import optuna
from optuna.samplers import GridSampler


search_space = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],   
    "degree": [3],                
}


n_trials = np.prod([len(v) for v in search_space.values()])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classes_global = np.unique(y)  


def objective(trial: optuna.Trial):
    # Pull params from Optuna grid
    C = trial.suggest_categorical("C", search_space["C"])
    kernel = trial.suggest_categorical("kernel", search_space["kernel"])
    gamma = trial.suggest_categorical("gamma", search_space["gamma"])
    degree = trial.suggest_categorical("degree", search_space["degree"])  

    
    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []
    fold_bal_acc = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = SVC(
            class_weight="balanced",
            C=C, kernel=kernel, gamma=gamma, degree=degree
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Balanced accuracy for the objective
        fold_bal_acc.append(balanced_accuracy_score(y_test, y_pred))

        # Confusion matrix using a consistent label order
        cm = confusion_matrix(y_test, y_pred, labels=classes_global)

        # Per-class TN, FP, FN, TP
        TN_FP_FN_TP = []
        for i_cls in range(len(classes_global)):
            TP = cm[i_cls, i_cls]
            FN = cm[i_cls, :].sum() - TP
            FP = cm[:, i_cls].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            TN_FP_FN_TP.append((TN, FP, FN, TP))

        # Sensitivity (recall) and specificity per class for this fold
        class_sens = [TP / (TP + FN) if (TP + FN) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        class_spec = [TN / (TN + FP) if (TN + FP) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        all_class_sensitivities.append(class_sens)
        all_class_specificities.append(class_spec)

        # Accuracy & macro-F1
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average="macro"))

    # Means across folds
    mean_bal_acc = float(np.mean(fold_bal_acc))
    mean_sens = np.mean(all_class_sensitivities, axis=0)
    mean_spec = np.mean(all_class_specificities, axis=0)
    mean_accuracy = float(np.mean(fold_accuracies))
    mean_f1 = float(np.mean(fold_f1s))

    
    trial.set_user_attr("mean_accuracy", mean_accuracy)
    trial.set_user_attr("mean_f1_macro", mean_f1)
    for idx, val in enumerate(mean_sens):
        trial.set_user_attr(f"sensitivity_class_{idx}", float(val))
    for idx, val in enumerate(mean_spec):
        trial.set_user_attr(f"specificity_class_{idx}", float(val))

    return mean_bal_acc  


sampler = GridSampler(search_space)
study = optuna.create_study(direction="maximize", sampler=sampler)

start_time = time.time()
pbar = tqdm(total=n_trials, desc="Optuna SVM (grid)")
def _callback(study, trial):
    pbar.update(1)

study.optimize(objective, n_trials=int(n_trials), callbacks=[_callback])
pbar.close()
elapsed_time = time.time() - start_time

# ===== Save results =====
# Full trials dataframe (includes params and value)
trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
trials_df = trials_df.rename(columns={"value": "mean_balanced_accuracy"})
trials_df["rank_balanced_accuracy"] = trials_df["mean_balanced_accuracy"].rank(ascending=False, method="min").astype(int)
trials_df = trials_df.sort_values("rank_balanced_accuracy")
trials_df.to_csv("SVM_Optuna_full_trials.csv", index=False)


svm_summary_df = trials_df[["params_C", "params_kernel", "params_gamma", "params_degree",
                            "mean_balanced_accuracy", "rank_balanced_accuracy"]]
svm_summary_df.to_csv("SVM_Optuna_balanced_accuracy_summary.csv", index=False)


extended_rows = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    row = {
        "mean_balanced_accuracy": t.value,
        "params_C": t.params.get("C"),
        "params_kernel": t.params.get("kernel"),
        "params_gamma": t.params.get("gamma"),
        "params_degree": t.params.get("degree"),
        "mean_accuracy": t.user_attrs.get("mean_accuracy"),
        "mean_f1_macro": t.user_attrs.get("mean_f1_macro"),
    }
    # Add class-wise metrics
    k = 0
    while f"sensitivity_class_{k}" in t.user_attrs:
        row[f"sensitivity_class_{k}"] = t.user_attrs[f"sensitivity_class_{k}"]
        k += 1
    k = 0
    while f"specificity_class_{k}" in t.user_attrs:
        row[f"specificity_class_{k}"] = t.user_attrs[f"specificity_class_{k}"]
        k += 1
    extended_rows.append(row)

svm_full_metrics_df = pd.DataFrame(extended_rows)
# Add rank for balanced accuracy like before
svm_full_metrics_df["rank_balanced_accuracy"] = svm_full_metrics_df["mean_balanced_accuracy"].rank(
    ascending=False, method="min"
).astype(int)
svm_full_metrics_df = svm_full_metrics_df.sort_values("rank_balanced_accuracy")
svm_full_metrics_df.to_csv("SVM_Optuna_sens_spec_f1_acc_report.csv", index=False)


print(f"\nSVM Optuna (Grid) took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("Best SVM parameters:", study.best_trial.params)
print("Best SVM Balanced Accuracy:", study.best_value)
print("\nTop Rows from Extended Metrics (SVM, Optuna):")
print(svm_full_metrics_df.head())
