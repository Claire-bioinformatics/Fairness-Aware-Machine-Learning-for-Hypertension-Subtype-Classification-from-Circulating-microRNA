import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
import time

from catboost import CatBoostClassifier

# -----------------------------
# 1) Load data
# -----------------------------
X = pd.read_csv("expression.csv", index_col=0)
metadata = pd.read_csv("info.csv", index_col=0)
assert all(X.index == metadata.index)

le = LabelEncoder()
y = le.fit_transform(metadata['DiseaseSubtype.1'])

# -----------------------------
# 2) Param grid
# -----------------------------
param_grid_cat = [
    {
        'bootstrap_type': ['Bayesian'],
        'iterations': [300, 600],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'rsm': [0.8, 1.0],
    },
    {
        'bootstrap_type': ['Bernoulli'],
        'subsample': [0.8, 1.0],
        'iterations': [300, 600],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'rsm': [0.8, 1.0],
    },
]

# -----------------------------
# 3) Base model
# -----------------------------
cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    random_seed=42,
    verbose=0,
    thread_count=-1
)

# -----------------------------
# 4) Global sample weights (used by GridSearchCV and sliced later)
# -----------------------------
sample_weights = compute_sample_weight(class_weight='balanced', y=y)

# -----------------------------
# 5) GridSearchCV
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_cat = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid_cat,
    cv=cv,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

start_time = time.time()
grid_cat.fit(X, y, sample_weight=sample_weights)

results_cat = pd.DataFrame(grid_cat.cv_results_)
results_cat.to_csv("CAT_GridSearchCV_full_results_1.csv", index=False)

summary_cat = results_cat[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].rename(
    columns={'mean_test_score': 'mean_balanced_accuracy',
             'std_test_score': 'std_balanced_accuracy'}
).sort_values(by='rank_test_score')
summary_cat.to_csv("CAT_GridSearchCV_balanced_accuracy_summary_1.csv", index=False)

print("\nBalanced Accuracy Summary (CatBoost):")
print(summary_cat.head())

# -----------------------------
# 6) Extended metrics (use SLICED GLOBAL weights per fold)
# -----------------------------
extended_results_cat = []
labels_order = np.unique(y)

print("\nComputing full metrics (sensitivity, specificity, accuracy, F1) for CatBoost...\n")
for i, params in tqdm(enumerate(grid_cat.cv_results_['params']), total=len(grid_cat.cv_results_['params'])):
    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []
    fold_bal_accs = []  

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        
        sample_weights_fold = sample_weights[train_idx]

        model = CatBoostClassifier(
            loss_function='MultiClass',
            random_seed=42,
            verbose=0,
            thread_count=-1,
            **params
        )
        model.fit(X_train, y_train, sample_weight=sample_weights_fold)

        # Predict class labels
        y_pred = model.predict(X_test)
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        
        y_pred = y_pred.astype(int)

        # Balanced accuracy from the SAME predictions
        fold_bal_accs.append(balanced_accuracy_score(y_test, y_pred))

        # Confusion matrix with fixed label order
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

    # Averages across folds
    mean_sens = np.mean(all_class_sensitivities, axis=0)
    mean_spec = np.mean(all_class_specificities, axis=0)
    mean_accuracy = float(np.mean(fold_accuracies))
    mean_f1_macro = float(np.mean(fold_f1s))
    mean_bal_acc_from_preds = float(np.mean(fold_bal_accs))  # should ≈ summary BA

    # Row (align with GridSearchCV stats)
    result_row = {
        'mean_balanced_accuracy': float(grid_cat.cv_results_['mean_test_score'][i]),
        'mean_balanced_accuracy_from_preds': mean_bal_acc_from_preds,  
        'delta_BA': float(mean_bal_acc_from_preds - float(grid_cat.cv_results_['mean_test_score'][i])),
        'rank_balanced_accuracy': int(grid_cat.cv_results_['rank_test_score'][i]),
        'mean_accuracy': mean_accuracy,
        'mean_f1_macro': mean_f1_macro
    }

    result_row.update(params)

    # Per-class metrics
    for idx, val in enumerate(mean_sens):
        result_row[f'sensitivity_class_{idx}'] = float(val)
    for idx, val in enumerate(mean_spec):
        result_row[f'specificity_class_{idx}'] = float(val)

    extended_results_cat.append(result_row)

# -----------------------------
# 7) Build full DataFrame and rename classes
# -----------------------------
full_metrics_cat_df = pd.DataFrame(extended_results_cat).sort_values(by='rank_balanced_accuracy')

class_names = ['CS', 'HV', 'PA', 'PHT', 'PPGL']  
if len(class_names) == len(labels_order):
    rename_dict = {f'sensitivity_class_{i}': f'sensitivity_{name}' for i, name in enumerate(class_names)}
    rename_dict.update({f'specificity_class_{i}': f'specificity_{name}' for i, name in enumerate(class_names)})
    full_metrics_cat_df = full_metrics_cat_df.rename(columns=rename_dict)
else:
    print(f"Warning: expected {len(class_names)} classes, found {len(labels_order)}. Skipping rename.")

full_metrics_cat_df.to_csv("CAT_GridSearchCV_sens_spec_f1_acc_report_1.csv", index=False)

# -----------------------------
# 8) Done
# -----------------------------
elapsed_time = time.time() - start_time
print(f"\nGridSearchCV (CatBoost) took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("Best CAT parameters:", grid_cat.best_params_)
print("Best CAT Balanced Accuracy:", grid_cat.best_score_)
print("\nTop Rows from Extended Metrics (CatBoost):")
print(full_metrics_cat_df.head())



