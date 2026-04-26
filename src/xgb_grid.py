import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import time


from sklearn.utils.class_weight import compute_sample_weight

from sklearn.model_selection import RandomizedSearchCV

X = pd.read_csv("expression.csv", index_col=0)
metadata = pd.read_csv("info.csv", index_col=0)

# Check alignment
assert all(X.index == metadata.index)

le = LabelEncoder()
y = le.fit_transform(metadata['DiseaseSubtype.1'])

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# 1. Define hyperparameter grid
param_grid_xgb = {
    'learning_rate': [0.15, 0.2, 0.25],
    'max_depth': [4, 5, 6, 7],
    'n_estimators': [80, 100, 120, 150],
    'colsample_bytree': [0.9, 1.0],
    'subsample': [0.7, 0.8, 0.9]
}

# 2. Define base model
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    random_state=42
)

# 3. Handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y)

# 4. GridSearchCV setup
grid_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

# 5. Fit GridSearchCV
start_time = time.time()
grid_xgb.fit(X, y, sample_weight=sample_weights)

# 6. Save full results
results_xgb = pd.DataFrame(grid_xgb.cv_results_)
results_xgb.to_csv("XGB_GridSearchCV_full_results.csv", index=False)

# 7. Create summary table
summary_xgb = results_xgb[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
summary_xgb = summary_xgb.rename(columns={
    'mean_test_score': 'mean_balanced_accuracy',
    'std_test_score': 'std_balanced_accuracy'
})
summary_xgb = summary_xgb.sort_values(by='rank_test_score')
summary_xgb.to_csv("XGB_GridSearchCV_balanced_accuracy_summary.csv", index=False)
print("\nBalanced Accuracy Summary:")
print(summary_xgb.head())

# 8. Extended metrics per parameter set
extended_results_xgb = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nComputing full metrics (sensitivity, specificity, accuracy, F1)...\n")
for i, params in tqdm(enumerate(grid_xgb.cv_results_['params']), total=len(grid_xgb.cv_results_['params'])):
    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []

    sample_weights_global = compute_sample_weight(class_weight='balanced', y=y)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
       # sample_weights_fold = compute_sample_weight(class_weight='balanced', y=y_train)
        sample_weights_fold = sample_weights_global[train_idx]

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            random_state=42,
            **params
        )
        model.fit(X_train, y_train, sample_weight=sample_weights_fold)
        y_pred = model.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        TN_FP_FN_TP = []
        for class_idx in range(len(model.classes_)):
            TP = cm[class_idx, class_idx]
            FN = cm[class_idx, :].sum() - TP
            FP = cm[:, class_idx].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            TN_FP_FN_TP.append((TN, FP, FN, TP))

        # Class-wise metrics
        class_sens = [TP / (TP + FN) if (TP + FN) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        class_spec = [TN / (TN + FP) if (TN + FP) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        all_class_sensitivities.append(class_sens)
        all_class_specificities.append(class_spec)

        # Overall metrics
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='macro'))

    # Averages
    mean_sens = np.mean(all_class_sensitivities, axis=0)
    mean_spec = np.mean(all_class_specificities, axis=0)
    mean_accuracy = np.mean(fold_accuracies)
    mean_f1_macro = np.mean(fold_f1s)

    # Result row
    result_row = {
        'mean_balanced_accuracy': grid_xgb.cv_results_['mean_test_score'][i],
        'rank_balanced_accuracy': grid_xgb.cv_results_['rank_test_score'][i],
        'mean_accuracy': mean_accuracy,
        'mean_f1_macro': mean_f1_macro
    }

    result_row.update(params)

    # Per-class metrics
    for idx, val in enumerate(mean_sens):
        result_row[f'sensitivity_class_{idx}'] = val
    for idx, val in enumerate(mean_spec):
        result_row[f'specificity_class_{idx}'] = val

    extended_results_xgb.append(result_row)

# 9. Create full DataFrame
full_metrics_xgb_df = pd.DataFrame(extended_results_xgb)
full_metrics_xgb_df = full_metrics_xgb_df.sort_values(by='rank_balanced_accuracy')

# 10. Rename class metrics
class_names = ['CS', 'HV', 'PA', 'PHT', 'PPGL']
rename_dict = {
    f'sensitivity_class_{i}': f'sensitivity_{name}' for i, name in enumerate(class_names)
}
rename_dict.update({
    f'specificity_class_{i}': f'specificity_{name}' for i, name in enumerate(class_names)
})
full_metrics_xgb_df = full_metrics_xgb_df.rename(columns=rename_dict)

# 11. Save full report
full_metrics_xgb_df.to_csv("XGB_GridSearchCV_sens_spec_f1_acc_report.csv", index=False)

# 12. Done
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nGridSearchCV took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("Best XGB parameters:", grid_xgb.best_params_)
print("Best XGB Balanced Accuracy:", grid_xgb.best_score_)
print("\nTop Rows from Extended Metrics:")
print(full_metrics_xgb_df.head())
