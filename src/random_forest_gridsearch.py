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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score
)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']  
}



# Initialize RF model
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# GridSearchCV
grid_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

start_time = time.time()

# Fit
grid_rf.fit(X, y)

# Save full GridSearch results
results_df = pd.DataFrame(grid_rf.cv_results_)
results_df.to_csv("RF_GridSearchCV_full_results_1.csv", index=False)

# Extract balanced accuracy summary
summary_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
summary_df = summary_df.rename(columns={
    'mean_test_score': 'mean_balanced_accuracy',
    'std_test_score': 'std_balanced_accuracy'
})
summary_df = summary_df.sort_values(by='rank_test_score')
summary_df.to_csv("RF_GridSearchCV_balanced_accuracy_summary_1.csv", index=False)

print("\nBalanced Accuracy Summary:")
print(summary_df.head())

# Initialize list to store extended metrics
extended_results = []

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute sensitivity, specificity, accuracy, and F1
print("\nComputing full metrics (sensitivity, specificity, accuracy, F1)...\n")
for i, params in tqdm(enumerate(grid_rf.cv_results_['params']), total=len(grid_rf.cv_results_['params'])):
    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = RandomForestClassifier(class_weight='balanced', random_state=42, **params)
        model.fit(X_train, y_train)
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

        # Sensitivity and specificity
        class_sensitivities = [TP / (TP + FN) if (TP + FN) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        class_specificities = [TN / (TN + FP) if (TN + FP) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        all_class_sensitivities.append(class_sensitivities)
        all_class_specificities.append(class_specificities)

        # Accuracy and F1
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='macro'))

    # Mean metrics across folds
    mean_sens = np.mean(all_class_sensitivities, axis=0)
    mean_spec = np.mean(all_class_specificities, axis=0)
    mean_accuracy = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)

    # Store all results
    result_row = {
        'mean_balanced_accuracy': grid_rf.cv_results_['mean_test_score'][i],
        'rank_balanced_accuracy': grid_rf.cv_results_['rank_test_score'][i],
        'mean_accuracy': mean_accuracy,
        'mean_f1_macro': mean_f1
    }

    # Add params as separate columns
    result_row.update(params)

    # Add class-wise metrics
    for idx, val in enumerate(mean_sens):
        result_row[f'sensitivity_class_{idx}'] = val
    for idx, val in enumerate(mean_spec):
        result_row[f'specificity_class_{idx}'] = val

    extended_results.append(result_row)

# Create and save full metrics DataFrame
full_metrics_df = pd.DataFrame(extended_results)
full_metrics_df = full_metrics_df.sort_values(by='rank_balanced_accuracy')
full_metrics_df.to_csv("RF_GridSearchCV_sens_spec_f1_acc_report_1.csv", index=False)

# Output best parameters and time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nGridSearchCV took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("Best RF parameters:", grid_rf.best_params_)
print("Best RF Balanced Accuracy:", grid_rf.best_score_)
print("\nTop Rows from Extended Metrics:")
print(full_metrics_df.head())
