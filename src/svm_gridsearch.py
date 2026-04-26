from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import time


param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],   
    'degree': [3]                 
}


svm_model = SVC(class_weight='balanced')


grid_svm = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid_svm,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=2
)

start_time = time.time()


grid_svm.fit(X, y)

# Save full GridSearch results
svm_results_df = pd.DataFrame(grid_svm.cv_results_)
svm_results_df.to_csv("SVM_GridSearchCV_full_results.csv", index=False)

# Extract balanced accuracy summary
svm_summary_df = svm_results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].rename(
    columns={
        'mean_test_score': 'mean_balanced_accuracy',
        'std_test_score': 'std_balanced_accuracy'
    }
).sort_values(by='rank_test_score')
svm_summary_df.to_csv("SVM_GridSearchCV_balanced_accuracy_summary.csv", index=False)

print("\nBalanced Accuracy Summary (SVM):")
print(svm_summary_df.head())

# Compute extended metrics (sensitivity, specificity, accuracy, macro-F1)
print("\nComputing full metrics (sensitivity, specificity, accuracy, F1) for SVM...\n")

extended_results_svm = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, params in tqdm(enumerate(grid_svm.cv_results_['params']),
                      total=len(grid_svm.cv_results_['params'])):
    all_class_sensitivities = []
    all_class_specificities = []
    fold_accuracies = []
    fold_f1s = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = SVC(class_weight='balanced', **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Confusion matrix with consistent label order
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        TN_FP_FN_TP = []
        for class_idx in range(len(model.classes_)):
            TP = cm[class_idx, class_idx]
            FN = cm[class_idx, :].sum() - TP
            FP = cm[:, class_idx].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            TN_FP_FN_TP.append((TN, FP, FN, TP))

        # Sensitivity and specificity per class
        class_sensitivities = [TP / (TP + FN) if (TP + FN) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]
        class_specificities = [TN / (TN + FP) if (TN + FP) > 0 else 0 for TN, FP, FN, TP in TN_FP_FN_TP]

        all_class_sensitivities.append(class_sensitivities)
        all_class_specificities.append(class_specificities)

        # Accuracy and macro-F1
        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='macro'))

    # Mean metrics across folds
    mean_sens = np.mean(all_class_sensitivities, axis=0)
    mean_spec = np.mean(all_class_specificities, axis=0)
    mean_accuracy = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)

    # Store all results
    result_row = {
        'mean_balanced_accuracy': grid_svm.cv_results_['mean_test_score'][i],
        'rank_balanced_accuracy': grid_svm.cv_results_['rank_test_score'][i],
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

    extended_results_svm.append(result_row)

# Create and save full metrics DataFrame
svm_full_metrics_df = pd.DataFrame(extended_results_svm).sort_values(by='rank_balanced_accuracy')
svm_full_metrics_df.to_csv("SVM_GridSearchCV_sens_spec_f1_acc_report.csv", index=False)

# Output best parameters and time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nSVM GridSearchCV took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("Best SVM parameters:", grid_svm.best_params_)
print("Best SVM Balanced Accuracy:", grid_svm.best_score_)
print("\nTop Rows from Extended Metrics (SVM):")
print(svm_full_metrics_df.head())
