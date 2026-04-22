# Fairness-Aware-Machine-Learning-for-Hypertension-Subtype-Classification-from-Circulating-microRNA
ML pipeline for imbalanced multi-class classification of endocrine hypertension using circulating microRNA, **improving rare subtype detection (0.00 → 0.71 sensitivity)** with fairness-aware evaluation, hyperparameter optimization, and ensemble learning

## Problem

Endocrine hypertension subtypes (e.g., Cushing’s syndrome, primary aldosteronism) are frequently misdiagnosed as primary hypertension, delaying appropriate treatment.

Circulating microRNAs offer a non-invasive biomarker — but classification is challenging due to:
- severe class imbalance
- small sample sizes for rare subtypes
- high-dimensional biological data

## Why This Is Challenging

- Minority classes (e.g., Cushing’s syndrome) are underrepresented
- **Standard accuracy is misleading in imbalanced datasets**
- Models tend to ignore clinically important rare subtypes
- Trade-off between overall performance and minority detection

## Key Idea

Instead of optimizing for overall accuracy, this project prioritizes **fairness-aware evaluation**:

- Balanced accuracy → equal weight per class
- Minimum sensitivity → protects the worst-performing class

This ensures rare but clinically critical subtypes are not overlooked.

## Approach

### Models
- Random Forest
- XGBoost
- CatBoost
- Support Vector Machine (SVM)

### Optimization
- GridSearchCV (exhaustive search)
- Optuna (Bayesian optimization)

### Imbalance Handling
- Class weighting / sample weighting
- Stratified cross-validation

### Ensemble
- Stacking with Logistic Regression meta-learner

### Interpretability
- SHAP for feature importance

## Key Results

- Improved sensitivity for smallest class (Cushing’s syndrome):
  - **0.00 → 0.71**
- Minimum sensitivity improved across all models
- Final stacked model:
  - **Balanced accuracy: ~0.65**
- Identified biologically relevant microRNA driving CS classification (via SHAP)
