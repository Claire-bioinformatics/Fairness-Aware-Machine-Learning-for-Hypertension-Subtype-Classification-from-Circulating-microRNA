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

## Technical Highlights

- Imbalanced multi-class classification
- Fairness-aware model evaluation
- Hyperparameter optimization at scale (HPC)
- Ensemble learning (stacking)
- Model interpretability (SHAP)
- Reproducible ML pipelines with SLURM

## Tech Stack

- Python (scikit-learn, XGBoost, CatBoost)
- Optuna
- SHAP
- NumPy, pandas
- Matplotlib, seaborn
- SLURM (HPC)

## Key Takeaways

- Accuracy is misleading in imbalanced biomedical datasets
- Improving minority-class performance often reduces majority-class performance
- Hyperparameter tuning can significantly improve rare-class detection
- Stacking provides marginal gains relative to computational cost
- Interpretability is essential for trust in clinical ML systems

## Reproducibility

This project was designed to ensure reproducible machine learning experiments across models, datasets, and computational environments.

---

### Data Requirements

The pipeline expects two input files:

- `expression.csv` → microRNA ΔCT expression matrix (samples × features)  
- `info.csv` → metadata file containing class labels (`DiseaseSubtype.1`)  

Both files must:
- share identical sample indices  
- be preprocessed to remove batch effects and normalize ΔCT values  

> Note: Due to data privacy and consortium restrictions, raw data is not publicly available. A synthetic or example dataset can be used to demonstrate the pipeline structure.

---

### Environment

Experiments were conducted using:

- Python 3.12  
- scikit-learn  
- XGBoost  
- CatBoost  
- Optuna  
- SHAP  

For reproducibility:
- All random seeds fixed (`random_state = 42`)  
- Stratified cross-validation used throughout  

You can recreate the environment with:

```bash
conda create -n hypertension_ml python=3.12
conda activate hypertension_ml
pip install -r requirements.txt
