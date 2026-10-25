# Pistachio Classification Using Machine Learning

Binary classification of two Turkish pistachio varieties, **Kirmizi** and **Siit**, from 16 morphological kernel features, comparing seven classical machine-learning classifiers.

![Pipeline](figures/fig1_pipeline.png)

## Overview

Pistachio varieties differ in market value, but sorting is often done by hand. This project shows that simple shape measurements extracted from kernel images, combined with standard scikit-learn classifiers, can identify the variety automatically with close to 89% accuracy.

**Best model: Gradient Boosting** with 88.95% hold-out accuracy (AUC 0.952) and 88.36% ± 2.31% mean accuracy in stratified 10-fold cross-validation.

## Dataset

| Property | Value |
|---|---|
| File | `pistachio.csv` (also provided as `archive (3).zip`) |
| Samples | 1,718 |
| Classes | Kirmizi_Pistachio (998, 58.1%), Siit_Pistachio (720, 41.9%) |
| Features | 16 numerical morphological features |
| Missing values | None |

The features are `AREA`, `PERIMETER`, `MAJOR_AXIS`, `MINOR_AXIS`, `ECCENTRICITY`, `EQDIASQ`, `SOLIDITY`, `CONVEX_AREA`, `EXTENT`, `ASPECT_RATIO`, `ROUNDNESS`, `COMPACTNESS` and `SHAPEFACTOR_1` to `SHAPEFACTOR_4`. The dataset is a redistributed version of the pistachio data introduced by Ozkan, Koklu and Saracoglu (2021).

## Method

1. **Preprocessing:** label encoding of the class (Kirmizi = 0, Siit = 1) and z-score standardisation of all features.
2. **Split:** 80/20 train/test hold-out (`random_state=42`, 344 test samples), plus stratified 10-fold cross-validation with scaling inside each fold.
3. **Models (scikit-learn defaults):** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine (RBF), k-Nearest Neighbours (k = 5), Gaussian Naive Bayes.
4. **Metrics:** accuracy, precision, recall, F1-score, ROC-AUC, MCC and confusion matrices; permutation feature importance for the best model.

## Results

### Hold-out test set (n = 344)

| Classifier | Accuracy | Precision | Recall | F1 | AUC | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8837 | 0.8552 | 0.8671 | 0.8611 | 0.9419 | 0.7612 |
| Decision Tree | 0.8052 | 0.7714 | 0.7552 | 0.7633 | 0.7980 | 0.5979 |
| Random Forest | 0.8750 | 0.8521 | 0.8462 | 0.8491 | 0.9381 | 0.7424 |
| Gradient Boosting | 0.8895 | 0.8889 | 0.8392 | 0.8633 | 0.9516 | 0.7717 |
| Support Vector Machine | 0.8808 | 0.8493 | 0.8671 | 0.8581 | 0.9487 | 0.7555 |
| K-Nearest Neighbors | 0.8634 | 0.8333 | 0.8392 | 0.8362 | 0.9173 | 0.7190 |
| Gaussian Naive Bayes | 0.8721 | 0.8153 | 0.8951 | 0.8533 | 0.9417 | 0.7429 |

### Stratified 10-fold cross-validation

| Classifier | Accuracy (%) | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 86.91 ± 2.97 | 0.8418 ± 0.0358 | 0.9395 ± 0.0211 |
| Decision Tree | 81.26 ± 2.80 | 0.7765 ± 0.0360 | 0.8079 ± 0.0303 |
| Random Forest | 87.49 ± 2.25 | 0.8491 ± 0.0282 | 0.9388 ± 0.0206 |
| Gradient Boosting | 88.36 ± 2.31 | 0.8604 ± 0.0259 | 0.9446 ± 0.0212 |
| Support Vector Machine | 87.78 ± 2.95 | 0.8524 ± 0.0349 | 0.9363 ± 0.0210 |
| K-Nearest Neighbors | 86.61 ± 2.58 | 0.8385 ± 0.0294 | 0.9197 ± 0.0221 |
| Gaussian Naive Bayes | 86.50 ± 2.86 | 0.8418 ± 0.0344 | 0.9350 ± 0.0227 |

![Confusion matrices](figures/fig5_confusion_matrices.png)

![ROC curves](figures/fig7_roc.png)

### Key findings

- Gradient Boosting is the best and most stable model; SVM, Random Forest and Logistic Regression are within about one percentage point.
- The single Decision Tree overfits (100% training accuracy, about 80% test accuracy).
- `MINOR_AXIS` is by far the most important feature; many features are highly redundant (for example `AREA` and `EQDIASQ`, r ≈ 0.997).

![Feature importance](figures/fig10_importance.png)

## Repository Structure

```
├── PISTACHIO_CLASSIFICATION_USING_MACHINE_LEARNING.ipynb   # Colab notebook with outputs
├── pistachio_classification_using_machine_learning.py      # Script version of the notebook
├── pistachio.csv                                           # Dataset
├── archive (3).zip                                         # Zipped copy of the dataset
└── figures/                                                # Figures used in this README (optional)
```

## How to Run

```bash
pip install pandas scikit-learn seaborn matplotlib
python pistachio_classification_using_machine_learning.py
```

The script was exported from Google Colab and reads the data from `/content/sample_data/pistachio.csv`. When running locally, change that line to:

```python
data = pd.read_csv('pistachio.csv')
```

Alternatively, open the notebook in Jupyter or Google Colab and run all cells.

## Notes and Possible Improvements

- Set `random_state` on the Decision Tree, Random Forest and Gradient Boosting models for fully repeatable results.
- In the script, the "train accuracy" line in the accuracy plot currently reuses the test accuracies; compute it with `clf.score(X_train, y_train)` to show the real training accuracy.
- Fit the `StandardScaler` on the training split only (for example with a `Pipeline`) to avoid data leakage.
- Hyperparameter tuning, feature selection or PCA, and image-based deep learning are natural next steps.

## References

- I. A. Ozkan, M. Koklu, R. Saracoglu, "Classification of pistachio species using improved k-NN classifier," *Progress in Nutrition*, 23(2), e2021044, 2021.
- D. Singh et al., "Classification and analysis of pistachio species with pre-trained deep learning models," *Electronics*, 11(7), 981, 2022.
- F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 12, 2825–2830, 2011.
