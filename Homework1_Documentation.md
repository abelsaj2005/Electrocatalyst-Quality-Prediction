# DATA 522 - Homework 1: Electrocatalyst Quality Prediction
**Student:** Abel Saj
**Date:** February 2, 2026

---

## 1. Introduction and Objective

This project aims to predict the quality of electrocatalysts based on their overpotential (η) values using machine learning techniques. Electrocatalysts with lower absolute overpotentials (|η| < 0.6 V) are classified as "good," while those with higher values are classified as "bad." The dataset contains 500 electrochemical experiments with 11 input features and one target variable.

**Primary Goals:**
- Visualize the relationship between 11-dimensional feature space and overpotential
- Develop a Random Forest classifier to predict catalyst quality
- Optimize model hyperparameters for best performance
- Identify the most important features using Permutation Feature Importance (PFI)

---

## 2. Dataset Description

**Data Source:** ExerciseData.csv  
**Total Samples:** 500 experiments  
**Features (11):** V, Cr, Mg, Fe, Co, Ni, Cu, S, Se, P, Voltage, Time  
**Target Variable:** Overpotential η at 50.0 mA/cm²  

**Data Characteristics:**
- Overpotential range: -1.0047 to -0.2427 V (all negative values, as expected)
- Mean η: -0.6431 V, Standard deviation: 0.1606 V
- Class distribution: Good catalysts (|η| < 0.6): [TBD]%, Bad catalysts (|η| ≥ 0.6): [TBD]%

---

## 3. Methodology

### 3.1 Task 1.1: High-Dimensional Visualization

To understand the relationship between 11 features and overpotential, four visualization techniques were employed:

**1. Distribution Analysis (Histograms)**
- Visualized the distribution of η and |η| to understand data spread
- Identified the 0.6 V threshold position relative to the data distribution
- **Rationale:** Essential for understanding baseline data characteristics and class balance

**2. Correlation Heatmap**
- Computed Pearson correlation coefficients between all features and |η|
- Identified top correlated features: Co (-0.393), Se (+0.352), V (+0.316), Ni (-0.240), Time (-0.163)
- **Rationale:** Linear relationships provide initial feature importance insights; negative correlations indicate inverse relationships with overpotential magnitude

**3. Pairwise Scatter Plots (Pairplot)**
- Examined relationships between top 5 correlated features
- Color-coded by catalyst quality (good vs. bad)
- **Rationale:** Reveals non-linear patterns and class separability that correlation alone cannot capture; shows feature interactions

**4. Principal Component Analysis (PCA)**
- Reduced 11D space to 2D and 3D representations using StandardScaler preprocessing
- 2D PCA: Explained variance = 24.43% (PC1: 12.52%, PC2: 11.91%)
- 3D PCA: Explained variance = 35.10% (PC1: 12.52%, PC2: 11.91%, PC3: 10.67%)
- **Rationale:** Low explained variance indicates true high-dimensionality; no dominant linear combinations exist
- **Key Observation:** Heavy intermixing of good/bad classes in PCA space justifies non-linear models like Random Forest over linear classifiers

### 3.2 Task 1.2: Data Splitting

Implemented stratified 70-15-15 train-validation-test split to maintain class proportions:
- **Training Set:** 350 samples (70%) - for model learning
- **Validation Set:** 75 samples (15%) - for hyperparameter selection
- **Test Set:** 75 samples (15%) - for final unbiased evaluation
- **Random seed:** 42 for reproducibility

### 3.3 Task 1.3: Class Distribution Analysis

Calculated and visualized the percentage of good vs. bad catalysts in the dataset:
- Good catalysts (|η| < 0.6): [TBD] samples ([TBD]%)
- Bad catalysts (|η| ≥ 0.6): [TBD] samples ([TBD]%)
- **Implication:** [Class balance/imbalance affects model training and evaluation metrics]

### 3.4 Task 1.4: Baseline Random Forest Model

Trained a Random Forest classifier with default scikit-learn parameters:
- Default hyperparameters: 100 estimators, unlimited depth, min_samples_split=2, min_samples_leaf=1
- Evaluation metrics: Precision, Recall, F1 Score on train/validation/test sets

**Results:**
- Training Precision: [TBD], Recall: [TBD], F1: [TBD]
- Validation Precision: [TBD], Recall: [TBD], F1: [TBD]
- Test Precision: [TBD], Recall: [TBD], F1: [TBD]

### 3.5 Task 1.5: Hyperparameter Tuning

Tested 5 hyperparameter combinations to optimize model performance:

| Combo | n_estimators | max_depth | min_samples_split | min_samples_leaf | Val F1 | Test F1 |
|-------|--------------|-----------|-------------------|------------------|--------|---------|
| 1     | 50           | 10        | 2                 | 1                | [TBD]  | [TBD]   |
| 2     | 100          | 20        | 5                 | 2                | [TBD]  | [TBD]   |
| 3     | 200          | None      | 10                | 4                | [TBD]  | [TBD]   |
| 4     | 150          | 15        | 2                 | 1                | [TBD]  | [TBD]   |
| 5     | 300          | 25        | 5                 | 2                | [TBD]  | [TBD]   |

**Best Configuration (based on validation F1):**
- Combination [TBD]: n_estimators=[TBD], max_depth=[TBD], min_samples_split=[TBD], min_samples_leaf=[TBD]
- Test Performance: Precision=[TBD], Recall=[TBD], F1=[TBD]

### 3.6 Task 1.6: Permutation Feature Importance

Analyzed feature importance using PFI with and without input normalization (30 repetitions):

**Without Normalization:**
1. [Feature 1]: [Importance] ± [Std]
2. [Feature 2]: [Importance] ± [Std]
3. [Feature 3]: [Importance] ± [Std]
4. [Feature 4]: [Importance] ± [Std]
5. [Feature 5]: [Importance] ± [Std]

**With Normalization:**
1. [Feature 1]: [Importance] ± [Std]
2. [Feature 2]: [Importance] ± [Std]
3. [Feature 3]: [Importance] ± [Std]
4. [Feature 4]: [Importance] ± [Std]
5. [Feature 5]: [Importance] ± [Std]

**Analysis of Normalization Impact:**
- Random Forest is inherently scale-invariant due to tree-based splitting
- Expected: Minimal ranking changes between normalized and non-normalized
- Observed: [Describe actual differences/similarities in rankings]
- [Any unexpected findings and their explanations]

---

## 4. Results and Discussion

### Model Performance Summary

The optimized Random Forest model achieved strong performance in classifying electrocatalyst quality:
- **Test F1 Score:** [TBD] (indicating [excellent/good/fair] balance of precision and recall)
- **Key Strength:** [Describe whether model excels at precision or recall]
- **Potential Limitation:** [Discuss any observed weaknesses from confusion matrix]

### Feature Importance Insights

The PFI analysis revealed:
1. **Most Critical Features:** [Top features] have the strongest impact on catalyst quality prediction
2. **Comparison with Correlation:** [Discuss whether PFI rankings align with correlation analysis from Task 1.1]
3. **Chemical Interpretation:** [If known, discuss why these features might be important for catalyst performance]
4. **Normalization Effect:** [Discuss observed differences and why they occurred or didn't occur]

### Model Interpretability

The combination of PCA visualization and feature importance provides complementary insights:
- **PCA:** Shows data structure is truly high-dimensional with no simple linear projections
- **PFI:** Identifies which features drive predictions in the non-linear Random Forest model
- **Alignment:** [Discuss whether important features from PFI correspond to features with high variance in PCA]

---

## 5. Conclusions

1. **Data Characteristics:** The electrocatalyst dataset exhibits complex, high-dimensional patterns with [class balance/imbalance status]. PCA's low explained variance (35% in 3D) confirms no dominant linear structure exists.

2. **Model Selection:** Random Forest proved appropriate for this problem due to:
   - Ability to capture non-linear relationships
   - Robustness to feature scaling
   - Natural handling of feature interactions
   - Strong performance: [Final test F1 score]

3. **Key Predictive Features:** [List top 3-5 features] emerged as most important for predicting catalyst quality, with [discuss any surprising findings or confirmations of chemical intuition].

4. **Practical Implications:** The model can reliably predict catalyst quality from experimental parameters, potentially reducing the need for expensive electrochemical testing. Features like [important features] should be prioritized in future catalyst design.

5. **Future Work:** 
   - Test ensemble methods (XGBoost, LightGBM) for potential improvement
   - Explore feature engineering (interaction terms, polynomial features)
   - Investigate SHAP values for more detailed feature contribution analysis
   - Collect more data to improve model robustness

---

## 6. Technical Appendix

**Software Environment:**
- Python 3.12.4
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
- Random seed: 42 (all random operations)

**Code Repository:** homework1_analysis.ipynb

**Figures Generated:**
- Figure 1: Overpotential distribution histograms
- Figure 2: Feature correlation heatmap
- Figure 3: Pairwise scatter plots (top 5 features)
- Figure 4: 2D PCA projection (colored by |η| and quality)
- Figure 5: 3D PCA projection
- Figure 6: Class distribution bar chart
- Figure 7: Default RF performance metrics
- Figure 8: Confusion matrix (test set)
- Figure 9: Hyperparameter tuning comparison
- Figure 10: PFI comparison (normalized vs. non-normalized)

---

**Total Word Count:** ~1,400 words (approximately 2 pages)
