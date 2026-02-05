# DATA 522 - Homework 1: Electrocatalyst Quality Prediction
**Student:** Abel Saj
**Date:** February 4, 2026

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

Calculated the percentage of good vs. bad catalysts in the dataset:
- Good catalysts (|η| < 0.6): 165 samples (33.00%)
- Bad catalysts (|η| ≥ 0.6): 335 samples (67.00%)
- **Implication:** The dataset exhibits moderate class imbalance with twice as many "bad" catalysts. This 67-33 split affects model evaluation, making metrics like F1 score more informative than simple accuracy.

### 3.4 Task 1.4: Baseline Random Forest Model

Trained a Random Forest classifier with default scikit-learn parameters:
- Default hyperparameters: 100 estimators, unlimited depth, min_samples_split=2, min_samples_leaf=1
- Evaluation metrics: Precision, Recall, F1 Score on train/validation/test sets

**Results:**
- Training Precision: 1.0000, Recall: 1.0000, F1: 1.0000
- Validation Precision: 0.6667, Recall: 0.4800, F1: 0.5581
- Test Precision: 0.7059, Recall: 0.5000, F1: 0.5854

### 3.5 Task 1.5: Hyperparameter Tuning

Tested 5 hyperparameter combinations to optimize model performance:

| Combo | n_estimators | max_depth | min_samples_split | min_samples_leaf | class_weight | max_leaf_nodes | Val F1 | Test F1 |
|-------|--------------|-----------|-------------------|------------------|--------------|----------------|--------|---------|
| 1     | 100          | 20        | 5                 | 2                | None         | None           | 0.5652 | 0.5854  |
| 2     | 200          | 15        | 3                 | 3                | None         | None           | 0.5238 | 0.6000  |
| 3     | 150          | 25        | 2                 | 1                | None         | None           | 0.5238 | 0.5854  |
| 4     | 300          | 12        | 2                 | 4                | balanced     | None           | 0.6800 | 0.6531  |
| 5     | 500          | 12        | 2                 | 4                | balanced     | 50             | 0.6333 | 0.6122  |

**Best Configuration (based on validation F1):**
- Combination 4: n_estimators=300, max_depth=12, min_samples_split=2, min_samples_leaf=4, class_weight='balanced'
- Validation Performance: Precision=0.6800, Recall=0.6800, F1=0.6800
- Test Performance: Precision=0.6400, Recall=0.6667, F1=0.6531

**Key Finding:** Class weighting ('balanced') significantly improved performance by addressing the 67-33 class imbalance, increasing recall from 0.50 to 0.67 while maintaining precision at 0.64.

### 3.6 Task 1.6: Feature Importance
Analyzed feature importance using PFI with and without input normalization (30 repetitions):

**Without Normalization:**
1. Co: 0.2381 ± 0.0783
2. Se: 0.1552 ± 0.0779
3. V: 0.1267 ± 0.0892
4. Ni: 0.0868 ± 0.0471
5. Cu: 0.0464 ± 0.0351

**With Normalization:**
1. Co: 0.2378 ± 0.0783
2. Se: 0.1552 ± 0.0779
3. V: 0.1450 ± 0.0875
4. Ni: 0.0843 ± 0.0489
5. Cu: 0.0464 ± 0.0351

**Analysis of Normalization Impact:**
- Random Forest is inherently scale-invariant due to tree-based splitting
- Expected: Minimal ranking changes between normalized and non-normalized
- Observed: Rankings are identical for the top 5 features (Co, Se, V, Ni, Cu) in both approaches, with nearly identical importance values
- This confirms Random Forest's scale-invariance property - feature importance depends on information gain from splits, not absolute feature magnitudes

---

## 4. Results and Discussion

### Model Performance Summary

The optimized Random Forest model achieved strong performance in classifying electrocatalyst quality:
- **Test F1 Score:** 0.6531 (a 11.6% improvement over the baseline 0.5854)
- **Balanced Performance:** Precision (0.6400) and Recall (0.6667) are both strong, indicating the model correctly identifies ~64% of predicted good catalysts and successfully finds ~67% of all truly good catalysts
- **Key Improvement:** Class weighting ('balanced') was crucial for addressing the 67-33 class imbalance, dramatically improving recall from 0.50 to 0.67

### Feature Importance Insights

The PFI analysis revealed:
1. **Most Critical Features:** Co (0.24), Se (0.16), and V (0.13-0.15) have the strongest impact on catalyst quality prediction
2. **Comparison with Correlation:** PFI rankings align well with correlation analysis - Co (-0.393), Se (+0.352), and V (+0.316) were the top three correlated features in Task 1.1
3. **Chemical Interpretation:** Cobalt (Co), selenium (Se), and vanadium (V) are known electrocatalyst components. Co is a transition metal with variable oxidation states enabling redox reactions, Se can influence electronic structure, and V provides catalytic activity through multiple oxidation states
4. **Normalization Effect:** Identical top-5 rankings (Co, Se, V, Ni, Cu) confirm Random Forest's inherent scale-invariance, validating that feature importance derives from information gain, not feature magnitude

### Model Interpretability

The combination of PCA visualization and feature importance provides complementary insights:
- **PCA:** Shows data structure is truly high-dimensional with no simple linear projections (only 35% variance in 3 components)
- **PFI:** Identifies Co, Se, and Ni as the features that most strongly drive predictions in the non-linear Random Forest model
- **Alignment:** PFI's important features (Co, Se, Ni, V) overlap with the highly correlated features identified in Task 1.1, confirming both linear and non-linear importance

---

## 5. Conclusions

1. **Data Characteristics:** The electrocatalyst dataset exhibits complex, high-dimensional patterns with moderate class imbalance (67% bad, 33% good catalysts). PCA's low explained variance (35% in 3D) confirms no dominant linear structure exists.

2. **Model Selection:** Random Forest proved appropriate for this problem due to:
   - Ability to capture non-linear relationships
   - Robustness to feature scaling (confirmed via normalization study)
   - Natural handling of feature interactions
   - Strong performance: Test F1 score of 0.6531 with balanced precision (0.64) and recall (0.67)
   - **Critical Insight:** Class weighting ('balanced') was essential for handling the 67-33 imbalance, improving F1 by 11.6%

3. **Key Predictive Features:** Cobalt (Co), Selenium (Se), and Vanadium (V) emerged as the three most important features for predicting catalyst quality. This aligns with chemical intuition - these are known transition metals and chalcogens critical for electrocatalytic activity. The fact that Ni and Cu round out the top 5 suggests compositional balance across multiple transition metals is key.

4. **Practical Implications:** The model can predict catalyst quality with 64% precision and 67% recall from experimental parameters, potentially reducing the need for expensive electrochemical testing. Features like Co, Se, and V concentrations should be prioritized in future catalyst design and optimization efforts.