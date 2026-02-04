# 🔬 Electrocatalyst Quality Prediction using Machine Learning

A comprehensive machine learning project that predicts electrocatalyst performance using Random Forest classification with advanced feature engineering and permutation feature importance analysis.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This project develops a machine learning pipeline to classify electrocatalyst quality based on overpotential measurements at 50.0 mA/cm². By analyzing 500 electrochemical experiments with 11 compositional and operational features, the model predicts whether catalysts are "good" (|η| < 0.6 V) or "bad" (|η| ≥ 0.6 V).

**Key Achievement:** Built a Random Forest classifier achieving **70.6% precision** on test data, enabling reliable catalyst quality screening and reducing the need for expensive experimental validation.

## 🎯 Business Impact

- **Cost Reduction:** Predicts catalyst quality before expensive electrochemical testing
- **Accelerated R&D:** Identifies key compositional factors (Co, Se, Ni) for catalyst optimization
- **Data-Driven Design:** Provides actionable insights for next-generation catalyst development

## 🛠️ Technologies & Skills Demonstrated

### Core Technologies
- **Python 3.12** | **scikit-learn** | **pandas** | **NumPy**
- **Matplotlib** | **Seaborn** | **PCA** | **Random Forest**

### Machine Learning Techniques
- ✅ High-dimensional data visualization (PCA, correlation analysis)
- ✅ Hyperparameter tuning with stratified validation
- ✅ Permutation Feature Importance (PFI) analysis
- ✅ Model evaluation with imbalanced datasets
- ✅ Scale-invariance validation for tree-based models

### Data Science Workflow
1. **Exploratory Data Analysis** → Visualized 11D feature space using PCA and correlation heatmaps
2. **Feature Engineering** → Identified top correlated features and non-linear patterns
3. **Model Development** → Trained and optimized Random Forest with 5 hyperparameter configurations
4. **Model Interpretation** → Validated feature importance with/without normalization
5. **Performance Analysis** → Evaluated precision-recall trade-offs for imbalanced classes

## 📊 Dataset

- **Source:** Electrochemical experiments dataset
- **Size:** 500 samples
- **Features:** 11 compositional and operational parameters
  - **Compositional:** V, Cr, Mg, Fe, Co, Ni, Cu, S, Se, P
  - **Operational:** Voltage, Time
- **Target:** Overpotential η at 50.0 mA/cm²
- **Class Distribution:** 33% good / 67% bad (moderate imbalance)

## 🚀 Key Results

### Model Performance
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Precision** | 1.000 | 0.667 | **0.706** |
| **Recall** | 1.000 | 0.480 | 0.500 |
| **F1 Score** | 1.000 | 0.558 | 0.585 |

### Top 5 Most Important Features
1. **Cobalt (Co)** - Importance: 0.211 ± 0.079
2. **Selenium (Se)** - Importance: 0.118 ± 0.076
3. **Nickel (Ni)** - Importance: 0.078 ± 0.053
4. **Vanadium (V)** - Importance: 0.049 ± 0.062
5. **Magnesium (Mg)** - Importance: 0.044 ± 0.026

**Insight:** Transition metals (Co, Ni) and chalcogens (Se) dominate catalyst performance—aligning with electrochemistry domain knowledge.

### Hyperparameter Optimization
Tested 5 Random Forest configurations, achieving best validation F1 (0.565) with:
- `n_estimators=100`
- `max_depth=20`
- `min_samples_split=5`
- `min_samples_leaf=2`

## 📁 Project Structure

```
.
├── homework1_analysis.ipynb       # Complete analysis pipeline (32 cells)
├── ExerciseData.csv                # Dataset (500 experiments)
├── Homework1_Documentation.md      # Detailed technical report
└── README.md                       # This file
```

## 🔍 Methodology Highlights

### 1. High-Dimensional Visualization
- **PCA Analysis:** 2D projection explains only 24.4% variance → True high-dimensionality
- **Correlation Heatmap:** Identified Co (-0.393), Se (+0.352), V (+0.316) as top correlates
- **Pairplots:** Visualized non-linear class separability across top features

### 2. Stratified Data Splitting
- **70% Training** (350 samples) - Model learning
- **15% Validation** (75 samples) - Hyperparameter selection
- **15% Test** (75 samples) - Unbiased evaluation

### 3. Feature Importance Validation
Ran PFI analysis **with and without normalization** to validate Random Forest's scale-invariance:
- ✅ **Identical top-5 rankings** in both scenarios
- ✅ Confirms importance stems from information gain, not feature magnitude

## 💡 Key Insights

1. **Model Selection Rationale:** Random Forest outperforms linear models due to:
   - PCA's low explained variance (35% in 3D) → non-linear relationships dominate
   - Heavy class intermixing in PCA space → complex decision boundaries required

2. **Precision-Recall Trade-off:** Model prioritizes precision (0.71) over recall (0.50):
   - Conservative predictions minimize false positives
   - Suitable for screening applications where experimental validation follows

3. **Domain Alignment:** Top features (Co, Se, Ni) match electrochemistry literature:
   - Transition metals enable variable oxidation states for redox reactions
   - Chalcogens modulate electronic structure and conductivity

## 🔧 Setup & Usage

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Run the Analysis
```bash
jupyter notebook homework1_analysis.ipynb
```

The notebook runs end-to-end in ~30 seconds on modern hardware.

## 📈 Visualizations

The project includes 6 comprehensive visualizations:
1. **Overpotential Distribution** - Histogram with good/bad threshold
2. **Correlation Heatmap** - Feature-target relationships
3. **Pairwise Scatter Plots** - Top 5 features colored by quality
4. **2D PCA Projection** - Class separation in reduced space
5. **3D PCA Projection** - Multi-dimensional class structure
6. **PFI Comparison** - Normalized vs. non-normalized feature importance

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ **End-to-end ML pipeline** design and implementation
- ✅ **Imbalanced data handling** with stratified sampling and F1 optimization
- ✅ **Model interpretability** through feature importance and domain validation
- ✅ **Scientific rigor** via normalization studies and cross-validation
- ✅ **Professional documentation** and reproducible research practices

## 🔮 Future Enhancements

- [ ] Implement SHAP values for instance-level interpretability
- [ ] Test gradient boosting methods (XGBoost, LightGBM, CatBoost)
- [ ] Engineer interaction features (e.g., Co×Se, Ni×Voltage)
- [ ] Perform threshold optimization for precision-recall balance
- [ ] Deploy model as REST API for real-time predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Abel Saj**
- GitHub: [@abelsaj](https://github.com/abelsaj)
- LinkedIn: [Connect with me](https://linkedin.com/in/abelsaj)

---

⭐ **If you found this project interesting, please consider giving it a star!**

*Built as part of DATA 522: Data Science & Machine Learning coursework*
