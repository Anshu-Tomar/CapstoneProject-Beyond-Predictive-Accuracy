# Beyond Predictive Accuracy
 
**Evaluating sample-size sensitivity, explainability, missing-data robustness, calibration and computational efficiency of machine learning models for diabetes classification.**
 
> ⚠️ This is a research and learning project. It is **not** a medical device and must **not** be used to diagnose patients.
 
This README has two parts:
 
- **[Part 1 – Business overview](#part-1--business-overview):** for managers, clinicians and reviewers. No code.
- **[Part 2 – Technical guide](#part-2--technical-guide):** for developers and data scientists who want to run the code.
---
 
## Part 1 – Business overview
 
### The problem
 
Diabetes is one of the biggest public health challenges in the world. Finding it early helps prevent complications and improves patient outcomes.
 
Today, diagnosis mostly depends on lab tests such as fasting blood glucose and HbA1c. But diabetes is also linked to age, body weight (BMI), blood pressure, lifestyle and smoking. Machine learning (ML) can look at all of these factors together and find patterns that simple rules can miss.
 
### Why accuracy is not enough
 
Many studies report ML models with very high accuracy for diabetes prediction. **High accuracy alone does not mean a model is safe or useful in a hospital.**
 
A simple example from this project's data: only about **1 in 10 patients** has diabetes. A model that always says "no diabetes" is about **90% accurate**, and it misses every diabetic patient.
 
Before a model can be trusted in healthcare, we also need to know:
 
| Question | Why it matters to a hospital or clinic |
|---|---|
| **Does it still work with less data?** (sample size) | Small clinics do not have millions of patient records. |
| **Can a doctor understand why it decided?** (explainability) | Doctors will not trust or act on a "black box". |
| **Does it still work when some values are missing?** (missing data) | Real patient records are often incomplete. |
| **Can we trust its risk percentage?** (calibration) | If the model says "70% risk", about 70 in 100 such patients should really have diabetes. |
| **Is it fast and cheap to run?** (computational efficiency) | Many healthcare settings have limited computers and budgets. |
| **Does balancing the data help?** (SMOTENC) | Rare cases (diabetic patients) are easy for a model to ignore. |
 
### What this project delivers
 
A side-by-side comparison of 5 common ML models on **all** of the questions above, not only on accuracy. The goal is to find models that are **accurate, reliable, explainable and practical** for diabetes classification.
 
### Who can use the results
 
- **Clinicians:** to see which patient factors drive a prediction.
- **Hospital IT and data teams:** to choose a model that fits their data size and hardware.
- **Health-tech product teams:** to understand the trade-offs before building a screening tool.
- **Researchers and students:** as a template for evaluating medical ML models beyond accuracy.
---
 
## Part 2 – Technical guide
 
### Aim
 
To compare selected supervised ML algorithms for diabetes classification beyond accuracy, with focus on sample-size sensitivity, explainability, robustness to missing data, probability calibration and computational efficiency, and to identify reliable, interpretable and computationally feasible models for healthcare use.
 
### Objectives
 
| # | Objective | Branch |
|---|---|---|
| 1 | Build and compare supervised ML models using demographic, body-measurement, clinical and lab features | `baseline-model` |
| 2 | Measure the effect of **SMOTENC** (class balancing for mixed number and category data) on performance | `baseline-model` |
| 3 | Measure the effect of **sample size** on performance and stability | *to be created* |
| 4 | Explain models with **SHAP**, at population level and for single patients | *to be created* |
| 5 | Test **robustness to missing data** with different missing-data strategies | `missing-data-robustness` |
| 6 | Check **calibration** with calibration curves, Brier score, calibration intercept and slope | `calibration` |
| 7 | Compare **computational efficiency**: training time, prediction time, memory | `Computational-efficiency` |
| 8 | Compare accuracy, sensitivity, specificity, precision, F1-score, ROC-AUC and PR-AUC | `baseline-model` |
 
### Dataset
 
**Diabetes Simple Diagnosis**: 88,380 patient records, 7 input features and 1 target. About **9.5%** of patients have diabetes, so the classes are imbalanced.
 
| Column | Meaning | Type |
|---|---|---|
| `Age` | Age in years | Number |
| `Gender` | Patient gender | Category |
| `BMI` | Body Mass Index | Number |
| `High_BP` | High blood pressure (1 = yes, 0 = no) | Category |
| `FBS` | Fasting blood sugar (mg/dL) | Number |
| `HbA1c_level` | Glycated haemoglobin (%) | Number |
| `Smoking` | Smoker (1 = yes, 0 = no) | Category |
| `Diagnosis` | **Target:** 1 = diabetes, 0 = no diabetes | Target |
 
> 📁 **Data files are not stored in this repository.** Put the data file in the project folder before you run a notebook (see [Getting started](#getting-started)).
 
### Models
 
Logistic Regression · Decision Tree · Random Forest · XGBoost · Support Vector Machine (SVM)
 
### Evaluation method
 
Every model is tested on data it has **never seen**. The final test set is used **only once**.
 
```
Full data (100%)
│
├── Final test data (10%)   → final_test_data_10.csv   (used only once, at the end)
│
└── Development data (90%)  → development_data_90.csv
        ├── Training data   (80%) → models learn here; SMOTENC is applied here only
        └── Validation data (20%) → choose the best settings and the best model
```
 
1. **Stratified 90/10 split** into development and final test data. Both parts keep the same % of diabetic patients.
2. **Stratified 80/20 split** of the development data into training and validation data.
3. **Hyperparameter tuning:** train every setting on the training data and score it on the validation data.
4. **5-fold cross-validation** on the training data, to check that the best settings are stable.
5. **Select** the best model and setting by validation **PR-AUC**.
6. **Retrain** the selected model on training + validation data (90%).
7. **Report** the results on the 10% final test data.
8. **Repeat** steps 3–7 with **SMOTENC** applied to the training data only.
**Main metric:** PR-AUC, because it focuses on finding the rare diabetic patients.
**Other metrics:** accuracy, precision, recall (sensitivity), F1-score, ROC-AUC, Brier score and the confusion matrix.
 
### Repository structure
 
Each part of the study lives on its own branch.
 
| Branch | Notebook | What it does | Status |
|---|---|---|---|
| `main` | – | Project home: README and requirements | – |
| `feature-data-cleaning` | `DataProcessing.ipynb` | Data validation: shape, types, nulls, duplicates, odd values | ✅ Done |
| `data-eda` | `DataEDA.ipynb` | Charts, class balance, features vs diagnosis, correlation | ✅ Done |
| `baseline-model` | `BaseLineModel.ipynb` | 5 models, tuning, cross-validation, SMOTENC, final test | 🔄 In progress |
| `missing-data-robustness` | – | Missing-data strategies | ⏳ Planned |
| `calibration` | – | Calibration curves, Brier score, intercept and slope | ⏳ Planned |
| `Computational-efficiency` | – | Training time, prediction time, memory | ⏳ Planned |
| *to be created* | – | Sample-size sensitivity | ⏳ Planned |
| *to be created* | – | SHAP explainability | ⏳ Planned |
 
### Getting started
 
#### Option A – Google Colab (recommended)
 
1. Open a new Colab notebook and run:
```python
!git clone https://github.com/Anshu-Tomar/CapstoneProject-Beyond-Predictive-Accuracy.git
%cd /content/CapstoneProject-Beyond-Predictive-Accuracy
!git checkout baseline-model          # or another branch from the table above
!pip install -q -r requirements.txt
```
 
2. Upload your data file (for example `project_sample_15000.csv`) into `/content/CapstoneProject-Beyond-Predictive-Accuracy` using the Colab **Files** panel.
3. Open the notebook, set `DATA_FILE` in the **Settings** cell, and choose **Runtime → Run all**.
#### Option B – Your own computer
 
```bash
git clone https://github.com/Anshu-Tomar/CapstoneProject-Beyond-Predictive-Accuracy.git
cd CapstoneProject-Beyond-Predictive-Accuracy
git checkout baseline-model
 
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
 
jupyter notebook BaseLineModel.ipynb
```
 
> ⏱️ On 15,000 rows the baseline notebook takes about 3–4 minutes. SVM is very slow on the full 88,380-row file. Remove the SVM entry from the `MODELS` cell if you use the full file.
 
### Output files (baseline notebook)
 
| File | Content |
|---|---|
| `development_data_90.csv` | The 90% development data (training + validation) |
| `final_test_data_10.csv` | The 10% final test data, kept apart until the end |
| `final_test_results.csv` | Final test results: Baseline vs SMOTENC |
 
### Tech stack
 
| Library | Used for |
|---|---|
| `pandas`, `numpy` | Loading and handling data |
| `matplotlib` | Charts and confusion matrices |
| `scikit-learn` | Splits, preprocessing, models, cross-validation, metrics |
| `xgboost` | XGBoost model |
| `imbalanced-learn` | SMOTENC and resampling pipelines |
| `openpyxl` | Reading `.xlsx` data files |
 
### Git workflow
 
```bash
git checkout -b <new-branch-name>     # start a new part of the study
git add .
git commit -m "Short message about the change"
git push -u origin <new-branch-name>
```
 
Open a pull request into `main` when a part is finished.
 
### Disclaimer
 
This project is for research and education only. The models are not clinically validated and must not be used for diagnosis or treatment decisions.
 
### Maintainer
 
Anshu Tomar – [github.com/Anshu-Tomar](https://github.com/Anshu-Tomar)
 
