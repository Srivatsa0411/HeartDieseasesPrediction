# 🪀 Heart Disease Prediction with XGBoost

This project builds a robust machine learning pipeline to **predict the presence of heart disease** using patient medical data. It leverages advanced preprocessing, detailed exploratory analysis, and a highly tuned XGBoost model to deliver high predictive accuracy.

---

## 🚀 Highlights

* ✅ Achieved **F1 Score of 0.91089** on the test set (`test_X.csv`)
* 📊 Clean and explainable **EDA notebook** included
* 🧪 Robust pipeline using **XGBoost** + `RandomizedSearchCV`
* 🧼 Smart preprocessing: outlier capping, null handling, and optional SMOTE
* 🔍 Designed for production use with clean `src/` module layout

---

## 🧠 Project Structure

```
heartdiseaseprediction/
├── data/                        # Input data files
│   ├── train.csv
│   └── test_X.csv
├── notebooks/
│   └── exploratory_checks.ipynb  # EDA and feature sanity checks
├── src/                        # Modular codebase
│   ├── config.py               # Feature lists
│   ├── preprocess.py           # Cleaning and transformation
│   ├── model.py                # Model and hyperparameter tuning
│   └── utils.py                # Evaluation and submission helper
├── y_predict.csv               # Final predictions for test_X
├── requirements.txt            # Dependencies
├── .gitignore
└── README.md
```

---

## 📓 Notebooks

The notebook `exploratory_checks.ipynb` performs:

* Null value inspection
* Feature distribution plots
* Outlier and anomaly detection
* Correlation heatmap
* Class imbalance visualization

It helped reveal irregular values like **non-positive cholesterol** and guided the data cleaning strategy.

---

## 🧪 Model Pipeline

The core model is implemented using:

* `XGBoostClassifier` with:

  * Grid-tuned hyperparameters
  * Stratified 10-fold cross-validation
* `Pipeline` + `ColumnTransformer` for modular preprocessing
* Optional `SMOTE` block (currently disabled)
* Final model retrained on full dataset before generating predictions

---

## 🛠 Tools & Libraries

* Python 3.x
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn
* XGBoost
* Imbalanced-learn (for optional SMOTE)

---

## ✅ Output

The final prediction file `y_predict.csv` contains the model's output on `test_X.csv`.
Model performance:

* **Validation F1 Score**: \~0.91
* Predictions align closely with actual outcomes, demonstrating strong generalization.

---

## 📊 Evaluation

The evaluation metric for this competition is **Mean F1-Score**. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision (**p**) and recall (**r**):

* **Precision (p)** = TP / (TP + FP)
* **Recall (r)** = TP / (TP + FN)

The F1 score is the harmonic mean of precision and recall:

```
F1 = 2 * (p * r) / (p + r)
```

This metric balances both false positives and false negatives and is ideal for imbalanced datasets like this one.

---

## 📌 How to Run

1. Install dependencies

   ```
   pip install -r requirements.txt
   ```

2. Place your `train.csv` and `test_X.csv` in the `data/` directory.

3. Run:

   ```
   python main.py
   ```

4. Output saved to `y_predict.csv`.

---

## 🙋 About the Author

This project was independently developed as part of my machine learning portfolio to demonstrate end-to-end data science capabilities — from cleaning raw data to deploying high-performing models.
For inquiries, feel free to connect via [LinkedIn](https://www.linkedin.com/in/srivatsa-bhamidipati/) or explore other projects on my GitHub profile.

---
