# Customer Churn Prediction (Supervised Learning - Classification)

An end-to-end Machine Learning classification pipeline built with **Scikit-Learn** and **Random Forest Classifier** to predict customer churn (churn vs. retention) based on customer demographic, financial, and behavioral attributes.

---

## 📌 Key Highlights
* **Synthetic / Real-world Customer Dataset:** 1,000 customer samples featuring credit score, age, tenure, balance, number of products, and active membership status.
* **Feature Engineering & Separation:** Isolated behavioral features from the target label (`Exited`).
* **Random Forest Modeling:** Utilized `RandomForestClassifier` for ensemble-based binary classification.
* **Comprehensive Evaluation:** Evaluated using **Accuracy Score** (~70.5%) and **Confusion Matrix** analysis.
* **Modular & Dynamic Pipeline:** Features automated path resolution (`BASE_DIR`) and model serialization (`.pkl`).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib

---

## 📁 Repository Structure
```text
Customer-Churn-Prediction-ML/
│── churn_prediction.py       # Main Python script for training and evaluation
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
│── .gitignore                # Git ignore configuration
└── outputs/                  # Saved artifacts
    ├── churn_model.pkl       # Trained Random Forest model
    └── churn_data.csv        # Processed dataset
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction-ML.git
   cd Customer-Churn-Prediction-ML
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the classification pipeline:**
   ```bash
   python churn_prediction.py
   ```
