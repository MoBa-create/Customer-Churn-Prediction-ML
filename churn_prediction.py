from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR,"outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

np.random.seed(42)
n_samples = 1000

credit_score = np.random.randint(350, 850, n_samples)
age = np.random.randint(18, 70, n_samples)
tenure = np.random.randint(0, 10, n_samples)
balance = np.random.uniform(0, 200000, n_samples).round(2)
num_of_products = np.random.randint(1, 5, n_samples)
is_active_member = np.random.randint(0, 2, n_samples)

churn_prob = (
    0.1
    + (age > 45) * 0.3
    + (is_active_member == 0) * 0.25
    + (num_of_products > 2) * 0.2
)

churn_prob = np.clip(churn_prob, 0, 0.9)
exited = (np.random.rand(n_samples) < churn_prob).astype(int)

df = pd.DataFrame({
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "IsActiveMember": is_active_member,
    "Exited": exited
})

x = df.drop(columns=["Exited"])
y = df["Exited"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


print(f"accuracy score : {acc * 100} %")
print(f"confusion matrix : {cm}")

joblib.dump(model, os.path.join(OUTPUTS_DIR, "churn_model.pkl"))
df.to_csv(os.path.join(OUTPUTS_DIR, "churn_data.csv"), index=False)