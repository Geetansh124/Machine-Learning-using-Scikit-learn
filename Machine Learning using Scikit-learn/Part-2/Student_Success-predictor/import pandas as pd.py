import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Student_success_dataset.csv")

le = LabelEncoder()
df["Internet"] = le.fit_transform(df["Internet"])
df["Passed"] = le.fit_transform(df["Passed"])

features = [
    "StudyHours",
    "Attendance",
    "PastScore",
    "Internet",
    "SleepHours",
]
scaler = StandardScaler()
df_scaled = df.copy()

df_scaled[features] = scaler.fit_transform(df[features])
X = df_scaled[features]
y = df["Passed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()

print("--------Predict Your Result--------")
try:
    StudyHours = float(input("Enter Study Hours: ").strip())
    Attendance = float(input("Enter attendance (0-100): ").strip())
    PastScore = float(input("Enter past score (0-100): ").strip())
    SleepHours = float(input("Enter sleep hours: ").strip())
    Internet = input("Internet access (Yes/No): ").strip()

    user_input_df = pd.DataFrame(
        [
            {
                "StudyHours": StudyHours,
                "Attendance": Attendance,
                "PastScore": PastScore,
                "Internet": Internet,
                "SleepHours": SleepHours,
            }
        ]
    )

    # Encode and scale user input using the same transformers as the training data
    user_input_df["Internet"] = le.transform(user_input_df["Internet"])
    user_input_scaled = scaler.transform(user_input_df[features])

    prediction = model.predict(user_input_scaled)[0]

    result = "Pass" if prediction == 1 else "Fail"

    print(f"Prediction Based on input: {result}")

except Exception as e:
    print("An error occurred:", e)
