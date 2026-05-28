import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['study_hours', 'attendance', 'assignments', 'previous_marks']]
y = data['final_score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'student_model.pkl')

print("Model trained successfully!")