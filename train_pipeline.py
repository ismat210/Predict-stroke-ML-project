import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Paths
model_path = r"D:\End to end stroke ML model\Notebook\best_stroke_model.pkl"
scaler_path = r"D:\End to end stroke ML model\Notebook\scaler.pkl"

# Load
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

numeric_features = ["age", "avg_glucose_level", "bmi"]
numeric_transformer = scaler  # Already trained scaler
from sklearn.preprocessing import OneHotEncoder

categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
# Create artifacts folder
os.makedirs(r"D:\End to end stroke ML model\artifacts", exist_ok=True)

# Save pipeline
full_pipeline_path = r"D:\End to end stroke ML model\artifacts\stroke_pipeline.pkl"
with open(full_pipeline_path, "wb") as f:
    pickle.dump(pipeline, f)

print("Full pipeline saved successfully in artifacts!")