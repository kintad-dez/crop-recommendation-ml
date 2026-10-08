import joblib
import pandas as pd


# Load the trained model
model = joblib.load("models/crop_model.pkl")

# Load the scaler used during training
scaler = joblib.load("models/scaler.pkl")

# Load the label encoder
label_encoder = joblib.load("models/label_encoder.pkl")


# Example soil and weather measurements
# The column names and order must match the training data
sample = pd.DataFrame([
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }
])


# Scale the sample using the same scaler
sample_scaled = scaler.transform(sample)


# Make a prediction
prediction = model.predict(sample_scaled)


# Convert the predicted number back into the crop name
crop = label_encoder.inverse_transform(prediction)


print("Recommended crop:", crop[0])