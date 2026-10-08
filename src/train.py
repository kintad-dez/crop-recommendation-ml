import joblib
import numpy as np


# Load the trained model
model = joblib.load("models/crop_model.pkl")

# Load the scaler used during training
scaler = joblib.load("models/scaler.pkl")

# Load the label encoder
label_encoder = joblib.load("models/label_encoder.pkl")


# Example soil and weather measurements
# Order:
# N, P, K, temperature, humidity, ph, rainfall

sample = np.array([
    [90, 42, 43, 20.8, 82.0, 6.5, 202.9]
])


# Scale the sample using the same scaler
sample_scaled = scaler.transform(sample)


# Make a prediction
prediction = model.predict(sample_scaled)


# Convert the predicted number back to the crop name
crop = label_encoder.inverse_transform(prediction)


print("Recommended crop:", crop[0])