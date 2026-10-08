from flask import Flask, render_template, request
import joblib
import pandas as pd


# Create the Flask application
app = Flask(__name__)


# Load the trained machine learning model
model = joblib.load("models/crop_model.pkl")

# Load the scaler used during training
scaler = joblib.load("models/scaler.pkl")

# Load the label encoder
label_encoder = joblib.load("models/label_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    crop = None
    confidence = None
    error = None

    if request.method == "POST":

        try:
            # Get values entered by the user
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Validate the input values
            if not 0 <= N <= 140:
                raise ValueError("Nitrogen must be between 0 and 140.")

            if not 0 <= P <= 145:
                raise ValueError("Phosphorus must be between 0 and 145.")

            if not 0 <= K <= 205:
                raise ValueError("Potassium must be between 0 and 205.")

            if not -10 <= temperature <= 50:
                raise ValueError("Temperature must be between -10°C and 50°C.")

            if not 0 <= humidity <= 100:
                raise ValueError("Humidity must be between 0% and 100%.")

            if not 0 <= ph <= 14:
                raise ValueError("pH must be between 0 and 14.")

            if not 0 <= rainfall <= 500:
                raise ValueError("Rainfall must be between 0 and 500 mm.")

            # Put the values into a DataFrame
            sample = pd.DataFrame([
                {
                    "N": N,
                    "P": P,
                    "K": K,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall
                }
            ])

            # Scale the input using the same scaler used during training
            sample_scaled = scaler.transform(sample)

            # Make the prediction
            prediction = model.predict(sample_scaled)

            # Convert the numerical prediction back to a crop name
            crop = label_encoder.inverse_transform(prediction)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(sample_scaled)

            # Get the highest prediction probability
            confidence = probabilities.max() * 100

        except ValueError as e:

            # Show validation error to the user
            error = str(e)

        except Exception:

            # Handle unexpected errors
            error = "Please enter valid values for all fields."

    return render_template(
        "index.html",
        crop=crop,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)