# 🌱 Crop Recommendation System

A machine learning web application that recommends a suitable crop based on soil nutrients and environmental conditions.

## 📌 Overview

The Crop Recommendation System uses machine learning to analyze soil and weather conditions and recommend the crop most suitable for the provided conditions.

Users enter:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The trained machine learning model processes these values and returns a recommended crop along with the model's confidence.

## 🚀 Features

- 🌱 Crop recommendation based on soil and environmental conditions
- 🤖 Machine learning classification model
- 📊 Data preprocessing and feature scaling
- 🌐 Flask web interface
- 📈 Model evaluation
- 💾 Saved trained model and preprocessing objects
- ⚠️ Input validation for unrealistic values

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Matplotlib
- Seaborn
- Joblib
- HTML/CSS

## 📂 Project Structure

```text
crop-recommendation-ml/
│
├── data/
│   └── raw/
│       └── Crop_recommendation.csv
│
├── models/
│   ├── crop_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── crop_recommendation.ipynb
│
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── requirements.txt
├── LICENSE
└── README.md