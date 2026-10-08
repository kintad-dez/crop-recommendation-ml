import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
df = pd.read_csv("data/raw/Crop_recommendation.csv")


# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]


# Convert crop names into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)
print("Number of classes:", len(label_encoder.classes_))