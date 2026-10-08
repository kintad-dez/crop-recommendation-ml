import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/raw/Crop_recommendation.csv")


# ==========================================
# 2. BASIC DATASET INSPECTION
# ==========================================

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCrop Classes:")
print(df["label"].value_counts())

print("\nStatistical Summary:")
print(df.describe())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nNumber of Crop Types:")
print(df["label"].nunique())

print("\nCrop Types:")
print(sorted(df["label"].unique()))


# ==========================================
# 3. FEATURE CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.drop("label", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==========================================
# 4. FEATURE DISTRIBUTIONS
# ==========================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

df[features].hist(
    figsize=(12, 10),
    bins=20
)

plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()


# ==========================================
# 5. FEATURE DISTRIBUTIONS BY CROP
# ==========================================

fig, axes = plt.subplots(4, 2, figsize=(18, 24))

for ax, feature in zip(axes.flat, features):

    sns.boxplot(
        data=df,
        x="label",
        y=feature,
        ax=ax
    )

    ax.set_title(
        f"{feature} Distribution by Crop",
        fontsize=14
    )

    ax.set_xlabel("Crop")
    ax.set_ylabel(feature)

    ax.tick_params(
        axis="x",
        rotation=45
    )


# Hide the unused eighth subplot
axes.flat[-1].set_visible(False)

plt.tight_layout()
plt.show() 

# ==========================================
# 6. PREPARE FEATURES AND TARGET
# ==========================================

# Features used to make the crop prediction
X = df[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

# Target variable: the crop we want to predict
y = df["label"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())