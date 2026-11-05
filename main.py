
# EXPERIMENT 2
# Exploratory Data Analysis of Healthcare Data

# STEP 1: Import Libraries

!pip install kagglehub -q

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

print("Libraries imported successfully!")


# STEP 2: Download and Load Healthcare Dataset

p = kagglehub.dataset_download("prasad22/healthcare-dataset")

csv_path = os.path.join(p, "healthcare_dataset.csv")

df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")


# STEP 3: Dataset Information

print("Shape of the dataset:", df.shape)

print("\nColumn names:\n", df.columns.tolist())

print("\nData types:\n", df.dtypes)

print("\nMissing values:\n", df.isnull().sum())


# STEP 4: Missing Values Heatmap

sns.heatmap(df.isnull(), cbar=False)

plt.title("Missing Values Heatmap")

plt.show()


# STEP 5: Drop Missing Values and Remove Duplicates

df = df.dropna()

print("\nNumber of duplicate records:", df.duplicated().sum())

df = df.drop_duplicates()


# STEP 6: Descriptive Statistics

print("\nDescriptive Statistics (Numerical):\n", df.describe())

print("\nDescriptive Statistics (Categorical):\n", df.describe(include='object'))


# STEP 7: Histograms of Numerical Features

numerical_cols = df.select_dtypes(include='number').columns

df[numerical_cols].hist(figsize=(12, 8))

plt.suptitle("Histograms of Numerical Features")

plt.show()


# STEP 8: Boxplots of Numerical Features

for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# STEP 9: Outlier Detection Using Boxplots

categorical_cols = df.select_dtypes(include='object').columns

for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Outlier Detection in {col}")
    plt.show()


# STEP 10: Distribution of Medical Condition

if 'Medical Condition' in df.columns:

    print("\nDistribution of Medical Condition:\n",
          df['Medical Condition'].value_counts())

    df['Medical Condition'].value_counts().plot(kind='bar')

    plt.title("Distribution of Target Variable (Medical Condition)")

    plt.ylabel("Count")

    plt.show()


# STEP 11: Summary of Cleaned Dataset

print("\n--- Summary ---")

print("Dataset has", df.shape[0], "rows and",
      df.shape[1], "columns after cleaning.")

print("Target variable distribution:")

print(df['Medical Condition'].value_counts())


# FINAL RESULT

print("\nExperiment 2 Completed Successfully!")
