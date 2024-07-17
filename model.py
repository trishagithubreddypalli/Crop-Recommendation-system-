import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import pickle

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

# Validation
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style("whitegrid")
pio.templates.default = "plotly_white"


# Analyze Data
def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print("\nDataset columns:", df.columns)
    print("\nData types of each column:\n", df.dtypes)
    print("\nFirst few rows of the dataset:\n", df.head())
    print("\nMissing values in each column:\n", df.isnull().sum())
    print("\nClass distribution:\n", df["label"].value_counts())
    print("\nSummary statistics of numeric columns:\n", df.describe())


# Checking for Duplicates
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print("Duplicate values removed!")
    else:
        print("No Duplicate values")
    return df


# Split Data to Training and Validation set
def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


# Train Model
def fit_model(X_train, y_train, X_test, y_test, model):
    # Fit the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion Matrix", fontsize=20, y=1.1)
    plt.ylabel("Actual label", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)
    plt.show()

    # Classification Report
    print(classification_report(y_test, y_pred))


# Load Dataset
df = pd.read_csv("SmartCrop-Dataset.csv")

# Explore Data
explore_data(df)

# Checking and Removing Duplicates
df = checking_removing_duplicates(df)

# Split Data to Training and Validation set
target = "label"
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

# Train model
pipeline = make_pipeline(StandardScaler(), GaussianNB())
fit_model(X_train, y_train, X_test, y_test, pipeline)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)
