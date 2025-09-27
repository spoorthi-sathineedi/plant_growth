# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv("plant_growth_data.csv")

# %%
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.columns

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.describe()

# %%
df.nunique()

# %%
# Let's inspect unique values / samples from each column to see if any has a pattern like "value/value"
samples = {col: df[col].astype(str).unique()[:10] for col in df.columns}
samples


# %%
df.columns

# %%
# Calculate averages for plant growth dataset
average_temp = df['Temperature'].mean()
average_humidity = df['Humidity'].mean()
average_sunlight = df['Sunlight_Hours'].mean()

print(f"Average Temperature: {average_temp:.2f} °C")
print(f"Average Humidity: {average_humidity:.2f} %")
print(f"Average Sunlight Hours: {average_sunlight:.2f} hours/day")


# %%
# Create a copy of the DataFrame with only the selected columns
df_selected = df.copy()

# %%
# If you want to drop some columns from plant growth dataset
df_selected = df.drop(['Soil_Type', 'Fertilizer_Type'], axis=1)  # example

print(df_selected.head())


# %%
# Encode categorical features from plant growth dataset
df_encoded = pd.get_dummies(
    df, 
    columns=['Soil_Type', 'Water_Frequency', 'Fertilizer_Type'],
    drop_first=True
)

print(df_encoded.head())


# %%
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
label_encoder = LabelEncoder()

# Encode categorical features one by one
for col in ['Soil_Type', 'Water_Frequency', 'Fertilizer_Type']:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

print(df_encoded.head())


# %%
df_encoded.columns

# %%
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Features (X) and target (y)
X = df_encoded.drop('Growth_Milestone', axis=1)  # Drop target column
y = df_encoded['Growth_Milestone']              # Target variable

# Create a random forest classifier
clf = RandomForestClassifier(random_state=42)

# Fit the classifier to the data
clf.fit(X, y)

# Get feature importance scores
feature_importance = clf.feature_importances_

# Create a DataFrame to display feature importance scores
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Print the feature importance scores
print(feature_importance_df)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# %%
logreg = LogisticRegression()

# %%
logreg.fit(X_train_resampled, y_train_resampled)

# %%
y_pred = logreg.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# %%
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train_resampled, y_train_resampled)

# %%
y_pred = svc.predict(X_test)

# %%
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# %%
import pickle
with open('logreg_model.pkl', 'wb') as file:
    pickle.dump(logreg, file)
print("Model saved as logreg_model.pkl")

# %%



