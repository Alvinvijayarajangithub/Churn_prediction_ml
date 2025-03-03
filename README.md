# Churn_prediction_ml
This project focuses on predicting customer churn in the banking sector using machine learning techniques. Customer churn refers to customers leaving a bank, which can lead to significant financial losses. 
# Loading Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
path = "/content/Untitled spreadsheet - BankChurners.csv"
df = pd.read_csv(path)
df.describe()
#Data Preprocessing
df.isnull().sum()
data = pd.read_csv("/content/Untitled spreadsheet - BankChurners.csv")
missing_values = data.isnull().sum()
print(missing_values)
#Handling Missing Values
# handling missing data
df.fillna(df.mean(numeric_only=True), inplace=True)
#Checking for outliers using boxplots
# Checking for outliers using boxplots
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_cols])
plt.xticks(rotation=90)
plt.title("Boxplot for Outlier Detection")
plt.show()
# Handling Outliers using IQR Method
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    # Checking for skewness and sorting in ascending order
skewness = df[numerical_cols].skew().sort_values(ascending=True)

print("\nSkewness in numerical columns (Sorted in Ascending Order):")
print(skewness)

# Visualizing distributions with histograms
plt.figure(figsize=(15, 8))
# Adjust the layout to accommodate all numerical columns
# Calculate the number of rows needed:
num_rows = int(np.ceil(len(numerical_cols) / 4))  # Assuming 4 columns

df[numerical_cols].hist(bins=30, figsize=(15, 10), layout=(num_rows, 4))  # Changed layout
plt.suptitle("Histograms of Numerical Variables")
plt.tight_layout()  # Adjust subplot parameters for a tight layout
plt.show()
# Univariate Analysis: Countplot for categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
plt.figure(figsize=(15, 8))
for i, col in enumerate(categorical_cols[:4]):  # Limiting to the first 4 columns
    plt.subplot(2, 2, i+1)
    sns.countplot(x=df[col])
    plt.xticks(rotation=45)
    plt.title(f"Countplot of {col}")
plt.tight_layout()
plt.show()

# Bivariate Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x=df["Attrition_Flag"], y=df["Total_Trans_Amt"], palette="coolwarm")
plt.title("Churn vs. Total Transaction Amount")
plt.show()


sns.violinplot(x=df["Attrition_Flag"], y=df["Total_Trans_Amt"], palette="coolwarm")
plt.title("Churn vs. Transaction Amount (Violin Plot)")
plt.show()


# Heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

sns.kdeplot(df[df["Attrition_Flag"] == "Existing Customer"]["Total_Trans_Amt"], label="Active Customers", shade=True, color="blue")

#Credit Limit vs. Total Transaction Amount (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Credit_Limit"], y=df["Total_Trans_Amt"], hue=df["Attrition_Flag"], palette="coolwarm", alpha=0.7)
plt.title("Credit Limit vs. Total Transaction Amount")
plt.xlabel("Credit Limit")
plt.ylabel("Total Transaction Amount")
plt.legend(title="Customer Status")
plt.show()

#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

# Encoding categorical variables
label_encoder = LabelEncoder()
df["Attrition_Flag"] = label_encoder.fit_transform(df["Attrition_Flag"])  # 0 = Existing Customer, 1 = Attrited Customer

# One-Hot Encoding for other categorical features
df = pd.get_dummies(df, drop_first=True)

#Feature Selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Train a Random Forest model to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract feature importance from Random Forest
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Select the top K features using SelectKBest
k = 10  # Choose the number of top features
selector = SelectKBest(score_func=f_classif, k=k)
X_new = selector.fit_transform(X_train, y_train)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]

# Print selected features
print("Top Selected Features:\n", selected_features)

# Plot feature importance (Top 10 from Random Forest)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances.Importance[:10], y=feature_importances.Feature[:10], palette="coolwarm")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()

#Splitting Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# Defining features (X) and target variable (y)
X = df.drop(columns=["Attrition_Flag"])  # Features
y = df["Attrition_Flag"]  # Target

# Splitting into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model Selection & Training

#Logistic Regression (Baseline Model)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

#Support Vector Machine (SVM)
from sklearn.svm import SVC

svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test)

#Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate Logistic Regression
print("Logistic Regression Report:\n", classification_report(y_test, y_pred))

# Evaluate Random Forest
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# Evaluate SVM
print("SVM Report:\n", classification_report(y_test, y_pred_svm))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

#Hyperparameter Tuning & Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameters
param_grid = {
    'RandomForest': {
        'model': [RandomForestClassifier()],
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'model': [GradientBoostingClassifier()],
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 10]
    },
    'SVM': {
        'model': [SVC()],
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },
    'LogisticRegression': {
        'model': [LogisticRegression()],
        'model__C': [0.1, 1, 10],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear']
    }
}

# Iterate over models and perform GridSearchCV
best_models = {}
for model_name, params in param_grid.items():
    print(f"\nTuning {model_name}...")

    # Create pipeline with StandardScaler and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', params['model'][0])  # Placeholder model for GridSearch
    ])

    # Run GridSearch
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    # Save best model and score
    best_models[model_name] = {'best_model': grid.best_estimator_, 'best_score': grid.best_score_}

    print(f"Best Parameters for {model_name}: {grid.best_params_}")
    print(f"Best Accuracy: {grid.best_score_:.4f}")

# Evaluate the best model on test data
best_model_name = max(best_models, key=lambda x: best_models[x]['best_score'])
final_model = best_models[best_model_name]['best_model']

print(f"\nBest Model Selected: {best_model_name}")
y_pred = final_model.predict(X_test)

# Final Evaluation
from sklearn.metrics import accuracy_score, classification_report

print("\nFinal Model Performance on Test Data:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine (SVM)": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "MLP Classifier": MLPClassifier(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boost": GradientBoostingClassifier(),
    "Adaboost": AdaBoostClassifier()
}

# Train and evaluate classification models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Model:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

import joblib

# Save model
joblib.dump(rf_model, "bank_churn_model.pkl")

# Load model
loaded_model = joblib.load("bank_churn_model.pkl")

# Creating new features based on existing data
df["Credit_Utilization"] = df["Total_Trans_Amt"] / df["Credit_Limit"]  # Credit usage ratio
df["Avg_Transaction_Value"] = df["Total_Trans_Amt"] / df["Total_Trans_Ct"]  # Avg transaction value
df["Transaction_Frequency"] = df["Total_Trans_Ct"] / 12  # Monthly transaction frequency

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Check on unseen data if available
if 'X_unseen' in locals():
    y_unseen_pred = rf.predict(X_unseen)
    print("Predictions on unseen data:\n", y_unseen_pred)

#Conclusion:
"The Random Forest model effectively identifies key factors influencing customer attrition, with top features such as transaction amount, credit limit, and transaction count playing significant roles. The classification report and confusion matrix indicate the model's effectiveness, and testing on unseen data provides insights into its generalization ability."
#Next Steps & Future Work
Enhance model performance by fine-tuning hyperparameters and incorporating more diverse datasets. Deploy the model as a web application for real-time customer churn predictions. Explore advanced machine learning techniques such as XGBoost and deep learning models for improved accuracy. Implement customer segmentation strategies to personalize retention efforts.
This project provides valuable insights into customer attrition and highlights the power of machine learning in predicting churn. With further improvements, this model can be leveraged for proactive customer engagement, ultimately reducing churn rates and boosting business profitability.
#Best Model: The Random Forest Classifier, achieving an accuracy of 1.00, has been saved as Random_Forest_Best.pkl for future use.
