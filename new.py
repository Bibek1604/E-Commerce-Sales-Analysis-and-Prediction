# Step 1: Problem Definition
# Retail stores struggle to optimize sales strategies based on product category and region.
# Analyzing sales data helps identify key factors that affect sales performance across categories and regions.
# Traditional sales analysis methods miss the complex relationships between product categories, shipping modes, and regional demographics.

# This research will analyze correlations and patterns to understand how sales are influenced by these factors
# and whether targeted strategies for each category or region can improve overall sales performance.

# Step 2: Data Collection and Preprocessing
from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
df = pd.read_csv('SampleSuperstore.csv')

# Display the first few rows to confirm it loaded correctly
df.head()

# Data Preprocessing
# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop missing values (if necessary)
df.dropna(inplace=True)

# Convert 'Postal Code' to string (if applicable)
df['Postal Code'] = df['Postal Code'].astype(str)

# Create a new 'Unit Price' column
df['Unit Price'] = df['Sales'] / df['Quantity']

# Display summary statistics
print("Summary Statistics:\n", df.describe())

# Step 3: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Sales by Region
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Sales', data=df, estimator=sum)
plt.title('Total Sales by Region')
plt.tight_layout()
plt.show()

# Sales by Category
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Sales', data=df, estimator=sum)
plt.title('Total Sales by Category')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sales', 'Profit', 'Discount', 'Quantity']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Machine Learning Models with GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Feature Selection
X = df[['Region', 'Category', 'Discount', 'Profit', 'Quantity']]
y = df['Sales']
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Definitions
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Hyperparameter Grid for Optimization
param_grids = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
    'XGBoost': {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 10], 'n_estimators': [50, 100]}
}

results = {}

for name, model in models.items():
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

# Display Model Results
results_df = pd.DataFrame(results).T
print("\nMachine Learning Model Performance (Optimized):")
print(results_df)

# Visualization: Model Performance Comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['R²'], palette='viridis')
plt.title("Model Comparison: R² Scores")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['RMSE'], palette='magma')
plt.title("Model Comparison: RMSE Scores")
plt.tight_layout()
plt.show()

# Best Model Selection
best_model_name = results_df['R²'].idxmax()
best_model = models[best_model_name]

# Save the Best Model
import joblib
model_path = f'/content/drive/MyDrive/Colab Notebooks/Retail_Sales_Prediction/models/best_{best_model_name}_optimized_model.pkl'
joblib.dump(best_model, model_path)

print(f"✅ The best optimized model is {best_model_name} and has been saved at: {model_path}")
