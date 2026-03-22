# Databricks notebook source
# COMMAND ----------

# Databricks notebook source

# COMMAND ----------

# Install required libraries
%pip install xgboost scikit-learn mlflow --quiet
dbutils.library.restartPython()

# COMMAND ----------

# Import libraries
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------

# Create widgets
dbutils.widgets.text("catalog", "lakehouse_dev")
dbutils.widgets.text("output_schema", "ml_accelerator")
dbutils.widgets.text("model_name", "next_product_purchase_recommendation_model")
dbutils.widgets.text("experiment_name", "/ml_accelerator/next_product_purchase_recommendation")

# COMMAND ----------

# Get widget values
catalog = dbutils.widgets.get("catalog")
output_schema = dbutils.widgets.get("output_schema")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")

print(f"Catalog: {catalog}")
print(f"Output Schema: {output_schema}")
print(f"Model Name: {model_name}")
print(f"Experiment Name: {experiment_name}")

# COMMAND ----------

# Set MLflow registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set experiment
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# Read feature table
feature_table_name = f"{catalog}.{output_schema}.next_product_purchase_recommendation_features"
print(f"Reading feature table: {feature_table_name}")

df = spark.table(feature_table_name)
print(f"Total rows: {df.count()}")
df.printSchema()

# COMMAND ----------

# Convert to Pandas (after filtering to manageable size if needed)
# Drop run_date partition column
columns_to_drop = ["run_date"]
feature_columns = [col for col in df.columns if col not in columns_to_drop]

df_filtered = df.select(feature_columns)
print(f"Columns after dropping run_date: {len(df_filtered.columns)}")

# COMMAND ----------

# Convert to Pandas for sklearn
pdf = df_filtered.toPandas()
print(f"Pandas DataFrame shape: {pdf.shape}")
print(f"Columns: {pdf.columns.tolist()}")

# COMMAND ----------

# Identify target column and feature columns
target_column = "product_id"

# Verify target column exists
if target_column not in pdf.columns:
    raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {pdf.columns.tolist()}")

# COMMAND ----------

# Sort by date_id for temporal split
# Assuming there's a date-related column for sorting
date_columns = [col for col in pdf.columns if 'date' in col.lower() or 'days_since' in col.lower()]
print(f"Date-related columns found: {date_columns}")

# Use the first date-related column for sorting, or create a row index
if 'days_since_last_order' in pdf.columns:
    sort_column = 'days_since_last_order'
    pdf_sorted = pdf.sort_values(by=sort_column, ascending=True)
elif 'days_since_first_order' in pdf.columns:
    sort_column = 'days_since_first_order'
    pdf_sorted = pdf.sort_values(by=sort_column, ascending=True)
else:
    # If no date column, use row order as proxy
    pdf_sorted = pdf.copy()
    print("Warning: No date column found for temporal sorting. Using row order.")

# COMMAND ----------

# Prepare features and target
X = pdf_sorted.drop(columns=[target_column])
y = pdf_sorted[target_column]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target unique values: {y.nunique()}")

# COMMAND ----------

# Encode target variable (product_id) as it's likely categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Target classes: {len(label_encoder.classes_)}")
print(f"Sample encoded values: {y_encoded[:10]}")

# COMMAND ----------

# Handle non-numeric features
# Identify numeric and non-numeric columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Non-numeric features: {len(non_numeric_features)}")

if non_numeric_features:
    print(f"Non-numeric columns: {non_numeric_features}")
    # Drop non-numeric features or encode them
    # For simplicity, dropping non-numeric features
    X = X[numeric_features]
    print(f"Features after dropping non-numeric: {X.shape}")

# COMMAND ----------

# Temporal train/test split (last 20% as test)
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y_encoded[:split_index]
y_test = y_encoded[split_index:]

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train target distribution: {np.bincount(y_train)[:10]}")
print(f"Test target distribution: {np.bincount(y_test)[:10]}")

# COMMAND ----------

# Calculate scale_pos_weight for class imbalance
# For multiclass, XGBoost doesn't use scale_pos_weight directly
# Instead, we'll use sample weights or class weights
# For simplicity, we'll compute class weights

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = np.array([class_weights[label] for label in y_train])

print(f"Class weights computed for {len(class_weights)} classes")
print(f"Sample weights shape: {sample_weights.shape}")

# COMMAND ----------

# Create sklearn pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        tree_method='hist',
        enable_categorical=False
    ))
])

print("Pipeline created")

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name=f"xgboost_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    
    # Log parameters
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("num_classes", len(label_encoder.classes_))
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("num_features", X_train.shape[1])
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("imputer_strategy", "median")
    mlflow.log_param("split_ratio", 0.8)
    mlflow.log_param("class_weight", "balanced")
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    print("Model training completed")
    
    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    train_accuracy = (y_train_pred == y_train).mean()
    test_accuracy = (y_test_pred == y_test).mean()
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # For multiclass, calculate top-k accuracy
    y_test_proba = pipeline.predict_proba(X_test)
    
    # Top-5 accuracy
    top5_preds = np.argsort(y_test_proba, axis=1)[:, -5:]
    top5_accuracy = np.mean([y_test[i] in top5_preds[i] for i in range(len(y_test))])
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    mlflow.log_metric("top5_accuracy", top5_accuracy)
    
    # Top-10 accuracy
    top10_preds = np.argsort(y_test_proba, axis=1)[:, -10:]
    top10_accuracy = np.mean([y_test[i] in top10_preds[i] for i in range(len(y_test))])
    print(f"Top-10 Accuracy: {top10_accuracy:.4f}")
    mlflow.log_metric("top10_accuracy", top10_accuracy)
    
    # Calculate AUC for multiclass (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
        
        # Handle case where test set doesn't have all classes
        if y_test_bin.shape[1] == y_test_proba.shape[1]:
            auc_ovr = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')
            print(f"AUC (OVR, weighted): {auc_ovr:.4f}")
            mlflow.log_metric("auc_ovr_weighted", auc_ovr)
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
    
    # Feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    feature_names = X_train.columns.tolist()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Feature Importances:")
    print(importance_df.head(15).to_string(index=False))
    
    # Log feature importance as artifact
    importance_path = "/tmp/feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path)
    
    # Create input example
    input_example = X_test.head(5)
    
    # Log model
    print("\nLogging model to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None  # Will register separately
    )
    print("Model logged successfully")

# COMMAND ----------

# Register model to Unity Catalog
print(f"\nRegistering model to Unity Catalog: {catalog}.{output_schema}.{model_name}")

model_uri = f"runs:/{run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=f"{catalog}.{output_schema}.{model_name}"
)

print(f"Model registered: {registered_model.name}, Version: {registered_model.version}")

# COMMAND ----------

# Set Champion alias
client = MlflowClient()

print(f"Setting 'Champion' alias for version {registered_model.version}")
client.set_registered_model_alias(
    name=f"{catalog}.{output_schema}.{model_name}",
    alias="Champion",
    version=registered_model.version
)

print("Champion alias set successfully")

# COMMAND ----------

# Grant permissions on model
grant_sql = f"""
GRANT ALL PRIVILEGES ON MODEL `{catalog}`.`{output_schema}`.`{model_name}` TO `account users`
"""

try:
    spark.sql(grant_sql)
    print("Permissions granted successfully")
except Exception as e:
    print(f"Warning: Could not grant permissions: {e}")

# COMMAND ----------

# Print final summary
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Model: {catalog}.{output_schema}.{model_name}")
print(f"Version: {registered_model.version}")
print(f"Alias: Champion")
print(f"Run ID: {run_id}")
print(f"\nMetrics:")
print(f"  Train Accuracy: {train_accuracy:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
print(f"  Top-10 Accuracy: {top10_accuracy:.4f}")
print(f"\nTarget Classes: {len(label_encoder.classes_)}")
print(f"Features: {X_train.shape[1]}")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")
print("="*80)

# COMMAND ----------

# Display top 15 features
display(importance_df.head(15))