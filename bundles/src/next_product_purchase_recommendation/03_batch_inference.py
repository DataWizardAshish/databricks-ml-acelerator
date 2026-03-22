# Databricks notebook source
# Batch Inference Notebook: Next Product Purchase Recommendation
# Loads Champion model from Unity Catalog and scores feature table

# COMMAND ----------

import mlflow
import pandas as pd
from datetime import datetime
from pyspark.sql import functions as F

# Set MLflow registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Create widgets for parameterization
dbutils.widgets.text("catalog", "lakehouse_dev")
dbutils.widgets.text("output_schema", "ml_accelerator")
dbutils.widgets.text("model_name", "next_product_purchase_recommendation_model")
dbutils.widgets.text("run_date", datetime.now().strftime("%Y-%m-%d"))

# COMMAND ----------

# Get widget values
catalog = dbutils.widgets.get("catalog")
output_schema = dbutils.widgets.get("output_schema")
model_name = dbutils.widgets.get("model_name")
run_date = dbutils.widgets.get("run_date")

print(f"Catalog: {catalog}")
print(f"Output Schema: {output_schema}")
print(f"Model Name: {model_name}")
print(f"Run Date: {run_date}")

# COMMAND ----------

# Load Champion model from Unity Catalog
model_uri = f"models:/{catalog}.{output_schema}.{model_name}@Champion"
print(f"Loading model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded successfully")

# COMMAND ----------

# Read feature table filtered to run_date
feature_table_name = f"{catalog}.{output_schema}.next_product_purchase_recommendation_features"
print(f"Reading features from: {feature_table_name}")

features_df = spark.table(feature_table_name).filter(F.col("run_date") == run_date)

feature_count = features_df.count()
print(f"Features loaded: {feature_count} records for run_date={run_date}")

if feature_count == 0:
    raise ValueError(f"No features found for run_date={run_date}. Please check feature table.")

# COMMAND ----------

# Display sample of features
display(features_df.limit(5))

# COMMAND ----------

# Identify key columns to preserve
key_columns = ["customer_id", "product_id", "run_date"]

# Get feature columns (exclude keys and metadata)
all_columns = features_df.columns
feature_columns = [col for col in all_columns if col not in key_columns]

print(f"Key columns: {key_columns}")
print(f"Feature columns ({len(feature_columns)}): {feature_columns[:10]}...")

# COMMAND ----------

# Prepare data for scoring
# Keep keys separate for joining back later
keys_df = features_df.select(*key_columns)

# Extract only feature columns for model input
model_input_df = features_df.select(*feature_columns)

# COMMAND ----------

# Convert to pandas for batch prediction
print("Converting to pandas for prediction...")
model_input_pd = model_input_df.toPandas()

print(f"Pandas DataFrame shape: {model_input_pd.shape}")
print(f"Memory usage: {model_input_pd.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# COMMAND ----------

# Generate predictions
print("Generating predictions...")
predictions = model.predict(model_input_pd)

print(f"Predictions generated: {len(predictions)} records")
print(f"Prediction type: {type(predictions)}")

# COMMAND ----------

# Convert predictions to DataFrame
if isinstance(predictions, pd.DataFrame):
    predictions_pd = predictions
else:
    predictions_pd = pd.DataFrame({"prediction": predictions})

print(f"Predictions DataFrame shape: {predictions_pd.shape}")
print(f"Predictions columns: {predictions_pd.columns.tolist()}")

# COMMAND ----------

# Convert predictions back to Spark DataFrame
predictions_spark = spark.createDataFrame(predictions_pd)

# COMMAND ----------

# Add row numbers to both DataFrames for joining
keys_with_index = keys_df.withColumn("row_id", F.monotonically_increasing_id())
predictions_with_index = predictions_spark.withColumn("row_id", F.monotonically_increasing_id())

# COMMAND ----------

# Join keys with predictions
scores_df = keys_with_index.join(predictions_with_index, on="row_id", how="inner").drop("row_id")

# COMMAND ----------

# Add metadata columns
scores_df = scores_df.withColumn("score_date", F.lit(run_date).cast("date"))
scores_df = scores_df.withColumn("model_name", F.lit(model_name))
scores_df = scores_df.withColumn("scored_at", F.current_timestamp())

# COMMAND ----------

# Display sample of scores
print("Sample of scored records:")
display(scores_df.limit(10))

# COMMAND ----------

# Verify schema before writing
print("Scores DataFrame schema:")
scores_df.printSchema()

scored_count = scores_df.count()
print(f"Total records to write: {scored_count}")

# COMMAND ----------

# Write scores to output table
output_table = f"{catalog}.{output_schema}.next_product_purchase_recommendation_scores"
print(f"Writing scores to: {output_table}")

scores_df.write.format("delta").mode("append").partitionBy("score_date").saveAsTable(output_table)

print(f"Successfully wrote {scored_count} records to {output_table}")

# COMMAND ----------

# Grant SELECT permissions to account users
grant_sql = f"GRANT SELECT ON TABLE {catalog}.{output_schema}.next_product_purchase_recommendation_scores TO `account users`"
print(f"Executing: {grant_sql}")

spark.sql(grant_sql)
print("Permissions granted successfully")

# COMMAND ----------

# Summary statistics
print("=" * 80)
print("BATCH INFERENCE SUMMARY")
print("=" * 80)
print(f"Run Date: {run_date}")
print(f"Model: {model_uri}")
print(f"Feature Table: {feature_table_name}")
print(f"Output Table: {output_table}")
print(f"Records Scored: {scored_count}")
print(f"Scored At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# COMMAND ----------

# Verify data was written correctly
verification_df = spark.table(output_table).filter(F.col("score_date") == run_date)
verification_count = verification_df.count()

print(f"Verification: {verification_count} records found in output table for score_date={run_date}")

if verification_count != scored_count:
    print(f"WARNING: Mismatch between scored ({scored_count}) and verified ({verification_count}) counts")
else:
    print("✓ Verification successful: All records written correctly")

# COMMAND ----------

# Display prediction distribution
print("Prediction distribution:")
display(verification_df.groupBy("prediction").count().orderBy(F.desc("count")).limit(20))

# COMMAND ----------

# Display sample of final output
print("Sample of final scored output:")
display(verification_df.orderBy(F.desc("scored_at")).limit(10))