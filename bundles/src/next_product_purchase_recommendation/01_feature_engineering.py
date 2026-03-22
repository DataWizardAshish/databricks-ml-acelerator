# Databricks notebook source
# Feature Engineering: Next Product Purchase Recommendation
# Use Case: Predict next product a customer will purchase based on historical behavior
# ML Type: Multiclass Classification
# Target: product_id (label encoded)

# COMMAND ----------

# Import required libraries
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import hashlib

# COMMAND ----------

# Create widgets for parameterization
dbutils.widgets.text("catalog", "lakehouse_dev")
dbutils.widgets.text("discovery_schema", "ds_brand_funnel_ai_lhdev")
dbutils.widgets.text("output_schema", "ml_accelerator")
dbutils.widgets.text("run_date", datetime.now().strftime("%Y-%m-%d"))

# COMMAND ----------

# Get widget values
catalog = dbutils.widgets.get("catalog")
discovery_schema = dbutils.widgets.get("discovery_schema")
output_schema = dbutils.widgets.get("output_schema")
run_date = dbutils.widgets.get("run_date")

print(f"Catalog: {catalog}")
print(f"Discovery Schema: {discovery_schema}")
print(f"Output Schema: {output_schema}")
print(f"Run Date: {run_date}")

# COMMAND ----------

# Read source tables
orders_df = spark.table(f"{catalog}.{discovery_schema}.orders")
customers_df = spark.table(f"{catalog}.{discovery_schema}.customers")
products_df = spark.table(f"{catalog}.{discovery_schema}.products")
dates_df = spark.table(f"{catalog}.{discovery_schema}.dates")

print(f"Orders count: {orders_df.count()}")
print(f"Customers count: {customers_df.count()}")
print(f"Products count: {products_df.count()}")
print(f"Dates count: {dates_df.count()}")

# COMMAND ----------

# Display schema of source tables
print("=== Orders Schema ===")
orders_df.printSchema()
print("\n=== Customers Schema ===")
customers_df.printSchema()
print("\n=== Products Schema ===")
products_df.printSchema()
print("\n=== Dates Schema ===")
dates_df.printSchema()

# COMMAND ----------

# Filter out rows where target (product_id) is null
orders_df_clean = orders_df.filter(F.col("product_id").isNotNull())
print(f"Orders after filtering null product_id: {orders_df_clean.count()}")

# COMMAND ----------

# Join orders with dates to get temporal features
orders_with_dates = orders_df_clean.join(
    dates_df,
    orders_df_clean.date_id == dates_df.date_id,
    "left"
).select(
    orders_df_clean["*"],
    dates_df["year"],
    dates_df["month"],
    dates_df["quarter"]
)

print(f"Orders with dates count: {orders_with_dates.count()}")

# COMMAND ----------

# Join with customers to get customer attributes
orders_with_customers = orders_with_dates.join(
    customers_df,
    orders_with_dates.customer_id == customers_df.customer_id,
    "left"
).select(
    orders_with_dates["*"],
    customers_df["country"],
    customers_df["segment"]
)

print(f"Orders with customers count: {orders_with_customers.count()}")

# COMMAND ----------

# Join with products to get product attributes
orders_full = orders_with_customers.join(
    products_df,
    orders_with_customers.product_id == products_df.product_id,
    "left"
).select(
    orders_with_customers["*"],
    products_df["price"].alias("product_price"),
    products_df["category"].alias("product_category")
)

print(f"Orders full join count: {orders_full.count()}")

# COMMAND ----------

# Create reference date for recency calculations (use run_date)
reference_date = F.lit(run_date).cast("date")

# Convert date_id to date for temporal calculations
orders_full = orders_full.withColumn(
    "order_date",
    F.to_date(F.col("date_id").cast("string"), "yyyyMMdd")
)

# COMMAND ----------

# Calculate days since reference date for each order
orders_full = orders_full.withColumn(
    "days_since_order",
    F.datediff(reference_date, F.col("order_date"))
)

print(f"Orders with temporal features count: {orders_full.count()}")

# COMMAND ----------

# Customer-level aggregations (historical features up to each order)
# Window for customer historical aggregates
customer_window = Window.partitionBy("customer_id").orderBy("order_date").rowsBetween(Window.unboundedPreceding, -1)
customer_window_all = Window.partitionBy("customer_id").orderBy("order_date").rowsBetween(Window.unboundedPreceding, 0)

# COMMAND ----------

# Calculate customer aggregate features
orders_with_features = orders_full.withColumn(
    "customer_total_orders",
    F.count("order_id").over(customer_window)
).withColumn(
    "customer_total_revenue",
    F.sum("revenue").over(customer_window)
).withColumn(
    "customer_total_quantity",
    F.sum("quantity").over(customer_window)
).withColumn(
    "customer_avg_order_value",
    F.avg("revenue").over(customer_window)
).withColumn(
    "customer_avg_quantity",
    F.avg("quantity").over(customer_window)
).withColumn(
    "customer_max_quantity",
    F.max("quantity").over(customer_window)
).withColumn(
    "customer_revenue_std",
    F.stddev("revenue").over(customer_window)
).withColumn(
    "customer_unique_products",
    F.countDistinct("product_id").over(customer_window)
).withColumn(
    "customer_unique_categories",
    F.countDistinct("product_category").over(customer_window)
)

print(f"Orders with customer aggregates count: {orders_with_features.count()}")

# COMMAND ----------

# Calculate days since first and last order per customer
customer_first_last = orders_full.groupBy("customer_id").agg(
    F.min("order_date").alias("first_order_date"),
    F.max("order_date").alias("last_order_date")
)

orders_with_features = orders_with_features.join(
    customer_first_last,
    "customer_id",
    "left"
).withColumn(
    "days_since_first_order",
    F.datediff(F.col("order_date"), F.col("first_order_date"))
).withColumn(
    "days_since_last_order",
    F.datediff(reference_date, F.col("last_order_date"))
)

print(f"Orders with recency features count: {orders_with_features.count()}")

# COMMAND ----------

# Calculate order frequency (average days between orders)
orders_with_features = orders_with_features.withColumn(
    "order_frequency_days",
    F.when(
        F.col("customer_total_orders") > 1,
        F.col("days_since_first_order") / F.col("customer_total_orders")
    ).otherwise(0)
)

# COMMAND ----------

# Product-level aggregations
product_aggregates = orders_full.groupBy("product_id").agg(
    F.sum("revenue").alias("product_total_sales"),
    F.count("order_id").alias("product_purchase_count"),
    F.avg("quantity").alias("product_avg_quantity_per_order")
)

orders_with_features = orders_with_features.join(
    product_aggregates,
    "product_id",
    "left"
)

print(f"Orders with product aggregates count: {orders_with_features.count()}")

# COMMAND ----------

# Category-level aggregations for ranking
category_aggregates = orders_full.groupBy("product_category").agg(
    F.sum("revenue").alias("category_total_revenue"),
    F.count("order_id").alias("category_purchase_count")
)

# Add category rank
category_window = Window.orderBy(F.desc("category_total_revenue"))
category_aggregates = category_aggregates.withColumn(
    "category_popularity_rank",
    F.row_number().over(category_window)
)

orders_with_features = orders_with_features.join(
    category_aggregates.select("product_category", "category_popularity_rank"),
    orders_with_features.product_category == category_aggregates.product_category,
    "left"
).drop(category_aggregates.product_category)

print(f"Orders with category features count: {orders_with_features.count()}")

# COMMAND ----------

# Calculate price percentile within category
price_window = Window.partitionBy("product_category").orderBy("product_price")
orders_with_features = orders_with_features.withColumn(
    "price_percentile_in_category",
    F.percent_rank().over(price_window)
)

# COMMAND ----------

# Customer-product affinity features
# Check if customer purchased this category before
customer_category_window = Window.partitionBy("customer_id", "product_category").orderBy("order_date").rowsBetween(Window.unboundedPreceding, -1)

orders_with_features = orders_with_features.withColumn(
    "customer_purchased_this_category_before",
    F.when(F.count("order_id").over(customer_category_window) > 0, 1).otherwise(0)
).withColumn(
    "customer_category_purchase_count",
    F.count("order_id").over(customer_category_window)
)

# COMMAND ----------

# Calculate if customer purchased similar price range before
orders_with_features = orders_with_features.withColumn(
    "customer_avg_price_purchased",
    F.avg("product_price").over(customer_window)
).withColumn(
    "customer_purchased_similar_price_range",
    F.when(
        F.abs(F.col("product_price") - F.col("customer_avg_price_purchased")) / F.col("customer_avg_price_purchased") < 0.3,
        1
    ).otherwise(0)
)

# COMMAND ----------

# Hash encode customer_id to fixed dimension (100 buckets)
orders_with_features = orders_with_features.withColumn(
    "customer_id_hash",
    F.abs(F.hash(F.col("customer_id"))) % 100
)

print(f"Orders with hash encoded customer_id count: {orders_with_features.count()}")

# COMMAND ----------

# Fill null values in aggregate features with 0
aggregate_cols = [
    "customer_total_orders",
    "customer_total_revenue",
    "customer_total_quantity",
    "customer_avg_order_value",
    "customer_avg_quantity",
    "customer_max_quantity",
    "customer_revenue_std",
    "customer_unique_products",
    "customer_unique_categories",
    "order_frequency_days",
    "product_total_sales",
    "product_purchase_count",
    "product_avg_quantity_per_order",
    "category_popularity_rank",
    "customer_category_purchase_count",
    "customer_purchased_this_category_before",
    "customer_purchased_similar_price_range"
]

for col in aggregate_cols:
    orders_with_features = orders_with_features.withColumn(
        col,
        F.coalesce(F.col(col), F.lit(0))
    )

# COMMAND ----------

# One-hot encode categorical features: country, segment, month, quarter, product_category
# First, create temporary view for easier manipulation
orders_with_features.createOrReplaceTempView("orders_temp")

# COMMAND ----------

# Get unique values for one-hot encoding
countries = orders_with_features.select("country").distinct().rdd.flatMap(lambda x: x).collect()
segments = orders_with_features.select("segment").distinct().rdd.flatMap(lambda x: x).collect()
categories = orders_with_features.select("product_category").distinct().rdd.flatMap(lambda x: x).collect()

print(f"Unique countries: {len(countries)}")
print(f"Unique segments: {len(segments)}")
print(f"Unique categories: {len(categories)}")

# COMMAND ----------

# One-hot encode country
for country in countries:
    if country is not None:
        safe_country = country.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        orders_with_features = orders_with_features.withColumn(
            f"country_{safe_country}",
            F.when(F.col("country") == country, 1).otherwise(0)
        )

# COMMAND ----------

# One-hot encode segment
for segment in segments:
    if segment is not None:
        safe_segment = segment.replace(" ", "_").replace("-", "_")
        orders_with_features = orders_with_features.withColumn(
            f"segment_{safe_segment}",
            F.when(F.col("segment") == segment, 1).otherwise(0)
        )

# COMMAND ----------

# One-hot encode product_category
for category in categories:
    if category is not None:
        safe_category = category.replace(" ", "_").replace("-", "_").replace("&", "and")
        orders_with_features = orders_with_features.withColumn(
            f"category_{safe_category}",
            F.when(F.col("product_category") == category, 1).otherwise(0)
        )

# COMMAND ----------

# One-hot encode month (1-12)
for month in range(1, 13):
    orders_with_features = orders_with_features.withColumn(
        f"month_{month}",
        F.when(F.col("month") == month, 1).otherwise(0)
    )

# COMMAND ----------

# One-hot encode quarter (1-4)
for quarter in range(1, 5):
    orders_with_features = orders_with_features.withColumn(
        f"quarter_{quarter}",
        F.when(F.col("quarter") == quarter, 1).otherwise(0)
    )

# COMMAND ----------

# Label encode target: product_id
indexer = StringIndexer(inputCol="product_id", outputCol="product_id_label", handleInvalid="keep")
indexer_model = indexer.fit(orders_with_features)
orders_with_features = indexer_model.transform(orders_with_features)

print(f"Orders with label encoded target count: {orders_with_features.count()}")
print(f"Number of unique product labels: {orders_with_features.select('product_id_label').distinct().count()}")

# COMMAND ----------

# Add run_date partition column
orders_with_features = orders_with_features.withColumn(
    "run_date",
    F.lit(run_date)
)

# COMMAND ----------

# Select final feature columns
# Drop columns per feature plan: order_id, customer_id (keep hash), product_id (keep label), date_id, customer_name, product_name
# Keep numeric features: quantity, revenue, year, product_price, all aggregates
# Keep one-hot encoded features
# Keep target: product_id_label

feature_columns = [
    # Target
    "product_id_label",
    # Numeric features from orders
    "quantity",
    "revenue",
    # Temporal features
    "year",
    "days_since_order",
    "days_since_first_order",
    "days_since_last_order",
    # Customer hash
    "customer_id_hash",
    # Product features
    "product_price",
    "price_percentile_in_category",
    # Customer aggregates
    "customer_total_orders",
    "customer_total_revenue",
    "customer_total_quantity",
    "customer_avg_order_value",
    "customer_avg_quantity",
    "customer_max_quantity",
    "customer_revenue_std",
    "customer_unique_products",
    "customer_unique_categories",
    "order_frequency_days",
    # Product aggregates
    "product_total_sales",
    "product_purchase_count",
    "product_avg_quantity_per_order",
    "category_popularity_rank",
    # Customer-product affinity
    "customer_purchased_this_category_before",
    "customer_category_purchase_count",
    "customer_purchased_similar_price_range",
    # Partition column
    "run_date"
]

# COMMAND ----------

# Add all one-hot encoded columns to feature list
one_hot_cols = [col for col in orders_with_features.columns if 
                col.startswith("country_") or 
                col.startswith("segment_") or 
                col.startswith("category_") or 
                col.startswith("month_") or 
                col.startswith("quarter_")]

feature_columns.extend(one_hot_cols)

print(f"Total feature columns: {len(feature_columns)}")
print(f"One-hot encoded columns: {len(one_hot_cols)}")

# COMMAND ----------

# Select final features
final_features = orders_with_features.select(feature_columns)

print(f"Final feature table count: {final_features.count()}")
print(f"Final feature table columns: {len(final_features.columns)}")

# COMMAND ----------

# Display sample of final features
display(final_features.limit(10))

# COMMAND ----------

# Check for null values in target
null_target_count = final_features.filter(F.col("product_id_label").isNull()).count()
print(f"Rows with null target: {null_target_count}")

# Filter out null targets if any
final_features = final_features.filter(F.col("product_id_label").isNotNull())
print(f"Final feature table after removing null targets: {final_features.count()}")

# COMMAND ----------

# Write feature table to Unity Catalog
output_table = f"{catalog}.{output_schema}.next_product_purchase_recommendation_features"

print(f"Writing feature table to: {output_table}")

final_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy("run_date").saveAsTable(output_table)

print(f"Feature table written successfully")

# COMMAND ----------

# Verify written table
written_df = spark.table(output_table)
print(f"Written table count: {written_df.count()}")
print(f"Written table columns: {len(written_df.columns)}")

# COMMAND ----------

# Grant SELECT permission to account users
spark.sql(f"GRANT SELECT ON TABLE {output_table} TO `account users`")
print(f"Granted SELECT permission on {output_table} to account users")

# COMMAND ----------

# Display feature table statistics
print("=== Feature Table Statistics ===")
print(f"Total rows: {written_df.count()}")
print(f"Total columns: {len(written_df.columns)}")
print(f"Unique products (labels): {written_df.select('product_id_label').distinct().count()}")
print(f"Unique customers (hash): {written_df.select('customer_id_hash').distinct().count()}")

# COMMAND ----------

# Display target distribution
print("=== Target Distribution (Top 20 Products) ===")
display(written_df.groupBy("product_id_label").count().orderBy(F.desc("count")).limit(20))

# COMMAND ----------

# Display feature summary statistics
print("=== Numeric Feature Summary ===")
numeric_features = [
    "quantity",
    "revenue",
    "year",
    "customer_total_orders",
    "customer_total_revenue",
    "customer_avg_order_value",
    "product_price",
    "product_total_sales"
]

display(written_df.select(numeric_features).summary())

# COMMAND ----------

# Final success message
print("=" * 80)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"Feature Table: {output_table}")
print(f"Total Features: {len(written_df.columns) - 2}")  # Exclude target and run_date
print(f"Total Observations: {written_df.count()}")
print(f"Run Date: {run_date}")
print("=" * 80)