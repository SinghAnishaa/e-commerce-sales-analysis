# Databricks notebook source
# COMMAND ------
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import gc



# COMMAND ------
# Initialization of Spark Session
spark = SparkSession.builder.appName("PySpark Analysis").getOrCreate()



# COMMAND ------
## AWS Data Setup and Reading
# Configure AWS credentials and Hadoop filesystem to enable PySpark to connect to an S3 bucket. Read data into a PySpark DataFrame.
# COMMAND ------
# Setting up AWS credentials
access_key = dbutils.secrets.get(scope="aws", key="access-key")
secret_key = dbutils.secrets.get(scope="aws", key="secret-key")
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
# Reading data from S3
data = spark.read.option("header", True).csv("s3://databricks-workspace-stack-db4f9-bucket/2019-Oct.csv")
display(data)




# COMMAND ------
# Displaying summary statistics
# Present descriptive statistics of the dataset to provide a quick overview of its key numerical attributes. Useful for understanding the distribution and scale of key variables.
# COMMAND ------
data.describe().show()




# COMMAND ------
# Prepare the data for analysis by handling missing values in the 'category_code' column. The choice of 'unknown' as a placeholder helps manage missing data.
# COMMAND ------
print("Before Cleaning:")
data.select([F.count(F.when(col(c).isNotNull(), c)).alias(c) for c in data.columns]).show()
data = data.na.fill('unknown', subset=['category_code'])
print("\nAfter Cleaning:")
data.select([F.count(F.when(col(c).isNotNull(), c)).alias(c) for c in data.columns]).show()




# COMMAND ------
# Product Intelligence
# Highest Grossing Product
# To start off our analysis, let's find out what are the top 10 highest-grossing products from our store, that is, the products that make the most amount of money in terms of revenue...
# COMMAND ------
# Calculate product performance
product_performance = data.filter(data['event_type'] == 'purchase') \
    .groupBy('product_id') \
    .agg(F.sum('price').alias('revenue')) \
    .orderBy('revenue', ascending=False)
# Display top 10 products by revenue
product_performance.limit(10).toPandas().plot(kind='bar', x='product_id', y='revenue', color='salmon')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Product ID')
plt.ylabel('Total Revenue ($)')
plt.show()




# COMMAND ------
# Highest Earnings Per Session
# Now, total revenue is not a meaningful metric on its own, especially if we ignore the number of sessions that landed on the product page. After all, if you are sending a lot of traffic to a page that is doing a poor job at converting users, you could easily be leaving money on the table. So, to account for the fact that different products have different conversion rates, let's find out what are the top 10 products by earnings per session.
# COMMAND ------
# 1) Calculate the total number of views per product_id
num_of_views = data.filter(col('event_type') == 'view') \
    .groupBy('product_id') \
    .agg(F.count('event_type').alias('total_views'))
# 2) Calculate how much each product made in revenue
revenue_per_product = data.filter(col('event_type') == 'purchase') \
    .groupBy('product_id') \
    .agg(F.sum('price').alias('total_revenue'))
# 3) Merge both dataframes and calculate earnings per session
revenue_and_views = revenue_per_product.join(num_of_views, 'product_id', 'left_outer')
revenue_and_views = revenue_and_views.fillna(0)  # Fill null values with 0

revenue_and_views = revenue_and_views.withColumn('earnings_per_session', col('total_revenue') / col('total_views'))
# Display top 10 products by earnings per session
top_10_earnings_per_session = revenue_and_views \
    .orderBy('earnings_per_session', ascending=False) \
    .limit(10).toPandas()
# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='product_id', y='earnings_per_session', data=top_10_earnings_per_session, color='salmon')
plt.title('Top 10 Products by Earnings Per Session')
plt.xlabel('Product ID')
plt.ylabel('Average Earnings Per Session ($)')
plt.show()





# COMMAND ------
# Products That Sell Well Together
# Upsells, downsells, and cross-sells are one of the most reliable ways to increase the average cart value of the customer. But in order to optimize them, the products offered cannot be picked at random. After all, each and every product caters to different customer avatars, and trying to use a one size fits all approach on the upsell path is a guaranteed way to leave money on the table. So let's dig into the data and find out what products users tend to buy in the same session.
# COMMAND ------
# Calculate the total number of purchases per product pair
product_pairs = data.filter(data['event_type'] == 'purchase') \
    .withColumn('prev_product_id', F.lag('product_id').over(Window.orderBy('event_time'))) \
    .groupBy('prev_product_id', 'product_id') \
    .agg(F.count('event_type').alias('count'))
# Display top product pairs that sell well together
top_product_pairs = product_pairs.orderBy('count', ascending=False).limit(10)
# Convert the PySpark DataFrame to Pandas for visualization
top_product_pairs_pd = top_product_pairs.toPandas()
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(top_product_pairs_pd['prev_product_id'] + ' & ' + top_product_pairs_pd['product_id'], top_product_pairs_pd['count'], color='skyblue')
plt.title('Top Product Pairs That Sell Well Together')
plt.xlabel('Product Pairs')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=45, ha='right')
plt.show()



# COMMAND ------
# Average Order Value
# Now, speaking of upsells and cross-sells, we can also use the data to find out what is the average order value. So let's find out:
# COMMAND ------
# Filter only purchase events
purchase_data = data.filter(col('event_type') == 'purchase')
# Group by user_id and calculate the total order value
total_order_value = purchase_data.groupBy('user_id').agg(F.sum('price').alias('total_order_value'))
# Calculate the average order value
average_order_value = total_order_value.select(F.avg('total_order_value').alias('average_order_value'))
# Display the average order value
average_order_value.show()



# COMMAND ------
# Number of purchases per customer
#Unfortunately, looking at the Average Order Value does not tell the whole story, since this metric doesn't account for the fact that some customers make more than one purchase and keep coming back. So let's take a look at two important metrics of customer retention: The average number of purchases per customer and the average Customer Lifetime Value.
# COMMAND ------
# Filter only purchase events
purchase_data = data.filter(col('event_type') == 'purchase')
# Group by user_id and count the number of purchases
purchases_per_customer = purchase_data.groupBy('user_id').agg(F.count('event_type').alias('num_purchases'))
# Calculate the average number of purchases per customer
average_purchases_per_customer = purchases_per_customer.selectExpr('avg(num_purchases) as avg_purchases_per_customer')
# Display the result
average_purchases_per_customer.show()




# COMMAND ------
# User Engagement Analysis
plt.figure(figsize=(8, 5))
# Count events by event type
event_type_counts = data.groupBy("event_type").count().toPandas()
# Plot bar chart
sns.barplot(x="event_type", y="count", data=event_type_counts, color='skyblue')
plt.title('User Engagement Analysis')
plt.xlabel('Event Type')
plt.ylabel('Count')
plt.show()



# COMMAND ------
# Brand Performance (Top 10 Brand)
# Calculate brand performance
brand_performance = data.filter(data['event_type'] == 'purchase') \
    .groupBy('brand') \
    .agg(F.sum('price').alias('revenue')) \
    .orderBy('revenue', ascending=False)
# Display top 10 brands by revenue
brand_performance.limit(10).toPandas().plot(kind='bar', x='brand', y='revenue', color='lightgreen')
plt.title('Top 10 Brands by Revenue')
plt.xlabel('Brand')
plt.ylabel('Revenue')
plt.show()



# COMMAND ------
# Pricing Strategy - Limit to Top 10 Categories
# Top 10 product categories with the highest average prices
# COMMAND ------
# Calculate pricing strategy
price_analysis = data.groupBy('category_code').agg(F.mean('price').alias('mean_price'),
                                                   F.min('price').alias('min_price'),
                                                   F.max('price').alias('max_price'))
# Select the top 10 categories based on mean_price
top_10_categories = price_analysis.orderBy('mean_price', ascending=False).limit(10)
# Plot bar chart for the top 10 categories
top_10_categories.toPandas().plot(kind='bar', x='category_code', colormap='viridis', alpha=0.7)
plt.title('Pricing Strategy - Top 10 Categories')
plt.xlabel('Category Code')
plt.ylabel('Price')
plt.show()



# COMMAND ------
# Time-based Analysis
# Analyze daily event counts over time. The chart reveals patterns and trends in daily event occurrences.
# COMMAND ------
# Resample data to daily frequency
time_analysis = data.groupBy(F.date_trunc('day', 'event_time').alias('date')).count().orderBy('date')
# Plot line chart
time_analysis.toPandas().plot(x='date', y='count', kind='line', marker='o', color='orange')
plt.title('Time-based Analysis')
plt.xlabel('Date')
plt.ylabel('Event Count')
plt.show()




# COMMAND ------
# Conversion Rate Analysis
# Calculate and visualize unique user counts for each event type. The chart displays the unique user counts for different event types, providing insights into conversion rates.
# COMMAND ------
# Calculate conversion rates
conversion_rates = data.groupBy('event_type').agg(F.countDistinct('user_id').alias('unique_users'))
# Plot bar chart
conversion_rates.toPandas().plot(kind='bar', x='event_type', y='unique_users', color='lightcoral')
plt.title('Conversion Rate Analysis')
plt.xlabel('Event Type')
plt.ylabel('Unique Users')
plt.show()




# COMMAND ------
# Plot the category code distribution
# Visualize the distribution of events by category code, focusing on the top 10 categories. The chart provides a clear view of the event distribution among the top 10 category codes.
# COMMAND ------
# Count events by category code
category_code_counts = data.groupBy("category_code").count()
# Get the top 10 categories
top_10_categories = category_code_counts.orderBy(F.desc("count")).limit(10)
# Filter out rows with missing or None values in the "category_code" column
top_10_categories_filtered = top_10_categories.filter(top_10_categories["category_code"].isNotNull())
# Plot the top 10 category code distribution with rotated x-axis labels
plt.figure(figsize=(10, 6))
plt.bar(top_10_categories_filtered.toPandas()["category_code"], top_10_categories_filtered.toPandas()["count"])
plt.xlabel("Category Code")
plt.ylabel("Count")
plt.title("Top 10 Category Code Distribution")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()




# COMMAND ------
# Plot the category ID distribution
# Visualize the distribution of events by category ID, focusing on the top 10 categories. The chart provides a clear view of the event distribution among the top 10 category IDs.
# COMMAND ------
# Count events by category ID
category_id_counts = data.groupBy("category_id").count()
# Get the top 10 categories
top_10_categories = category_id_counts.orderBy(F.desc("count")).limit(10)
# Filter out rows with missing or None values in the "category_id" column
top_10_categories_filtered = top_10_categories.filter(top_10_categories["category_id"].isNotNull())
# Plot the top 10 category ID distribution with rotated x-axis labels
plt.figure(figsize=(10, 6))
plt.bar(top_10_categories_filtered.toPandas()["category_id"], top_10_categories_filtered.toPandas()["count"])
plt.xlabel("Category ID")
plt.ylabel("Count")
plt.title("Top 10 Category ID Distribution")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()






# COMMAND ------
# Plot the brand distribution
# Visualize the distribution of events by brand, focusing on the top 10 brands. The chart provides insights into the event distribution among the top 10 brands, aiding in understanding brand popularity.
# COMMAND ------
# Count events by brand
brand_counts = data.groupBy("brand").count()
# Get the top 10 brands
top_10_brands = brand_counts.orderBy(F.desc("count")).limit(10)
# Filter out rows with missing or None values in the "brand" column
top_10_brands_filtered = top_10_brands.filter(top_10_brands["brand"].isNotNull())
# Plot the top 10 brand distribution with rotated x-axis labels
plt.figure(figsize=(10, 6))
plt.bar(top_10_brands_filtered.toPandas()["brand"], top_10_brands_filtered.toPandas()["count"])
plt.xlabel("Brand")
plt.ylabel("Count")
plt.title("Top 10 Brand Distribution")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()





# COMMAND ------
# Calculate average purchase price by brand
# Calculate and visualize the average purchase price for the top 10 brands. The chart provides insights into the average purchase prices for the top 10 brands, aiding in understanding pricing trends.
# COMMAND ------
average_purchase_price_by_brand = data.groupBy("brand").agg(F.mean("price").alias("average_purchase_price")).collect()
# Convert the Spark DataFrame to a Pandas DataFrame for easier plotting
average_purchase_price_df = pd.DataFrame(average_purchase_price_by_brand, columns=['brand', 'average_purchase_price'])
# Drop rows with missing values in 'brand' or 'average_purchase_price'
average_purchase_price_df = average_purchase_price_df.dropna()
# Select the top 10 products based on average purchase price
top_10_products = average_purchase_price_df.nlargest(10, 'average_purchase_price')
# Create a bar chart for the top 10 products
plt.figure(figsize=(12, 6))
plt.bar(top_10_products['brand'], top_10_products['average_purchase_price'], color='skyblue')
plt.xlabel('Brand')
plt.ylabel('Average Purchase Price')
plt.title('Average Purchase Price by Brand (Top 10 Products)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()





# COMMAND ------
# Calculate total revenue
# Calculate and visually represent the total revenue generated. The text annotation provides a quick overview of the total revenue, aiding in understanding the overall financial performance.
# COMMAND ------
total_revenue = data.filter(F.col("event_type") == "purchase").agg(F.sum("price").alias("total_revenue")).collect()[0]["total_revenue"]
# Create a bar chart or pie chart as needed
# For a simple representation, let's use a text annotation
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, f'Total Revenue: ${total_revenue:,.2f}', ha='center', va='center', fontsize=18)
plt.axis('off')  # Turn off the axis for a cleaner look
plt.title('Total Revenue', fontsize=20)
plt.show()