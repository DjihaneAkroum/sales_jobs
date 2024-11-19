from pyspark.sql import SparkSession
import os
import logging as logger
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import StorageLevel

AWS_BUCKET = "prod-datalake-tanf"
AWS_INPUT_DATA = "silver/processed/online_sales_dataset"
AWS_OUTPUT_DATA = "gold/online_sales_dataset"


def init_spark():
    logger.info("Initiating Spark session")
    spark = SparkSession.builder \
        .appName("Spark S3 Example") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.eu-north-1.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider') \
        .config('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.2.1') \
        .getOrCreate()

    return spark


def read_data(spark):
    path = f"s3a://{AWS_BUCKET}/{AWS_INPUT_DATA}"
    logger.info(f"Reading data from path {path}")
    salesDF = spark.read.parquet(path)
    salesDF = salesDF.repartition(8)
    logger.info(f"Schema of raw sales df is {str(salesDF.schema)}")

    salesDF.persist(StorageLevel.MEMORY_AND_DISK)
    return salesDF

def get_total_sales_by_category(salesDF):
    # Total sales by category
    total_sales_category = salesDF.groupBy("category").agg(
        sum("total_cost_invoice").alias("total_sales"),
        sum("quantity").alias("total_quantity")
    )
    return total_sales_category

def get_avg_order_by_customer(salesDF):
    avg_order_value_customer = salesDF.groupBy("fkid_customer").agg(
        avg("total_cost_invoice").alias("avg_order_value"),
        count("pkid_invoice").alias("num_orders"),
        sum("total_cost_invoice").alias("total_revenue")
    )
    return avg_order_value_customer

def get_sales_by_month(salesDF):
    # Sales by month/year (extracting month and year from date_of_invoice)
    sales_by_month = salesDF.withColumn("month", month("date_of_invoice")) \
        .withColumn("year", year("date_of_invoice")) \
        .groupBy("year", "month").agg(
        sum("total_cost_invoice").alias("monthly_sales"),
        avg("total_cost_invoice").alias("avg_order_value")
    )
    return sales_by_month

def get_sales_by_product(salesDF):
    sales_by_product = salesDF.groupBy("product_sku").agg(
        sum("quantity").alias("total_quantity_sold"),
        sum("total_cost_invoice").alias("total_sales")
    )
    return sales_by_product

def get_returns_by_product(salesDF):
    return_rate_by_product = salesDF.groupBy("product_sku").agg(
        sum(when(col("is_returned") == True, 1).otherwise(0)).alias("returns"),
        count("pkid_invoice").alias("total_orders")
    ).withColumn("return_rate", col("returns") / col("total_orders"))
    return return_rate_by_product

def write_data(salesDF, prefix):
    path = f"s3a://{AWS_BUCKET}/{prefix}"
    logger.info(f"Writing sales dataframe data to {path}")
    salesDF.write.mode("overwrite").parquet(path)


def main():
    spark = init_spark()
    salesDF = read_data(spark)

    total_sales_by_category = get_total_sales_by_category(salesDF)
    write_data(total_sales_by_category, f"{AWS_OUTPUT_DATA}/total_sales_by_category")

    avg_order_by_customer = get_avg_order_by_customer(salesDF)
    write_data(avg_order_by_customer, f"{AWS_OUTPUT_DATA}/avg_order_by_customer")

    sales_by_month = get_sales_by_month(salesDF)
    write_data(sales_by_month, f"{AWS_OUTPUT_DATA}/sales_by_month")

    sales_by_product = get_sales_by_product(salesDF)
    write_data(sales_by_product, f"{AWS_OUTPUT_DATA}/sales_by_product")

    returns_by_product = get_returns_by_product(salesDF)
    write_data(returns_by_product, f"{AWS_OUTPUT_DATA}/returns_by_product")


if __name__ == "__main__":
    main()