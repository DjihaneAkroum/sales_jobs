from pyspark.sql import SparkSession
import os
import logging as logger
from pyspark.sql.types import *
from pyspark.sql.functions import *

AWS_BUCKET = "prod-datalake-tanf"
AWS_RAW_DATA_PATH = "bronze/online_sales_dataset.csv"
AWS_OUTPUT_PATH = "silver/preprocessed/online_sales_dataset"


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
    path = f"s3a://{AWS_BUCKET}/{AWS_RAW_DATA_PATH}"
    logger.info(f"Reading data from path {path}")
    salesDF = spark.read.option("header", "true").csv(path)
    salesDF = salesDF.repartition(8)
    logger.info(f"Schema of raw sales df is {str(salesDF.schema)}")

    return salesDF


def preprocess_data(salesDF):
    preprocessedData = salesDF.select(
        col("InvoiceNo").alias("pkid_invoice"),
        col("StockCode").alias("product_sku"),
        col("Description").alias("product_description"),
        col("Category").alias("category"),
        col("Quantity").cast(IntegerType()).alias("quantity"),
        col("InvoiceDate").cast(TimestampType()).alias("date_of_invoice"),
        col("UnitPrice").cast(FloatType()).alias("unity_price"),
        col("CustomerID").alias("fkid_customer"),
        col("Country").alias("country"),
        col("Discount").cast(FloatType()).alias("percentage_discount"),
        col("PaymentMethod").alias("payment_method"),
        col("ShippingCost").cast(FloatType()).alias("shipping_cost"),
        col("SalesChannel").alias("slaes_channel"),
        when(col("ReturnStatus") == lit("Returned"), lit(True)) \
            .when(col("ReturnStatus") == lit("Not Returned"), lit(False)) \
            .otherwise(None).alias("is_returned"),
        col("ShipmentProvider").alias("shipment_provider"),
        col("WarehouseLocation").alias("warehouse_location"),
        when(col("OrderPriority") == lit("Low"), lit(1)) \
            .when(col("OrderPriority") == lit("Medium"), lit(2)) \
            .when(col("OrderPriority") == lit("High"), lit(3)) \
            .otherwise(lit(-1)).alias("order_priority"),
        ((col("UnitPrice") * col("Quantity"))*col("Discount") + col("ShippingCost")).alias("total_cost_invoice")
    )

    return preprocessedData


def write_data(salesDF):
    path = f"s3a://{AWS_BUCKET}/{AWS_OUTPUT_PATH}"
    logger.info(f"Writing sales dataframe data to {path}")
    salesDF.write.mode("overwrite").parquet(path)


def main():
    spark = init_spark()
    salesDF = read_data(spark)
    preprocessedSalesDF = preprocess_data(salesDF)
    write_data(preprocessedSalesDF)


if __name__ == "__main__":
    main()