from pyspark.sql import SparkSession
import os
import logging as logger
from pyspark.sql.types import *
from pyspark.sql.functions import col, count
from pyspark import StorageLevel
from datetime import datetime

AWS_BUCKET = "prod-datalake-tanf"
AWS_INPUT_DATA = "silver/preprocessed/online_sales_dataset"
AWS_OUTPUT_DATA = "silver/processed/online_sales_dataset"
AWS_OUTPUT_ERRORS = "wrong_data/online_sales_dataset"


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


def get_duplicated_invoice_id(salesDF, data_date):
    """
    Identify duplicated `pkid_invoice` in the DataFrame.
    """
    groupped_invoice = salesDF.groupBy("pkid_invoice").agg(count("*").alias("nb_invoice"))
    duplicates = groupped_invoice.where(col("nb_invoice") > 1).select(col("pkid_invoice").alias("pkid_invoice_unvalid"))
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/duplicates_invoice"
    write_data(duplicates, prefix)
    return duplicates


# Check for negative values in relevant columns
def get_negative_values(salesDF, data_date):
    """
    Identify rows with negative values in `quantity`, `unity_price`, `percentage_discount`, and `shipping_cost`.
    """
    negative_rows = salesDF.filter(
        (col("quantity") < 0) |
        (col("unity_price") < 0) |
        (col("percentage_discount") < 0) |
        (col("shipping_cost") < 0)
    )
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/negative_values"
    write_data(negative_rows, prefix)
    return negative_rows.select(col("pkid_invoice").alias("pkid_invoice_unvalid"))


# Check for incorrect discount percentages
def get_wrong_discount(salesDF, data_date):
    """
    Identify rows with incorrect `percentage_discount` (should be between 0 and 1).
    """
    invalid_discount = salesDF.filter(
        (col("percentage_discount") < 0) | (col("percentage_discount") > 1)
    )
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/wrong_discount"
    write_data(invalid_discount, prefix)
    return invalid_discount.select(col("pkid_invoice").alias("pkid_invoice_unvalid"))


def get_null_invoice_id(salesDF, data_date):
    null_invoice = salesDF.where(col("pkid_invoice").isNull())
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/null_invoice"
    write_data(null_invoice, prefix)


# Check for null or missing values in relevant columns
def get_nulls(salesDF, data_date):
    """
    Identify rows with null values in mandatory columns.
    """
    null_rows = salesDF.where(
        col("product_sku").isNull() |
        col("quantity").isNull() |
        col("date_of_invoice").isNull() |
        col("unity_price").isNull() |
        col("fkid_customer").isNull() |
        col("percentage_discount").isNull() |
        col("is_returned").isNull() |
        col("shipment_provider").isNull()
    )
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/null_rows"
    write_data(null_rows, prefix)
    return null_rows


def validate_logical_consistency(salesDF, data_date):
    inconsistent_rows = salesDF.filter(
        (col("is_returned") == True) & (col("quantity") >= 0)
    )
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/logical_constistency"
    write_data(inconsistent_rows, prefix)
    return inconsistent_rows.select(col("pkid_invoice").alias("pkid_invoice_unvalid"))


def validate_dates(salesDF, data_date):
    invalid_dates = salesDF.filter(col("date_of_invoice") > datetime.now())
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/unvalid_dates"
    write_data(invalid_dates, prefix)
    return invalid_dates.select(col("pkid_invoice").alias("pkid_invoice_unvalid"))


def validate_data(salesDF, data_date):
    get_null_invoice_id(salesDF, data_date)
    get_nulls(salesDF, data_date)
    duplicated_invoice = get_duplicated_invoice_id(salesDF, data_date)
    wrong_discount = get_wrong_discount(salesDF, data_date)
    negative_values = get_negative_values(salesDF, data_date)
    unvalid_dates = validate_dates(salesDF, data_date)
    unvalid_logic = validate_logical_consistency(salesDF, data_date)
    unified_invoice_ids = duplicated_invoice \
        .union(wrong_discount) \
        .union(negative_values) \
        .union(unvalid_dates) \
        .union(unvalid_logic)

    unified_invoice_ids = unified_invoice_ids.dropDuplicates()
    return unified_invoice_ids


def filter_out_non_valid_data(salesDf, unvalid_data):
    valid_data = salesDf \
        .where(col("pkid_invoice").isNotNull()) \
        .join(unvalid_data, salesDf.pkid_invoice == unvalid_data.pkid_invoice_unvalid, "left_anti")
    return valid_data


def write_data(salesDF, prefix):
    path = f"s3a://{AWS_BUCKET}/{prefix}"
    logger.info(f"Writing sales dataframe data to {path}")
    salesDF.write.mode("overwrite").parquet(path)


def main():
    spark = init_spark()
    salesDF = read_data(spark)
    data_date = datetime.now().strftime('%y-%m-%d')
    unified_invoice_ids = validate_data(salesDF, data_date)
    valid_data = filter_out_non_valid_data(salesDF, unified_invoice_ids)
    write_data(valid_data, AWS_OUTPUT_DATA)


if __name__ == "__main__":
    main()