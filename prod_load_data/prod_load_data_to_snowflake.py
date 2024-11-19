from pyspark.sql import SparkSession
import os
import logging as logger
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import StorageLevel
from datetime import datetime

AWS_BUCKET = "prod-datalake-tanf"
AWS_INPUT_SILVER_DATA = "silver/processed/online_sales_dataset"
AWS_INPUT_GOLD_DATA = "gold/online_sales_dataset"


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


def read_data(spark, prefix):
    path = f"s3a://{AWS_BUCKET}/{prefix}"
    logger.info(f"Reading data from path {path}")
    salesDF = spark.read.parquet(path)
    salesDF = salesDF.repartition(8)
    logger.info(f"Schema of raw sales df is {str(salesDF.schema)}")
    return salesDF


def get_duplicated_invoice_id(salesDF, data_date):
    """
    Identify duplicated `pkid_invoice` in the DataFrame.
    """
    groupped_invoice = salesDF.groupBy("pkid_invoice").agg(count("*").alias("nb_invoice"))
    duplicates = salesDF.where(col("nb_invoice") > 1).select(col("pkid_invoice_unvalid"))
    prefix = f"{AWS_OUTPUT_ERRORS}/{data_date}/duplicates_invoice"
    write_data(duplicates_invoice, prefix)
    return duplicates

def write_data_to_snowflake(salesDF, database, schema, table_name):
    # Snowflake connection parameters
    sfparams = {
        "sfURL": os.getenv("snowflake_url"),
        "sfUser": os.getenv("snowflake_user"),
        "sfPassword": os.getenv("snowflake_password"),
        "sfDatabase": database,
        "sfSchema": schema,
        "sfRole": "ACCOUNTADMIN",
        "sfWarehouse": "COMPUTE_WH",
        "truncate_table": "on",
        "usestagingtable": "off"
    }
    logger.info(f"Writing salesDF to table {table_name}")
    salesDF.write.format("net.snowflake.spark.snowflake").options(**sfparams).option("dbtable", table_name).mode("overwrite").save()

def main():
    spark = init_spark()
    silver_salesDF = read_data(spark)
    gold_salesDF = read_data(spark)
    write_data_to_snowflake(silver_salesDF, "sales", "sales", "invoices")
    write_data_to_snowflake(gold_salesDF, "sales", "insights", "invoices")


if __name__ == "__main__":
    main()