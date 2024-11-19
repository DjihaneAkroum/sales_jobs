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
        .config("spark.jars.packages",
                "net.snowflake:snowflake-jdbc-3.13.11,net.snowflake:spark-snowflake_2.12-2.9.0-spark_3.5") \
        .config("spark.driver.extraClassPath","snowflake-jdbc-3.13.11.jar:spark-snowflake_2.12-2.12.0-spark_3.5.jar") \
        .config("spark.executor.extraClassPath","snowflake-jdbc-3.13.11.jar:spark-snowflake_2.12-2.12.0-spark_3.5.jar")\
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
        "sfRole": "TEST",
        "sfWarehouse": "BI_ANALYTICS",
        "truncate_table": "on",
        "usestagingtable": "off"
    }
    logger.info(f"Writing salesDF to table {table_name}")
    salesDF.write.format("net.snowflake.spark.snowflake").options(**sfparams).option("dbtable", table_name).mode("overwrite").save()

def main():
    spark = init_spark()

    silver_salesDF = read_data(spark, f"{AWS_INPUT_SILVER_DATA}")
    write_data_to_snowflake(silver_salesDF, "sales", "sales", "invoices")

    total_sales_by_category = read_data(spark, f"{AWS_INPUT_GOLD_DATA}/total_sales_by_category")
    write_data_to_snowflake(total_sales_by_category, "sales", "insights", "total_sales_by_category")

    avg_order_by_customer = read_data(spark, f"{AWS_INPUT_GOLD_DATA}/avg_order_by_customer")
    write_data_to_snowflake(avg_order_by_customer, "sales", "insights", "avg_order_by_customer")

    sales_by_month = read_data(spark, f"{AWS_INPUT_GOLD_DATA}/sales_by_month")
    write_data_to_snowflake(sales_by_month, "sales", "insights", "sales_by_month")

    sales_by_product = read_data(spark, f"{AWS_INPUT_GOLD_DATA}/sales_by_product")
    write_data_to_snowflake(sales_by_product, "sales", "insights", "sales_by_product")

    returns_by_product = read_data(spark, f"{AWS_INPUT_GOLD_DATA}/returns_by_product")
    write_data_to_snowflake(returns_by_product, "sales", "insights", "returns_by_product")


if __name__ == "__main__":
    main()