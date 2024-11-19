from pyspark.sql import SparkSession
import os
import logging as logger
from pyspark.sql.types import *
from pyspark.sql.functions import *
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