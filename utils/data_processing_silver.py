import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DecimalType

def process_lms(snapshot_date_str, bronze_lms_directory, silver_lms_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# Function to process attribute table
def process_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    ## Replace Occupation ______ with "UnSpecified"
    
    df = df.withColumn(
    "Occupation",
    F.when(F.col("Occupation") == "_______", "Unspecified").otherwise(F.col("Occupation"))
    )

    ## Remove "_" and "-" in Age, cast to int, and get absolute value
    df = df.withColumn(
    "Age",
    F.abs(F.regexp_replace(F.col("Age"), "[_-]", "").cast(IntegerType()))
    )
    # values that survive stripping but are still not a real age (e.g. "-500" -> 500, "3843_" -> 3843)
    # get set to null rather than kept as a wrong number
    df = df.withColumn(
        "Age",
        F.when(F.col("Age").between(14, 100), F.col("Age"))
    )
    
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))



    # save silver table - IRL connect to database to write
    partition_name = "silver_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# Function to process clickstream table
def process_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    ## Replace Occupation ______ with "UnSpecified"
    
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))



    # save silver table - IRL connect to database to write
    partition_name = "silver_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# Function to process silver financials table
def process_financials(snapshot_date_str, bronze_financials_directory, silver_financials_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    ## Replace Occupation ______ with "UnSpecified"

    ## Remove "_" and "-" in Annual_Income, cast to int, and get absolute value
    df = df.withColumn(
    "Annual_Income",
    F.regexp_replace(F.col("Annual_Income"), "[_-]", "").cast(DecimalType(10,2))
    )

    ## Rounding Monthly_Inhand_Salary to 2 digits
    df = df.withColumn(
    "Monthly_Inhand_Salary", F.col("Monthly_Inhand_Salary").cast(DecimalType(10,2))
    )

    ## Re-calculate Num_of_Loan
    #df = df.withColumn(
    # 1. Clean out the oxford / conjunction 'and ' so delimiters become purely commas
    #"cleaned_loans",
    #F.regexp_replace(F.trim(F.col("Type_of_Loan")), r",?\s+and\s+", ", ")
    #).withColumn(
    # 2. Split by comma and count elements, handling nulls/empty strings
    #"Num_of_Loan",
    #F.when(
    #    (F.col("Type_of_Loan").isNull()) | (F.trim(F.col("Type_of_Loan")) == ""), 0
    #).otherwise(
    #    F.size(F.split(F.col("cleaned_loans"), r"\s*,\s*"))
    #)
    #).drop("cleaned_loans")


    ## Remove "_" and "-" in Num_of_Delayed_Payment, cast to int, and get absolute value
    df = df.withColumn(
    "Num_of_Delayed_Payment",
    F.regexp_replace(F.col("Num_of_Delayed_Payment"), "[_-]", "").cast(IntegerType())
    )

    # Fill blank Changed_Credit_Limit by "Not Specified"
    df = df.withColumn(
    "Changed_Credit_Limit",
        F.when(
        F.col("Changed_Credit_Limit").isNull()
        | (F.trim(F.col("Changed_Credit_Limit")) == "")
        | (F.col("Changed_Credit_Limit").rlike("^[_\\s]+$")),
        0
    ).otherwise(F.col("Changed_Credit_Limit")).cast(DecimalType(10,2))
    )

    ## Remove "_" and "-" in Outstanding_Debt, cast to int, and get absolute value
    df = df.withColumn(
    "Outstanding_Debt",
    F.regexp_replace(F.col("Outstanding_Debt"), "_", "").cast(DecimalType(10,2))
    )

    ## Remove "_" and "-" in Credit_Utilization_Ratio, cast to int, and get absolute value
    df = df.withColumn(
    "Credit_Utilization_Ratio",
    F.regexp_replace(F.col("Credit_Utilization_Ratio"), "_", "").cast(DecimalType(10,2))
    )

    # Convert Credit History Age to months

    df = df.withColumn(
        "Credit_History_Age_Months",
        (
            F.coalesce(F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s*Years?", 1).cast("int"), F.lit(0)) * 12
            + F.coalesce(F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s*Months?", 1).cast("int"), F.lit(0))
        )
    )

    # replace min payment from "NM" to "No"
    df = df.withColumn(
    "Payment_of_Min_Amount",
    F.regexp_replace(F.col("Payment_of_Min_Amount"), "NM", "No") 
    )

    # round total EMI 
    df = df.withColumn(
    "Total_EMI_per_month",
    F.col("Total_EMI_per_month").cast(DecimalType(10,2))
    )

    #Amount invest monthly
    df = df.withColumn(
    "Amount_invested_monthly",
    F.regexp_replace(F.col("Amount_invested_monthly"), "_", "").cast(DecimalType(10,2))
    )

    # round monthly balance
    df = df.withColumn(
    "Monthly_Balance",
    F.col("Monthly_Balance").cast(DecimalType(10,2))
    )
    


    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": DecimalType(10,2),
        "Monthly_Inhand_Salary": DecimalType(10,2),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": DecimalType(10,2),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DecimalType(10,2),
        "Credit_Utilization_Ratio": DecimalType(10,2),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": DecimalType(10,2),
        "Amount_invested_monthly": DecimalType(10,2),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DecimalType(10,2),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))



    # save silver table - IRL connect to database to write
    partition_name = "silver_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df