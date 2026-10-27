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

def process_gold_table(date_str, silver_lms_daily_directory, silver_attributes_daily_directory, silver_financials_daily_directory,
                       silver_clickstream_daily_directory,
                       gold_feature_store_directory, spark, dpd, mob):

    # prepare arguments
    #date_str = datetime.strptime(date_str, "%Y-%m-%d")

    # connect to silver lms table
    partition_name = "silver_loan_daily_" + date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_daily_directory + partition_name
    lms = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', lms.count())

    
    # read every partition of each silver table (wildcard covers all snapshot_date files)
    #lms = spark.read.parquet(silver_lms_daily_directory + "*")
    attr = spark.read.parquet(silver_attributes_daily_directory + "*")
    fin = spark.read.parquet(silver_financials_daily_directory + "*")

    # read clickstream table
    partition_name = "silver_daily_" + date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_daily_directory + partition_name
    click = spark.read.parquet(filepath)
    print('loaded lms:', lms.count(), 'attributes:', attr.count(), 'financials:', fin.count(), 'clickstream:', click.count())

    # PII: Name and SSN live in silver attributes but must never reach a feature store
    attr = attr.drop("Name", "SSN")

    # attributes/financials each carry their OWN snapshot_date (the customer's origination date) -
    # a different thing from lms's snapshot_date (this specific installment's date). Since we're
    # deliberately joining on Customer_ID only (not date), drop these first: left unmerged, both
    # would collide with lms's snapshot_date and break the later clickstream join and the write.
    attr = attr.drop("snapshot_date")
    fin = fin.drop("snapshot_date")

    # attributes/financials: one row per customer -> safe many-to-one join on Customer_ID alone,
    # no snapshot_date needed (every lms installment for a customer gets the same, static values)
    lms = lms.join(attr, on="Customer_ID", how="inner")
    lms = lms.join(fin, on="Customer_ID", how="inner")

    # ---- feature engineering: project-specific, built for the loan default model ----
    # (same logic as gold_ft_store_customer - kept here too since this table has its own,
    # separate grain; duplicated rather than imported so this script has no dependency on
    # the other gold script's internals)
    LOAN_TYPES = [
        "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
        "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
    ]
    # coalesce null Type_of_Loan to "" first: F.size() on a null array returns -1 (Spark legacy
    # behavior), not 0, which would corrupt Total_Loans for every customer with no loans on record
    loan_type_array = F.split(F.regexp_replace(F.coalesce(col("Type_of_Loan"), F.lit("")), r",?\s+and\s+", ", "), r",\s*")

    all_loan_type_cols = []
    for loan_type in LOAN_TYPES:
        col_name = "n_" + loan_type.replace("-", "_").replace(" ", "_")
        all_loan_type_cols.append(col_name)
        lms = lms.withColumn(col_name, F.size(F.filter(loan_type_array, lambda t: t == loan_type)).cast(IntegerType()))

    secured = ["n_Auto_Loan", "n_Mortgage_Loan", "n_Home_Equity_Loan"]
    unsecured = ["n_Personal_Loan", "n_Payday_Loan", "n_Credit_Builder_Loan", "n_Student_Loan"]

    lms = lms.withColumn("Total_Loans", sum(F.col(c) for c in all_loan_type_cols))
    lms = lms.withColumn("n_secured", sum(F.col(c) for c in secured))
    lms = lms.withColumn("n_unsecured", sum(F.col(c) for c in unsecured))
    lms = lms.withColumn("n_distinct_loan_types", sum(F.when(F.col(c) > 0, 1).otherwise(0) for c in all_loan_type_cols))

    lms = lms.withColumn("debt_to_income", col("Outstanding_Debt").cast("double") / col("Annual_Income").cast("double"))
    lms = lms.withColumn("emi_to_income", col("Total_EMI_per_month").cast("double") / col("Monthly_Inhand_Salary").cast("double"))

    lms = lms.withColumn("Spend_Level", F.regexp_extract(col("Payment_Behaviour"), r"^(Low|High)_spent", 1))
    lms = lms.withColumn("Payment_Value", F.regexp_extract(col("Payment_Behaviour"), r"(Small|Medium|Large)_value", 1))
    lms = lms.withColumn("Spend_Level", F.when(col("Spend_Level") == "", "Unknown").otherwise(col("Spend_Level")))
    lms = lms.withColumn("Payment_Value", F.when(col("Payment_Value") == "", "Unknown").otherwise(col("Payment_Value")))

    lms = lms.withColumn("Credit_Mix_MinAmt", F.concat_ws("_", col("Credit_Mix"), col("Payment_of_Min_Amount")))

    # clickstream: one row per customer PER MONTH -> must join on (Customer_ID, snapshot_date),
    # LEFT join so a customer/month with no clickstream still keeps its lms row (clickstream
    # columns come through as null instead of dropping the row)
    lms = lms.join(click, on=["Customer_ID", "snapshot_date"], how="left")

    print('final row count:', lms.count(), '(should equal the lms row count after the cutoff filter)')

    # save gold table - IRL connect to database to write
    os.makedirs(gold_feature_store_directory, exist_ok=True)
    filepath = gold_feature_store_directory + "gold_ft_store_loan_panel.parquet"
    lms.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return lms