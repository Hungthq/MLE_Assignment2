import os

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

def process_feature_store(date_str, silver_attributes_daily_directory, silver_financials_daily_directory,
                       silver_clickstream_daily_directory,
                       gold_feature_store_directory, spark):

    # prepare arguments
    #date_str = datetime.strptime(date_str, "%Y-%m-%d"
    
    # one partition = the customers who applied for a loan in this month (snapshot_date = application date)
    partition_name = "silver_daily_" + date_str.replace('-','_') + '.parquet'
    attr = spark.read.parquet(silver_attributes_daily_directory + partition_name)
    fin = spark.read.parquet(silver_financials_daily_directory + partition_name)

    # PII: Name and SSN live in silver attributes but must never reach a feature store
    attr = attr.drop("Name", "SSN")
    fin = fin.drop("snapshot_date")  # same date as attr; keep a single copy

    # attributes + financials: one row per applicant (1:1 on Customer_ID)
    click = attr.join(fin, on="Customer_ID", how="inner")

    # clickstream: point-in-time, SAME month as the application only (never a later month),
    # LEFT join so applicants without clickstream (loans from Jul 2024, file ends Dec 2024) are kept
    click_path = silver_clickstream_daily_directory + partition_name
    if os.path.exists(click_path):
        clk = spark.read.parquet(click_path)
        click = click.join(clk, on=["Customer_ID", "snapshot_date"], how="left")
    else:
        for i in range(1, 21):
            click = click.withColumn(f"fe_{i}", F.lit(None).cast(IntegerType()))
    print('attributes+financials:', attr.count(), '| rows with clickstream:', click.filter(col("fe_1").isNotNull()).count())

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
        click = click.withColumn(col_name, F.size(F.filter(loan_type_array, lambda t: t == loan_type)).cast(IntegerType()))

    secured = ["n_Auto_Loan", "n_Mortgage_Loan", "n_Home_Equity_Loan"]
    unsecured = ["n_Personal_Loan", "n_Payday_Loan", "n_Credit_Builder_Loan", "n_Student_Loan"]

    click = click.withColumn("Total_Loans", sum(F.col(c) for c in all_loan_type_cols))
    click = click.withColumn("n_secured", sum(F.col(c) for c in secured))
    click = click.withColumn("n_unsecured", sum(F.col(c) for c in unsecured))
    click = click.withColumn("n_distinct_loan_types", sum(F.when(F.col(c) > 0, 1).otherwise(0) for c in all_loan_type_cols))

    click = click.withColumn("debt_to_income", col("Outstanding_Debt").cast("double") / col("Annual_Income").cast("double"))
    click = click.withColumn("emi_to_income", col("Total_EMI_per_month").cast("double") / col("Monthly_Inhand_Salary").cast("double"))

    click = click.withColumn("Spend_Level", F.regexp_extract(col("Payment_Behaviour"), r"^(Low|High)_spent", 1))
    click = click.withColumn("Payment_Value", F.regexp_extract(col("Payment_Behaviour"), r"(Small|Medium|Large)_value", 1))
    click = click.withColumn("Spend_Level", F.when(col("Spend_Level") == "", "Unknown").otherwise(col("Spend_Level")))
    click = click.withColumn("Payment_Value", F.when(col("Payment_Value") == "", "Unknown").otherwise(col("Payment_Value")))

    click = click.withColumn("Credit_Mix_MinAmt", F.concat_ws("_", col("Credit_Mix"), col("Payment_of_Min_Amount")))


    # raw text columns have been parsed into numeric features above
    click = click.drop("Type_of_Loan", "Credit_History_Age")

    print('final row count:', click.count())

    # save gold table - IRL connect to database to write
    os.makedirs(gold_feature_store_directory, exist_ok=True)
    filepath = gold_feature_store_directory + "gold_feature_store_" + date_str.replace('-','_') + '.parquet'
    click.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return click


def process_label_store(date_str, silver_lms_daily_directory,
                       gold_label_store_directory, spark, dpd=30, mob=6):

    # connect to silver lms table
    partition_name = "silver_loan_daily_" + date_str.replace('-','_') + '.parquet'
    filepath = silver_lms_daily_directory + partition_name
    lms = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', lms.count())

    # label = 1 if the loan is dpd+ days past due at month-on-book `mob`
    lms = lms.filter(col("mob") == mob)
    lms = lms.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    lms = lms.withColumn("label_def", F.lit(str(dpd) + "dpd_" + str(mob) + "mob").cast(StringType()))

    # loan_start_date = application date = the feature store's snapshot_date (join key)
    lms = lms.select("loan_id", "Customer_ID", "loan_start_date", "label", "label_def", "snapshot_date")
    print('final row count:', lms.count())

    # save gold table - IRL connect to database to write
    os.makedirs(gold_label_store_directory, exist_ok=True)
    filepath = gold_label_store_directory + "gold_label_store_" + date_str.replace('-','_') + ".parquet"
    lms.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return lms
