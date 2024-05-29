# Databricks notebook source
import dlt
import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame
from pyspark.ml.feature import (
    VectorAssembler,
    MinMaxScaler,
    StandardScaler,
    StringIndexer,
)
from schemas import (
    bronze_schema,
    gold_schema,
    silver_1_schema,
    silver_2_schema,
    silver_3_schema,
)
from validations import (
    bronze_validations,
    silver_1_validation,
    silver_2_validation,
    silver_3_validation,
    gold_validation,
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to create a UDF for mapping dictionary values
def create_map_udf(mapping_dict):
    return F.udf(lambda key: mapping_dict.get(key, "Unknown"), T.StringType())


# Decorator for error handling
def handle_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {e}")
            raise

    return wrapper


# Function to summarize data type conversions
def summarize_data_type_conversions(df: DataFrame):
    summary = {}
    for field in df.schema.fields:
        summary[field.name] = str(field.dataType)
    return summary


# Function to summarize normalization and standardization
def summarize_normalization(df: DataFrame, columns: list):
    summary = {}
    for col in columns:
        summary[col] = {
            "min": df.select(F.min(col)).collect()[0][0],
            "max": df.select(F.max(col)).collect()[0][0],
            "mean": df.select(F.mean(col)).collect()[0][0],
            "stddev": df.select(F.stddev(col)).collect()[0][0],
        }
    return summary


# Bronze Layer
@dlt.expect_all_or_drop(bronze_validations)
@dlt.table(
    name="bronze",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=bronze_schema,
)
@handle_error
def bronze_layer() -> DataFrame:
    raw_path = "dbfs:/FileStore/tables/mhcld_puf_2021.csv"
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(raw_path)
        .na.drop()
    )
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == bronze_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("bronze")
    logger.info("Bronze layer processing complete.")
    logger.info(f"Schema: {summarize_data_type_conversions(df)}")
    return df


# Silver 1 Layer
@dlt.expect_all_or_drop(silver_1_validation)
@dlt.table(
    name="silver_1",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=silver_1_schema,
)
@handle_error
def silver_1() -> DataFrame:
    df = spark.read.table("bronze")

    # Mappings
    gender = {1: "Male", 2: "Female", -9: "Missing/unknown/not collected/invalid"}
    race = {
        1: "American Indian/Alaska Native",
        2: "Asian",
        3: "Black or African American",
        4: "Native Hawaiian or Other Pacific Islander",
        5: "White",
        6: "Some other race alone/two or more races",
        -9: "Missing/unknown/not collected/invalid",
    }
    marital_status = {
        1: "Never married",
        2: "Now married",
        3: "Separated",
        4: "Divorced, widowed",
        -9: "Missing/unknown/not collected/invalid",
    }
    employment_status = {
        1: "Full-time",
        2: "Part-time",
        3: "Employed full-time/part-time not differentiated",
        4: "Unemployed",
        5: "Not in labor force",
        -9: "Missing/unknown/not collected/invalid",
    }
    ethnicity = {
        1: "Mexican",
        2: "Puerto Rican",
        3: "Other Hispanic or Latino origin",
        4: "Not of Hispanic or Latino origin",
        -9: "Missing/unknown/not collected/invalid",
    }

    # Apply mapping UDFs
    race_map_udf = create_map_udf(race)
    gender_map_udf = create_map_udf(gender)
    marital_status_map_udf = create_map_udf(marital_status)
    employment_status_map_udf = create_map_udf(employment_status)
    ethnicity_map_udf = create_map_udf(ethnicity)

    df = df.withColumn("GENDER_mapped", gender_map_udf("GENDER"))
    df = df.withColumn("RACE_mapped", race_map_udf("RACE"))
    df = df.withColumn("MARSTAT_mapped", marital_status_map_udf("MARSTAT"))
    df = df.withColumn("EMPLOY_mapped", employment_status_map_udf("EMPLOY"))
    df = df.withColumn("ETHNIC_mapped", ethnicity_map_udf("ETHNIC"))

    df = df.withColumn(
        "CASEID_int",
        F.substring(F.col("CASEID").cast("string"), 5, 10).cast(T.IntegerType()),
    )
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == silver_1_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_1")
    logger.info("Silver_1 layer processing complete.")
    logger.info(f"Schema: {summarize_data_type_conversions(df)}")
    return df


# Silver 2 Layer
@dlt.expect_all_or_drop(silver_2_validation)
@dlt.table(
    name="silver_2",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=silver_2_schema,
)
@handle_error
def silver_2() -> DataFrame:
    df = spark.read.table("silver_1")
    numeric_columns = ["NUMMHS"]

    for col in numeric_columns:
        assembler = VectorAssembler(inputCols=[col], outputCol=f"{col}_vec")
        df = assembler.transform(df)

        min_max_scaler = MinMaxScaler(
            inputCol=f"{col}_vec", outputCol=f"{col}_normalized_vec"
        )
        df = min_max_scaler.fit(df).transform(df)

        standard_scaler = StandardScaler(
            inputCol=f"{col}_normalized_vec",
            outputCol=f"{col}_standardized_vec",
            withMean=True,
            withStd=True,
        )
        df = standard_scaler.fit(df).transform(df)

        extract_first_element = F.udf(lambda x: float(x[0]), T.FloatType())
        df = df.withColumn(
            f"{col}_normalized", extract_first_element(F.col(f"{col}_normalized_vec"))
        )
        df = df.withColumn(
            f"{col}_standardized",
            extract_first_element(F.col(f"{col}_standardized_vec")),
        )

        df = (
            df.drop(f"{col}_vec")
            .drop(f"{col}_normalized_vec")
            .drop(f"{col}_standardized_vec")
        )

    df = df.na.drop()
    assert df.schema == silver_2_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_2")
    logger.info("Silver_2 layer processing complete.")
    logger.info(
        f"Normalization Summary: {summarize_normalization(df, numeric_columns)}"
    )
    return df


# Silver 3 Layer
@dlt.expect_all_or_drop(silver_3_validation)
@dlt.table(
    name="silver_3",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=silver_3_schema,
)
@handle_error
def silver_3() -> DataFrame:
    df = spark.read.table("silver_2")

    df = df.withColumn(
        "demographic_strata",
        F.concat_ws(
            "_",
            F.col("GENDER"),
            F.col("RACE"),
            F.col("ETHNIC"),
            F.col("MARSTAT"),
            F.col("EMPLOY"),
        ),
    )

    indexer = StringIndexer(inputCol="demographic_strata", outputCol="strataIndex")
    df = indexer.fit(df).transform(df)

    SEED = 42
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train, validation = train.randomSplit([0.75, 0.25], seed=SEED)

    train.write.format("delta").mode("overwrite").saveAsTable("silver_train")
    test.write.format("delta").mode("overwrite").saveAsTable("silver_test")
    validation.write.format("delta").mode("overwrite").saveAsTable("silver_validation")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == silver_3_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_3_full")
    logger.info("Silver_3 layer processing complete.")
    logger.info("Data partitioning and sampling complete.")
    return df


# Gold Full Layer
@dlt.expect_all_or_drop(gold_validation)
@dlt.table(
    name="gold_full",
    # schema=gold_schema
)
@handle_error
def gold_full_layer() -> DataFrame:
    df = spark.read.table("silver_3_full").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == gold_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_full")
    logger.info("Gold_full layer processing complete.")
    return df


# Gold Train Layer
@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_train", schema=gold_schema)
@handle_error
def gold_train_layer() -> DataFrame:
    df = spark.read.table("silver_train").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == gold_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_train")
    logger.info("Gold_train layer processing complete.")
    return df


# Gold Validation Layer
@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_validation", schema=gold_schema)
@handle_error
def gold_validation_layer() -> DataFrame:
    df = spark.read.table("silver_validation").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    assert df.schema == gold_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_validation")
    logger.info("Gold_validation layer processing complete.")
    return df


# Gold Test Layer
@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_test", schema=gold_schema)
@handle_error
def gold_test_layer() -> DataFrame:
    df = spark.read.table("silver_test").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    assert df.schema == gold_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_test")
    logger.info("Gold_test layer processing complete.")
    return df


# Summary Reporting Functions
def generate_summary_report():
    bronze_summary = summarize_data_type_conversions(spark.read.table("bronze"))
    silver_1_summary = summarize_data_type_conversions(spark.read.table("silver_1"))
    silver_2_normalization_summary = summarize_normalization(
        spark.read.table("silver_2"), ["NUMMHS_normalized", "NUMMHS_standardized"]
    )
    silver_3_summary = "Data partitioning and sampling complete."

    logger.info(f"Bronze Layer Summary: {bronze_summary}")
    logger.info(f"Silver_1 Layer Summary: {silver_1_summary}")
    logger.info(f"Silver_2 Normalization Summary: {silver_2_normalization_summary}")
    logger.info(f"Silver_3 Summary: {silver_3_summary}")


generate_summary_report()
