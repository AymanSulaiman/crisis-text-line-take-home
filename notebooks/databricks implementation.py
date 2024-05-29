# Databricks notebook source
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS bronze;
# MAGIC DROP TABLE IF EXISTS silver_1;
# MAGIC DROP TABLE IF EXISTS silver_2;
# MAGIC DROP TABLE IF EXISTS silver_3;
# MAGIC DROP TABLE IF EXISTS silver_test;
# MAGIC DROP TABLE IF EXISTS silver_train;
# MAGIC DROP TABLE IF EXISTS silver_validation;
# MAGIC DROP TABLE IF EXISTS gold_full;
# MAGIC DROP TABLE IF EXISTS gold_test;
# MAGIC DROP TABLE IF EXISTS gold_train;
# MAGIC DROP TABLE IF EXISTS gold_validation;
# MAGIC

# COMMAND ----------

import dlt

from pyspark.sql import types as T

# bronze_schema = T.StructType([
#     T.StructField("YEAR", T.IntegerType(), True),
#     T.StructField("AGE", T.IntegerType(), True),
#     T.StructField("EDUC", T.IntegerType(), True),
#     T.StructField("ETHNIC", T.IntegerType(), True),
#     T.StructField("RACE", T.IntegerType(), True),
#     T.StructField("GENDER", T.IntegerType(), True),
#     T.StructField("SPHSERVICE", T.IntegerType(), True),
#     T.StructField("CMPSERVICE", T.IntegerType(), True),
#     T.StructField("OPISERVICE", T.IntegerType(), True),
#     T.StructField("RTCSERVICE", T.IntegerType(), True),
#     T.StructField("IJSSERVICE", T.IntegerType(), True),
#     T.StructField("MH1", T.IntegerType(), True),
#     T.StructField("MH2", T.IntegerType(), True),
#     T.StructField("MH3", T.IntegerType(), True),
#     T.StructField("SUB", T.IntegerType(), True),
#     T.StructField("MARSTAT", T.IntegerType(), True),
#     T.StructField("SMISED", T.IntegerType(), True),
#     T.StructField("SAP", T.IntegerType(), True),
#     T.StructField("EMPLOY", T.IntegerType(), True),
#     T.StructField("DETNLF", T.IntegerType(), True),
#     T.StructField("VETERAN", T.IntegerType(), True),
#     T.StructField("LIVARAG", T.IntegerType(), True),
#     T.StructField("NUMMHS", T.IntegerType(), True),
#     T.StructField("TRAUSTREFLG", T.IntegerType(), True),
#     T.StructField("ANXIETYFLG", T.IntegerType(), True),
#     T.StructField("ADHDFLG", T.IntegerType(), True),
#     T.StructField("CONDUCTFLG", T.IntegerType(), True),
#     T.StructField("DELIRDEMFLG", T.IntegerType(), True),
#     T.StructField("BIPOLARFLG", T.IntegerType(), True),
#     T.StructField("DEPRESSFLG", T.IntegerType(), True),
#     T.StructField("ODDFLG", T.IntegerType(), True),
#     T.StructField("PDDFLG", T.IntegerType(), True),
#     T.StructField("PERSONFLG", T.IntegerType(), True),
#     T.StructField("SCHIZOFLG", T.IntegerType(), True),
#     T.StructField("ALCSUBFLG", T.IntegerType(), True),
#     T.StructField("OTHERDISFLG", T.IntegerType(), True),
#     T.StructField("STATEFIP", T.IntegerType(), True),
#     T.StructField("DIVISION", T.IntegerType(), True),
#     T.StructField("REGION", T.IntegerType(), True),
#     T.StructField("CASEID", T.LongType(), True),
# ])


@dlt.expect_all_or_drop(
    {
        "non_negative_age": "AGE >= 0 or AGE == -9",
        "valid_age_type": "cast(AGE as int) == AGE",
        "valid_year_type": "cast(YEAR as int) == YEAR",
        "valid_caseid_type": "cast(CASEID as long) == CASEID",
        "valid_educ_values": "EDUC >= -9",
        "valid_ethnic_values": "ETHNIC >= -9",
        "valid_race_values": "RACE >= -9",
        "valid_gender_values": "GENDER >= -9",
        "valid_sphservice_values": "SPHSERVICE >= -9",
        "valid_cmpservice_values": "CMPSERVICE >= -9",
        "valid_opiservice_values": "OPISERVICE >= -9",
        "valid_rtcservice_values": "RTCSERVICE >= -9",
        "valid_ijsservice_values": "IJSSERVICE >= -9",
        "valid_mh1_values": "MH1 >= -9",
        "valid_mh2_values": "MH2 >= -9",
        "valid_mh3_values": "MH3 >= -9",
        "valid_sub_values": "SUB >= -9",
        "valid_marstat_values": "MARSTAT >= -9",
        "valid_smised_values": "SMISED >= -9",
        "valid_sap_values": "SAP >= -9",
        "valid_employ_values": "EMPLOY >= -9",
        "valid_detnlf_values": "DETNLF >= -9",
        "valid_veteran_values": "VETERAN >= -9",
        "valid_livarag_values": "LIVARAG >= -9",
        "valid_nummhs_values": "NUMMHS >= -9",
        "valid_traustref_values": "TRAUSTREFLG >= -9",
        "valid_anxiety_values": "ANXIETYFLG >= -9",
        "valid_adhd_values": "ADHDFLG >= -9",
        "valid_conduct_values": "CONDUCTFLG >= -9",
        "valid_delirdem_values": "DELIRDEMFLG >= -9",
        "valid_bipolar_values": "BIPOLARFLG >= -9",
        "valid_depress_values": "DEPRESSFLG >= -9",
        "valid_odd_values": "ODDFLG >= -9",
        "valid_pdd_values": "PDDFLG >= -9",
        "valid_person_values": "PERSONFLG >= -9",
        "valid_schizo_values": "SCHIZOFLG >= -9",
        "valid_alcsub_values": "ALCSUBFLG >= -9",
        "valid_otherdis_values": "OTHERDISFLG >= -9",
        "valid_statefip_values": "STATEFIP >= -9",
        "valid_division_values": "DIVISION >= -9",
        "valid_region_values": "REGION >= -9",
    }
)
@dlt.table(
    name="bronze",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    # schema=bronze_schema
)
def bronze_layer():
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

    return df


# COMMAND ----------

# MAGIC %sql select * from bronze LIMIT 10

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import types as T


# silver_1_schema = T.StructType([
#     T.StructField("YEAR", T.IntegerType(), True),
#     T.StructField("AGE", T.IntegerType(), True),
#     T.StructField("EDUC", T.IntegerType(), True),
#     T.StructField("ETHNIC", T.IntegerType(), True),
#     T.StructField("RACE", T.IntegerType(), True),
#     T.StructField("GENDER", T.IntegerType(), True),
#     T.StructField("SPHSERVICE", T.IntegerType(), True),
#     T.StructField("CMPSERVICE", T.IntegerType(), True),
#     T.StructField("OPISERVICE", T.IntegerType(), True),
#     T.StructField("RTCSERVICE", T.IntegerType(), True),
#     T.StructField("IJSSERVICE", T.IntegerType(), True),
#     T.StructField("MH1", T.IntegerType(), True),
#     T.StructField("MH2", T.IntegerType(), True),
#     T.StructField("MH3", T.IntegerType(), True),
#     T.StructField("SUB", T.IntegerType(), True),
#     T.StructField("MARSTAT", T.IntegerType(), True),
#     T.StructField("SMISED", T.IntegerType(), True),
#     T.StructField("SAP", T.IntegerType(), True),
#     T.StructField("EMPLOY", T.IntegerType(), True),
#     T.StructField("DETNLF", T.IntegerType(), True),
#     T.StructField("VETERAN", T.IntegerType(), True),
#     T.StructField("LIVARAG", T.IntegerType(), True),
#     T.StructField("NUMMHS", T.IntegerType(), True),
#     T.StructField("TRAUSTREFLG", T.IntegerType(), True),
#     T.StructField("ANXIETYFLG", T.IntegerType(), True),
#     T.StructField("ADHDFLG", T.IntegerType(), True),
#     T.StructField("CONDUCTFLG", T.IntegerType(), True),
#     T.StructField("DELIRDEMFLG", T.IntegerType(), True),
#     T.StructField("BIPOLARFLG", T.IntegerType(), True),
#     T.StructField("DEPRESSFLG", T.IntegerType(), True),
#     T.StructField("ODDFLG", T.IntegerType(), True),
#     T.StructField("PDDFLG", T.IntegerType(), True),
#     T.StructField("PERSONFLG", T.IntegerType(), True),
#     T.StructField("SCHIZOFLG", T.IntegerType(), True),
#     T.StructField("ALCSUBFLG", T.IntegerType(), True),
#     T.StructField("OTHERDISFLG", T.IntegerType(), True),
#     T.StructField("STATEFIP", T.IntegerType(), True),
#     T.StructField("DIVISION", T.IntegerType(), True),
#     T.StructField("REGION", T.IntegerType(), True),
#     T.StructField("CASEID", T.LongType(), True),
#     T.StructField("GENDER_mapped", T.StringType(), True),
#     T.StructField("RACE_mapped", T.StringType(), True),
#     T.StructField("MARSTAT_mapped", T.StringType(), True),
#     T.StructField("EMPLOY_mapped", T.StringType(), True),
#     T.StructField("ETHNIC_mapped", T.StringType(), True),
#     T.StructField("CASEID_int", T.IntegerType(), True),
# ])


@dlt.expect_all(
    {
        "valid_gender": "GENDER IS NOT NULL",
        "valid_race": "RACE IS NOT NULL",
        "valid_ethnic": "ETHNIC IS NOT NULL",
        "valid_marstat": "MARSTAT IS NOT NULL",
        "valid_employ": "EMPLOY IS NOT NULL",
    }
)
@dlt.table(
    name="silver_1", partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def silver_1():
    df = spark.read.table("bronze")

    def create_map_udf(mapping_dict):
        return F.udf(lambda key: mapping_dict.get(key, "Unknown"), T.StringType())

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
    # assert df.schema == silver_1_schema
    df.printSchema()
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_1")
    return df


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_1 limit 10

# COMMAND ----------

from pyspark.ml.feature import (
    VectorAssembler,
    MinMaxScaler,
    StandardScaler,
    StringIndexer,
)
from pyspark.ml import Pipeline

# silver_2_schema = T.StructType([
#     T.StructField("YEAR", T.IntegerType(), True),
#     T.StructField("AGE", T.IntegerType(), True),
#     T.StructField("EDUC", T.IntegerType(), True),
#     T.StructField("ETHNIC", T.IntegerType(), True),
#     T.StructField("RACE", T.IntegerType(), True),
#     T.StructField("GENDER", T.IntegerType(), True),
#     T.StructField("SPHSERVICE", T.IntegerType(), True),
#     T.StructField("CMPSERVICE", T.IntegerType(), True),
#     T.StructField("OPISERVICE", T.IntegerType(), True),
#     T.StructField("RTCSERVICE", T.IntegerType(), True),
#     T.StructField("IJSSERVICE", T.IntegerType(), True),
#     T.StructField("MH1", T.IntegerType(), True),
#     T.StructField("MH2", T.IntegerType(), True),
#     T.StructField("MH3", T.IntegerType(), True),
#     T.StructField("SUB", T.IntegerType(), True),
#     T.StructField("MARSTAT", T.IntegerType(), True),
#     T.StructField("SMISED", T.IntegerType(), True),
#     T.StructField("SAP", T.IntegerType(), True),
#     T.StructField("EMPLOY", T.IntegerType(), True),
#     T.StructField("DETNLF", T.IntegerType(), True),
#     T.StructField("VETERAN", T.IntegerType(), True),
#     T.StructField("LIVARAG", T.IntegerType(), True),
#     T.StructField("NUMMHS", T.IntegerType(), True),
#     T.StructField("TRAUSTREFLG", T.IntegerType(), True),
#     T.StructField("ANXIETYFLG", T.IntegerType(), True),
#     T.StructField("ADHDFLG", T.IntegerType(), True),
#     T.StructField("CONDUCTFLG", T.IntegerType(), True),
#     T.StructField("DELIRDEMFLG", T.IntegerType(), True),
#     T.StructField("BIPOLARFLG", T.IntegerType(), True),
#     T.StructField("DEPRESSFLG", T.IntegerType(), True),
#     T.StructField("ODDFLG", T.IntegerType(), True),
#     T.StructField("PDDFLG", T.IntegerType(), True),
#     T.StructField("PERSONFLG", T.IntegerType(), True),
#     T.StructField("SCHIZOFLG", T.IntegerType(), True),
#     T.StructField("ALCSUBFLG", T.IntegerType(), True),
#     T.StructField("OTHERDISFLG", T.IntegerType(), True),
#     T.StructField("STATEFIP", T.IntegerType(), True),
#     T.StructField("DIVISION", T.IntegerType(), True),
#     T.StructField("REGION", T.IntegerType(), True),
#     T.StructField("CASEID", T.LongType(), True),
#     T.StructField("GENDER_mapped", T.StringType(), True),
#     T.StructField("RACE_mapped", T.StringType(), True),
#     T.StructField("MARSTAT_mapped", T.StringType(), True),
#     T.StructField("EMPLOY_mapped", T.StringType(), True),
#     T.StructField("ETHNIC_mapped", T.StringType(), True),
#     T.StructField("CASEID_int", T.IntegerType(), True),
#     T.StructField("NUMMHS_normalized", T.FloatType(), True),
#     T.StructField("NUMMHS_standardized", T.FloatType(), True),
# ])


@dlt.expect_all(
    {
        "valid_gender": "GENDER IS NOT NULL",
        "valid_race": "RACE IS NOT NULL",
        "valid_ethnic": "ETHNIC IS NOT NULL",
        "valid_marstat": "MARSTAT IS NOT NULL",
        "valid_employ": "EMPLOY IS NOT NULL",
    }
)
@dlt.table(
    name="silver_2", partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def silver_2():
    df = spark.read.table("silver_1")
    numeric_columns = ["NUMMHS"]

    for col in numeric_columns:
        # Assemble the column into a vector
        assembler = VectorAssembler(inputCols=[col], outputCol=f"{col}_vec")
        df = assembler.transform(df)

        # Apply MinMaxScaler
        min_max_scaler = MinMaxScaler(
            inputCol=f"{col}_vec", outputCol=f"{col}_normalized_vec"
        )
        df = min_max_scaler.fit(df).transform(df)

        # Apply StandardScaler
        standard_scaler = StandardScaler(
            inputCol=f"{col}_normalized_vec",
            outputCol=f"{col}_standardized_vec",
            withMean=True,
            withStd=True,
        )
        df = standard_scaler.fit(df).transform(df)

        # Extract the first element from the vector columns
        extract_first_element = F.udf(lambda x: float(x[0]), T.FloatType())
        df = df.withColumn(
            f"{col}_normalized", extract_first_element(F.col(f"{col}_normalized_vec"))
        )
        df = df.withColumn(
            f"{col}_standardized",
            extract_first_element(F.col(f"{col}_standardized_vec")),
        )

        # Drop intermediate vector columns
        df = (
            df.drop(f"{col}_vec")
            .drop(f"{col}_normalized_vec")
            .drop(f"{col}_standardized_vec")
        )

    df = df.na.drop()

    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_2")
    return df


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_2 limit 10

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# silver_3_schema = T.StructType([
#     T.StructField("YEAR", T.IntegerType(), True),
#     T.StructField("AGE", T.IntegerType(), True),
#     T.StructField("EDUC", T.IntegerType(), True),
#     T.StructField("ETHNIC", T.IntegerType(), True),
#     T.StructField("RACE", T.IntegerType(), True),
#     T.StructField("GENDER", T.IntegerType(), True),
#     T.StructField("SPHSERVICE", T.IntegerType(), True),
#     T.StructField("CMPSERVICE", T.IntegerType(), True),
#     T.StructField("OPISERVICE", T.IntegerType(), True),
#     T.StructField("RTCSERVICE", T.IntegerType(), True),
#     T.StructField("IJSSERVICE", T.IntegerType(), True),
#     T.StructField("MH1", T.IntegerType(), True),
#     T.StructField("MH2", T.IntegerType(), True),
#     T.StructField("MH3", T.IntegerType(), True),
#     T.StructField("SUB", T.IntegerType(), True),
#     T.StructField("MARSTAT", T.IntegerType(), True),
#     T.StructField("SMISED", T.IntegerType(), True),
#     T.StructField("SAP", T.IntegerType(), True),
#     T.StructField("EMPLOY", T.IntegerType(), True),
#     T.StructField("DETNLF", T.IntegerType(), True),
#     T.StructField("VETERAN", T.IntegerType(), True),
#     T.StructField("LIVARAG", T.IntegerType(), True),
#     T.StructField("NUMMHS", T.IntegerType(), True),
#     T.StructField("TRAUSTREFLG", T.IntegerType(), True),
#     T.StructField("ANXIETYFLG", T.IntegerType(), True),
#     T.StructField("ADHDFLG", T.IntegerType(), True),
#     T.StructField("CONDUCTFLG", T.IntegerType(), True),
#     T.StructField("DELIRDEMFLG", T.IntegerType(), True),
#     T.StructField("BIPOLARFLG", T.IntegerType(), True),
#     T.StructField("DEPRESSFLG", T.IntegerType(), True),
#     T.StructField("ODDFLG", T.IntegerType(), True),
#     T.StructField("PDDFLG", T.IntegerType(), True),
#     T.StructField("PERSONFLG", T.IntegerType(), True),
#     T.StructField("SCHIZOFLG", T.IntegerType(), True),
#     T.StructField("ALCSUBFLG", T.IntegerType(), True),
#     T.StructField("OTHERDISFLG", T.IntegerType(), True),
#     T.StructField("STATEFIP", T.IntegerType(), True),
#     T.StructField("DIVISION", T.IntegerType(), True),
#     T.StructField("REGION", T.IntegerType(), True),
#     T.StructField("CASEID", T.LongType(), True),
#     T.StructField("GENDER_mapped", T.StringType(), True),
#     T.StructField("RACE_mapped", T.StringType(), True),
#     T.StructField("MARSTAT_mapped", T.StringType(), True),
#     T.StructField("EMPLOY_mapped", T.StringType(), True),
#     T.StructField("ETHNIC_mapped", T.StringType(), True),
#     T.StructField("CASEID_int", T.IntegerType(), True),
#     T.StructField("demographic_strata", T.StringType(), False),
#     T.StructField("strataIndex", T.DoubleType(), False),
# ])


@dlt.expect_all_or_drop(
    {
        "non_negative_age": "AGE >= 0 or AGE == -9",
        "valid_age_type": "cast(AGE as int) == AGE",
        "valid_year_type": "cast(YEAR as int) == YEAR",
        "valid_caseid_type": "cast(CASEID as long) == CASEID",
        "valid_educ_values": "EDUC >= -9",
        "valid_ethnic_values": "ETHNIC >= -9",
        "valid_race_values": "RACE >= -9",
        "valid_gender_values": "GENDER >= -9",
        "valid_sphservice_values": "SPHSERVICE >= -9",
        "valid_cmpservice_values": "CMPSERVICE >= -9",
        "valid_opiservice_values": "OPISERVICE >= -9",
        "valid_rtcservice_values": "RTCSERVICE >= -9",
        "valid_ijsservice_values": "IJSSERVICE >= -9",
        "valid_mh1_values": "MH1 >= -9",
        "valid_mh2_values": "MH2 >= -9",
        "valid_mh3_values": "MH3 >= -9",
        "valid_sub_values": "SUB >= -9",
        "valid_marstat_values": "MARSTAT >= -9",
        "valid_smised_values": "SMISED >= -9",
        "valid_sap_values": "SAP >= -9",
        "valid_employ_values": "EMPLOY >= -9",
        "valid_detnlf_values": "DETNLF >= -9",
        "valid_veteran_values": "VETERAN >= -9",
        "valid_livarag_values": "LIVARAG >= -9",
        "valid_nummhs_values": "NUMMHS >= -9",
        "valid_traustref_values": "TRAUSTREFLG >= -9",
        "valid_anxiety_values": "ANXIETYFLG >= -9",
        "valid_adhd_values": "ADHDFLG >= -9",
        "valid_conduct_values": "CONDUCTFLG >= -9",
        "valid_delirdem_values": "DELIRDEMFLG >= -9",
        "valid_bipolar_values": "BIPOLARFLG >= -9",
        "valid_depress_values": "DEPRESSFLG >= -9",
        "valid_odd_values": "ODDFLG >= -9",
        "valid_pdd_values": "PDDFLG >= -9",
        "valid_person_values": "PERSONFLG >= -9",
        "valid_schizo_values": "SCHIZOFLG >= -9",
        "valid_alcsub_values": "ALCSUBFLG >= -9",
        "valid_otherdis_values": "OTHERDISFLG >= -9",
        "valid_statefip_values": "STATEFIP >= -9",
        "valid_division_values": "DIVISION >= -9",
        "valid_region_values": "REGION >= -9",
        "valid_demographic_strata": "demographic_strata IS NOT NULL AND demographic_strata != ''",
        "valid_strataIndex": "strataIndex IS NOT NULL AND cast(strataIndex as double) == strataIndex",
    }
)
@dlt.table(
    name="silver_3", partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def silver_3():
    df = spark.read.table("silver_2")

    # Create demographic_strata column
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

    # Index the demographic_strata column
    indexer = StringIndexer(inputCol="demographic_strata", outputCol="strataIndex")
    df = indexer.fit(df).transform(df)

    SEED = 42
    # Split the data into train, test, and validation sets
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train, validation = train.randomSplit([0.75, 0.25], seed=SEED)

    # Write train, test, and validation sets to their respective paths
    train.write.format("delta").mode("overwrite").saveAsTable("silver_train")
    test.write.format("delta").mode("overwrite").saveAsTable("silver_test")
    validation.write.format("delta").mode("overwrite").saveAsTable("silver_validation")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]

    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_3")
    return df


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_3 LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_train LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_test LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_validation LIMIT 10;

# COMMAND ----------

# gold_schema = T.StructType([
#     T.StructField("YEAR", T.IntegerType(), True),
#     T.StructField("AGE", T.IntegerType(), True),
#     T.StructField("EDUC", T.IntegerType(), True),
#     T.StructField("ETHNIC", T.IntegerType(), True),
#     T.StructField("RACE", T.IntegerType(), True),
#     T.StructField("GENDER", T.IntegerType(), True),
#     T.StructField("SPHSERVICE", T.IntegerType(), True),
#     T.StructField("CMPSERVICE", T.IntegerType(), True),
#     T.StructField("OPISERVICE", T.IntegerType(), True),
#     T.StructField("RTCSERVICE", T.IntegerType(), True),
#     T.StructField("IJSSERVICE", T.IntegerType(), True),
#     T.StructField("MH1", T.IntegerType(), True),
#     T.StructField("MH2", T.IntegerType(), True),
#     T.StructField("MH3", T.IntegerType(), True),
#     T.StructField("SUB", T.IntegerType(), True),
#     T.StructField("MARSTAT", T.IntegerType(), True),
#     T.StructField("SMISED", T.IntegerType(), True),
#     T.StructField("SAP", T.IntegerType(), True),
#     T.StructField("EMPLOY", T.IntegerType(), True),
#     T.StructField("DETNLF", T.IntegerType(), True),
#     T.StructField("VETERAN", T.IntegerType(), True),
#     T.StructField("LIVARAG", T.IntegerType(), True),
#     T.StructField("NUMMHS", T.IntegerType(), True),
#     T.StructField("TRAUSTREFLG", T.IntegerType(), True),
#     T.StructField("ANXIETYFLG", T.IntegerType(), True),
#     T.StructField("ADHDFLG", T.IntegerType(), True),
#     T.StructField("CONDUCTFLG", T.IntegerType(), True),
#     T.StructField("DELIRDEMFLG", T.IntegerType(), True),
#     T.StructField("BIPOLARFLG", T.IntegerType(), True),
#     T.StructField("DEPRESSFLG", T.IntegerType(), True),
#     T.StructField("ODDFLG", T.IntegerType(), True),
#     T.StructField("PDDFLG", T.IntegerType(), True),
#     T.StructField("PERSONFLG", T.IntegerType(), True),
#     T.StructField("SCHIZOFLG", T.IntegerType(), True),
#     T.StructField("ALCSUBFLG", T.IntegerType(), True),
#     T.StructField("OTHERDISFLG", T.IntegerType(), True),
#     T.StructField("STATEFIP", T.IntegerType(), True),
#     T.StructField("DIVISION", T.IntegerType(), True),
#     T.StructField("REGION", T.IntegerType(), True),
#     T.StructField("CASEID", T.LongType(), True),
#     T.StructField("GENDER_mapped", T.StringType(), True),
#     T.StructField("RACE_mapped", T.StringType(), True),
#     T.StructField("MARSTAT_mapped", T.StringType(), True),
#     T.StructField("EMPLOY_mapped", T.StringType(), True),
#     T.StructField("ETHNIC_mapped", T.StringType(), True),
#     T.StructField("CASEID_int", T.IntegerType(), True),
# ])


@dlt.expect_all(
    {
        "valid_gender": "GENDER IS NOT NULL",
        "valid_race": "RACE IS NOT NULL",
        "valid_ethnic": "ETHNIC IS NOT NULL",
        "valid_marstat": "MARSTAT IS NOT NULL",
        "valid_employ": "EMPLOY IS NOT NULL",
    }
)
@dlt.table(name="gold")  # , schema=gold_schema)
def gold_layer():
    df = spark.read.table("silver_3").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_full")
    return df


# COMMAND ----------

# MAGIC %sql select * from gold_full

# COMMAND ----------

gold_schema = T.StructType(
    [
        T.StructField("YEAR", T.IntegerType(), True),
        T.StructField("AGE", T.IntegerType(), True),
        T.StructField("EDUC", T.IntegerType(), True),
        T.StructField("ETHNIC", T.IntegerType(), True),
        T.StructField("RACE", T.IntegerType(), True),
        T.StructField("GENDER", T.IntegerType(), True),
        T.StructField("SPHSERVICE", T.IntegerType(), True),
        T.StructField("CMPSERVICE", T.IntegerType(), True),
        T.StructField("OPISERVICE", T.IntegerType(), True),
        T.StructField("RTCSERVICE", T.IntegerType(), True),
        T.StructField("IJSSERVICE", T.IntegerType(), True),
        T.StructField("MH1", T.IntegerType(), True),
        T.StructField("MH2", T.IntegerType(), True),
        T.StructField("MH3", T.IntegerType(), True),
        T.StructField("SUB", T.IntegerType(), True),
        T.StructField("MARSTAT", T.IntegerType(), True),
        T.StructField("SMISED", T.IntegerType(), True),
        T.StructField("SAP", T.IntegerType(), True),
        T.StructField("EMPLOY", T.IntegerType(), True),
        T.StructField("DETNLF", T.IntegerType(), True),
        T.StructField("VETERAN", T.IntegerType(), True),
        T.StructField("LIVARAG", T.IntegerType(), True),
        T.StructField("NUMMHS", T.IntegerType(), True),
        T.StructField("TRAUSTREFLG", T.IntegerType(), True),
        T.StructField("ANXIETYFLG", T.IntegerType(), True),
        T.StructField("ADHDFLG", T.IntegerType(), True),
        T.StructField("CONDUCTFLG", T.IntegerType(), True),
        T.StructField("DELIRDEMFLG", T.IntegerType(), True),
        T.StructField("BIPOLARFLG", T.IntegerType(), True),
        T.StructField("DEPRESSFLG", T.IntegerType(), True),
        T.StructField("ODDFLG", T.IntegerType(), True),
        T.StructField("PDDFLG", T.IntegerType(), True),
        T.StructField("PERSONFLG", T.IntegerType(), True),
        T.StructField("SCHIZOFLG", T.IntegerType(), True),
        T.StructField("ALCSUBFLG", T.IntegerType(), True),
        T.StructField("OTHERDISFLG", T.IntegerType(), True),
        T.StructField("STATEFIP", T.IntegerType(), True),
        T.StructField("DIVISION", T.IntegerType(), True),
        T.StructField("REGION", T.IntegerType(), True),
        T.StructField("CASEID", T.LongType(), True),
        T.StructField("GENDER_mapped", T.StringType(), True),
        T.StructField("RACE_mapped", T.StringType(), True),
        T.StructField("MARSTAT_mapped", T.StringType(), True),
        T.StructField("EMPLOY_mapped", T.StringType(), True),
        T.StructField("ETHNIC_mapped", T.StringType(), True),
        T.StructField("CASEID_int", T.IntegerType(), True),
    ]
)


@dlt.expect_all(
    {
        "valid_gender": "GENDER IS NOT NULL",
        "valid_race": "RACE IS NOT NULL",
        "valid_ethnic": "ETHNIC IS NOT NULL",
        "valid_marstat": "MARSTAT IS NOT NULL",
        "valid_employ": "EMPLOY IS NOT NULL",
    }
)
@dlt.table(name="gold")
def gold_layer():
    df = spark.read.table("silver_train").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    df.printSchema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_train")

    return df


# COMMAND ----------

gold_schema = T.StructType(
    [
        T.StructField("YEAR", T.IntegerType(), True),
        T.StructField("AGE", T.IntegerType(), True),
        T.StructField("EDUC", T.IntegerType(), True),
        T.StructField("ETHNIC", T.IntegerType(), True),
        T.StructField("RACE", T.IntegerType(), True),
        T.StructField("GENDER", T.IntegerType(), True),
        T.StructField("SPHSERVICE", T.IntegerType(), True),
        T.StructField("CMPSERVICE", T.IntegerType(), True),
        T.StructField("OPISERVICE", T.IntegerType(), True),
        T.StructField("RTCSERVICE", T.IntegerType(), True),
        T.StructField("IJSSERVICE", T.IntegerType(), True),
        T.StructField("MH1", T.IntegerType(), True),
        T.StructField("MH2", T.IntegerType(), True),
        T.StructField("MH3", T.IntegerType(), True),
        T.StructField("SUB", T.IntegerType(), True),
        T.StructField("MARSTAT", T.IntegerType(), True),
        T.StructField("SMISED", T.IntegerType(), True),
        T.StructField("SAP", T.IntegerType(), True),
        T.StructField("EMPLOY", T.IntegerType(), True),
        T.StructField("DETNLF", T.IntegerType(), True),
        T.StructField("VETERAN", T.IntegerType(), True),
        T.StructField("LIVARAG", T.IntegerType(), True),
        T.StructField("NUMMHS", T.IntegerType(), True),
        T.StructField("TRAUSTREFLG", T.IntegerType(), True),
        T.StructField("ANXIETYFLG", T.IntegerType(), True),
        T.StructField("ADHDFLG", T.IntegerType(), True),
        T.StructField("CONDUCTFLG", T.IntegerType(), True),
        T.StructField("DELIRDEMFLG", T.IntegerType(), True),
        T.StructField("BIPOLARFLG", T.IntegerType(), True),
        T.StructField("DEPRESSFLG", T.IntegerType(), True),
        T.StructField("ODDFLG", T.IntegerType(), True),
        T.StructField("PDDFLG", T.IntegerType(), True),
        T.StructField("PERSONFLG", T.IntegerType(), True),
        T.StructField("SCHIZOFLG", T.IntegerType(), True),
        T.StructField("ALCSUBFLG", T.IntegerType(), True),
        T.StructField("OTHERDISFLG", T.IntegerType(), True),
        T.StructField("STATEFIP", T.IntegerType(), True),
        T.StructField("DIVISION", T.IntegerType(), True),
        T.StructField("REGION", T.IntegerType(), True),
        T.StructField("CASEID", T.LongType(), True),
        T.StructField("GENDER_mapped", T.StringType(), True),
        T.StructField("RACE_mapped", T.StringType(), True),
        T.StructField("MARSTAT_mapped", T.StringType(), True),
        T.StructField("EMPLOY_mapped", T.StringType(), True),
        T.StructField("ETHNIC_mapped", T.StringType(), True),
        T.StructField("CASEID_int", T.IntegerType(), True),
    ]
)


@dlt.expect_all(
    {
        "valid_gender": "GENDER IS NOT NULL",
        "valid_race": "RACE IS NOT NULL",
        "valid_ethnic": "ETHNIC IS NOT NULL",
        "valid_marstat": "MARSTAT IS NOT NULL",
        "valid_employ": "EMPLOY IS NOT NULL",
    }
)
@dlt.table(name="gold", schema=gold_schema)
def gold_layer():
    df = spark.read.table("silver_validation").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    assert df.schema == gold_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_validation")

    return df


# COMMAND ----------

gold_schema = T.StructType(
    [
        T.StructField("YEAR", T.IntegerType(), True),
        T.StructField("AGE", T.IntegerType(), True),
        T.StructField("EDUC", T.IntegerType(), True),
        T.StructField("ETHNIC", T.IntegerType(), True),
        T.StructField("RACE", T.IntegerType(), True),
        T.StructField("GENDER", T.IntegerType(), True),
        T.StructField("SPHSERVICE", T.IntegerType(), True),
        T.StructField("CMPSERVICE", T.IntegerType(), True),
        T.StructField("OPISERVICE", T.IntegerType(), True),
        T.StructField("RTCSERVICE", T.IntegerType(), True),
        T.StructField("IJSSERVICE", T.IntegerType(), True),
        T.StructField("MH1", T.IntegerType(), True),
        T.StructField("MH2", T.IntegerType(), True),
        T.StructField("MH3", T.IntegerType(), True),
        T.StructField("SUB", T.IntegerType(), True),
        T.StructField("MARSTAT", T.IntegerType(), True),
        T.StructField("SMISED", T.IntegerType(), True),
        T.StructField("SAP", T.IntegerType(), True),
        T.StructField("EMPLOY", T.IntegerType(), True),
        T.StructField("DETNLF", T.IntegerType(), True),
        T.StructField("VETERAN", T.IntegerType(), True),
        T.StructField("LIVARAG", T.IntegerType(), True),
        T.StructField("NUMMHS", T.IntegerType(), True),
        T.StructField("TRAUSTREFLG", T.IntegerType(), True),
        T.StructField("ANXIETYFLG", T.IntegerType(), True),
        T.StructField("ADHDFLG", T.IntegerType(), True),
        T.StructField("CONDUCTFLG", T.IntegerType(), True),
        T.StructField("DELIRDEMFLG", T.IntegerType(), True),
        T.StructField("BIPOLARFLG", T.IntegerType(), True),
        T.StructField("DEPRESSFLG", T.IntegerType(), True),
        T.StructField("ODDFLG", T.IntegerType(), True),
        T.StructField("PDDFLG", T.IntegerType(), True),
        T.StructField("PERSONFLG", T.IntegerType(), True),
        T.StructField("SCHIZOFLG", T.IntegerType(), True),
        T.StructField("ALCSUBFLG", T.IntegerType(), True),
        T.StructField("OTHERDISFLG", T.IntegerType(), True),
        T.StructField("STATEFIP", T.IntegerType(), True),
        T.StructField("DIVISION", T.IntegerType(), True),
        T.StructField("REGION", T.IntegerType(), True),
        T.StructField("CASEID", T.LongType(), True),
        T.StructField("GENDER_mapped", T.StringType(), True),
        T.StructField("RACE_mapped", T.StringType(), True),
        T.StructField("MARSTAT_mapped", T.StringType(), True),
        T.StructField("EMPLOY_mapped", T.StringType(), True),
        T.StructField("ETHNIC_mapped", T.StringType(), True),
        T.StructField("CASEID_int", T.IntegerType(), True),
    ]
)


@dlt.expect_all(
    {
        "valid_gender": "GENDER IS NOT NULL",
        "valid_race": "RACE IS NOT NULL",
        "valid_ethnic": "ETHNIC IS NOT NULL",
        "valid_marstat": "MARSTAT IS NOT NULL",
        "valid_employ": "EMPLOY IS NOT NULL",
    }
)
@dlt.table(name="gold", schema=gold_schema)
def gold_layer():
    df = spark.read.table("silver_test").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    assert df.schema == gold_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_test")

    return df


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_full;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_train LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_test LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM gold_validation LIMIT 10;

# COMMAND ----------
