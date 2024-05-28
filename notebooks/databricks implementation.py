# Databricks notebook source
# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS bronze;
# MAGIC DROP TABLE IF EXISTS silver;
# MAGIC DROP TABLE IF EXISTS gold;
# MAGIC

# COMMAND ----------

import dlt

# from pyspark.sql import types as T

# bronze_schema = T.StructType(
#     [
#         T.StructField("YEAR", T.IntegerType(), True),
#         T.StructField("AGE", T.IntegerType(), True),
#         T.StructField("EDUC", T.IntegerType(), True),
#         T.StructField("SPHSERVICE", T.IntegerType(), True),
#         T.StructField("CMPSERVICE", T.IntegerType(), True),
#         T.StructField("OPISERVICE", T.IntegerType(), True),
#         T.StructField("RTCSERVICE", T.IntegerType(), True),
#         T.StructField("IJSSERVICE", T.IntegerType(), True),
#         T.StructField("MH1", T.IntegerType(), True),
#         T.StructField("MH2", T.IntegerType(), True),
#         T.StructField("MH3", T.IntegerType(), True),
#         T.StructField("SUB", T.IntegerType(), True),
#         T.StructField("SMISED", T.IntegerType(), True),
#         T.StructField("SAP", T.IntegerType(), True),
#         T.StructField("DETNLF", T.IntegerType(), True),
#         T.StructField("VETERAN", T.IntegerType(), True),
#         T.StructField("LIVARAG", T.IntegerType(), True),
#         T.StructField("NUMMHS", T.IntegerType(), True),
#         T.StructField("TRAUSTREFLG", T.IntegerType(), True),
#         T.StructField("ANXIETYFLG", T.IntegerType(), True),
#         T.StructField("ADHDFLG", T.IntegerType(), True),
#         T.StructField("CONDUCTFLG", T.IntegerType(), True),
#         T.StructField("DELIRDEMFLG", T.IntegerType(), True),
#         T.StructField("BIPOLARFLG", T.IntegerType(), True),
#         T.StructField("DEPRESSFLG", T.IntegerType(), True),
#         T.StructField("ODDFLG", T.IntegerType(), True),
#         T.StructField("PDDFLG", T.IntegerType(), True),
#         T.StructField("PERSONFLG", T.IntegerType(), True),
#         T.StructField("SCHIZOFLG", T.IntegerType(), True),
#         T.StructField("ALCSUBFLG", T.IntegerType(), True),
#         T.StructField("OTHERDISFLG", T.IntegerType(), True),
#         T.StructField("STATEFIP", T.IntegerType(), True),
#         T.StructField("DIVISION", T.IntegerType(), True),
#         T.StructField("REGION", T.IntegerType(), True),
#         T.StructField("CASEID", T.LongType(), True),
#         T.StructField("GENDER", T.IntegerType(), True),
#         T.StructField("RACE", T.IntegerType(), True),
#         T.StructField("ETHNIC", T.IntegerType(), True),
#         T.StructField("MARSTAT", T.IntegerType(), True),
#         T.StructField("EMPLOY", T.IntegerType(), True),
#     ]
# )

@dlt.table(
    name="bronze", 
    # schema=bronze_schema,
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def bronze_layer():
    # raw_path = "gs://crisis-text-line/mhcld_puf_2021.csv"
    raw_path = "gs://crisis-text-line/mhcld_puf_2021.csv"
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(raw_path)
        .na.drop()
    )
    
    df.write.format("delta").mode("overwrite").saveAsTable("bronze")
    
    return df

# COMMAND ----------

# MAGIC %sql select * from bronze LIMIT 10

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import StringIndexer

@dlt.table(
    name="silver_data_type_conversion",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def silver_data_type_conversion():
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

    df.write.format("delta").mode("overwrite").saveAsTable("silver_data_type_conversion")
    return df


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.ml import Pipeline

@dlt.table(
    name="silver_normalization_standardization",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def silver_normalization_standardization():
    df = dlt.read("silver_data_type_conversion")

    numeric_columns = [
        "NUMMHS",
    ]

    # Assemble and scale numeric columns
    stages = []
    for col in numeric_columns:
        assembler = VectorAssembler(inputCols=[col], outputCol=f"{col}_vec")
        min_max_scaler = MinMaxScaler(inputCol=f"{col}_vec", outputCol=f"{col}_normalized_vec")
        standard_scaler = StandardScaler(inputCol=f"{col}_vec", outputCol=f"{col}_standardized_vec", withMean=True, withStd=True)
        stages.extend([assembler, min_max_scaler, standard_scaler])

    df = Pipeline(stages=stages).fit(df).transform(df)

    extract_first_element = F.udf(lambda x: float(x[0]), T.FloatType())
    for col in numeric_columns:
        df = df.withColumn(f"{col}_normalized", extract_first_element(F.col(f"{col}_normalized_vec")))
        df = df.withColumn(f"{col}_standardized", extract_first_element(F.col(f"{col}_standardized_vec")))
        df = df.drop(f"{col}_vec").drop(f"{col}_normalized_vec").drop(f"{col}_standardized_vec")

    df = df.na.drop()
    df.write.format("delta").mode("overwrite").saveAsTable("silver_normalization_standardization")
    return df


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

@dlt.table(
    name="silver_data_partitioning_sampling", 
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
)
def silver_data_partitioning_sampling():
    df = dlt.read("silver_normalization_standardization")

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
    df.write.format("delta").mode("overwrite").saveAsTable("silver_data_partitioning_sampling")
    return df


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_full LIMIT 10;
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

@dlt.table(name="gold")
def gold_layer():
    df = dlt.read("silver").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")

    # Split the data into train, test, and validation sets
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train, validation = train.randomSplit([0.75, 0.25], seed=SEED)

    # Write the DataFrame to the gold output path
    df.write.format("delta").mode("overwrite").saveAsTable("gold_full")

    # Write train, test, and validation sets to their respective paths
    train.write.format("delta").mode("overwrite").saveAsTable("gold_train")
    test.write.format("delta").mode("overwrite").saveAsTable("gold_test")
    validation.write.format("delta").mode("overwrite").saveAsTable("gold_validation")

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
