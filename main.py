import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    MinMaxScaler,
    StandardScaler,
)
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
import yaml
import os

SEED = 42

config = yaml.safe_load(open("config.yaml", "r"))

spark = (
    SparkSession.builder.appName("Medallion Architecture")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.instances", "4")
    .config("spark.executor.cores", "4")
    .config("spark.executor.memoryOverhead", "4g")
    .config("spark.sql.shuffle.partitions", "800")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.2")
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4040")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.executor.heartbeatInterval", "30s")
    .config("spark.network.timeout", "600s")
    .config("spark.rpc.message.maxSize", "512")
    .config("spark.broadcast.compress", "true")
    .config("spark.sql.broadcastTimeout", "36000")
    .config("spark.cleaner.referenceTracking.blocking", "true")
    .config("spark.cleaner.referenceTracking.blocking.shuffleTimeout", "600s")
    .getOrCreate()
)

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


def bronze_layer(raw_input_path: str, bronze_sql_name: str) -> None:
    print("Reading raw data into the bronze layer...")
    df = spark.read.csv(raw_input_path, header=True, inferSchema=True).na.drop()
    df.repartition(
        "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    ).createOrReplaceTempView(bronze_sql_name)
    print(f"Bronze layer view '{bronze_sql_name}' created.")


def silver_layer(bronze_input_name: str, silver_sql_name: str) -> None:
    print("Transforming data from bronze to silver layer...")

    df = spark.sql(f"SELECT * FROM {bronze_input_name}").na.drop()

    # Select numeric columns
    numeric_types = [T.IntegerType]
    numeric_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, tuple(numeric_types))
    ]

    print(f"Numeric columns selected: {numeric_columns}")

    # Define UDFs for mappings
    gender_udf = F.udf(lambda x: gender.get(x, "Unknown"), T.StringType())
    race_udf = F.udf(lambda x: race.get(x, "Unknown"), T.StringType())
    marital_status_udf = F.udf(
        lambda x: marital_status.get(x, "Unknown"), T.StringType()
    )
    employment_status_udf = F.udf(
        lambda x: employment_status.get(x, "Unknown"), T.StringType()
    )
    ethnicity_udf = F.udf(lambda x: ethnicity.get(x, "Unknown"), T.StringType())

    # Apply UDFs
    df = df.withColumn("GENDER_mapped", gender_udf(F.col("GENDER")))
    df = df.withColumn("RACE_mapped", race_udf(F.col("RACE")))
    df = df.withColumn("MARSTAT_mapped", marital_status_udf(F.col("MARSTAT")))
    df = df.withColumn("EMPLOY_mapped", employment_status_udf(F.col("EMPLOY")))
    df = df.withColumn("ETHNIC_mapped", ethnicity_udf(F.col("ETHNIC")))

    stages = []

    # Index categorical columns
    for column in [
        "GENDER_mapped",
        "RACE_mapped",
        "MARSTAT_mapped",
        "EMPLOY_mapped",
        "ETHNIC_mapped",
    ]:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index")
        stages.append(indexer)

    # Convert CASEID to integer
    df = df.withColumn(
        "CASEID_int",
        F.lpad(F.regexp_replace(F.col("CASEID"), "2021", ""), 10, "0").cast(
            T.IntegerType()
        ),
    )

    # VectorAssembler for numeric columns
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numeric_features")
    stages.append(assembler)

    # MinMaxScaler
    min_max_scaler = MinMaxScaler(
        inputCol="numeric_features", outputCol="scaled_features_min_max"
    )
    stages.append(min_max_scaler)

    # StandardScaler
    standard_scaler = StandardScaler(
        inputCol="numeric_features", outputCol="scaled_features_standard"
    )
    stages.append(standard_scaler)

    # Create and fit pipeline
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)
    df_transformed.printSchema()

    # Define a UDF to convert Vector to Array
    def vector_to_array(v):
        return v.toArray().tolist() if isinstance(v, T.DenseVector) else list(v)

    vector_to_array_udf = F.udf(vector_to_array, T.ArrayType(T.DoubleType()))

    # Apply the UDF to create a new array column
    df_transformed = df_transformed.withColumn("CASEID_int", vector_to_array_udf(df_transformed["scaled_features_standard"]))

    # Show the resulting DataFrame
    df_transformed.select("scaled_features_standard", "CASEID_int").show(truncate=False)

    columns = ['CASEID_int', 'scaled_features_standard']

    df_transformed_v2 = df_transformed.select(columns)

    # Use posexplode to split array into multiple rows with position and value
    exploded_df = df_transformed_v2.selectExpr("CASEID_int", "posexplode(scaled_features_standard) as (pos, value)")

    # Pivot the DataFrame to get individual columns for each position
    pivot_df = exploded_df.groupBy("CASEID_int").pivot("pos").agg(F.first("value"))

    # Rename the columns
    for i, col in enumerate(numeric_columns):
        pivot_df = pivot_df.withColumnRenamed(str(i), col)

    # Show operation
    pivot_df.show()

    # # Remove temporary columns
    # df_transformed_final = df_transformed_standard.drop(
    #     "numeric_features", "scaled_features_min_max", "scaled_features_standard"
    # )

    # print("Splitting data into train, test, and validation sets...")
    # train, test = df_transformed_final.randomSplit([0.8, 0.2], seed=SEED)
    # train, validation = train.randomSplit([0.75, 0.25], seed=SEED)

    # print(f"Creating silver layer view '{silver_sql_name}'...")
    # df_transformed_final.repartition(
    #     "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    # ).createOrReplaceTempView(silver_sql_name)

    # print("Creating train, test, and validation views...")
    # train.repartition(
    #     "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    # ).createOrReplaceTempView(f"{silver_sql_name}_train")
    # test.repartition(
    #     "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    # ).createOrReplaceTempView(f"{silver_sql_name}_test")
    # validation.repartition(
    #     "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    # ).createOrReplaceTempView(f"{silver_sql_name}_validation")
    # print("Silver layer transformation complete.")


def gold_layer(
    silver_input_name: str,
    silver_train_name: str,
    silver_test_name: str,
    silver_validation_name: str,
    gold_sql_name: str,
) -> None:
    print("Transforming data from silver to gold layer...")
    df = spark.sql(f"SELECT * FROM {silver_input_name}").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    df.repartition(
        "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    ).createOrReplaceTempView(gold_sql_name)
    print(f"Gold layer view '{gold_sql_name}' created.")

    print("Creating train, test, and validation views...")
    train_df = spark.sql(f"SELECT * FROM {silver_train_name}").na.drop()
    train_df = train_df.drop("demographic_strata").drop("strataIndex")
    train_df.repartition(
        "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    ).createOrReplaceTempView(f"{gold_sql_name}_train")

    test_df = spark.sql(f"SELECT * FROM {silver_test_name}").na.drop()
    test_df = test_df.drop("demographic_strata").drop("strataIndex")
    test_df.repartition(
        "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    ).createOrReplaceTempView(f"{gold_sql_name}_test")

    validation_df = spark.sql(f"SELECT * FROM {silver_validation_name}").na.drop()
    validation_df = validation_df.drop("demographic_strata").drop("strataIndex")
    validation_df.repartition(
        "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    ).createOrReplaceTempView(f"{gold_sql_name}_validation")
    print("Gold layer transformation complete.")


def schema_validator(df: DataFrame, schema: T.StructType) -> bool:
    return df.schema == schema


def data_validator():
    pass


def main():
    raw_path = os.path.join("data", "*.csv")

    # Configured paths for each medallion layer
    bronze_sql_name = "bronze"
    silver_sql_name = "silver"
    gold_sql_name = "gold"

    silver_train_name = f"{silver_sql_name}_train"
    silver_test_name = f"{silver_sql_name}_test"
    silver_validation_name = f"{silver_sql_name}_validation"

    # Run the bronze layer processing
    print("Starting bronze layer processing...")
    bronze_layer(raw_input_path=raw_path, bronze_sql_name=bronze_sql_name)

    # Run the silver layer processing
    print("Starting silver layer processing...")
    silver_layer(bronze_input_name=bronze_sql_name, silver_sql_name=silver_sql_name)

    # Run the gold layer processing
    print("Starting gold layer processing...")
    gold_layer(
        silver_input_name=silver_sql_name,
        silver_train_name=silver_train_name,
        silver_test_name=silver_test_name,
        silver_validation_name=silver_validation_name,
        gold_sql_name=gold_sql_name,
    )

    # Queries to display a sample of the data at each layer
    bronze_sql = f"SELECT * FROM {bronze_sql_name} LIMIT 10"
    silver_sql = f"SELECT * FROM {silver_sql_name} LIMIT 10"
    gold_sql = f"SELECT * FROM {gold_sql_name} LIMIT 10"

    print("Displaying sample data from bronze layer...")
    spark.sql(bronze_sql).show(truncate=False)
    print("Displaying sample data from silver layer...")
    spark.sql(silver_sql).show(truncate=False)
    print("Displaying sample data from gold layer...")
    spark.sql(gold_sql).show(truncate=False)


if __name__ == "__main__":
    main()
    spark.stop()
