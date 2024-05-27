import pyspark
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
# Multiprocessing library for cores

# .config("spark.memory.fraction", "0.8")
# .config("spark.memory.storageFraction", "0.3")
spark = (
    SparkSession.builder.appName("Medallion Architecture")
    .config("spark.executor.memory", "12g")
    .config("spark.driver.memory", "12g")
    .config("spark.executor.instances", "1")
    .config("spark.executor.cores", "8")
    .config("spark.executor.memoryOverhead", "2g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.2")
    .config("spark.ui.enabled", "true")
    .config("spark.ui.port", "4040")
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


def bronze_layer(
    raw_input_path: str, bronze_output_path: str, bronze_sql_name: str
) -> None:
    df = spark.read.csv(raw_input_path, header=True, inferSchema=True).na.drop()

    df.createOrReplaceTempView(bronze_sql_name)

    df.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(bronze_output_path, compression="snappy")


def silver_layer(
    bronze_input_path: str,
    silver_output_path: str,
    silver_sql_name: str,
    train_path: str,
    test_path: str,
    validation_path: str,
) -> None:
    def create_map_udf(mapping_dict):
        return F.udf(lambda key: mapping_dict.get(key, "Unknown"), T.StringType())

    race_map_udf = create_map_udf(race)
    gender_map_udf = create_map_udf(gender)
    marital_status_map_udf = create_map_udf(marital_status)
    employment_status_map_udf = create_map_udf(employment_status)
    ethnicity_map_udf = create_map_udf(ethnicity)

    df = spark.read.parquet(bronze_input_path).na.drop()

    print("mapping")

    df = df.withColumn("GENDER_mapped", gender_map_udf("GENDER"))
    df = df.withColumn("RACE_mapped", race_map_udf("RACE"))
    df = df.withColumn("MARSTAT_mapped", marital_status_map_udf("MARSTAT"))
    df = df.withColumn("EMPLOY_mapped", employment_status_map_udf("EMPLOY"))
    df = df.withColumn("ETHNIC_mapped", ethnicity_map_udf("ETHNIC"))

    print("finish mapping")

    print("case to int")
    df = df.withColumn(
        "CASEID_int",
        F.substring(F.col("CASEID").cast("string"), 5, 10).cast(T.IntegerType()),
    )

    print("finish case to int")

    numeric_columns = [
        "NUMMHS",
        "TRAUSTREFLG",
        "ANXIETYFLG",
        "ADHDFLG",
        "CONDUCTFLG",
        "DELIRDEMFLG",
        "BIPOLARFLG",
        "DEPRESSFLG",
        "ODDFLG",
        "PDDFLG",
        "PERSONFLG",
        "SCHIZOFLG",
        "ALCSUBFLG",
        "OTHERDISFLG",
    ]

    print("to numeric")

    for column in numeric_columns:
        df = df.withColumn(column, F.col(column).cast(T.FloatType()))

    print("finish numeric")

    cols_to_scale = numeric_columns

    extract_first_element = F.udf(lambda x: float(x[0]), T.FloatType())

    print("scaling")

    for col in cols_to_scale:
        assembler = VectorAssembler(inputCols=[col], outputCol=f"{col}_vec")
        scaler = MinMaxScaler(inputCol=f"{col}_vec", outputCol=f"{col}_normalized_vec")
        df = assembler.transform(df)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        df = df.withColumn(
            f"{col}_normalized", extract_first_element(F.col(f"{col}_normalized_vec"))
        )
        df = df.drop(f"{col}_vec").drop(f"{col}_normalized_vec")

    print("finish scaling")

    print("scaling")
    for col in cols_to_scale:
        assembler = VectorAssembler(inputCols=[col], outputCol=f"{col}_vec")
        scaler = StandardScaler(
            inputCol=f"{col}_vec",
            outputCol=f"{col}_standardized_vec",
            withMean=True,
            withStd=True,
        )
        df = assembler.transform(df)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        df = df.withColumn(
            f"{col}_standardized",
            extract_first_element(F.col(f"{col}_standardized_vec")),
        )
        df = df.drop(f"{col}_vec").drop(f"{col}_standardized_vec")
    print("finish scaling")

    df = df.na.drop()

    # Create demographic_strata column
    print("demographic_strata column")
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
    print("finish demographic_strata column")

    # Index the demographic_strata column
    print("indexer")
    indexer = StringIndexer(inputCol="demographic_strata", outputCol="strataIndex")
    df = indexer.fit(df).transform(df)
    print("finish indexer")

    # Split the data into train, test, and validation sets
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train, validation = train.randomSplit([0.75, 0.25], seed=SEED)

    # Create or replace the temporary view
    df.createOrReplaceTempView(silver_sql_name)

    # Write the DataFrame to the silver output path
    df.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(silver_output_path, compression="snappy")

    # Write train, test, and validation sets to their respective paths
    train.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(train_path, compression="snappy")

    test.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(test_path, compression="snappy")

    validation.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(validation_path, compression="snappy")

    train.createOrReplaceTempView(f"{silver_sql_name}_train")
    test.createOrReplaceTempView(f"{silver_sql_name}_test")
    validation.createOrReplaceTempView(f"{silver_sql_name}_validation")


def gold_layer(
    silver_input_path: str,
    silver_train_path: str,
    silver_test_path: str,
    silver_validation_path: str,
    gold_output_path: str,
    gold_train_path: str,
    gold_test_path: str,
    gold_validation_path: str,
    gold_sql_name: str,
) -> None:
    df = spark.read.parquet(silver_input_path).na.drop()
    df = df.drop("demographic_strata")
    df = df.drop("strataIndex")
    df.createOrReplaceTempView(gold_sql_name)
    df.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(gold_output_path, compression="snappy")

    train_df = spark.read.parquet(silver_train_path).na.drop()
    train_df = train_df.drop("demographic_strata")
    train_df = train_df.drop("strataIndex")
    train_df.createOrReplaceTempView(f"{gold_sql_name}_train")
    train_df.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(gold_train_path, compression="snappy")

    test_df = spark.read.parquet(silver_test_path).na.drop()
    test_df = df.drop("demographic_strata")
    test_df = df.drop("strataIndex")
    test_df.createOrReplaceTempView(f"{gold_sql_name}_test")
    test_df.write.partitionBy("GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY").mode(
        "overwrite"
    ).parquet(gold_test_path, compression="snappy")

    validation_df = spark.read.parquet(silver_validation_path).na.drop()
    validation_df = df.drop("demographic_strata")
    validation_df = df.drop("strataIndex")
    validation_df.createOrReplaceTempView(f"{gold_sql_name}_validation")
    validation_df.write.partitionBy(
        "GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"
    ).mode("overwrite").parquet(gold_validation_path, compression="snappy")


def schema_validator(df: DataFrame, schema: T.StructType) -> bool:
    return df.schema == schema


def data_validator():
    pass


def main():
    raw_path = os.path.join("data", "*.csv")
    bronze_path = os.path.join("data", "medallion_layers_main", "bronze")
    silver_path = os.path.join("data", "medallion_layers_main", "silver", "full")
    silver_train_path = os.path.join("data", "medallion_layers_main", "silver", "train")
    silver_test_path = os.path.join("data", "medallion_layers_main", "silver", "test")
    silver_validation_path = os.path.join(
        "data", "medallion_layers_main", "silver", "validation"
    )
    gold_path = os.path.join("data", "medallion_layers_main", "gold", "full")
    gold_train_path = os.path.join("data", "medallion_layers_main", "gold", "train")
    gold_test_path = os.path.join("data", "medallion_layers_main", "gold", "test")
    gold_validation_path = os.path.join(
        "data", "medallion_layers_main", "gold", "validation"
    )

    bronze_layer(
        raw_input_path=raw_path,
        bronze_output_path=bronze_path,
        bronze_sql_name="bronze",
    )

    silver_layer(
        bronze_input_path=bronze_path,
        silver_output_path=silver_path,
        silver_sql_name="silver",
        train_path=silver_train_path,
        test_path=silver_test_path,
        validation_path=silver_validation_path,
    )

    gold_layer(
        silver_input_path=silver_path,
        silver_train_path=silver_train_path,
        silver_test_path=silver_test_path,
        silver_validation_path=silver_validation_path,
        gold_output_path=gold_path,
        gold_train_path=gold_train_path,
        gold_test_path=gold_test_path,
        gold_validation_path=gold_validation_path,
        gold_sql_name="gold",
    )

    bronze_sql = """SELECT * FROM bronze LIMIT 10"""
    silver_sql = """SELECT * FROM silver LIMIT 10"""
    gold_sql = """SELECT * FROM gold LIMIT 10"""

    spark.sql(bronze_sql).show()
    spark.sql(silver_sql).show()
    spark.sql(gold_sql).show()


if __name__ == "__main__":
    main()
    spark.stop()
