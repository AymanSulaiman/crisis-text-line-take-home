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


@dlt.expect_all_or_drop(bronze_validations)
@dlt.table(
    name="bronze",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=bronze_schema,
)
def bronze_layer() -> DataFrame:
    """Load and process raw data into the bronze layer.

    Returns:
        DataFrame: Processed bronze layer DataFrame.
    """
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


@dlt.expect_all_or_drop(silver_1_validation)
@dlt.table(
    name="silver_1",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=silver_1_schema,
)
def silver_1() -> DataFrame:
    """Process bronze data to create the first silver layer.

    Returns:
        DataFrame: Processed silver layer DataFrame.
    """
    df = spark.read.table("bronze")

    def create_map_udf(mapping_dict):
        """Create a UDF for mapping dictionary values.

        Args:
            mapping_dict (dict): Mapping dictionary.

        Returns:
            function: UDF for mapping dictionary values.
        """
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
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == silver_1_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_1")
    return df


@dlt.expect_all_or_drop(silver_2_validation)
@dlt.table(
    name="silver_2",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=silver_2_schema,
)
def silver_2() -> DataFrame:
    """Process silver_1 data to create the second silver layer.

    Returns:
        DataFrame: Processed silver layer DataFrame with normalized and standardized columns.
    """
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
    assert df.schema == silver_2_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_2")
    return df


@dlt.expect_all_or_drop(silver_3_validation)
@dlt.table(
    name="silver_3",
    partition_cols=["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"],
    schema=silver_3_schema,
)
def silver_3() -> DataFrame:
    """Process silver_2 data to create the third silver layer.

    Returns:
        DataFrame: Processed silver layer DataFrame with demographic strata and splits for training, testing, and validation.
    """
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
    assert df.schema == silver_3_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("silver_3_full")
    return df


@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_full", schema=gold_schema)
def gold_full_layer() -> DataFrame:
    """Process silver_3_full data to create the full gold layer.

    Returns:
        DataFrame: Processed gold layer DataFrame.
    """
    df = spark.read.table("silver_3_full").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == gold_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_full")
    return df


@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_train", schema=gold_schema)
def gold_train_layer() -> DataFrame:
    """Process silver_train data to create the training gold layer.

    Returns:
        DataFrame: Processed gold layer DataFrame for training.
    """
    df = spark.read.table("silver_train").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    assert df.schema == gold_schema
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_train")

    return df


@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_validation", schema=gold_schema)
def gold_validation_layer() -> DataFrame:
    """Process silver_validation data to create the validation gold layer.

    Returns:
        DataFrame: Processed gold layer DataFrame for validation.
    """
    df = spark.read.table("silver_validation").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    assert df.schema == gold_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_validation")

    return df


@dlt.expect_all_or_drop(gold_validation)
@dlt.table(name="gold_test", schema=gold_schema)
def gold_test_layer() -> DataFrame:
    """Process silver_test data to create the testing gold layer.

    Returns:
        DataFrame: Processed gold layer DataFrame for testing.
    """
    df = spark.read.table("silver_test").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    assert df.schema == gold_schema
    partition_columns = ["GENDER", "RACE", "ETHNIC", "MARSTAT", "EMPLOY"]
    df.write.format("delta").mode("overwrite").partitionBy(
        partition_columns
    ).saveAsTable("gold_test")

    return df
