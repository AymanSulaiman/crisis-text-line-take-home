import dlt
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler, StandardScaler

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

@dlt.table(name="bronze")
def bronze_layer():
    raw_path = "dbfs:/path/to/your/data/*.csv"
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(raw_path)
        .na.drop()
    )
    return df

@dlt.table(name="silver")
def silver_layer():
    df = dlt.read("bronze").na.drop()
    numeric_types = [T.IntegerType]
    numeric_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, tuple(numeric_types))
    ]

    # Define UDFs for mappings
    gender_udf = F.udf(lambda x: gender.get(x, "Unknown"), T.StringType())
    race_udf = F.udf(lambda x: race.get(x, "Unknown"), T.StringType())
    marital_status_udf = F.udf(lambda x: marital_status.get(x, "Unknown"), T.StringType())
    employment_status_udf = F.udf(lambda x: employment_status.get(x, "Unknown"), T.StringType())
    ethnicity_udf = F.udf(lambda x: ethnicity.get(x, "Unknown"), T.StringType())

    # Apply UDFs
    df = df.withColumn("GENDER_mapped", gender_udf(F.col("GENDER")))
    df = df.withColumn("RACE_mapped", race_udf(F.col("RACE")))
    df = df.withColumn("MARSTAT_mapped", marital_status_udf(F.col("MARSTAT")))
    df = df.withColumn("EMPLOY_mapped", employment_status_udf(F.col("EMPLOY")))
    df = df.withColumn("ETHNIC_mapped", ethnicity_udf(F.col("ETHNIC")))

    stages = []

    # Index categorical columns
    for column in ["GENDER_mapped", "RACE_mapped", "MARSTAT_mapped", "EMPLOY_mapped", "ETHNIC_mapped"]:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}_index")
        stages.append(indexer)

    # Convert specific columns to float
    for col in numeric_columns:
        df = df.withColumn(col, F.col(col).cast(T.FloatType()))

    # VectorAssembler for numeric columns
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numeric_features")
    stages.append(assembler)

    # MinMaxScaler
    min_max_scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_features_min_max")
    stages.append(min_max_scaler)

    # StandardScaler
    standard_scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features_standard")
    stages.append(standard_scaler)

    # Create and fit pipeline
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    # Define a UDF to convert Vector to Array
    def vector_to_array(v):
        return v.toArray().tolist() if isinstance(v, DenseVector) else list(v)

    vector_to_array_udf = F.udf(vector_to_array, T.ArrayType(T.DoubleType()))

    # Apply the UDF to create a new array column
    df_transformed = df_transformed.withColumn("scaled_features_array", vector_to_array_udf(df_transformed["scaled_features_standard"]))

    # Use posexplode to split array into multiple rows with position and value
    exploded_df = df_transformed.selectExpr("CASEID", "posexplode(scaled_features_array) as (pos, value)")

    # Pivot the DataFrame to get individual columns for each position
    pivot_df = exploded_df.groupBy("CASEID").pivot("pos").agg(F.first("value"))

    return pivot_df

@dlt.table(name="gold")
def gold_layer():
    df = dlt.read("silver").na.drop()
    df = df.drop("demographic_strata").drop("strataIndex")
    return df
