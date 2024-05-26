from pyspark.sql import types as T

silver_schema = T.StructType(
    [
        T.StructField("YEAR", T.IntegerType(), True),
        T.StructField("AGE", T.IntegerType(), True),
        T.StructField("EDUC", T.IntegerType(), True),
        T.StructField("SPHSERVICE", T.IntegerType(), True),
        T.StructField("CMPSERVICE", T.IntegerType(), True),
        T.StructField("OPISERVICE", T.IntegerType(), True),
        T.StructField("RTCSERVICE", T.IntegerType(), True),
        T.StructField("IJSSERVICE", T.IntegerType(), True),
        T.StructField("MH1", T.IntegerType(), True),
        T.StructField("MH2", T.IntegerType(), True),
        T.StructField("MH3", T.IntegerType(), True),
        T.StructField("SUB", T.IntegerType(), True),
        T.StructField("SMISED", T.IntegerType(), True),
        T.StructField("SAP", T.IntegerType(), True),
        T.StructField("DETNLF", T.IntegerType(), True),
        T.StructField("VETERAN", T.IntegerType(), True),
        T.StructField("LIVARAG", T.IntegerType(), True),
        T.StructField("NUMMHS", T.FloatType(), True),
        T.StructField("TRAUSTREFLG", T.FloatType(), True),
        T.StructField("ANXIETYFLG", T.FloatType(), True),
        T.StructField("ADHDFLG", T.FloatType(), True),
        T.StructField("CONDUCTFLG", T.FloatType(), True),
        T.StructField("DELIRDEMFLG", T.FloatType(), True),
        T.StructField("BIPOLARFLG", T.FloatType(), True),
        T.StructField("DEPRESSFLG", T.FloatType(), True),
        T.StructField("ODDFLG", T.FloatType(), True),
        T.StructField("PDDFLG", T.FloatType(), True),
        T.StructField("PERSONFLG", T.FloatType(), True),
        T.StructField("SCHIZOFLG", T.FloatType(), True),
        T.StructField("ALCSUBFLG", T.FloatType(), True),
        T.StructField("OTHERDISFLG", T.FloatType(), True),
        T.StructField("STATEFIP", T.IntegerType(), True),
        T.StructField("DIVISION", T.IntegerType(), True),
        T.StructField("REGION", T.IntegerType(), True),
        T.StructField("CASEID", T.LongType(), True),
        T.StructField("GENDER", T.IntegerType(), True),
        T.StructField("RACE", T.IntegerType(), True),
        T.StructField("ETHNIC", T.IntegerType(), True),
        T.StructField("MARSTAT", T.IntegerType(), True),
        T.StructField("EMPLOY", T.IntegerType(), True),
        T.StructField("GENDER_mapped", T.StringType(), True),
        T.StructField("RACE_mapped", T.StringType(), True),
        T.StructField("MARSTAT_mapped", T.StringType(), True),
        T.StructField("EMPLOY_mapped", T.StringType(), True),
        T.StructField("ETHNIC_mapped", T.StringType(), True),
        T.StructField("CASEID_int", T.IntegerType(), True),
        T.StructField("NUMMHS_normalized", T.FloatType(), True),
        T.StructField("TRAUSTREFLG_normalized", T.FloatType(), True),
        T.StructField("ANXIETYFLG_normalized", T.FloatType(), True),
        T.StructField("ADHDFLG_normalized", T.FloatType(), True),
        T.StructField("CONDUCTFLG_normalized", T.FloatType(), True),
        T.StructField("DELIRDEMFLG_normalized", T.FloatType(), True),
        T.StructField("BIPOLARFLG_normalized", T.FloatType(), True),
        T.StructField("DEPRESSFLG_normalized", T.FloatType(), True),
        T.StructField("ODDFLG_normalized", T.FloatType(), True),
        T.StructField("PDDFLG_normalized", T.FloatType(), True),
        T.StructField("PERSONFLG_normalized", T.FloatType(), True),
        T.StructField("SCHIZOFLG_normalized", T.FloatType(), True),
        T.StructField("ALCSUBFLG_normalized", T.FloatType(), True),
        T.StructField("OTHERDISFLG_normalized", T.FloatType(), True),
        T.StructField("NUMMHS_standardized", T.FloatType(), True),
        T.StructField("TRAUSTREFLG_standardized", T.FloatType(), True),
        T.StructField("ANXIETYFLG_standardized", T.FloatType(), True),
        T.StructField("ADHDFLG_standardized", T.FloatType(), True),
        T.StructField("CONDUCTFLG_standardized", T.FloatType(), True),
        T.StructField("DELIRDEMFLG_standardized", T.FloatType(), True),
        T.StructField("BIPOLARFLG_standardized", T.FloatType(), True),
        T.StructField("DEPRESSFLG_standardized", T.FloatType(), True),
        T.StructField("ODDFLG_standardized", T.FloatType(), True),
        T.StructField("PDDFLG_standardized", T.FloatType(), True),
        T.StructField("PERSONFLG_standardized", T.FloatType(), True),
        T.StructField("SCHIZOFLG_standardized", T.FloatType(), True),
        T.StructField("ALCSUBFLG_standardized", T.FloatType(), True),
        T.StructField("OTHERDISFLG_standardized", T.FloatType(), True),
        T.StructField("demographic_strata", T.StringType(), False),
        T.StructField("strataIndex", T.DoubleType(), False),
    ]
)
