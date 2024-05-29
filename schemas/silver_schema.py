from pyspark.sql import types as T

silver_1_schema = T.StructType(
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

silver_2_schema = T.StructType(
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
        T.StructField("NUMMHS_normalized", T.FloatType(), True),
        T.StructField("NUMMHS_standardized", T.FloatType(), True),
    ]
)

silver_3_schema = T.StructType(
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
        T.StructField("NUMMHS_normalized", T.FloatType(), True),
        T.StructField("NUMMHS_standardized", T.FloatType(), True),
        T.StructField("demographic_strata", T.StringType(), True),
        T.StructField("strataIndex", T.DoubleType(), True),
    ]
)
