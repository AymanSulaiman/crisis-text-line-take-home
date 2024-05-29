# README

Table of Contents

1. Introduction
2. Resources
3. Challenge Scenario
4. Preparation
5. Challenge Tasks
    - Reading the Data
    - Medallion Architecture Implementation
        - Bronze Layer
        - Silver Layer
        - Gold Layer
    - SQL Queries
6. Transformation Exercise for Senior Data Engineer
    - Data Type Conversion
    - Data Normalization and Standardization
    - Data Partitioning and Sampling
7. Documentation and Presentation
8. How to Run and Test

Introduction
This project demonstrates the implementation of the Medallion Architecture using the MH-CLD-2021-DS0001 SAMHSA dataset with PySpark. The goal is to showcase data engineering practices focusing on robustness and scalability.

Resources

- DataBricks Free Trial
- Dataset: MH-CLD-2021-DS0001 SAMHSA dataset
- Dataset Source: MH-CLD-2021-DS0001-bndl-data-csv_v1.zip
- Data Dictionary: Dataset Documentation (link-to-dataset-documentation)

Challenge Scenario
As a senior data engineer, my task is to process and transform the given dataset using PySpark, implementing the Medallion Architecture to ensure data quality, robustness, and scalability.

Preparation

1. Start up a DataBricks workspace.
2. Use the provided dataset: MH-CLD-2021-DS0001-bndl-data-csv_v1.zip.
3. Review the dataset documentation.

Challenge Tasks

Reading the Data

1. Use PySpark to read the and transform the dataset (CSV/Parquet).
2. Explore and understand the schema and data types.
3. Implement schema checks to ensure data integrity.

Medallion Architecture Implementation

Insert a diagram of the Medallion Archtecture here.

Bronze Layer

1. Create a bronze table to store raw data.
2. Perform basic data validation and cleaning.
3. Partition the bronze table for improved query performance.
4. Discuss logging and exception handling for the bronze layer.

Silver Layer

1. Create a silver table to store transformed and enriched data.
2. Perform complex transformations that are the following.

    Data Type Conversion

    1. Convert categorical variables to appropriate types.
    2. Ensure CASEID is stored as an integer.
    3. Convert numeric variables to float types.
    4. Validate data types for all variables.
    5. Summarize data type conversions.

    Data Normalization and Standardization
    1. Normalize numeric variables using min-max scaling.
    2. Standardize numeric variables using z-score normalization.
    3. Store normalized and standardized variables in new columns.
    4. Summarize techniques and rationale.

    Data Partitioning and Sampling
    1. Split dataset into training and testing sets based on demographic variables.
    2. Implement stratified sampling for training and testing sets.
    3. Create a validation set from the training set.
    4. Store dataset splits with appropriate naming conventions.
    5. Provide a detailed report on partitioning and sampling process.

3. Partition the silver table for improved query performance.
4. Implement schema checks for transformed data.
5. Discuss monitoring for the silver layer.

Gold Layer

1. Create a gold table for final, curated data.
2. Perform additional transformations for business-ready datasets.
3. Partition the gold table for improved query performance.
4. Implement schema checks for final data.
5. Discuss monitoring for the gold layer.

Transformation Exercise for Senior Data Engineer
Data Type Conversion

How to Run and Test

1. Set up your databricks enviornment.
2. Git clone this repo.
3. Run the `main.py`, the pipeline should run and the data should be saved as Delta Live Tables.
