import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.context import GlueContext
from awsglue.job import Job

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

datasource0 = glueContext.create_dynamic_frame.from_catalog(
    database="datalake_submissions",
    table_name="streaming_submissions",
    transformation_ctx="datasource0",
)

apply_mapping = ApplyMapping.apply(
    frame=datasource0,
    mappings=[
        ("col0", "long", "unit_number", "long"),
        ("col1", "long", "cycle", "long"),
        ("col2", "double", "op_1", "double"),
        ("col3", "double", "op_2", "double"),
        ("col4", "double", "op_3", "double"),
        ("col5", "double", "sensor_measurement_1", "double"),
        ("col6", "double", "sensor_measurement_2", "double"),
        ("col7", "double", "sensor_measurement_3", "double"),
        ("col8", "double", "sensor_measurement_4", "double"),
        ("col9", "double", "sensor_measurement_5", "double"),
        ("col10", "double", "sensor_measurement_6", "double"),
        ("col11", "double", "sensor_measurement_7", "double"),
        ("col12", "double", "sensor_measurement_8", "double"),
        ("col13", "double", "sensor_measurement_9", "double"),
        ("col14", "double", "sensor_measurement_10", "double"),
        ("col15", "double", "sensor_measurement_11", "double"),
        ("col16", "double", "sensor_measurement_12", "double"),
        ("col17", "double", "sensor_measurement_13", "double"),
        ("col18", "double", "sensor_measurement_14", "double"),
        ("col19", "double", "sensor_measurement_15", "double"),
        ("col20", "double", "sensor_measurement_16", "double"),
        ("col21", "long", "sensor_measurement_17", "long"),
        ("col22", "long", "sensor_measurement_18", "long"),
        ("col23", "double", "sensor_measurement_19", "double"),
        ("col24", "double", "sensor_measurement_20", "double"),
        ("col25", "double", "sensor_measurement_21", "double"),
        ("partition_0", "string", "year", "string"),
        ("partition_1", "string", "month", "string"),
        ("partition_2", "string", "day", "string"),
        ("partition_3", "string", "hour", "string"),
    ],
    transformation_ctx="apply_mapping",
)

df = apply_mapping.toDF()
results_df = (
    df.groupBy("unit_number")
    .max("cycle")
    .withColumnRenamed("max(cycle)", "failure_cycle")
)

joined = results_df.join(df, ["unit_number"])

rul_df = joined.withColumn("failure_cycle", joined["failure_cycle"] - joined["cycle"])

final = DynamicFrame.fromDF(rul_df, glueContext, "final").repartition(1)

finished = glueContext.write_dynamic_frame.from_options(
    frame=final,
    connection_type="s3",
    connection_options={
        "path": "s3://datalake-curated-datasets-907317471167-us-east-1-qkg9331",
        "partitionKeys": ["year", "month", "day", "hour"],
    },
    format="glueparquet",
    transformation_ctx="finished",
)

job.commit()
