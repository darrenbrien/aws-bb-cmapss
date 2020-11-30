import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)
## @type: DataSource
## @args: [database = "datalake-submissions", table_name = "cmapss_streaming_submissions", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(
    database="datalake-submissions",
    table_name="cmapss_streaming_submissions",
    transformation_ctx="datasource0",
)
## @type: ApplyMapping
## @args: [mapping = [("col0", "long", "unit_number", "long"), ("col1", "long", "cycle", "long"), ("col2", "double", "op_1", "double"), ("col3", "double", "op_2", "double"), ("col4", "double", "op_3", "double"), ("col5", "double", "col5", "double"), ("col6", "double", "col6", "double"), ("col7", "double", "col7", "double"), ("col8", "double", "col8", "double"), ("col9", "double", "col9", "double"), ("col10", "double", "col10", "double"), ("col11", "double", "col11", "double"), ("col12", "double", "col12", "double"), ("col13", "double", "col13", "double"), ("col14", "double", "col14", "double"), ("col15", "double", "col15", "double"), ("col16", "double", "col16", "double"), ("col17", "double", "col17", "double"), ("col18", "double", "col18", "double"), ("col19", "double", "col19", "double"), ("col20", "double", "col20", "double"), ("col21", "long", "col21", "long"), ("col22", "long", "col22", "long"), ("col23", "double", "col23", "double"), ("col24", "double", "col24", "double"), ("col25", "double", "col25", "double"), ("partition_0", "string", "year", "string"), ("partition_1", "string", "month", "string"), ("partition_2", "string", "day", "string"), ("partition_3", "string", "hour", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(
    frame=datasource0,
    mappings=[
        ("col0", "long", "unit_number", "long"),
        ("col1", "long", "cycle", "long"),
        ("col2", "double", "op_1", "double"),
        ("col3", "double", "op_2", "double"),
        ("col4", "double", "op_3", "double"),
        ("col5", "double", "col5", "double"),
        ("col6", "double", "col6", "double"),
        ("col7", "double", "col7", "double"),
        ("col8", "double", "col8", "double"),
        ("col9", "double", "col9", "double"),
        ("col10", "double", "col10", "double"),
        ("col11", "double", "col11", "double"),
        ("col12", "double", "col12", "double"),
        ("col13", "double", "col13", "double"),
        ("col14", "double", "col14", "double"),
        ("col15", "double", "col15", "double"),
        ("col16", "double", "col16", "double"),
        ("col17", "double", "col17", "double"),
        ("col18", "double", "col18", "double"),
        ("col19", "double", "col19", "double"),
        ("col20", "double", "col20", "double"),
        ("col21", "long", "col21", "long"),
        ("col22", "long", "col22", "long"),
        ("col23", "double", "col23", "double"),
        ("col24", "double", "col24", "double"),
        ("col25", "double", "col25", "double"),
        ("partition_0", "string", "year", "string"),
        ("partition_1", "string", "month", "string"),
        ("partition_2", "string", "day", "string"),
        ("partition_3", "string", "hour", "string"),
    ],
    transformation_ctx="applymapping1",
)
## @type: ResolveChoice
## @args: [choice = "make_struct", transformation_ctx = "resolvechoice2"]
## @return: resolvechoice2
## @inputs: [frame = applymapping1]

resolvechoice2 = ResolveChoice.apply(
    frame=applymapping1, choice="make_struct", transformation_ctx="resolvechoice2"
)
## @type: DropNullFields
## @args: [transformation_ctx = "dropnullfields3"]
## @return: dropnullfields3
## @inputs: [frame = resolvechoice2]
dropnullfields3 = DropNullFields.apply(
    frame=resolvechoice2, transformation_ctx="dropnullfields3"
)

df = dropnullfields3.toDF()
results_df = (
    df.groupBy("unit_number")
    .max("cycle")
    .withColumnRenamed("max(cycle)", "failure_cycle")
)

joined = results_df.join(df, ["unit_number"])

rul_df = joined.withColumn("failure_cycle", joined["failure_cycle"] - joined["cycle"])

final = DynamicFrame.fromDF(rul_df, glueContext, "final")

glueContext.write_dynamic_frame.from_catalog(
    connection_type="s3", namespace="datalake-curated-datasets", table_name="cmapss-rul"
)

job.commit()
