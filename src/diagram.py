from diagrams import Cluster, Diagram
from diagrams.aws.storage import S3
from diagrams.aws.iot import IotRule
from diagrams.aws.compute import Lambda
from diagrams.aws.network import Endpoint
from diagrams.aws.analytics import (
    Athena,
    Glue,
    GlueCrawlers,
    GlueDataCatalog,
    KinesisDataFirehose,
    Kinesis,
    EMR,
)
from diagrams.aws.ml import SagemakerNotebook, SagemakerTrainingJob, SagemakerModel

with Diagram("AWS ML Lab", show=False):
    iot = IotRule("Engine devices")
    inference = Kinesis("real-time")
    source = KinesisDataFirehose("batches")

    with Cluster("VPC"):
        with Cluster("Training"):
            submissions = S3("Submissions")
            curated = S3("CuratedData")

            submissions_crawler = GlueCrawlers("submissions crawler")
            curated_crawler = Glue("ETL")

            ctas = Athena("train/eval split")

            catalog = GlueDataCatalog("data catalog")

            notebooks = SagemakerNotebook("Build Model")
            job = SagemakerTrainingJob("Train Model")
            model = SagemakerModel("Fitted Model")

        with Cluster("Inference"):

            endpointLambda = Lambda("call endpoint")
            with Cluster("Multi AZ endpoints") as az:
                endpoints = [
                    Endpoint("us-east-1a"),
                    Endpoint("us-east-1b"),
                    Endpoint("us-east-1c"),
                ]

            published = S3("Monitor data")
            monitor_sched = EMR("model monitor")

    source >> submissions >> submissions_crawler >> curated_crawler >> curated
    submissions >> catalog
    iot >> inference >> source

    curated >> ctas >> [catalog, job]

    notebooks >> job >> model

    model >> endpoints >> published >> monitor_sched

    inference >> endpointLambda >> endpoints
    monitor_sched >> job
