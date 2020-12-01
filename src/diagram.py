from diagrams import Cluster, Diagram
from diagrams.aws.compute import ECS, Lambda
from diagrams.aws.storage import S3
from diagrams.aws.analytics import (
    Athena,
    Glue,
    GlueCrawlers,
    GlueDataCatalog,
    KinesisDataFirehose,
)
from diagrams.aws.network import VPC
from diagrams.aws.ml import SagemakerNotebook, SagemakerTrainingJob, SagemakerModel

with Diagram("AWS ML Lab", show=False):
    source = KinesisDataFirehose("TurboFan engine data")

    with Cluster("VPC"):
        submissions = S3("Submissions")
        curated = S3("CuratedData")
        published = S3("PublishedData")

        submissions_crawler = GlueCrawlers("submissions crawler")
        curated_crawler = Glue("ETL")

        ctas = Athena("train/eval split")

        catalog = GlueDataCatalog("data catalog")

        notebooks = SagemakerNotebook("Build Model")
        job = SagemakerTrainingJob("Train Model")
        model = SagemakerModel("Fitted Model")

    source >> submissions >> submissions_crawler >> curated_crawler >> curated
    submissions >> catalog

    curated >> ctas >> [catalog, published]

    [published, notebooks] >> job >> model
