# AWS CMAPSS ML Lab  

A lab to demo AWS data and machine learning services

## Contents

* Templates contains CloudFormation scripts to deploy the necessary infrastructure for this lab to an AWS environment 
  * tested mostly on us-east-1, should work on all major regions with services like AWS glue and Sagemaker
* During the lab we'll use the scripts and notebooks in src
* This lab uses the nasa turbo fan failure dataset see [data/readme.txt](here) for a description. We'll be using this dataset to make predictions for the remaining cycles an engine has until it fails.

## Getting started

* Deploy the CloudFormation stack in the templates folder
  * This deployment will incur some modest charges while deployed
  * It'll be easy to delete all the services as the end

1. Create an s3 bucket we can use for our deployment in the us-east-1 region
use the console or from the CLI
`aws s3 mb s3://aws-bb-darren-predictive-maintenance --region us-east-1`
2. We'll now use this bucket to host our deployment files for cloudformation
use the cloud formation in the console or
`aws cloudformation package --template-file templates/data-lake-master.yaml --s3-bucket aws-bb-darren-predictive-maintenance-capstone --output-template-file packaged-template.yaml --force-upload --region us-east-1`
then
`aws cloudformation deploy --template-file file://templates/packaged-template.yaml --stack-name aws-bb-cmapss --parameter-overrides AvailabilityZones=us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f --region us-east-1 --capabilities CAPABILITY_IAM`
be sure to change the bucket to the bucket you created during 1. 
3. The deployment will take about 5 minutes

## So what did we just do?

* Deploy a VPC
* Create some S3 buckets for our raw, etl and processed data
* Create some glue resources which we'll use to perform our ETL
* Created various IAM roles which can control parts of our infrastructure based on the least access principle.
* Created a SageMaker notebook to do some data analysis and submit SageMaker training jobs from
* Created a Cloud9 deployment we can use as a terminal environment.



`cd SageMaker`

`git clone https://github.com/darrenbrien/aws-bb-cmapss.git`

`python 'src/publisher.py data/train_*'



