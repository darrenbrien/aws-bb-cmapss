# AWS CMAPSS ML Lab  

A lab to demo AWS data and machine learning services

## Agenda

1. Intros
2. Login to event engine
3. Discuss architecture
4. Work through lab
5. Questions
6. Wrap up, next steps

## So what have we deployed

* VPC - A secure network for our resources to run where we can control access from the public internet
* Created some S3 buckets for our raw, etl and processed data
* Created a Cloud9 deployment we can use as a terminal environment, a handy terminal enviroment to run scripts.
* Created some glue resources which we'll use to perform our ETL
* Created various IAM roles which can control parts of our infrastructure based on the least access principle.
* Created a SageMaker notebook to do some data analysis and submit SageMaker training jobs from

### Important

This lab has been deployed to the `us-east-1` region (North Virginia), make sure you're access resources from this region!

## Architecture
![alt text](https://github.com/darrenbrien/aws-bb-cmapss/blob/master/src/aws_ml_lab.png "Logo Title Text 1")

## Lab Exercise 

### Publish data onto our delivery stream

* This will just be a simple python script, but imagine we're collecting this data in realtime over IOT core, publishing the data onto kinesis stream and using firehose to buffer the data onto S3 for us!
* Go to "Cloud9" in the AWS console page
* Open the only Cloud9 instance (a micro EC2 instance)
* We need to install the python boto3 package

`sudo pip install boto3`

* Next we need to clone this git repo so we can use the resources

`git clone https://github.com/darrenbrien/aws-bb-cmapss.git`

* Publish some of our data onto a delivery stream

`cd aws-bb-cmapss`

`python src/publisher.py 'data/train_*'  <- quotes matter or you'll just publish a single file :( 

### Navigate into the S3 area of the console

1. Review the contents and structure of the submissions bucket

2. We can see our data has arrived, we haven't really done much with it yet, just simulate a realtime (ish) data arrival 

3. AWS glue is a serverless ETL service built on top of Apache Spark. Glue is great for working with arbitraryily large datasets as it support parallelism and dividing work amongst workers.
  * Before we can perform etl we need to know what shape (columns, types, partitions) the data has, this is what a glue Crawler can do for us

Navigate to AWS Glue in the Console, click "Crawlers", take a look at the cmapss_submissions crawler

The sole job of a glue crawler is to discover the metadata associated with data repository.

Tick the crawler and select run, this should take ~1 minute.

4. Our crawler should have discovered some of the metadata of our csv data which firehose has saved to S3 for us

* Note we haven't looked at this data yet, our crawler has discovered the partitions and column structure of the data

5. Now we know a little about the submissions data, we can perform some ETL to make it usable

### Our Challenge

* We're collecting data in realtime about some engine tests that are going on
* We collect data until the engine "fails" and requires maintenance
* Maybe Nasa can get back to the Moon if they can do a better job of predicting when engines are close to failing
* That way they could perform maintenance before the engine fails, keeping it online more of the time!
* Today we'll collect this data, structure it so that we can run a machine learning model over the data and show what our predictions are for engines we haven't seen fail yet!

1. Navigate to the jobs section of the glue console. 
2. Open the script tab, we'll each need to make some changes specific to our own AWS environment.
3. Go back to the Cloud9 window and open the glue_job.py file (should be the same), we need to edit it to reference your s3 bucket in your environment with the "*curated*" pattern
4. Go to the Cloud9 terminal

`aws s3 ls`

You should have a bucket with curated in the name `datalake-curated-dataset-123456789-us-east-1-qwertyu`

5. Copy this bucket name and paste it over the similarly named s3 bucket in the glue_job.py file
6. Now we need to make this file available for AWS Glue, to do this copy lets first create a new bucket

`aws s3 mb s3://aws-glue-$(openssl rand -hex 5)`

### Important your bucket must start with the aws-glue prefix or bad things will happen

7. Back in the glue console click Action => Save As

Enter the `s3://<the-bucket-you-just-created`

Click save

8. Finally Selection run-job

3. Hit the Edit Script button
Run the 
`git clone https://github.com/darrenbrien/aws-bb-cmapss.git`

`python 'src/publisher.py data/train_*'



