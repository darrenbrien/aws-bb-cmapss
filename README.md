# AWS Blackbelt CMAPPSS dataset ML Capstone project

See (writeup)[writeup.md] for detailed breakdown of project delivery.

## Getting started

* An S3 bucket where resources can be deployed using Cloudformation
```
aws s3 mb SOME_BUCKET
```

* Given appropriate env variables deploy.sh will deploy the required infrastructure
```
export S3_DEPLOY_BUCKET=SOME_BUCKET
export STACK_NAME=stack-name
```

* Deploy the infrastructure in the templates folder
```
./deploy.sh
```

* Src has glue jobs and notebooks used to complete project.
* src/publisher.py will write data into firehose to get the pipeline started
```
pip install boto3
python publisher.py 'data/train_*'
```

* The deployed glue workflow can then be executed to start data preparation.
* Pull this repo to the sagemaker notebook instance to run the notebooks.

