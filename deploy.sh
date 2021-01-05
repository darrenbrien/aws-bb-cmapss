#!/bin/sh
set -e
if [[ -z "${STACK_NAME}" ]]; then
    printf "STACK_NAME env var not set" >&2
    exit 1
elif [[ -z "${S3_DEPLOY_BUCKET}" ]]; then
    printf "S3_DEPLOY_BUCKET env var not set" >&2
    exit 1
fi
cd templates
aws cloudformation package --template-file data-lake-master.yaml --s3-bucket $S3_DEPLOY_BUCKET --output-template-file packaged-template.yaml
aws cloudformation deploy --template-file packaged-template.yaml --stack-name $STACK_NAME --force-upload --capabilities CAPABILITY_IAM --parameter-overrides AvailabilityZones=us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f
export S3PATH=$(aws s3 ls --output table | cut -d ' ' -f3 | grep curated)
sed -i '' -e 's/datalake-curated-datasets-907317471167-us-east-1-gismq40/'"$S3PATH"'/g' ../src/glue_job.py
aws cloudformation package --template-file data-lake-master.yaml --s3-bucket $S3_DEPLOY_BUCKET --output-template-file packaged-template.yaml --force-upload
aws cloudformation deploy --template-file packaged-template.yaml --stack-name $STACK_NAME --force-upload --capabilities CAPABILITY_IAM --parameter-overrides AvailabilityZones=us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f
cd ..
