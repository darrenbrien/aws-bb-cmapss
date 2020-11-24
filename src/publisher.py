import boto3

if __name__ == "__main__":
    sess = boto3.Session(region_name="us-east-1",)
    client = sess.client("firehose")

    filename = "train_FD001.txt"
    streamname = "streaming-submissions-9sge7xlg"
    with open(filename) as csvf:
        for row in csvf.readlines():
            client.put_record(DeliveryStreamName=streamname, Record={"Data": row})
