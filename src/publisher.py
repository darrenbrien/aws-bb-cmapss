import sys
import time
from os.path import basename
import glob
import boto3


def publish_batch(client, streamname, batch):
    response = client.put_record_batch(DeliveryStreamName=streamname, Records=batch)
    if response["FailedPutCount"] > 0:
        print(f"{response['FailedPutCount']} failed, retrying")
        for i, obj in enumerate(response["RequestResponses"]):
            if "RecordId" in obj:
                del batch[i]
        time.sleep(1)
        publish_batch(client, streamname, batch)
    else:
        batch.clear()


if __name__ == "__main__":
    sess = boto3.Session(region_name="us-east-1",)
    client = sess.client("firehose")

    batchsize = 500
    res = client.list_delivery_streams()
    streamname, *b = [
        i for i in res["DeliveryStreamNames"] if i.startswith("streaming_submissions-")
    ]
    batch = []
    for filename in glob.glob(sys.argv[1]):
        with open(filename) as csvf:
            for i, row in enumerate(csvf.readlines()):
                if len(batch) >= batchsize:
                    publish_batch(client, streamname, batch)
                batch.append({"Data": basename(filename) + " " + row})
            else:
                publish_batch(client, streamname, batch)
                print(
                    f"Published {i + 1} records to {streamname} from {filename} in batches of {batchsize}"
                )
