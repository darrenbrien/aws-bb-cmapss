import sys
from os.path import basename
import glob
import boto3

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
                batch.append({"Data": basename(filename) + " " + row})
                if len(batch) >= batchsize:
                    client.put_record_batch(
                        DeliveryStreamName=streamname, Records=batch
                    )
                    batch.clear()
            else:
                client.put_record_batch(DeliveryStreamName=streamname, Records=batch)
                print(
                    f"Published {i} records to {streamname} from {filename} in batches of {batchsize}"
                )
