## Data Ingestion and Transformation
<details>
    <Summary>Click to expand</summary>

Given the multiple engines contained in the dataset one could envisage a scenario where device data arrived in real-time as cycles were completed. For this reason I decided to encorporate a setup using kinesis firehose to buffer data as it arrive and to write the data out to S3 in batches. In a production solution a customer could leverage AWS IOT Core to publish data from devices in the cloud and use an IOT Rule to forward data onto a kinesis data stream. Our firehose solution could easily consume from this stream, in the interests of expediency this project simulates the above using a simple python script running on a Cloud9 instance to publish the data.

Once data arrives in S3 a glue crawler is configured to discover metadata and add it to the dataset to the glue data catalog, this enables data preparation jobs to run against the discovered schema. The crawler could at a frequency as low as 5 minutes, enabling our system to provide soft real-time experience for users of the system. Our glue jobs take the raw csv data and calculate the RUL value per observation.

A constraint of the dataset is that until the final (failure) observation per unit number (engine) the RUL value cannot be calculated, we can't know how long an engine will last on cycle 1 until it subsequently fails. As such the notion of engine failure is implict to the way the data is structured, so for the purposes of this project data arrives in complete batches from the files provided.  

### Project architecture diagram

![png](../images/aws_ml_lab.png)

</details>

