## Data Preparation

The dataset requires us to calculate RUL for each observation, a Spark glue job is a scalable way to enable this as our data is tabular we can leverage the dataframe abstraction Spark provides. Leveraging a Glue Workflow enables us to chain our glue crawlers and jobs into a DAG which make the pipeline easier to manage and ensures we minimize the time between new data arriving and insight being available to our users. The dataset for this project is small, however, this approach can scale to an arbitrarily large datasize. Glue is a serverless tool and so we only pay for the processing which is required by the dataset.

Our glue job takes the raw data and calculates the RUL value per observation. A constraint of the dataset is that until the final (failure) observation per unit number (engine) the RUL value cannot be calculated, we can't know how long an engine will last on cycle 1 until it subsequently fails. As such the notion of engine failure is implict to the way the data is structured, so for the purposes of this project data arrives in complete batches from the files provided.

To ensure ETL and ML work are decoupled we separate raw data from the processed data which can be used for ML or dashboard visulisations. Three databases are setup in the data catalog which map to 3 separate S3 locations. This enables us to control which IAM roles have access to the data, for example ensuring the Data science team can only access data which has been prepared appropriately.

### Workflow diagram
![](../images/workflow.png)
