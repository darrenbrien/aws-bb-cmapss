CREATE TABLE "datalake_curated_datasets"."cmaps_rul_train_validation"
WITH ( format = 'TEXTFILE', field_delimiter = ',', external_location = 's3://datalake-published-data-907317471167-us-east-1-qkg9331/cmaps-ml2', partitioned_by = ARRAY['split']) AS
SELECT failure_cycle,
         cycle,
         op_1,
         op_2,
         op_3,
         sensor_measurement_1 ,
         sensor_measurement_2 ,
         sensor_measurement_3 ,
         sensor_measurement_4 ,
         sensor_measurement_5 ,
         sensor_measurement_6 ,
         sensor_measurement_7 ,
         sensor_measurement_8 ,
         sensor_measurement_9 ,
         sensor_measurement_10 ,
         sensor_measurement_11 ,
         sensor_measurement_12 ,
         sensor_measurement_13 ,
         sensor_measurement_14 ,
         sensor_measurement_15 ,
         sensor_measurement_16 ,
         sensor_measurement_17 ,
         sensor_measurement_18 ,
         sensor_measurement_19 ,
         sensor_measurement_20,
         sensor_measurement_21,  
         CASE unit_number % 3
           WHEN 0 THEN 'validation'
           ELSE 'train'
         END AS split
FROM "datalake_curated_datasets"."datalake_curated_datasets_907317471167_us_east_1_qkg9331"
