
# AWS Blackbelt Capstone Predictive maintenance project writeup 

## Intro
<details>
    <Summary>Click to expand</summary>

The Nasa Turbofan dataset is interesting to use for a machine learning project for a couple of reasons.
* Predictive maintenance has traditionally leveraged classical statistics to provide insights. Survival Analysis would be a common approach, however the many observations and exogenous variables in this dataset provide an opportunity to apply machine learning to discover more subtle patterns in the data.
* This data isn't provided with a clear set of labels to train a model, it potentially lends itself to either regression or classification. 
  * Given the number of observations, framing this as a regression problem makes sense as we are able to incorporate more information into our loss function as a result of the continuous (discrete) remaining useful life versus a binary observation. 
  * A classification model would have severe class imbalance as few observation result in engine failure. 
  * Furthermore framing the problem in this way enables the consumer of the predictions to instigate a "no suprises" policy where maintenance is actively performed on engines likely to have a fault in the near future e.g. all engines with less than 50 RUL will have maintenance performed. Assuming planned maintenance is cheaper to operate than reactive maintenance, *"a stitch in time saves 9"*. 

With the above in mind our data pipeline will need to calculate a remaining useful life (RUL) for each observation.

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=RUL_{u, i}=\max_{j=0}^n cycle_{u,j} - cycle_{u,i}">
</p>

The remaining useful life for unit number (u) on cycle (i) is the maximum cycle observed in the dataset for that unique unit number minus the current cycle number, i.
</details>

## Outline

This write-up covers the following sections, based on the project rubric.
1. [Data Ingestion and Transformation](writeup/ingestion.md)
2. [Data Preparation](writeup/data_preparation.md)
3. [Data Visualisation](writeup/visualisation.md)
4. [Training Models](writeup/training_models.md)
5. [Evaluate ML Models](writeup/evaluating_models.md)
6. [Improving ML models accuracy](writeup/improving_models.md)
7. [Machine Learning Implementation / Operations & Well-Architected](writeup/ml_ops.md)
