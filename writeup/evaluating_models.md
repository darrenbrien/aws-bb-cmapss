## Evaluate ML Models

* As we have formulated our project as a regression problem we use mean squared and a root mean squared error as a validation metric for hyper parameter turning

### Sagemaker Console hyper param jobs screens

Having run our hyper parameter optimization job we obtain a model which achieves 44.7 RMSE on the evaluation set.
We review the logs and observe the model appears to have converged, as train and test performance are similar we're confident we're not overfitting. 

![](../images/hyper_param_jobs.png)

![](../images/hyper_training_loss.png)
<br/>

Reading the best performing model from S3 enables us to evaluate its predictions and consider any issues.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
```


```python
!aws s3 cp s3://datalake-published-data-907317471167-us-east-1-pjkrtzr/sagemaker/cmapss-xgboost/hyper-xgboost/hyper-cmapss-2021-01-07-16-04-50-409-07f98126/output/model.tar.gz .

!tar xvzf model.tar.gz
```


```python
import pickle as pkl
with open("xgboost-model", "rb") as f:
    booster = pkl.load(f)   
```

### Residual plot of evaluation data

For early cycles are errors a high, we're predicting further in to the future and so have more uncertainty in the remaining cycles for any engine.

```python
fig, ax = plt.subplots(figsize=(30, 12))
ax.plot(residuals)
_ = ax.set_title('residual plot by filename, engine number')
```
    
![png](../images/evaluate_model_13_0.png)

### Distribution of errors by cycle and file for evaluation data

* Each point is a prediction error, coloured by the training file which it came from.
* As above our error is heteroskedastic across cycles.

```python
from matplotlib.colors import ListedColormap as lcm
labels, _ = df.loc[y_test.index, 'filename'].pipe(pd.factorize)
colours = ['red','orange','blue','green']
```



```python
fig, ax = plt.subplots(figsize=(30, 12))
ax.scatter(df.loc[y_test.index, 'cycle'], residuals, c=labels, cmap=lcm(colours), s=5, alpha=.5)
```
![png](../images/evaluate_model_15_1.png)

### Evaluate model against test data (unseen data for training or hyperparam optimization)

* We've held out our test data to gain a real understanding of how our model performs.

```python
! cat /home/ec2-user/SageMaker/aws-bb-cmapss/data/test_FD001.txt | cut -d ' ' -f2- > cmapss.test.1
! cat /home/ec2-user/SageMaker/aws-bb-cmapss/data/test_FD002.txt | cut -d ' ' -f2- > cmapss.test.2
! cat /home/ec2-user/SageMaker/aws-bb-cmapss/data/test_FD003.txt | cut -d ' ' -f2- > cmapss.test.3
! cat /home/ec2-user/SageMaker/aws-bb-cmapss/data/test_FD004.txt | cut -d ' ' -f2- > cmapss.test.4
```


```python
all_test_data = []
for i in range(1, 5):
    filename = f'cmapss.test.{i}'
    test_file_name = f'test_FD00{i}.txt'
    test_rul_name = f'RUL_FD00{i}.txt'
    test_data = pd.read_csv(f"/home/ec2-user/SageMaker/aws-bb-cmapss/data/{test_file_name}", header=None, delimiter=' ')

    labels = pd.read_csv(f"/home/ec2-user/SageMaker/aws-bb-cmapss/data/{test_rul_name}", names=['remaining_cycles'])
    labels.index += 1
    labels = labels.reset_index()
    labels = labels.rename(columns={'index' : 0})
    labels = test_data.groupby(0)[1].max().reset_index().merge(labels, left_on=0, right_on=0)
    labels['max_cycles'] = labels[1] + labels['remaining_cycles']

    test_data = test_data.merge(labels[[0, 'max_cycles']], left_on=0, right_on=0)

    test_data['RUL'] = test_data['max_cycles'] - test_data[1]
    test_data['filename'] = filename
    all_test_data.append(test_data)

all_test_data_df = pd.concat(all_test_data)
```


```python
residual_test = all_test_data_df.RUL.values - booster.predict(xgb.DMatrix(all_test_data_df.drop(columns=[0, 26, 27, 'max_cycles', 'RUL', 'filename']).values))
```

Our model performance is lower on this dataset, lets investigate why this is.
```python
mean_squared_error(all_test_data_df.RUL.values, booster.predict(xgb.DMatrix(all_test_data_df.drop(columns=[0, 26, 27, 'max_cycles', 'RUL', 'filename']).values)), squared=False)
```




    57.709556087846885


### Residuals on test data set (out of sample)

```python
fig, ax = plt.subplots(figsize=(30, 12))
ax.plot(residual_test)
_ = ax.set_title('residual plot by filename, engine number')
```
    
![png](../images/evaluate_model_22_0.png)


### Futher comparisons of the data reveal a fact which we are aware of and which explains the reason for higher errors on the test set
* the test files (all_test_data_df) are truncated at some point prior to the engine failure, where are the train data (x_test) are upto and including the final cycle.
Considering the distribution of errors on the scatter above this provides fewer low error (late cycle) prediction in the test data which would pull down our average error rate.
Looking at the distribution of cycle across each of the files we can see the train data has larger dispersion.
* the training data upper quartile starts at [152, 155, 193.75 and 180] cycles
* the test data upper quartile starts at [113, 119, 149 and 155] cycles
* This difference between the train/eval and test datasets could well explain the difference in model performance.


```python
x_test.groupby(['filename']).cycle.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>filename</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train_FD001.txt</th>
      <td>6608.0</td>
      <td>106.867736</td>
      <td>69.710392</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>101.0</td>
      <td>152.00</td>
      <td>362.0</td>
    </tr>
    <tr>
      <th>train_FD002.txt</th>
      <td>17506.0</td>
      <td>106.889181</td>
      <td>66.943381</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>102.0</td>
      <td>155.00</td>
      <td>340.0</td>
    </tr>
    <tr>
      <th>train_FD003.txt</th>
      <td>8098.0</td>
      <td>142.325142</td>
      <td>103.378154</td>
      <td>1.0</td>
      <td>63.0</td>
      <td>125.0</td>
      <td>193.75</td>
      <td>494.0</td>
    </tr>
    <tr>
      <th>train_FD004.txt</th>
      <td>19454.0</td>
      <td>126.601213</td>
      <td>83.612747</td>
      <td>1.0</td>
      <td>59.0</td>
      <td>117.0</td>
      <td>180.00</td>
      <td>417.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_test_data_df.groupby(['filename'])[1].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>filename</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cmapss.test.1</th>
      <td>13096.0</td>
      <td>76.836515</td>
      <td>53.057749</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>69.0</td>
      <td>113.0</td>
      <td>303.0</td>
    </tr>
    <tr>
      <th>cmapss.test.2</th>
      <td>33991.0</td>
      <td>81.223647</td>
      <td>58.892845</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>70.0</td>
      <td>119.0</td>
      <td>367.0</td>
    </tr>
    <tr>
      <th>cmapss.test.3</th>
      <td>16596.0</td>
      <td>105.999518</td>
      <td>83.286900</td>
      <td>1.0</td>
      <td>42.0</td>
      <td>87.0</td>
      <td>149.0</td>
      <td>475.0</td>
    </tr>
    <tr>
      <th>cmapss.test.4</th>
      <td>41214.0</td>
      <td>108.739094</td>
      <td>83.717459</td>
      <td>1.0</td>
      <td>43.0</td>
      <td>91.0</td>
      <td>155.0</td>
      <td>486.0</td>
    </tr>
  </tbody>
</table>
</div>


### Confirming explanation above 
A quick way to confirm our hypothesis for the discrepancies in eval vs test performance is to truncate the eval data from the train files similarly to the test files and see what our error rate looks like.

We truncate the evaluation data at 142 which is the median of the test dataset. We could be smarter and bootstrap the test set distribution, this comparison provides enough confidence we've explained the difference for a toy problem.


```python
all_test_data_df.groupby(['filename', 0])[1].max().describe()
```




    count    707.000000
    mean     148.369165
    std       78.471162
    min       19.000000
    25%       88.500000
    50%      142.000000
    75%      187.000000
    max      486.000000
    Name: 1, dtype: float64




```python
filt = x_test.cycle < 142
```


```python
mean_squared_error(y_test[filt].values, 
                   booster.predict(xgb.DMatrix(x_test.loc[filt, features].values)), 
                   squared=False)
```




    51.15290596835779

We could do more investigation however I'm comfortable that we've explained a significant difference is not down to our model overfitting but rather distributional differences between eval and test sets

