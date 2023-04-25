# Ex-06-Feature-Transformation

## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file

## CODE
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.isnull().sum()

df.describe()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
## OUTPUT:
## DATASET

![1](https://user-images.githubusercontent.com/118707079/233910238-757b6a97-9633-424b-8927-217abf81a7e2.png)

## ISNULL:
![2](https://user-images.githubusercontent.com/118707079/233911000-81401c74-fa10-464e-a0da-2ad18c8e5449.png)

## INFO:

![3](https://user-images.githubusercontent.com/118707079/233911222-05db806c-4717-45bf-bb90-7a317a319942.png)

## DESCRIBE:

![4](https://user-images.githubusercontent.com/118707079/233911325-948c1a11-737a-44e0-ae63-d03a03252480.png)

## HIGHLY POSITIVE SKEW:

![5](https://user-images.githubusercontent.com/118707079/233911425-60579b2c-0c79-4373-ad43-4e91b5745b8b.png)

## HIGHLY NEGATIVE SKEW:

![6](https://user-images.githubusercontent.com/118707079/233911992-f9358531-f24c-483e-8834-aadf070e16d7.png)

## MODERATE POSITIVE SKEW:

![7](https://user-images.githubusercontent.com/118707079/233912067-a4eea3b4-66d0-45cf-afc2-ed1f55c53a0e.png)

## MODERATE NEGATIVE SKEW:

![8](https://user-images.githubusercontent.com/118707079/233912782-2390b04c-cde3-4512-b7ab-8a9c9d40fb5b.png)

## LOG OF MODERATE POSITIVE SKEW::

![9](https://user-images.githubusercontent.com/118707079/233912902-7616bf7b-f2e2-45ad-bfd3-09d9f1ddbdba.png)

## LOG OF HIGHLY POSITIVE SKEW:

![10](https://user-images.githubusercontent.com/118707079/233913371-baf94e80-8519-4ecc-80c7-54ba079c4202.png)

## RECIPROCAL OF HIGHLY POSITIVE SKEW:

![11](https://user-images.githubusercontent.com/118707079/233913495-22761ef8-0ea9-4be6-ae8c-45146685a5d1.png)

## SQUARE ROOT TRANSFORMATION:

![12](https://user-images.githubusercontent.com/118707079/233913769-01b14ecd-9234-4693-9d7c-18066937c19d.png)

## POWER TRANSFORMATION OF MODERATE NEGATIVE SKEW:

![13](https://user-images.githubusercontent.com/118707079/233913887-6211cd3c-3aba-4b63-91cb-52711ec1437a.png)

## QUANTILE TRANSFORMATION:

![14](https://user-images.githubusercontent.com/118707079/233914057-c3e9ceed-b4cd-44d4-b0ac-ef87c33b5183.png)

## RESULT:

Thus, Feature transformation is performed and executed successfully for the given dataset












