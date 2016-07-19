---
priority: 0.6
title: Project 3
excerpt: An Investigation of Iowa Liquor Sales 2015-2016
categories: works
background-image: workspace.png
tags:
  - Linear Regression
  - Cross-Validation
  - Modeling
---

#### Scenario 2: Market Research for New Store Locations

- A liquor store owner in Iowa is looking to expand to new locations and has hired you to investigate the market data for potential new locations. The business owner is interested in the details of the best model you can fit to the data so that his team can evaluate potential locations for a new storefront.

Goal for Scenario #2: Your task is to:

- Build models of total sales based on location, price per bottle, total bottles sold. You may find it useful to build models for each county, zip code, or city.
- Provide a table of the best performing stores by location type of your choice (city, county, or zip code) and the predictions of your model(s).
- Based on your models and the table of data, recommend some general locations to the business owner, taking into account model performance. Validate your model's performance and ability to predict future sales using cross-validation.
- Bonus: Recommend targets for volume sold and price per bottle!

#### Basics: Load, Read, Clean
```python
import pandas as pd
import numpy as np

%matplotlib inline
from matplotlib import pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import statsmodels.api as sm
from sklearn import linear_model
from sklearn import metrics

from IPython.display import Image
from IPython.display import display
import seaborn as sns
'''

## Load the data into a DataFrame
df = pd.read_csv('/Users/JHYL/DSI-HK-1/projects/project-03/assets/Iowa_Liquor_sales_sample_10pct.csv')

df.head()
```

```python
df.describe()
```

```python
# Margin
df.dtypes
df['State Bottle Retail'].astype(float)
df['State Bottle Cost'].astype(float)

def diff(x, y):
    return x - y
df['Margin'] = diff(df['State Bottle Retail'].astype(float), df['State Bottle Cost'].astype(float))

#Price Per Liter
df['Price per Liter'] = df['Sale (Dollars)'].astype(float)/df['Volume Sold (Liters)']

#Price Per Bottle
df['Price per Bottle'] = df['Sale (Dollars)'].astype(float)/df['Bottles Sold']
df.head()

# Convert to numeric
df['Margin'] = df['Margin'].astype(float) 
df['Price per Liter'] = df['Price per Liter'].astype(float)
df.dropna(inplace=True)
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store Number</th>
      <th>County Number</th>
      <th>Category</th>
      <th>Vendor Number</th>
      <th>Item Number</th>
      <th>Bottle Volume (ml)</th>
      <th>Bottles Sold</th>
      <th>Sale (Dollars)</th>
      <th>Volume Sold (Liters)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>270955.000000</td>
      <td>269878.000000</td>
      <td>2.708870e+05</td>
      <td>270955.00000</td>
      <td>270955.000000</td>
      <td>270955.000000</td>
      <td>270955.000000</td>
      <td>270955.000000</td>
      <td>270955.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3590.263701</td>
      <td>57.231642</td>
      <td>1.043888e+06</td>
      <td>256.43443</td>
      <td>45974.963300</td>
      <td>924.830341</td>
      <td>9.871285</td>
      <td>128.902375</td>
      <td>8.981351</td>
    </tr>
    <tr>
      <th>std</th>
      <td>947.662050</td>
      <td>27.341205</td>
      <td>5.018211e+04</td>
      <td>141.01489</td>
      <td>52757.043086</td>
      <td>493.088489</td>
      <td>24.040912</td>
      <td>383.027369</td>
      <td>28.913690</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2106.000000</td>
      <td>1.000000</td>
      <td>1.011100e+06</td>
      <td>10.00000</td>
      <td>168.000000</td>
      <td>50.000000</td>
      <td>1.000000</td>
      <td>1.340000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2604.000000</td>
      <td>31.000000</td>
      <td>1.012200e+06</td>
      <td>115.00000</td>
      <td>26827.000000</td>
      <td>750.000000</td>
      <td>2.000000</td>
      <td>30.450000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3722.000000</td>
      <td>62.000000</td>
      <td>1.031200e+06</td>
      <td>260.00000</td>
      <td>38176.000000</td>
      <td>750.000000</td>
      <td>6.000000</td>
      <td>70.560000</td>
      <td>5.250000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4378.000000</td>
      <td>77.000000</td>
      <td>1.062310e+06</td>
      <td>380.00000</td>
      <td>64573.000000</td>
      <td>1000.000000</td>
      <td>12.000000</td>
      <td>135.000000</td>
      <td>10.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9023.000000</td>
      <td>99.000000</td>
      <td>1.701100e+06</td>
      <td>978.00000</td>
      <td>995507.000000</td>
      <td>6000.000000</td>
      <td>2508.000000</td>
      <td>36392.400000</td>
      <td>2508.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Modeling total sales based on location, price per bottle, total bottles sold:
```python
feature_cols = ['County Number', 'Price per Bottle', 'Bottles Sold']
X = df[['County Number', 'Price per Bottle', 'Bottles Sold']]
X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>County Number</th>
      <th>Price per Bottle</th>
      <th>Bottles Sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.0</td>
      <td>6.75</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>82.0</td>
      <td>20.63</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>18.89</td>
      <td>24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85.0</td>
      <td>14.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>99.0</td>
      <td>10.80</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
print type(X)
print X.shape
```

    <class 'pandas.core.frame.DataFrame'>
    (269258, 3)



```python
y = df['Sale (Dollars)']
y.head()
```




    0     81.00
    1     41.26
    2    453.36
    3     85.50
    4    129.60
    Name: Sale (Dollars), dtype: float64




```python
print type(y)
print y.shape
```

    <class 'pandas.core.series.Series'>
    (269258,)


### Splitting X and y into training and testing sets


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
```

    (201943, 3)
    (201943,)
    (67315, 3)
    (67315,)



```python
linreg= LinearRegression()
linreg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print linreg.intercept_
print linreg.coef_
```

    -105.56815407
    [  8.08048602e-03   6.78321177e+00   1.36198210e+01]



```python
zip(feature_cols, linreg.coef_)
```




    [('County Number', 0.0080804860228047126),
     ('Price per Bottle', 6.7832117671123537),
     ('Bottles Sold', 13.619821013589199)]




```python
y_pred = linreg.predict(X_test)
```


```python
#RMSE
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
```

    212.11562256



```python
X = df[['Price per Bottle', 'Bottles Sold']]
y = df['Sale (Dollars)']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
```

    212.117489827



```python
# cross-val: feature selection
X = df[['County Number', 'Price per Bottle', 'Bottles Sold']]
y = df['Sale (Dollars)']
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
print scores
mse_scores = -scores
print mse_scores
```

    [-38225.7981777  -81207.95978577 -30197.22167359 -59428.14744074
     -35299.13167418 -35539.1348213  -41913.76328788 -32789.13563896
     -33334.68436024 -30911.89449383]
    [ 38225.7981777   81207.95978577  30197.22167359  59428.14744074
      35299.13167418  35539.1348213   41913.76328788  32789.13563896
      33334.68436024  30911.89449383]



```python
rmse_scores = np.sqrt(mse_scores)
print rmse_scores
```

    [ 195.5141892   284.97010332  173.77347805  243.77889047  187.88063145
      188.51826124  204.72851117  181.07770608  182.57788574  175.81778776]



```python
print rmse_scores.mean()
```

    201.863744449



```python
# 10 fold cross val with two features excluding county
X = df[['Price per Bottle', 'Bottles Sold']]
print np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean()
```

    201.863449871



```python
# excluding price per bottle
X = df[['County Number', 'Bottles Sold']]
print np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean()
```

    214.537714094



```python
# excluding bottles sold
X = df[['Price per Bottle', 'County Number']]
print np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')).mean()
```

    377.280842037



```python
# Price per bottle against Sale
lm = linear_model.LinearRegression()

X = df[["Price per Bottle"]]

y = df["Sale (Dollars)"]


model = lm.fit(X, y)

predictions = lm.predict(X)


# Plot the model
plt.scatter(X,y)

plt.scatter(X, predictions, label='Linear Fit')

plt.xlabel("Predicted Values from Price per Bottle")

plt.ylabel("Actual Values Sale (Dollars)")

plt.show()

print "MSE:", mean_squared_error(y, predictions)
```


![png](output_39_0.png)


    MSE: 144336.018459



```python
# Total Bottles Sold Against Sale
lm = linear_model.LinearRegression()

X = df[["Bottles Sold"]]

y = df["Sale (Dollars)"]


model = lm.fit(X, y)

predictions = lm.predict(X)


# Plot the model
plt.scatter(X,y)

plt.scatter(X, predictions, label='Linear Fit')

plt.xlabel("Predicted Values from Bottles Sold")

plt.ylabel("Actual Values Sale (Dollars)")

plt.show()

print "MSE:", mean_squared_error(y, predictions)
```


![png](output_40_0.png)


    MSE: 46744.0651775



```python
# Location County Number Against Sale
lm = linear_model.LinearRegression()

X = df[["County Number"]]

y = df["Sale (Dollars)"]


model = lm.fit(X, y)

predictions = lm.predict(X)


# Plot the model
plt.scatter(X,y)

plt.scatter(X, predictions, label='Linear Fit')

plt.xlabel("Predicted Values from County Number")

plt.ylabel("Actual Values Sale (Dollars)")

plt.show()


print "MSE:", mean_squared_error(y, predictions)
```


![png](output_41_0.png)


    MSE: 146996.594944



```python
# Total Bottles Sold Against Sale
lm = linear_model.LinearRegression()

X = df[["Price per Bottle", "Bottles Sold"]]

y = df["Sale (Dollars)"]


model = lm.fit(X, y)

predictions = lm.predict(X)


# Plot the model

plt.scatter(predictions, y, label='Linear Fit')

plt.xlabel("Predicted Values from Bottles Sold")

plt.ylabel("Actual Values Sale (Dollars)")

plt.show()

print "MSE:", mean_squared_error(y, predictions)
```


![png](output_42_0.png)


    MSE: 41541.3227776



#### Tables of best performing stores by location type (city, county, or zip cope) and the predictions of the model:
## By Zip Code


```python
# Table of Best performing stores by location (city, zip, or county number)
# GROUP BY ZIP CODE TOTAL
sales_zip = df.groupby(by=["Zip Code"], as_index=False)


sales_zip = sales_zip.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Store Number": lambda x: x.iloc[0], # just extract once, should be the same
                   "County Number": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0]})
# Collapse the column indices
sales_zip.columns = [' '.join(col).strip() for col in sales_zip.columns.values]
# Rename columns
sales_zip = sales_zip.rename(columns={'County Number <lambda>': 'County Number', 'City <lambda>': 'City', 'Store Number <lambda>': 'Store Number'})
# Transform into DataFrame
sales_zip = pd.DataFrame(sales_zip)
# Quick check
sales_zip.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Zip Code</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>County Number</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50002</td>
      <td>ADAIR</td>
      <td>5851.20</td>
      <td>136.074419</td>
      <td>16.262104</td>
      <td>1.0</td>
      <td>391.35</td>
      <td>9.101163</td>
      <td>4417</td>
      <td>3.988140</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50003</td>
      <td>ADEL</td>
      <td>47716.10</td>
      <td>150.050629</td>
      <td>15.748419</td>
      <td>25.0</td>
      <td>3522.00</td>
      <td>11.075472</td>
      <td>4678</td>
      <td>4.247484</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50006</td>
      <td>ALDEN</td>
      <td>12280.24</td>
      <td>100.657705</td>
      <td>17.284564</td>
      <td>42.0</td>
      <td>860.24</td>
      <td>7.051148</td>
      <td>4172</td>
      <td>6.109426</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50009</td>
      <td>ALTOONA</td>
      <td>294558.39</td>
      <td>140.065806</td>
      <td>18.302846</td>
      <td>77.0</td>
      <td>20282.54</td>
      <td>9.644574</td>
      <td>4919</td>
      <td>4.813348</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50010</td>
      <td>AMES</td>
      <td>931101.58</td>
      <td>131.567271</td>
      <td>19.174701</td>
      <td>85.0</td>
      <td>63313.43</td>
      <td>8.946366</td>
      <td>2501</td>
      <td>5.252892</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Highest Sum of Sales by Zip Code
sales_zip.sort_values(by='Sale (Dollars) sum', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Zip Code</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>County Number</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>50314</td>
      <td>DES MOINES</td>
      <td>1325170.74</td>
      <td>293.244244</td>
      <td>22.579516</td>
      <td>77.0</td>
      <td>75200.28</td>
      <td>16.640912</td>
      <td>4829</td>
      <td>5.891215</td>
    </tr>
    <tr>
      <th>94</th>
      <td>50320</td>
      <td>DES MOINES</td>
      <td>1300498.14</td>
      <td>401.760315</td>
      <td>20.688775</td>
      <td>77.0</td>
      <td>81292.44</td>
      <td>25.113513</td>
      <td>2633</td>
      <td>5.749438</td>
    </tr>
    <tr>
      <th>358</th>
      <td>52402</td>
      <td>CEDAR RAPIDS</td>
      <td>1154143.14</td>
      <td>166.350986</td>
      <td>18.728158</td>
      <td>57.0</td>
      <td>81955.22</td>
      <td>11.812514</td>
      <td>2569</td>
      <td>4.739043</td>
    </tr>
    <tr>
      <th>323</th>
      <td>52240</td>
      <td>IOWA CITY</td>
      <td>1080449.51</td>
      <td>176.313562</td>
      <td>19.433833</td>
      <td>52.0</td>
      <td>68970.66</td>
      <td>11.255003</td>
      <td>2512</td>
      <td>5.045029</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50010</td>
      <td>AMES</td>
      <td>931101.58</td>
      <td>131.567271</td>
      <td>19.174701</td>
      <td>85.0</td>
      <td>63313.43</td>
      <td>8.946366</td>
      <td>2501</td>
      <td>5.252892</td>
    </tr>
    <tr>
      <th>409</th>
      <td>52807</td>
      <td>DAVENPORT</td>
      <td>730049.04</td>
      <td>206.812759</td>
      <td>19.223284</td>
      <td>82.0</td>
      <td>50119.91</td>
      <td>14.198275</td>
      <td>2614</td>
      <td>5.263677</td>
    </tr>
    <tr>
      <th>257</th>
      <td>51501</td>
      <td>COUNCIL BLUFFS</td>
      <td>712179.83</td>
      <td>153.091107</td>
      <td>18.117058</td>
      <td>78.0</td>
      <td>46063.27</td>
      <td>9.901821</td>
      <td>3963</td>
      <td>4.589979</td>
    </tr>
    <tr>
      <th>87</th>
      <td>50311</td>
      <td>DES MOINES</td>
      <td>686739.22</td>
      <td>202.937122</td>
      <td>18.795630</td>
      <td>77.0</td>
      <td>48275.49</td>
      <td>14.265807</td>
      <td>2626</td>
      <td>5.184143</td>
    </tr>
    <tr>
      <th>81</th>
      <td>50266</td>
      <td>WEST DES MOINES</td>
      <td>677930.29</td>
      <td>238.455958</td>
      <td>20.541719</td>
      <td>77.0</td>
      <td>39145.00</td>
      <td>13.768906</td>
      <td>3899</td>
      <td>5.551305</td>
    </tr>
    <tr>
      <th>386</th>
      <td>52722</td>
      <td>BETTENDORF</td>
      <td>672194.79</td>
      <td>181.723382</td>
      <td>20.628846</td>
      <td>82.0</td>
      <td>40866.71</td>
      <td>11.048043</td>
      <td>2603</td>
      <td>5.733014</td>
    </tr>
    <tr>
      <th>280</th>
      <td>52001</td>
      <td>DUBUQUE</td>
      <td>651576.01</td>
      <td>137.029655</td>
      <td>17.894118</td>
      <td>31.0</td>
      <td>44966.25</td>
      <td>9.456625</td>
      <td>4167</td>
      <td>5.006452</td>
    </tr>
    <tr>
      <th>152</th>
      <td>50613</td>
      <td>CEDAR FALLS</td>
      <td>625944.68</td>
      <td>118.842734</td>
      <td>18.474600</td>
      <td>7.0</td>
      <td>44101.73</td>
      <td>8.373216</td>
      <td>2106</td>
      <td>5.055141</td>
    </tr>
    <tr>
      <th>324</th>
      <td>52241</td>
      <td>CORALVILLE</td>
      <td>616268.88</td>
      <td>178.836007</td>
      <td>19.929634</td>
      <td>52.0</td>
      <td>35783.23</td>
      <td>10.383990</td>
      <td>2670</td>
      <td>5.486512</td>
    </tr>
    <tr>
      <th>80</th>
      <td>50265</td>
      <td>WEST DES MOINES</td>
      <td>507370.65</td>
      <td>116.476274</td>
      <td>20.090326</td>
      <td>77.0</td>
      <td>32666.06</td>
      <td>7.499096</td>
      <td>2648</td>
      <td>5.124040</td>
    </tr>
    <tr>
      <th>7</th>
      <td>50021</td>
      <td>ANKENY</td>
      <td>503888.30</td>
      <td>199.876359</td>
      <td>18.033967</td>
      <td>77.0</td>
      <td>35146.09</td>
      <td>13.941329</td>
      <td>2502</td>
      <td>5.264625</td>
    </tr>
    <tr>
      <th>221</th>
      <td>51106</td>
      <td>SIOUX CITY</td>
      <td>498307.31</td>
      <td>211.505649</td>
      <td>17.720443</td>
      <td>97.0</td>
      <td>34465.47</td>
      <td>14.628807</td>
      <td>3879</td>
      <td>5.065212</td>
    </tr>
    <tr>
      <th>258</th>
      <td>51503</td>
      <td>COUNCIL BLUFFS</td>
      <td>479608.99</td>
      <td>141.812238</td>
      <td>17.404441</td>
      <td>78.0</td>
      <td>33673.05</td>
      <td>9.956549</td>
      <td>3443</td>
      <td>4.951576</td>
    </tr>
    <tr>
      <th>184</th>
      <td>50702</td>
      <td>WATERLOO</td>
      <td>478738.03</td>
      <td>150.783631</td>
      <td>18.236020</td>
      <td>7.0</td>
      <td>34883.36</td>
      <td>10.986885</td>
      <td>2538</td>
      <td>4.743609</td>
    </tr>
    <tr>
      <th>101</th>
      <td>50401</td>
      <td>MASON CITY</td>
      <td>477804.83</td>
      <td>116.000202</td>
      <td>17.201952</td>
      <td>17.0</td>
      <td>36788.36</td>
      <td>8.931381</td>
      <td>4376</td>
      <td>4.959905</td>
    </tr>
    <tr>
      <th>185</th>
      <td>50703</td>
      <td>WATERLOO</td>
      <td>476790.63</td>
      <td>165.265383</td>
      <td>19.373641</td>
      <td>7.0</td>
      <td>29035.83</td>
      <td>10.064412</td>
      <td>2130</td>
      <td>4.103477</td>
    </tr>
    <tr>
      <th>360</th>
      <td>52404</td>
      <td>CEDAR RAPIDS</td>
      <td>464167.69</td>
      <td>109.421898</td>
      <td>17.933469</td>
      <td>57.0</td>
      <td>33884.16</td>
      <td>7.987779</td>
      <td>2552</td>
      <td>4.500396</td>
    </tr>
    <tr>
      <th>122</th>
      <td>50501</td>
      <td>FORT DODGE</td>
      <td>463841.86</td>
      <td>156.070612</td>
      <td>17.194485</td>
      <td>94.0</td>
      <td>35191.09</td>
      <td>11.840878</td>
      <td>2644</td>
      <td>4.959727</td>
    </tr>
    <tr>
      <th>407</th>
      <td>52804</td>
      <td>DAVENPORT</td>
      <td>448314.96</td>
      <td>150.795479</td>
      <td>18.548698</td>
      <td>82.0</td>
      <td>30023.58</td>
      <td>10.098749</td>
      <td>2637</td>
      <td>4.546525</td>
    </tr>
    <tr>
      <th>93</th>
      <td>50317</td>
      <td>DES MOINES</td>
      <td>432750.93</td>
      <td>97.796820</td>
      <td>18.397847</td>
      <td>77.0</td>
      <td>30280.08</td>
      <td>6.842956</td>
      <td>2532</td>
      <td>4.487437</td>
    </tr>
    <tr>
      <th>96</th>
      <td>50322</td>
      <td>URBANDALE</td>
      <td>384362.28</td>
      <td>133.459125</td>
      <td>19.854648</td>
      <td>77.0</td>
      <td>23556.68</td>
      <td>8.179403</td>
      <td>2663</td>
      <td>5.143708</td>
    </tr>
    <tr>
      <th>51</th>
      <td>50158</td>
      <td>MARSHALLTOWN</td>
      <td>338061.38</td>
      <td>126.048240</td>
      <td>17.243994</td>
      <td>64.0</td>
      <td>23261.83</td>
      <td>8.673315</td>
      <td>2544</td>
      <td>4.991171</td>
    </tr>
    <tr>
      <th>374</th>
      <td>52601</td>
      <td>BURLINGTON</td>
      <td>335973.17</td>
      <td>113.812049</td>
      <td>19.033672</td>
      <td>29.0</td>
      <td>22173.26</td>
      <td>7.511267</td>
      <td>4898</td>
      <td>4.834610</td>
    </tr>
    <tr>
      <th>91</th>
      <td>50315</td>
      <td>DES MOINES</td>
      <td>335370.63</td>
      <td>108.499071</td>
      <td>18.195592</td>
      <td>77.0</td>
      <td>22757.93</td>
      <td>7.362643</td>
      <td>2528</td>
      <td>4.292071</td>
    </tr>
    <tr>
      <th>398</th>
      <td>52761</td>
      <td>MUSCATINE</td>
      <td>324474.85</td>
      <td>95.743538</td>
      <td>18.066233</td>
      <td>70.0</td>
      <td>23394.85</td>
      <td>6.903172</td>
      <td>2662</td>
      <td>4.791664</td>
    </tr>
    <tr>
      <th>390</th>
      <td>52732</td>
      <td>CLINTON</td>
      <td>322112.38</td>
      <td>114.143296</td>
      <td>17.696046</td>
      <td>23.0</td>
      <td>23488.63</td>
      <td>8.323398</td>
      <td>2616</td>
      <td>4.582760</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>286</th>
      <td>52037</td>
      <td>DELMAR</td>
      <td>2118.00</td>
      <td>73.034483</td>
      <td>16.707874</td>
      <td>23.0</td>
      <td>161.38</td>
      <td>5.564828</td>
      <td>4541</td>
      <td>4.055517</td>
    </tr>
    <tr>
      <th>16</th>
      <td>50046</td>
      <td>CAMBRIDGE</td>
      <td>2116.42</td>
      <td>41.498431</td>
      <td>14.630457</td>
      <td>77.0</td>
      <td>178.03</td>
      <td>3.490784</td>
      <td>4940</td>
      <td>3.888431</td>
    </tr>
    <tr>
      <th>344</th>
      <td>52337</td>
      <td>STANWOOD</td>
      <td>2087.76</td>
      <td>231.973333</td>
      <td>14.157143</td>
      <td>16.0</td>
      <td>119.70</td>
      <td>13.300000</td>
      <td>5020</td>
      <td>3.146667</td>
    </tr>
    <tr>
      <th>132</th>
      <td>50540</td>
      <td>FONDA</td>
      <td>2077.80</td>
      <td>346.300000</td>
      <td>18.088571</td>
      <td>76.0</td>
      <td>132.00</td>
      <td>22.000000</td>
      <td>4610</td>
      <td>4.866667</td>
    </tr>
    <tr>
      <th>78</th>
      <td>50261</td>
      <td>VAN METER</td>
      <td>2061.12</td>
      <td>137.408000</td>
      <td>12.476952</td>
      <td>25.0</td>
      <td>209.70</td>
      <td>13.980000</td>
      <td>4623</td>
      <td>3.485333</td>
    </tr>
    <tr>
      <th>199</th>
      <td>51005</td>
      <td>AURELIA</td>
      <td>2061.06</td>
      <td>187.369091</td>
      <td>15.513593</td>
      <td>18.0</td>
      <td>157.50</td>
      <td>14.318182</td>
      <td>4923</td>
      <td>3.942727</td>
    </tr>
    <tr>
      <th>26</th>
      <td>50071</td>
      <td>DOWS</td>
      <td>2011.62</td>
      <td>154.740000</td>
      <td>13.818874</td>
      <td>99.0</td>
      <td>154.80</td>
      <td>11.907692</td>
      <td>4515</td>
      <td>3.520000</td>
    </tr>
    <tr>
      <th>139</th>
      <td>50569</td>
      <td>OTHO</td>
      <td>1962.12</td>
      <td>98.106000</td>
      <td>13.120976</td>
      <td>94.0</td>
      <td>160.50</td>
      <td>8.025000</td>
      <td>3845</td>
      <td>3.092000</td>
    </tr>
    <tr>
      <th>197</th>
      <td>51002</td>
      <td>ALTA</td>
      <td>1919.76</td>
      <td>137.125714</td>
      <td>12.375510</td>
      <td>11.0</td>
      <td>161.25</td>
      <td>11.517857</td>
      <td>4982</td>
      <td>3.845714</td>
    </tr>
    <tr>
      <th>15</th>
      <td>50044</td>
      <td>BUSSEY</td>
      <td>1896.00</td>
      <td>118.500000</td>
      <td>10.072708</td>
      <td>63.0</td>
      <td>217.50</td>
      <td>13.593750</td>
      <td>4789</td>
      <td>2.171875</td>
    </tr>
    <tr>
      <th>195</th>
      <td>50864</td>
      <td>VILLISCA</td>
      <td>1827.87</td>
      <td>83.085000</td>
      <td>17.039394</td>
      <td>69.0</td>
      <td>149.25</td>
      <td>6.784091</td>
      <td>4325</td>
      <td>4.510909</td>
    </tr>
    <tr>
      <th>133</th>
      <td>50541</td>
      <td>GILMORE CITY</td>
      <td>1773.60</td>
      <td>110.850000</td>
      <td>12.539940</td>
      <td>46.0</td>
      <td>195.00</td>
      <td>12.187500</td>
      <td>4527</td>
      <td>2.961250</td>
    </tr>
    <tr>
      <th>402</th>
      <td>52777</td>
      <td>WHEATLAND</td>
      <td>1665.08</td>
      <td>64.041538</td>
      <td>16.524666</td>
      <td>23.0</td>
      <td>134.37</td>
      <td>5.168077</td>
      <td>4948</td>
      <td>4.216923</td>
    </tr>
    <tr>
      <th>44</th>
      <td>50136</td>
      <td>KELLOG</td>
      <td>1472.67</td>
      <td>73.633500</td>
      <td>12.445951</td>
      <td>50.0</td>
      <td>133.67</td>
      <td>6.683500</td>
      <td>4071</td>
      <td>3.399000</td>
    </tr>
    <tr>
      <th>375</th>
      <td>52623</td>
      <td>DANVILLE</td>
      <td>1361.74</td>
      <td>90.782667</td>
      <td>13.992762</td>
      <td>29.0</td>
      <td>103.50</td>
      <td>6.900000</td>
      <td>4992</td>
      <td>3.070000</td>
    </tr>
    <tr>
      <th>336</th>
      <td>52316</td>
      <td>NORTH ENGLISH</td>
      <td>1301.32</td>
      <td>40.666250</td>
      <td>12.048961</td>
      <td>48.0</td>
      <td>161.63</td>
      <td>5.050937</td>
      <td>5026</td>
      <td>3.571563</td>
    </tr>
    <tr>
      <th>112</th>
      <td>50452</td>
      <td>LATIMER</td>
      <td>1227.42</td>
      <td>153.427500</td>
      <td>9.061190</td>
      <td>35.0</td>
      <td>145.50</td>
      <td>18.187500</td>
      <td>5055</td>
      <td>4.746250</td>
    </tr>
    <tr>
      <th>213</th>
      <td>51053</td>
      <td>SCHALLER</td>
      <td>1056.37</td>
      <td>70.424667</td>
      <td>14.488349</td>
      <td>81.0</td>
      <td>90.75</td>
      <td>6.050000</td>
      <td>3864</td>
      <td>5.106000</td>
    </tr>
    <tr>
      <th>208</th>
      <td>51038</td>
      <td>MERRILL</td>
      <td>1001.25</td>
      <td>100.125000</td>
      <td>13.728000</td>
      <td>75.0</td>
      <td>78.00</td>
      <td>7.800000</td>
      <td>4985</td>
      <td>3.978000</td>
    </tr>
    <tr>
      <th>54</th>
      <td>50162</td>
      <td>MELBOURNE</td>
      <td>984.30</td>
      <td>164.050000</td>
      <td>13.540635</td>
      <td>64.0</td>
      <td>82.50</td>
      <td>13.750000</td>
      <td>4338</td>
      <td>4.515000</td>
    </tr>
    <tr>
      <th>268</th>
      <td>51553</td>
      <td>MINDEN</td>
      <td>898.71</td>
      <td>99.856667</td>
      <td>22.557765</td>
      <td>78.0</td>
      <td>54.74</td>
      <td>6.082222</td>
      <td>5177</td>
      <td>4.433333</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50150</td>
      <td>LOVILIA</td>
      <td>812.44</td>
      <td>73.858182</td>
      <td>18.660346</td>
      <td>68.0</td>
      <td>48.15</td>
      <td>4.377273</td>
      <td>4811</td>
      <td>3.660909</td>
    </tr>
    <tr>
      <th>238</th>
      <td>51338</td>
      <td>EVERLY</td>
      <td>641.01</td>
      <td>91.572857</td>
      <td>14.714014</td>
      <td>21.0</td>
      <td>58.25</td>
      <td>8.321429</td>
      <td>3677</td>
      <td>5.042857</td>
    </tr>
    <tr>
      <th>318</th>
      <td>52223</td>
      <td>DELHI</td>
      <td>524.80</td>
      <td>47.709091</td>
      <td>21.691818</td>
      <td>28.0</td>
      <td>34.40</td>
      <td>3.127273</td>
      <td>5175</td>
      <td>3.244545</td>
    </tr>
    <tr>
      <th>264</th>
      <td>51535</td>
      <td>GRISWOLD</td>
      <td>420.81</td>
      <td>84.162000</td>
      <td>14.286000</td>
      <td>15.0</td>
      <td>34.50</td>
      <td>6.900000</td>
      <td>4990</td>
      <td>4.750000</td>
    </tr>
    <tr>
      <th>161</th>
      <td>50634</td>
      <td>GILBERTVILLE</td>
      <td>407.61</td>
      <td>67.935000</td>
      <td>16.857083</td>
      <td>7.0</td>
      <td>23.62</td>
      <td>3.936667</td>
      <td>5202</td>
      <td>3.921667</td>
    </tr>
    <tr>
      <th>404</th>
      <td>52801</td>
      <td>DAVENPORT</td>
      <td>246.40</td>
      <td>123.200000</td>
      <td>29.666667</td>
      <td>82.0</td>
      <td>10.50</td>
      <td>5.250000</td>
      <td>5130</td>
      <td>7.420000</td>
    </tr>
    <tr>
      <th>252</th>
      <td>51453</td>
      <td>LOHRVILLE</td>
      <td>139.50</td>
      <td>46.500000</td>
      <td>15.164444</td>
      <td>13.0</td>
      <td>9.00</td>
      <td>3.000000</td>
      <td>5193</td>
      <td>3.793333</td>
    </tr>
    <tr>
      <th>340</th>
      <td>52328</td>
      <td>ROBINS</td>
      <td>123.60</td>
      <td>61.800000</td>
      <td>14.480000</td>
      <td>57.0</td>
      <td>8.25</td>
      <td>4.125000</td>
      <td>5192</td>
      <td>3.620000</td>
    </tr>
    <tr>
      <th>262</th>
      <td>51530</td>
      <td>COUNCIL BLUFFS</td>
      <td>92.50</td>
      <td>30.833333</td>
      <td>22.691111</td>
      <td>78.0</td>
      <td>5.10</td>
      <td>1.700000</td>
      <td>5195</td>
      <td>3.836667</td>
    </tr>
  </tbody>
</table>
<p>412 rows × 10 columns</p>
</div>


```python
df_pivot = pd.pivot_table(df,index=["Zip Code","Store Number"], values=["Sale (Dollars)"], aggfunc=np.sum)
df_pivot
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Sale (Dollars)</th>
    </tr>
    <tr>
      <th>Zip Code</th>
      <th>Store Number</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">50002</th>
      <th>4417</th>
      <td>4214.10</td>
    </tr>
    <tr>
      <th>4753</th>
      <td>1637.10</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">50003</th>
      <th>4384</th>
      <td>8839.82</td>
    </tr>
    <tr>
      <th>4678</th>
      <td>37836.48</td>
    </tr>
    <tr>
      <th>4929</th>
      <td>1039.80</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">50006</th>
      <th>4172</th>
      <td>11225.08</td>
    </tr>
    <tr>
      <th>4497</th>
      <td>1055.16</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">50009</th>
      <th>2238</th>
      <td>12171.88</td>
    </tr>
    <tr>
      <th>2478</th>
      <td>24303.27</td>
    </tr>
    <tr>
      <th>2548</th>
      <td>126947.84</td>
    </tr>
    <tr>
      <th>3644</th>
      <td>70208.10</td>
    </tr>
    <tr>
      <th>3870</th>
      <td>11193.60</td>
    </tr>
    <tr>
      <th>4135</th>
      <td>24556.02</td>
    </tr>
    <tr>
      <th>4695</th>
      <td>3751.92</td>
    </tr>
    <tr>
      <th>4819</th>
      <td>16433.01</td>
    </tr>
    <tr>
      <th>4919</th>
      <td>4992.75</td>
    </tr>
    <tr>
      <th rowspan="14" valign="top">50010</th>
      <th>2500</th>
      <td>182871.72</td>
    </tr>
    <tr>
      <th>2501</th>
      <td>174349.30</td>
    </tr>
    <tr>
      <th>2609</th>
      <td>7925.05</td>
    </tr>
    <tr>
      <th>3524</th>
      <td>202673.94</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>14978.52</td>
    </tr>
    <tr>
      <th>4004</th>
      <td>50066.94</td>
    </tr>
    <tr>
      <th>4015</th>
      <td>33505.76</td>
    </tr>
    <tr>
      <th>4102</th>
      <td>17045.72</td>
    </tr>
    <tr>
      <th>4103</th>
      <td>23885.68</td>
    </tr>
    <tr>
      <th>4129</th>
      <td>87409.42</td>
    </tr>
    <tr>
      <th>4436</th>
      <td>11284.76</td>
    </tr>
    <tr>
      <th>4437</th>
      <td>6358.35</td>
    </tr>
    <tr>
      <th>4438</th>
      <td>5962.83</td>
    </tr>
    <tr>
      <th>4509</th>
      <td>73108.32</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>52803</th>
      <th>5198</th>
      <td>10864.17</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">52804</th>
      <th>2625</th>
      <td>220741.89</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>95209.60</td>
    </tr>
    <tr>
      <th>2839</th>
      <td>3817.19</td>
    </tr>
    <tr>
      <th>3853</th>
      <td>16974.66</td>
    </tr>
    <tr>
      <th>3917</th>
      <td>22111.62</td>
    </tr>
    <tr>
      <th>4638</th>
      <td>39841.63</td>
    </tr>
    <tr>
      <th>4749</th>
      <td>4242.95</td>
    </tr>
    <tr>
      <th>4841</th>
      <td>6532.78</td>
    </tr>
    <tr>
      <th>4952</th>
      <td>10192.44</td>
    </tr>
    <tr>
      <th>5003</th>
      <td>28650.20</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">52806</th>
      <th>2567</th>
      <td>55416.55</td>
    </tr>
    <tr>
      <th>2838</th>
      <td>6199.40</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>60708.72</td>
    </tr>
    <tr>
      <th>3858</th>
      <td>23758.56</td>
    </tr>
    <tr>
      <th>4196</th>
      <td>21164.36</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>12456.03</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>2002.20</td>
    </tr>
    <tr>
      <th>5029</th>
      <td>36511.78</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">52807</th>
      <th>2614</th>
      <td>179035.77</td>
    </tr>
    <tr>
      <th>2635</th>
      <td>129285.56</td>
    </tr>
    <tr>
      <th>3354</th>
      <td>295039.20</td>
    </tr>
    <tr>
      <th>3540</th>
      <td>23588.16</td>
    </tr>
    <tr>
      <th>3715</th>
      <td>20615.43</td>
    </tr>
    <tr>
      <th>3731</th>
      <td>56709.60</td>
    </tr>
    <tr>
      <th>4640</th>
      <td>20149.86</td>
    </tr>
    <tr>
      <th>4748</th>
      <td>2771.50</td>
    </tr>
    <tr>
      <th>4750</th>
      <td>2853.96</td>
    </tr>
    <tr>
      <th>56201</th>
      <th>4722</th>
      <td>4396.20</td>
    </tr>
    <tr>
      <th>712-2</th>
      <th>4307</th>
      <td>15185.99</td>
    </tr>
  </tbody>
</table>
<p>1378 rows × 1 columns</p>
</div>


## By City


```python
# GROUP BY CITY TOTAL

sales_city = df.groupby(by=["City"], as_index=False)

sales_city = sales_city.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Store Number": lambda x: x.iloc[0], # just extract once, should be the same
                   "Zip Code": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0]})
# Collapse the column indices
sales_city.columns = [' '.join(col).strip() for col in sales_city.columns.values]
# Rename columns
sales_city = sales_city.rename(columns={'Zip Code <lambda>': 'Zip Code', 'County Number <lambda>': 'County Number', 'Store Number <lambda>': 'Store Number'})
# Transform into DataFrame
sales_city = pd.DataFrame(sales_city)
# Quick check
sales_city.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>County Number</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACKLEY</td>
      <td>6721.25</td>
      <td>73.859890</td>
      <td>42.0</td>
      <td>16.010529</td>
      <td>50601</td>
      <td>577.53</td>
      <td>6.346484</td>
      <td>4415</td>
      <td>4.460659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ADAIR</td>
      <td>5851.20</td>
      <td>136.074419</td>
      <td>1.0</td>
      <td>16.262104</td>
      <td>50002</td>
      <td>391.35</td>
      <td>9.101163</td>
      <td>4417</td>
      <td>3.988140</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ADEL</td>
      <td>47716.10</td>
      <td>150.050629</td>
      <td>25.0</td>
      <td>15.748419</td>
      <td>50003</td>
      <td>3522.00</td>
      <td>11.075472</td>
      <td>4678</td>
      <td>4.247484</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFTON</td>
      <td>3262.08</td>
      <td>271.840000</td>
      <td>88.0</td>
      <td>17.551905</td>
      <td>50830</td>
      <td>265.50</td>
      <td>22.125000</td>
      <td>4531</td>
      <td>3.913333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AKRON</td>
      <td>5469.42</td>
      <td>79.266957</td>
      <td>75.0</td>
      <td>16.724127</td>
      <td>51001</td>
      <td>429.63</td>
      <td>6.226522</td>
      <td>4525</td>
      <td>4.173333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Highest Sum of Sales by City
sales_city.sort_values(by='Sale (Dollars) sum', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>County Number</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>DES MOINES</td>
      <td>4380886.26</td>
      <td>185.489299</td>
      <td>77.0</td>
      <td>19.995426</td>
      <td>50311</td>
      <td>276124.61</td>
      <td>11.691278</td>
      <td>2626</td>
      <td>5.002371</td>
    </tr>
    <tr>
      <th>51</th>
      <td>CEDAR RAPIDS</td>
      <td>2486949.08</td>
      <td>132.736394</td>
      <td>57.0</td>
      <td>19.041381</td>
      <td>52402</td>
      <td>171144.02</td>
      <td>9.134501</td>
      <td>2569</td>
      <td>4.743563</td>
    </tr>
    <tr>
      <th>81</th>
      <td>DAVENPORT</td>
      <td>1697702.84</td>
      <td>148.025359</td>
      <td>82.0</td>
      <td>18.735957</td>
      <td>52807</td>
      <td>115944.01</td>
      <td>10.109339</td>
      <td>2614</td>
      <td>4.578247</td>
    </tr>
    <tr>
      <th>175</th>
      <td>IOWA CITY</td>
      <td>1250666.19</td>
      <td>157.554320</td>
      <td>52.0</td>
      <td>19.198857</td>
      <td>52246</td>
      <td>80672.60</td>
      <td>10.162837</td>
      <td>3565</td>
      <td>5.018575</td>
    </tr>
    <tr>
      <th>358</th>
      <td>WATERLOO</td>
      <td>1211603.55</td>
      <td>144.651809</td>
      <td>7.0</td>
      <td>18.979850</td>
      <td>50703</td>
      <td>81164.33</td>
      <td>9.690106</td>
      <td>2130</td>
      <td>4.495614</td>
    </tr>
    <tr>
      <th>315</th>
      <td>SIOUX CITY</td>
      <td>1200607.02</td>
      <td>152.206772</td>
      <td>97.0</td>
      <td>18.592685</td>
      <td>51104</td>
      <td>80637.49</td>
      <td>10.222806</td>
      <td>2623</td>
      <td>4.994855</td>
    </tr>
    <tr>
      <th>73</th>
      <td>COUNCIL BLUFFS</td>
      <td>1191881.32</td>
      <td>148.299281</td>
      <td>78.0</td>
      <td>17.818893</td>
      <td>51501</td>
      <td>79741.42</td>
      <td>9.921789</td>
      <td>3963</td>
      <td>4.741859</td>
    </tr>
    <tr>
      <th>369</th>
      <td>WEST DES MOINES</td>
      <td>1181449.32</td>
      <td>165.283900</td>
      <td>77.0</td>
      <td>20.297211</td>
      <td>50265</td>
      <td>71544.33</td>
      <td>10.009000</td>
      <td>2648</td>
      <td>5.299187</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AMES</td>
      <td>979880.51</td>
      <td>130.061124</td>
      <td>85.0</td>
      <td>19.203869</td>
      <td>50010</td>
      <td>66786.44</td>
      <td>8.864672</td>
      <td>2501</td>
      <td>5.291353</td>
    </tr>
    <tr>
      <th>94</th>
      <td>DUBUQUE</td>
      <td>965011.30</td>
      <td>140.795346</td>
      <td>31.0</td>
      <td>17.967045</td>
      <td>52001</td>
      <td>70102.17</td>
      <td>10.227921</td>
      <td>4167</td>
      <td>5.098634</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ANKENY</td>
      <td>716381.14</td>
      <td>148.534344</td>
      <td>77.0</td>
      <td>19.521893</td>
      <td>50021</td>
      <td>48744.98</td>
      <td>10.106776</td>
      <td>2502</td>
      <td>5.494259</td>
    </tr>
    <tr>
      <th>31</th>
      <td>BETTENDORF</td>
      <td>672194.79</td>
      <td>181.723382</td>
      <td>82.0</td>
      <td>20.628846</td>
      <td>52722</td>
      <td>40866.71</td>
      <td>11.048043</td>
      <td>2603</td>
      <td>5.733014</td>
    </tr>
    <tr>
      <th>50</th>
      <td>CEDAR FALLS</td>
      <td>647023.66</td>
      <td>113.135803</td>
      <td>7.0</td>
      <td>18.459159</td>
      <td>50613</td>
      <td>45584.73</td>
      <td>7.970752</td>
      <td>2106</td>
      <td>5.016442</td>
    </tr>
    <tr>
      <th>69</th>
      <td>CORALVILLE</td>
      <td>616268.88</td>
      <td>178.836007</td>
      <td>52.0</td>
      <td>19.929634</td>
      <td>52241</td>
      <td>35783.23</td>
      <td>10.383990</td>
      <td>2670</td>
      <td>5.486512</td>
    </tr>
    <tr>
      <th>376</th>
      <td>WINDSOR HEIGHTS</td>
      <td>599211.44</td>
      <td>214.233622</td>
      <td>77.0</td>
      <td>19.378585</td>
      <td>50311</td>
      <td>42257.93</td>
      <td>15.108305</td>
      <td>2620</td>
      <td>5.350665</td>
    </tr>
    <tr>
      <th>226</th>
      <td>MASON CITY</td>
      <td>477804.83</td>
      <td>116.000202</td>
      <td>17.0</td>
      <td>17.201952</td>
      <td>50401</td>
      <td>36788.36</td>
      <td>8.931381</td>
      <td>4376</td>
      <td>4.959905</td>
    </tr>
    <tr>
      <th>130</th>
      <td>FORT DODGE</td>
      <td>463841.86</td>
      <td>156.070612</td>
      <td>94.0</td>
      <td>17.194485</td>
      <td>50501</td>
      <td>35191.09</td>
      <td>11.840878</td>
      <td>2644</td>
      <td>4.959727</td>
    </tr>
    <tr>
      <th>41</th>
      <td>BURLINGTON</td>
      <td>366889.34</td>
      <td>116.955480</td>
      <td>29.0</td>
      <td>19.176112</td>
      <td>52601</td>
      <td>23874.58</td>
      <td>7.610641</td>
      <td>4898</td>
      <td>4.770086</td>
    </tr>
    <tr>
      <th>62</th>
      <td>CLINTON</td>
      <td>352625.17</td>
      <td>114.600315</td>
      <td>23.0</td>
      <td>17.586640</td>
      <td>52732</td>
      <td>25575.30</td>
      <td>8.311765</td>
      <td>2616</td>
      <td>4.539815</td>
    </tr>
    <tr>
      <th>345</th>
      <td>URBANDALE</td>
      <td>339997.65</td>
      <td>140.263057</td>
      <td>77.0</td>
      <td>18.934166</td>
      <td>50322</td>
      <td>20985.00</td>
      <td>8.657178</td>
      <td>2663</td>
      <td>4.886663</td>
    </tr>
    <tr>
      <th>224</th>
      <td>MARSHALLTOWN</td>
      <td>338061.38</td>
      <td>126.048240</td>
      <td>64.0</td>
      <td>17.243994</td>
      <td>50158</td>
      <td>23261.83</td>
      <td>8.673315</td>
      <td>2544</td>
      <td>4.991171</td>
    </tr>
    <tr>
      <th>247</th>
      <td>MUSCATINE</td>
      <td>324474.85</td>
      <td>95.743538</td>
      <td>70.0</td>
      <td>18.066233</td>
      <td>52761</td>
      <td>23394.85</td>
      <td>6.903172</td>
      <td>2662</td>
      <td>4.791664</td>
    </tr>
    <tr>
      <th>185</th>
      <td>KEOKUK</td>
      <td>319238.75</td>
      <td>187.127052</td>
      <td>56.0</td>
      <td>17.715374</td>
      <td>52632</td>
      <td>21272.27</td>
      <td>12.469091</td>
      <td>2555</td>
      <td>5.192075</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ALTOONA</td>
      <td>294558.39</td>
      <td>140.065806</td>
      <td>77.0</td>
      <td>18.302846</td>
      <td>50009</td>
      <td>20282.54</td>
      <td>9.644574</td>
      <td>4919</td>
      <td>4.813348</td>
    </tr>
    <tr>
      <th>223</th>
      <td>MARION</td>
      <td>291375.51</td>
      <td>117.253726</td>
      <td>57.0</td>
      <td>16.621566</td>
      <td>50129</td>
      <td>21903.38</td>
      <td>8.814237</td>
      <td>2514</td>
      <td>4.626922</td>
    </tr>
    <tr>
      <th>46</th>
      <td>CARROLL</td>
      <td>278581.51</td>
      <td>188.740860</td>
      <td>14.0</td>
      <td>17.021868</td>
      <td>51401</td>
      <td>20843.65</td>
      <td>14.121714</td>
      <td>4158</td>
      <td>4.917859</td>
    </tr>
    <tr>
      <th>244</th>
      <td>MOUNT VERNON</td>
      <td>254311.02</td>
      <td>224.260159</td>
      <td>57.0</td>
      <td>19.431729</td>
      <td>52314</td>
      <td>17476.11</td>
      <td>15.411032</td>
      <td>5102</td>
      <td>5.788457</td>
    </tr>
    <tr>
      <th>321</th>
      <td>SPIRIT LAKE</td>
      <td>241553.69</td>
      <td>146.573841</td>
      <td>30.0</td>
      <td>17.274836</td>
      <td>51360</td>
      <td>18038.13</td>
      <td>10.945467</td>
      <td>4387</td>
      <td>5.169405</td>
    </tr>
    <tr>
      <th>271</th>
      <td>OTTUMWA</td>
      <td>235714.96</td>
      <td>102.932297</td>
      <td>90.0</td>
      <td>17.107524</td>
      <td>52501</td>
      <td>16686.58</td>
      <td>7.286716</td>
      <td>2596</td>
      <td>4.415284</td>
    </tr>
    <tr>
      <th>61</th>
      <td>CLEAR LAKE</td>
      <td>208523.33</td>
      <td>100.251601</td>
      <td>17.0</td>
      <td>17.847113</td>
      <td>50428</td>
      <td>15224.54</td>
      <td>7.319490</td>
      <td>3456</td>
      <td>5.312966</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>353</th>
      <td>WALL LAKE</td>
      <td>2256.84</td>
      <td>161.202857</td>
      <td>81.0</td>
      <td>17.630204</td>
      <td>51466</td>
      <td>132.75</td>
      <td>9.482143</td>
      <td>4957</td>
      <td>4.431429</td>
    </tr>
    <tr>
      <th>87</th>
      <td>DELMAR</td>
      <td>2118.00</td>
      <td>73.034483</td>
      <td>23.0</td>
      <td>16.707874</td>
      <td>52037</td>
      <td>161.38</td>
      <td>5.564828</td>
      <td>4541</td>
      <td>4.055517</td>
    </tr>
    <tr>
      <th>44</th>
      <td>CAMBRIDGE</td>
      <td>2116.42</td>
      <td>41.498431</td>
      <td>77.0</td>
      <td>14.630457</td>
      <td>50046</td>
      <td>178.03</td>
      <td>3.490784</td>
      <td>4940</td>
      <td>3.888431</td>
    </tr>
    <tr>
      <th>326</th>
      <td>STANWOOD</td>
      <td>2087.76</td>
      <td>231.973333</td>
      <td>16.0</td>
      <td>14.157143</td>
      <td>52337</td>
      <td>119.70</td>
      <td>13.300000</td>
      <td>5020</td>
      <td>3.146667</td>
    </tr>
    <tr>
      <th>126</th>
      <td>FONDA</td>
      <td>2077.80</td>
      <td>346.300000</td>
      <td>76.0</td>
      <td>18.088571</td>
      <td>50540</td>
      <td>132.00</td>
      <td>22.000000</td>
      <td>4610</td>
      <td>4.866667</td>
    </tr>
    <tr>
      <th>347</th>
      <td>VAN METER</td>
      <td>2061.12</td>
      <td>137.408000</td>
      <td>25.0</td>
      <td>12.476952</td>
      <td>50261</td>
      <td>209.70</td>
      <td>13.980000</td>
      <td>4623</td>
      <td>3.485333</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AURELIA</td>
      <td>2061.06</td>
      <td>187.369091</td>
      <td>18.0</td>
      <td>15.513593</td>
      <td>51005</td>
      <td>157.50</td>
      <td>14.318182</td>
      <td>4923</td>
      <td>3.942727</td>
    </tr>
    <tr>
      <th>93</th>
      <td>DOWS</td>
      <td>2011.62</td>
      <td>154.740000</td>
      <td>99.0</td>
      <td>13.818874</td>
      <td>50071</td>
      <td>154.80</td>
      <td>11.907692</td>
      <td>4515</td>
      <td>3.520000</td>
    </tr>
    <tr>
      <th>270</th>
      <td>OTHO</td>
      <td>1962.12</td>
      <td>98.106000</td>
      <td>94.0</td>
      <td>13.120976</td>
      <td>50569</td>
      <td>160.50</td>
      <td>8.025000</td>
      <td>3845</td>
      <td>3.092000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ALTA</td>
      <td>1919.76</td>
      <td>137.125714</td>
      <td>11.0</td>
      <td>12.375510</td>
      <td>51002</td>
      <td>161.25</td>
      <td>11.517857</td>
      <td>4982</td>
      <td>3.845714</td>
    </tr>
    <tr>
      <th>42</th>
      <td>BUSSEY</td>
      <td>1896.00</td>
      <td>118.500000</td>
      <td>63.0</td>
      <td>10.072708</td>
      <td>50044</td>
      <td>217.50</td>
      <td>13.593750</td>
      <td>4789</td>
      <td>2.171875</td>
    </tr>
    <tr>
      <th>349</th>
      <td>VILLISCA</td>
      <td>1827.87</td>
      <td>83.085000</td>
      <td>69.0</td>
      <td>17.039394</td>
      <td>50864</td>
      <td>149.25</td>
      <td>6.784091</td>
      <td>4325</td>
      <td>4.510909</td>
    </tr>
    <tr>
      <th>136</th>
      <td>GILMORE CITY</td>
      <td>1773.60</td>
      <td>110.850000</td>
      <td>46.0</td>
      <td>12.539940</td>
      <td>50541</td>
      <td>195.00</td>
      <td>12.187500</td>
      <td>4527</td>
      <td>2.961250</td>
    </tr>
    <tr>
      <th>373</th>
      <td>WHEATLAND</td>
      <td>1665.08</td>
      <td>64.041538</td>
      <td>23.0</td>
      <td>16.524666</td>
      <td>52777</td>
      <td>134.37</td>
      <td>5.168077</td>
      <td>4948</td>
      <td>4.216923</td>
    </tr>
    <tr>
      <th>183</th>
      <td>KELLOG</td>
      <td>1472.67</td>
      <td>73.633500</td>
      <td>50.0</td>
      <td>12.445951</td>
      <td>50136</td>
      <td>133.67</td>
      <td>6.683500</td>
      <td>4071</td>
      <td>3.399000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>DANVILLE</td>
      <td>1361.74</td>
      <td>90.782667</td>
      <td>29.0</td>
      <td>13.992762</td>
      <td>52623</td>
      <td>103.50</td>
      <td>6.900000</td>
      <td>4992</td>
      <td>3.070000</td>
    </tr>
    <tr>
      <th>256</th>
      <td>NORTH ENGLISH</td>
      <td>1301.32</td>
      <td>40.666250</td>
      <td>48.0</td>
      <td>12.048961</td>
      <td>52316</td>
      <td>161.63</td>
      <td>5.050937</td>
      <td>5026</td>
      <td>3.571563</td>
    </tr>
    <tr>
      <th>198</th>
      <td>LATIMER</td>
      <td>1227.42</td>
      <td>153.427500</td>
      <td>35.0</td>
      <td>9.061190</td>
      <td>50452</td>
      <td>145.50</td>
      <td>18.187500</td>
      <td>5055</td>
      <td>4.746250</td>
    </tr>
    <tr>
      <th>304</th>
      <td>SCHALLER</td>
      <td>1056.37</td>
      <td>70.424667</td>
      <td>81.0</td>
      <td>14.488349</td>
      <td>51053</td>
      <td>90.75</td>
      <td>6.050000</td>
      <td>3864</td>
      <td>5.106000</td>
    </tr>
    <tr>
      <th>232</th>
      <td>MERRILL</td>
      <td>1001.25</td>
      <td>100.125000</td>
      <td>75.0</td>
      <td>13.728000</td>
      <td>51038</td>
      <td>78.00</td>
      <td>7.800000</td>
      <td>4985</td>
      <td>3.978000</td>
    </tr>
    <tr>
      <th>230</th>
      <td>MELBOURNE</td>
      <td>984.30</td>
      <td>164.050000</td>
      <td>64.0</td>
      <td>13.540635</td>
      <td>50162</td>
      <td>82.50</td>
      <td>13.750000</td>
      <td>4338</td>
      <td>4.515000</td>
    </tr>
    <tr>
      <th>234</th>
      <td>MINDEN</td>
      <td>898.71</td>
      <td>99.856667</td>
      <td>78.0</td>
      <td>22.557765</td>
      <td>51553</td>
      <td>54.74</td>
      <td>6.082222</td>
      <td>5177</td>
      <td>4.433333</td>
    </tr>
    <tr>
      <th>212</th>
      <td>LOVILIA</td>
      <td>812.44</td>
      <td>73.858182</td>
      <td>68.0</td>
      <td>18.660346</td>
      <td>50150</td>
      <td>48.15</td>
      <td>4.377273</td>
      <td>4811</td>
      <td>3.660909</td>
    </tr>
    <tr>
      <th>117</th>
      <td>EVERLY</td>
      <td>641.01</td>
      <td>91.572857</td>
      <td>21.0</td>
      <td>14.714014</td>
      <td>51338</td>
      <td>58.25</td>
      <td>8.321429</td>
      <td>3677</td>
      <td>5.042857</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Carroll</td>
      <td>633.36</td>
      <td>633.360000</td>
      <td>14.0</td>
      <td>35.186667</td>
      <td>51401</td>
      <td>18.00</td>
      <td>18.000000</td>
      <td>9023</td>
      <td>8.800000</td>
    </tr>
    <tr>
      <th>86</th>
      <td>DELHI</td>
      <td>524.80</td>
      <td>47.709091</td>
      <td>28.0</td>
      <td>21.691818</td>
      <td>52223</td>
      <td>34.40</td>
      <td>3.127273</td>
      <td>5175</td>
      <td>3.244545</td>
    </tr>
    <tr>
      <th>150</th>
      <td>GRISWOLD</td>
      <td>420.81</td>
      <td>84.162000</td>
      <td>15.0</td>
      <td>14.286000</td>
      <td>51535</td>
      <td>34.50</td>
      <td>6.900000</td>
      <td>4990</td>
      <td>4.750000</td>
    </tr>
    <tr>
      <th>135</th>
      <td>GILBERTVILLE</td>
      <td>407.61</td>
      <td>67.935000</td>
      <td>7.0</td>
      <td>16.857083</td>
      <td>50634</td>
      <td>23.62</td>
      <td>3.936667</td>
      <td>5202</td>
      <td>3.921667</td>
    </tr>
    <tr>
      <th>210</th>
      <td>LOHRVILLE</td>
      <td>139.50</td>
      <td>46.500000</td>
      <td>13.0</td>
      <td>15.164444</td>
      <td>51453</td>
      <td>9.00</td>
      <td>3.000000</td>
      <td>5193</td>
      <td>3.793333</td>
    </tr>
    <tr>
      <th>295</th>
      <td>ROBINS</td>
      <td>123.60</td>
      <td>61.800000</td>
      <td>57.0</td>
      <td>14.480000</td>
      <td>52328</td>
      <td>8.25</td>
      <td>4.125000</td>
      <td>5192</td>
      <td>3.620000</td>
    </tr>
  </tbody>
</table>
<p>382 rows × 10 columns</p>
</div>



## By County


```python
# GROUP BY COUNTY NUMBER TOTAL
sales_county = df.groupby(by=["County Number"], as_index=False)

sales_county = sales_county.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Store Number": lambda x: x.iloc[0], # just extract once, should be the same
                   "Zip Code": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0]})
# Collapse the column indices
sales_county.columns = [' '.join(col).strip() for col in sales_county.columns.values]
# Rename columns
sales_county = sales_county.rename(columns={'Zip Code <lambda>': 'Zip Code', 'City <lambda>': 'City', 'Store Number <lambda>': 'Store Number'})
# Transform into DataFrame
sales_county = pd.DataFrame(sales_county)
# Quick check
sales_county.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>County Number</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>STUART</td>
      <td>55544.20</td>
      <td>95.109932</td>
      <td>15.659537</td>
      <td>50250</td>
      <td>4359.84</td>
      <td>7.465479</td>
      <td>3461</td>
      <td>4.673099</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>CORNING</td>
      <td>12441.71</td>
      <td>53.169701</td>
      <td>17.063570</td>
      <td>50841</td>
      <td>981.38</td>
      <td>4.193932</td>
      <td>2327</td>
      <td>5.091111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>WAUKON</td>
      <td>99000.00</td>
      <td>94.827586</td>
      <td>15.514064</td>
      <td>52172</td>
      <td>7868.04</td>
      <td>7.536437</td>
      <td>3857</td>
      <td>5.001964</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>CENTERVILLE</td>
      <td>98429.87</td>
      <td>94.644106</td>
      <td>16.325347</td>
      <td>52544</td>
      <td>8038.16</td>
      <td>7.729000</td>
      <td>4472</td>
      <td>4.778760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>EXIRA</td>
      <td>19655.54</td>
      <td>86.588282</td>
      <td>14.222018</td>
      <td>50076</td>
      <td>1717.80</td>
      <td>7.567401</td>
      <td>4523</td>
      <td>4.385859</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Highest Sum of Sales by County Number
sales_county.sort_values(by='Sale (Dollars) sum', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>County Number</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76</th>
      <td>77.0</td>
      <td>BONDURANT</td>
      <td>7.747219e+06</td>
      <td>158.287410</td>
      <td>19.596528</td>
      <td>50035</td>
      <td>502415.27</td>
      <td>10.265104</td>
      <td>4757</td>
      <td>5.077717</td>
    </tr>
    <tr>
      <th>56</th>
      <td>57.0</td>
      <td>CEDAR RAPIDS</td>
      <td>3.139999e+06</td>
      <td>133.833411</td>
      <td>18.609124</td>
      <td>52402</td>
      <td>219169.42</td>
      <td>9.341464</td>
      <td>2569</td>
      <td>4.754082</td>
    </tr>
    <tr>
      <th>81</th>
      <td>82.0</td>
      <td>DAVENPORT</td>
      <td>2.457277e+06</td>
      <td>147.761716</td>
      <td>19.104453</td>
      <td>52807</td>
      <td>162309.65</td>
      <td>9.760051</td>
      <td>2614</td>
      <td>4.821982</td>
    </tr>
    <tr>
      <th>51</th>
      <td>52.0</td>
      <td>CORALVILLE</td>
      <td>2.077858e+06</td>
      <td>157.855950</td>
      <td>19.351086</td>
      <td>52241</td>
      <td>129627.87</td>
      <td>9.847897</td>
      <td>2670</td>
      <td>5.136183</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>CEDAR FALLS</td>
      <td>1.928017e+06</td>
      <td>128.277892</td>
      <td>18.651688</td>
      <td>50613</td>
      <td>131951.74</td>
      <td>8.779224</td>
      <td>2106</td>
      <td>4.689621</td>
    </tr>
    <tr>
      <th>77</th>
      <td>78.0</td>
      <td>COUNCIL BLUFFS</td>
      <td>1.277581e+06</td>
      <td>140.578914</td>
      <td>17.686863</td>
      <td>51501</td>
      <td>85777.94</td>
      <td>9.438594</td>
      <td>3963</td>
      <td>4.724514</td>
    </tr>
    <tr>
      <th>96</th>
      <td>97.0</td>
      <td>SIOUX CITY</td>
      <td>1.249514e+06</td>
      <td>146.296023</td>
      <td>18.445257</td>
      <td>51104</td>
      <td>84348.74</td>
      <td>9.875745</td>
      <td>2623</td>
      <td>4.952187</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31.0</td>
      <td>DUBUQUE</td>
      <td>1.076389e+06</td>
      <td>139.086347</td>
      <td>17.681968</td>
      <td>52001</td>
      <td>79003.48</td>
      <td>10.208487</td>
      <td>4167</td>
      <td>5.032409</td>
    </tr>
    <tr>
      <th>84</th>
      <td>85.0</td>
      <td>AMES</td>
      <td>1.073666e+06</td>
      <td>120.043206</td>
      <td>18.667560</td>
      <td>50010</td>
      <td>74455.16</td>
      <td>8.324593</td>
      <td>2501</td>
      <td>5.140446</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17.0</td>
      <td>CLEAR LAKE</td>
      <td>6.949539e+05</td>
      <td>109.269481</td>
      <td>17.305428</td>
      <td>50428</td>
      <td>52733.77</td>
      <td>8.291473</td>
      <td>3456</td>
      <td>5.058730</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25.0</td>
      <td>WAUKEE</td>
      <td>6.259259e+05</td>
      <td>231.224943</td>
      <td>17.994723</td>
      <td>50263</td>
      <td>37375.05</td>
      <td>13.806816</td>
      <td>2665</td>
      <td>4.924728</td>
    </tr>
    <tr>
      <th>55</th>
      <td>56.0</td>
      <td>WEST POINT</td>
      <td>4.852360e+05</td>
      <td>146.199458</td>
      <td>18.077259</td>
      <td>52656</td>
      <td>33069.75</td>
      <td>9.963769</td>
      <td>4673</td>
      <td>5.026668</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29.0</td>
      <td>BURLINGTON</td>
      <td>4.668898e+05</td>
      <td>114.377714</td>
      <td>18.416256</td>
      <td>52601</td>
      <td>31462.72</td>
      <td>7.707673</td>
      <td>4898</td>
      <td>4.824831</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30.0</td>
      <td>MILFORD</td>
      <td>4.530934e+05</td>
      <td>132.910942</td>
      <td>17.572744</td>
      <td>51351</td>
      <td>33308.94</td>
      <td>9.770883</td>
      <td>3390</td>
      <td>5.132127</td>
    </tr>
    <tr>
      <th>93</th>
      <td>94.0</td>
      <td>FORT DODGE</td>
      <td>4.435444e+05</td>
      <td>141.076460</td>
      <td>17.690137</td>
      <td>50501</td>
      <td>33250.43</td>
      <td>10.575837</td>
      <td>2644</td>
      <td>4.914656</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23.0</td>
      <td>CLINTON</td>
      <td>4.197627e+05</td>
      <td>117.613542</td>
      <td>17.311216</td>
      <td>52732</td>
      <td>30940.57</td>
      <td>8.669255</td>
      <td>2616</td>
      <td>4.538748</td>
    </tr>
    <tr>
      <th>69</th>
      <td>70.0</td>
      <td>MUSCATINE</td>
      <td>3.571560e+05</td>
      <td>89.850574</td>
      <td>17.795800</td>
      <td>52761</td>
      <td>25751.51</td>
      <td>6.478367</td>
      <td>2662</td>
      <td>4.740953</td>
    </tr>
    <tr>
      <th>63</th>
      <td>64.0</td>
      <td>MARSHALLTOWN</td>
      <td>3.485070e+05</td>
      <td>116.791900</td>
      <td>17.177376</td>
      <td>50158</td>
      <td>24041.88</td>
      <td>8.056930</td>
      <td>2544</td>
      <td>4.935503</td>
    </tr>
    <tr>
      <th>89</th>
      <td>90.0</td>
      <td>OTTUMWA</td>
      <td>3.297726e+05</td>
      <td>93.632206</td>
      <td>17.386486</td>
      <td>52501</td>
      <td>23146.38</td>
      <td>6.571942</td>
      <td>2596</td>
      <td>4.484659</td>
    </tr>
    <tr>
      <th>90</th>
      <td>91.0</td>
      <td>INDIANOLA</td>
      <td>3.075335e+05</td>
      <td>125.013622</td>
      <td>15.750146</td>
      <td>50125</td>
      <td>25009.33</td>
      <td>10.166394</td>
      <td>2549</td>
      <td>4.631610</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.0</td>
      <td>CARROLL</td>
      <td>3.067951e+05</td>
      <td>160.541638</td>
      <td>16.885191</td>
      <td>51401</td>
      <td>22868.38</td>
      <td>11.966709</td>
      <td>4158</td>
      <td>4.922444</td>
    </tr>
    <tr>
      <th>62</th>
      <td>63.0</td>
      <td>PELLA</td>
      <td>2.411637e+05</td>
      <td>92.719600</td>
      <td>17.188992</td>
      <td>50219</td>
      <td>17565.75</td>
      <td>6.753460</td>
      <td>2642</td>
      <td>4.752422</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>OGDEN</td>
      <td>2.230642e+05</td>
      <td>105.968760</td>
      <td>16.299673</td>
      <td>50212</td>
      <td>17533.54</td>
      <td>8.329473</td>
      <td>3635</td>
      <td>4.645178</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42.0</td>
      <td>IOWA FALLS</td>
      <td>2.213883e+05</td>
      <td>133.769378</td>
      <td>15.605638</td>
      <td>50126</td>
      <td>17769.56</td>
      <td>10.736894</td>
      <td>2539</td>
      <td>5.063686</td>
    </tr>
    <tr>
      <th>54</th>
      <td>55.0</td>
      <td>ALGONA</td>
      <td>2.162142e+05</td>
      <td>129.083093</td>
      <td>17.068549</td>
      <td>50511</td>
      <td>17001.74</td>
      <td>10.150293</td>
      <td>3987</td>
      <td>5.310400</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11.0</td>
      <td>STORM LAKE</td>
      <td>2.136521e+05</td>
      <td>78.060691</td>
      <td>18.626529</td>
      <td>50588</td>
      <td>14976.80</td>
      <td>5.471977</td>
      <td>2290</td>
      <td>5.222210</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50.0</td>
      <td>NEWTON</td>
      <td>2.061819e+05</td>
      <td>72.907316</td>
      <td>16.797057</td>
      <td>50208</td>
      <td>16185.94</td>
      <td>5.723458</td>
      <td>4604</td>
      <td>4.626227</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>SUMNER</td>
      <td>2.046495e+05</td>
      <td>91.361402</td>
      <td>16.642927</td>
      <td>50674</td>
      <td>16009.78</td>
      <td>7.147223</td>
      <td>3717</td>
      <td>4.951129</td>
    </tr>
    <tr>
      <th>83</th>
      <td>84.0</td>
      <td>SIOUX CENTER</td>
      <td>1.964274e+05</td>
      <td>147.912176</td>
      <td>16.474216</td>
      <td>51250</td>
      <td>15002.72</td>
      <td>11.297229</td>
      <td>3981</td>
      <td>5.001227</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21.0</td>
      <td>SPENCER</td>
      <td>1.772498e+05</td>
      <td>92.462087</td>
      <td>16.811390</td>
      <td>51301</td>
      <td>14604.76</td>
      <td>7.618550</td>
      <td>4234</td>
      <td>4.829864</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>66</th>
      <td>67.0</td>
      <td>ONAWA</td>
      <td>6.777953e+04</td>
      <td>54.007594</td>
      <td>16.870042</td>
      <td>51040</td>
      <td>5024.82</td>
      <td>4.003841</td>
      <td>3723</td>
      <td>4.867068</td>
    </tr>
    <tr>
      <th>80</th>
      <td>81.0</td>
      <td>SAC CITY</td>
      <td>6.402323e+04</td>
      <td>71.614351</td>
      <td>15.744310</td>
      <td>50583</td>
      <td>5232.32</td>
      <td>5.852707</td>
      <td>2200</td>
      <td>5.154128</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43.0</td>
      <td>MISSOURI VALLEY</td>
      <td>6.195705e+04</td>
      <td>56.375842</td>
      <td>16.478385</td>
      <td>51555</td>
      <td>4728.92</td>
      <td>4.302930</td>
      <td>4152</td>
      <td>4.556160</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46.0</td>
      <td>HUMBOLDT</td>
      <td>6.142825e+04</td>
      <td>104.469813</td>
      <td>16.136856</td>
      <td>50548</td>
      <td>4826.52</td>
      <td>8.208367</td>
      <td>2606</td>
      <td>4.675000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35.0</td>
      <td>HAMPTON</td>
      <td>5.759978e+04</td>
      <td>81.817869</td>
      <td>16.536598</td>
      <td>50441</td>
      <td>4614.31</td>
      <td>6.554418</td>
      <td>4306</td>
      <td>4.969659</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>STUART</td>
      <td>5.554420e+04</td>
      <td>95.109932</td>
      <td>15.659537</td>
      <td>50250</td>
      <td>4359.84</td>
      <td>7.465479</td>
      <td>3461</td>
      <td>4.673099</td>
    </tr>
    <tr>
      <th>64</th>
      <td>65.0</td>
      <td>PACIFIC JUNCTION</td>
      <td>5.537766e+04</td>
      <td>109.226154</td>
      <td>15.245998</td>
      <td>51561</td>
      <td>4383.28</td>
      <td>8.645523</td>
      <td>4602</td>
      <td>4.810335</td>
    </tr>
    <tr>
      <th>65</th>
      <td>66.0</td>
      <td>OSAGE</td>
      <td>5.145385e+04</td>
      <td>50.944406</td>
      <td>15.708389</td>
      <td>50461</td>
      <td>4487.22</td>
      <td>4.442792</td>
      <td>4945</td>
      <td>4.831554</td>
    </tr>
    <tr>
      <th>58</th>
      <td>59.0</td>
      <td>CHARITON</td>
      <td>4.791119e+04</td>
      <td>100.865663</td>
      <td>17.293334</td>
      <td>50049</td>
      <td>3385.39</td>
      <td>7.127137</td>
      <td>2551</td>
      <td>4.748968</td>
    </tr>
    <tr>
      <th>75</th>
      <td>76.0</td>
      <td>LAURENS</td>
      <td>4.479333e+04</td>
      <td>85.320629</td>
      <td>14.413622</td>
      <td>50554</td>
      <td>4117.57</td>
      <td>7.842990</td>
      <td>5005</td>
      <td>4.386476</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19.0</td>
      <td>NEW HAMPTON</td>
      <td>4.445726e+04</td>
      <td>95.813060</td>
      <td>14.668877</td>
      <td>50659</td>
      <td>3735.23</td>
      <td>8.050065</td>
      <td>3766</td>
      <td>4.732220</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38.0</td>
      <td>GRUNDY CENTER</td>
      <td>3.895831e+04</td>
      <td>68.830936</td>
      <td>14.522364</td>
      <td>50638</td>
      <td>3405.30</td>
      <td>6.016431</td>
      <td>4495</td>
      <td>4.229117</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39.0</td>
      <td>GUTHRIE CENTER</td>
      <td>3.784590e+04</td>
      <td>86.603890</td>
      <td>14.511792</td>
      <td>50115</td>
      <td>2989.44</td>
      <td>6.840824</td>
      <td>5191</td>
      <td>4.375721</td>
    </tr>
    <tr>
      <th>97</th>
      <td>98.0</td>
      <td>NORTHWOOD</td>
      <td>3.754885e+04</td>
      <td>97.025452</td>
      <td>14.370646</td>
      <td>50459</td>
      <td>3240.30</td>
      <td>8.372868</td>
      <td>3664</td>
      <td>4.565840</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41.0</td>
      <td>BRITT</td>
      <td>3.701484e+04</td>
      <td>101.969256</td>
      <td>12.530454</td>
      <td>50423</td>
      <td>3622.25</td>
      <td>9.978650</td>
      <td>3045</td>
      <td>4.536198</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.0</td>
      <td>LOHRVILLE</td>
      <td>3.599253e+04</td>
      <td>84.888042</td>
      <td>13.882661</td>
      <td>51453</td>
      <td>3001.38</td>
      <td>7.078726</td>
      <td>5193</td>
      <td>4.225118</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12.0</td>
      <td>PARKERSBURG</td>
      <td>3.409963e+04</td>
      <td>84.824950</td>
      <td>14.234286</td>
      <td>50665</td>
      <td>3121.87</td>
      <td>7.765846</td>
      <td>4110</td>
      <td>4.378035</td>
    </tr>
    <tr>
      <th>67</th>
      <td>68.0</td>
      <td>ALBIA</td>
      <td>3.381553e+04</td>
      <td>96.066847</td>
      <td>15.920341</td>
      <td>52531</td>
      <td>2497.82</td>
      <td>7.096080</td>
      <td>2559</td>
      <td>4.541165</td>
    </tr>
    <tr>
      <th>57</th>
      <td>58.0</td>
      <td>COLUMBUS JUNCTION</td>
      <td>3.070261e+04</td>
      <td>63.435145</td>
      <td>16.030986</td>
      <td>52738</td>
      <td>2182.69</td>
      <td>4.509690</td>
      <td>3403</td>
      <td>4.033430</td>
    </tr>
    <tr>
      <th>71</th>
      <td>72.0</td>
      <td>SIBLEY</td>
      <td>3.031341e+04</td>
      <td>86.362991</td>
      <td>16.723692</td>
      <td>51249</td>
      <td>2251.62</td>
      <td>6.414872</td>
      <td>4891</td>
      <td>5.208063</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89.0</td>
      <td>FARMINGTON</td>
      <td>2.156597e+04</td>
      <td>88.024367</td>
      <td>16.018784</td>
      <td>52626</td>
      <td>1566.49</td>
      <td>6.393837</td>
      <td>4578</td>
      <td>4.732857</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27.0</td>
      <td>LAMONI</td>
      <td>1.974532e+04</td>
      <td>88.544036</td>
      <td>16.086738</td>
      <td>50140</td>
      <td>1452.91</td>
      <td>6.515291</td>
      <td>4419</td>
      <td>4.293722</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>EXIRA</td>
      <td>1.965554e+04</td>
      <td>86.588282</td>
      <td>14.222018</td>
      <td>50076</td>
      <td>1717.80</td>
      <td>7.567401</td>
      <td>4523</td>
      <td>4.385859</td>
    </tr>
    <tr>
      <th>53</th>
      <td>54.0</td>
      <td>SIGOURNEY</td>
      <td>1.829160e+04</td>
      <td>53.328280</td>
      <td>14.264715</td>
      <td>52591</td>
      <td>1615.71</td>
      <td>4.710525</td>
      <td>4883</td>
      <td>4.271866</td>
    </tr>
    <tr>
      <th>79</th>
      <td>80.0</td>
      <td>MOUNT AYR</td>
      <td>1.822317e+04</td>
      <td>90.662537</td>
      <td>12.611766</td>
      <td>50854</td>
      <td>1596.30</td>
      <td>7.941791</td>
      <td>4779</td>
      <td>4.682786</td>
    </tr>
    <tr>
      <th>86</th>
      <td>87.0</td>
      <td>LENOX</td>
      <td>1.498109e+04</td>
      <td>50.272114</td>
      <td>16.118487</td>
      <td>50851</td>
      <td>1158.93</td>
      <td>3.889027</td>
      <td>5110</td>
      <td>4.537181</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26.0</td>
      <td>BLOOMFIELD</td>
      <td>1.319752e+04</td>
      <td>65.012414</td>
      <td>14.345534</td>
      <td>52537</td>
      <td>1076.75</td>
      <td>5.304187</td>
      <td>3013</td>
      <td>4.636749</td>
    </tr>
    <tr>
      <th>92</th>
      <td>93.0</td>
      <td>CORYDON</td>
      <td>1.293362e+04</td>
      <td>80.835125</td>
      <td>14.912528</td>
      <td>50060</td>
      <td>1060.70</td>
      <td>6.629375</td>
      <td>2669</td>
      <td>4.874688</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>CORNING</td>
      <td>1.244171e+04</td>
      <td>53.169701</td>
      <td>17.063570</td>
      <td>50841</td>
      <td>981.38</td>
      <td>4.193932</td>
      <td>2327</td>
      <td>5.091111</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36.0</td>
      <td>HAMBURG</td>
      <td>3.230450e+03</td>
      <td>119.646296</td>
      <td>14.754744</td>
      <td>51640</td>
      <td>226.10</td>
      <td>8.374074</td>
      <td>4573</td>
      <td>3.431481</td>
    </tr>
  </tbody>
</table>
<p>99 rows × 10 columns</p>
</div>


#### Recommendations
What are great locations?
insert images

Evaluate model performace and its ability to predict future sales using cross-validation
# Cross-Validation Methods

### For Price per Bottle RidgeCV and LassoCV


```python
# Ridge CV
lm = linear_model.RidgeCV()

X = df[['Price per Bottle']]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(['Price per Bottle'], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.scatter(X, y)
plt.plot(X, predictions, label="Linear Fit")
plt.xlabel("Predicted Values from Price per bottle")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('Price per Bottle', 4.9374446722795255)]
    Intercept 56.4414188886



![png](output_58_1.png)


    MSE: 144336.018459



```python
# Lasso CV
lm = linear_model.LassoCV()

X = df[['Price per Bottle']]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(['Price per Bottle'], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.scatter(X, y)
plt.plot(X, predictions, label="Linear Fit")
plt.xlabel("Predicted Values from Store Number")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('Price per Bottle', 4.9233842495998896)]
    Intercept 56.6475566125



![png](output_59_1.png)


    MSE: 144336.040506


### For Bottles Sold RidgeCV and LassoCV


```python
# Ridge CV
lm = linear_model.RidgeCV()

X = df[['Bottles Sold']]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(['Store Number'], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.scatter(X, y)
plt.plot(X, predictions, label="Linear Fit")
plt.xlabel("Predicted Values from Bottles Sold")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('Store Number', 13.156936010345817)]
    Intercept -1.08340684474



![png](output_61_1.png)


    MSE: 46744.0651775



```python
# Lasso CV
lm = linear_model.LassoCV()

X = df[['Bottles Sold']]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(['Bottles Sold'], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.scatter(X, y)
plt.plot(X, predictions, label="Linear Fit")
plt.xlabel("Predicted Values from Bottles Sold")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('Bottles Sold', 13.143779402802783)]
    Intercept -0.953498135004



![png](output_62_1.png)


    MSE: 46744.1654876


### For County Number RidgeCV and LassoCV


```python
# Ridge CV
lm = linear_model.RidgeCV()

X = df[['County Number']]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(['County Number'], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.scatter(X, y)
plt.plot(X, predictions, label="Linear Fit")
plt.xlabel("Predicted Values from County Number")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('County Number', 0.2773759308620356)]
    Intercept 112.954420911



![png](output_64_1.png)


    MSE: 146996.594944



```python
# Lasso CV
lm = linear_model.LassoCV()

X = df[['County Number']]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(['County Number'], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.scatter(X, y)
plt.plot(X, predictions, label="Linear Fit")
plt.xlabel("Predicted Values from County Number")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('County Number', 0.27709856776439412)]
    Intercept 112.970294302



![png](output_65_1.png)


    MSE: 146996.595002



```python
# Ridge CV
lm = linear_model.RidgeCV()

X = df[["Price per Bottle", "Bottles Sold"]]
y = df['Sale (Dollars)']

model = lm.fit(X, y)
predictions = model.predict(X)

print "Coefficients", zip(["Price per Bottle", "Bottles Sold"], model.coef_)
print "Intercept", model.intercept_

# Plot the model
plt.plot(predictions, y, label="Linear Fit")
plt.xlabel("Predicted Values from Price per Bottle and Bottles Sold")
plt.ylabel("Actual Values Sale (Dollars)")
plt.show()
print "MSE:", mean_squared_error(y, predictions)
```

    Coefficients [('Price per Bottle', 6.8445069596054964), ('Bottles Sold', 13.345129694789648)]
    Intercept -103.287914716



![png](output_66_1.png)


    MSE: 41541.3227776


#### Bonus
Recommend target volume sold and price per bottle



```python
#sort by bottle volume
sales_volume = df.groupby(by=["Bottle Volume (ml)"], as_index=False)

sales_volume = sales_volume.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Store Number": lambda x: x.iloc[0], # just extract once, should be the same
                   "Zip Code": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0]})
# Collapse the column indices
sales_volume.columns = [' '.join(col).strip() for col in sales_volume.columns.values]
# Rename columns
sales_volume = sales_volume.rename(columns={'Zip Code <lambda>': 'Zip Code', 'City <lambda>': 'City', 'Store Number <lambda>': 'Store Number'})
# Transform into DataFrame
sales_volume = pd.DataFrame(sales_volume)
# Quick check
sales_volume.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bottle Volume (ml)</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>DES MOINES</td>
      <td>1367.58</td>
      <td>97.684286</td>
      <td>209.500000</td>
      <td>50312</td>
      <td>5.40</td>
      <td>0.385714</td>
      <td>4669</td>
      <td>3.493571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>DAVENPORT</td>
      <td>63723.93</td>
      <td>74.530912</td>
      <td>19.453216</td>
      <td>52804</td>
      <td>3899.00</td>
      <td>4.560234</td>
      <td>2625</td>
      <td>0.650398</td>
    </tr>
    <tr>
      <th>2</th>
      <td>150</td>
      <td>PRIMGHAR</td>
      <td>110.10</td>
      <td>36.700000</td>
      <td>52.666667</td>
      <td>51245</td>
      <td>1.80</td>
      <td>0.600000</td>
      <td>4878</td>
      <td>2.633333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200</td>
      <td>WATERLOO</td>
      <td>753064.32</td>
      <td>76.437710</td>
      <td>23.059653</td>
      <td>50703</td>
      <td>38090.80</td>
      <td>3.866301</td>
      <td>2130</td>
      <td>1.538674</td>
    </tr>
    <tr>
      <th>4</th>
      <td>250</td>
      <td>WAVERLY</td>
      <td>9.00</td>
      <td>9.000000</td>
      <td>36.000000</td>
      <td>50677</td>
      <td>0.25</td>
      <td>0.250000</td>
      <td>2651</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Highest Sum of Sales by Bottle Volume
sales_volume.sort_values(by='Sale (Dollars) sum', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bottle Volume (ml)</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>750</td>
      <td>SUMNER</td>
      <td>1.364331e+07</td>
      <td>111.840485</td>
      <td>20.925874</td>
      <td>50674</td>
      <td>717710.25</td>
      <td>5.883401</td>
      <td>3717</td>
      <td>5.242171</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1750</td>
      <td>AMES</td>
      <td>8.938938e+06</td>
      <td>165.239063</td>
      <td>10.066017</td>
      <td>50010</td>
      <td>915633.25</td>
      <td>16.925768</td>
      <td>2501</td>
      <td>5.897880</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1000</td>
      <td>CEDAR FALLS</td>
      <td>8.622779e+06</td>
      <td>235.357125</td>
      <td>15.268948</td>
      <td>50613</td>
      <td>589775.00</td>
      <td>16.097797</td>
      <td>2106</td>
      <td>5.091952</td>
    </tr>
    <tr>
      <th>6</th>
      <td>375</td>
      <td>AMES</td>
      <td>1.878093e+06</td>
      <td>69.913733</td>
      <td>20.291965</td>
      <td>50010</td>
      <td>111092.76</td>
      <td>4.135531</td>
      <td>4438</td>
      <td>2.539503</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200</td>
      <td>WATERLOO</td>
      <td>7.530643e+05</td>
      <td>76.437710</td>
      <td>23.059653</td>
      <td>50703</td>
      <td>38090.80</td>
      <td>3.866301</td>
      <td>2130</td>
      <td>1.538674</td>
    </tr>
    <tr>
      <th>8</th>
      <td>500</td>
      <td>IOWA CITY</td>
      <td>3.550629e+05</td>
      <td>29.732283</td>
      <td>18.912243</td>
      <td>52245</td>
      <td>20250.00</td>
      <td>1.695696</td>
      <td>2545</td>
      <td>3.153623</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3000</td>
      <td>DAVENPORT</td>
      <td>1.688157e+05</td>
      <td>105.509831</td>
      <td>14.699869</td>
      <td>52802</td>
      <td>11469.00</td>
      <td>7.168125</td>
      <td>4892</td>
      <td>14.699869</td>
    </tr>
    <tr>
      <th>9</th>
      <td>600</td>
      <td>INDEPENDENCE</td>
      <td>1.196924e+05</td>
      <td>36.558464</td>
      <td>21.693296</td>
      <td>50644</td>
      <td>6240.60</td>
      <td>1.906109</td>
      <td>5001</td>
      <td>4.331066</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>DAVENPORT</td>
      <td>6.372393e+04</td>
      <td>74.530912</td>
      <td>19.453216</td>
      <td>52804</td>
      <td>3899.00</td>
      <td>4.560234</td>
      <td>2625</td>
      <td>0.650398</td>
    </tr>
    <tr>
      <th>5</th>
      <td>300</td>
      <td>IOWA CITY</td>
      <td>4.430309e+04</td>
      <td>33.210712</td>
      <td>44.918991</td>
      <td>52246</td>
      <td>1076.10</td>
      <td>0.806672</td>
      <td>3565</td>
      <td>4.494813</td>
    </tr>
    <tr>
      <th>12</th>
      <td>800</td>
      <td>WATERLOO</td>
      <td>3.167470e+04</td>
      <td>99.920189</td>
      <td>12.280915</td>
      <td>50701</td>
      <td>2061.60</td>
      <td>6.503470</td>
      <td>2564</td>
      <td>3.275741</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1200</td>
      <td>CEDAR FALLS</td>
      <td>2.366798e+04</td>
      <td>94.671920</td>
      <td>40.876433</td>
      <td>50647</td>
      <td>543.60</td>
      <td>2.174400</td>
      <td>4947</td>
      <td>16.353000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2400</td>
      <td>SIOUX CITY</td>
      <td>1.649581e+04</td>
      <td>160.153495</td>
      <td>17.945833</td>
      <td>51106</td>
      <td>919.20</td>
      <td>8.924272</td>
      <td>3879</td>
      <td>14.360000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>850</td>
      <td>WAUKEE</td>
      <td>4.922330e+03</td>
      <td>205.097083</td>
      <td>34.681863</td>
      <td>50263</td>
      <td>147.05</td>
      <td>6.127083</td>
      <td>2665</td>
      <td>9.827917</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3600</td>
      <td>WAUKEE</td>
      <td>4.710000e+03</td>
      <td>362.307692</td>
      <td>4.166667</td>
      <td>50263</td>
      <td>1130.40</td>
      <td>86.953846</td>
      <td>2665</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6000</td>
      <td>ALGONA</td>
      <td>4.603500e+03</td>
      <td>657.642857</td>
      <td>24.750000</td>
      <td>50511</td>
      <td>186.00</td>
      <td>26.571429</td>
      <td>3987</td>
      <td>49.500000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2550</td>
      <td>DES MOINES</td>
      <td>4.207780e+03</td>
      <td>701.296667</td>
      <td>13.525490</td>
      <td>50312</td>
      <td>311.10</td>
      <td>51.850000</td>
      <td>2248</td>
      <td>11.500000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>400</td>
      <td>WEST DES MOINES</td>
      <td>3.299820e+03</td>
      <td>113.786897</td>
      <td>21.334483</td>
      <td>50266</td>
      <td>161.60</td>
      <td>5.572414</td>
      <td>2619</td>
      <td>2.844828</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1125</td>
      <td>SWISHER</td>
      <td>1.827000e+03</td>
      <td>365.400000</td>
      <td>58.103571</td>
      <td>52338</td>
      <td>31.49</td>
      <td>6.298000</td>
      <td>9001</td>
      <td>21.750000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4800</td>
      <td>JOHNSTON</td>
      <td>1.537920e+03</td>
      <td>90.465882</td>
      <td>15.529412</td>
      <td>50131</td>
      <td>100.80</td>
      <td>5.929412</td>
      <td>2587</td>
      <td>24.847059</td>
    </tr>
    <tr>
      <th>0</th>
      <td>50</td>
      <td>DES MOINES</td>
      <td>1.367580e+03</td>
      <td>97.684286</td>
      <td>209.500000</td>
      <td>50312</td>
      <td>5.40</td>
      <td>0.385714</td>
      <td>4669</td>
      <td>3.493571</td>
    </tr>
    <tr>
      <th>10</th>
      <td>603</td>
      <td>AMES</td>
      <td>6.804000e+02</td>
      <td>68.040000</td>
      <td>56.624932</td>
      <td>50010</td>
      <td>12.04</td>
      <td>1.204000</td>
      <td>2500</td>
      <td>11.340000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2250</td>
      <td>FORT MADISON</td>
      <td>4.567500e+02</td>
      <td>76.125000</td>
      <td>29.000000</td>
      <td>52627</td>
      <td>15.75</td>
      <td>2.625000</td>
      <td>3549</td>
      <td>21.750000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>950</td>
      <td>DES MOINES</td>
      <td>3.451000e+02</td>
      <td>57.516667</td>
      <td>15.792982</td>
      <td>50310</td>
      <td>21.85</td>
      <td>3.641667</td>
      <td>2627</td>
      <td>5.001667</td>
    </tr>
    <tr>
      <th>14</th>
      <td>900</td>
      <td>DAVENPORT</td>
      <td>2.647200e+02</td>
      <td>132.360000</td>
      <td>12.255556</td>
      <td>52807</td>
      <td>21.60</td>
      <td>10.800000</td>
      <td>4750</td>
      <td>3.680000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>150</td>
      <td>PRIMGHAR</td>
      <td>1.101000e+02</td>
      <td>36.700000</td>
      <td>52.666667</td>
      <td>51245</td>
      <td>1.80</td>
      <td>0.600000</td>
      <td>4878</td>
      <td>2.633333</td>
    </tr>
    <tr>
      <th>26</th>
      <td>4500</td>
      <td>AMES</td>
      <td>1.034100e+02</td>
      <td>103.410000</td>
      <td>22.980000</td>
      <td>50010</td>
      <td>4.50</td>
      <td>4.500000</td>
      <td>2500</td>
      <td>34.470000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1500</td>
      <td>BANCROFT</td>
      <td>5.229000e+01</td>
      <td>52.290000</td>
      <td>11.620000</td>
      <td>50517</td>
      <td>4.50</td>
      <td>4.500000</td>
      <td>3842</td>
      <td>5.810000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>250</td>
      <td>WAVERLY</td>
      <td>9.000000e+00</td>
      <td>9.000000</td>
      <td>36.000000</td>
      <td>50677</td>
      <td>0.25</td>
      <td>0.250000</td>
      <td>2651</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sort by price per bottle
sales_bp = df.groupby(by=["Price per Bottle"], as_index=False)

sales_bp = sales_bp.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Store Number": lambda x: x.iloc[0], # just extract once, should be the same
                   "Zip Code": lambda x: x.iloc[0],
                   "City": lambda x: x.iloc[0]})
# Collapse the column indices
sales_bp.columns = [' '.join(col).strip() for col in sales_bp.columns.values]
# Rename columns
sales_bp = sales_bp.rename(columns={'Zip Code <lambda>': 'Zip Code', 'City <lambda>': 'City', 'Store Number <lambda>': 'Store Number'})
# Transform into DataFrame
sales_bp = pd.DataFrame(sales_bp)
# Quick check
sales_bp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price per Bottle</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.34</td>
      <td>COUNCIL BLUFFS</td>
      <td>17283.32</td>
      <td>78.919269</td>
      <td>13.4</td>
      <td>51501</td>
      <td>1289.8</td>
      <td>5.889498</td>
      <td>3963</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.34</td>
      <td>ARNOLDS PARK</td>
      <td>8561.26</td>
      <td>182.154468</td>
      <td>13.4</td>
      <td>51331</td>
      <td>638.9</td>
      <td>13.593617</td>
      <td>5068</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.46</td>
      <td>WATERLOO</td>
      <td>7691.28</td>
      <td>51.968108</td>
      <td>14.6</td>
      <td>50703</td>
      <td>526.8</td>
      <td>3.559459</td>
      <td>4935</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.50</td>
      <td>DAVENPORT</td>
      <td>10284.00</td>
      <td>60.140351</td>
      <td>15.0</td>
      <td>52804</td>
      <td>685.6</td>
      <td>4.009357</td>
      <td>2625</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.56</td>
      <td>WATERLOO</td>
      <td>1506.96</td>
      <td>71.760000</td>
      <td>15.6</td>
      <td>50701</td>
      <td>96.6</td>
      <td>4.600000</td>
      <td>3993</td>
      <td>0.52</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales_bp.sort_values(by='Sale (Dollars) sum', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price per Bottle</th>
      <th>City</th>
      <th>Sale (Dollars) sum</th>
      <th>Sale (Dollars) mean</th>
      <th>Price per Liter mean</th>
      <th>Zip Code</th>
      <th>Volume Sold (Liters) sum</th>
      <th>Volume Sold (Liters) mean</th>
      <th>Store Number</th>
      <th>Margin mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>950</th>
      <td>22.50</td>
      <td>CEDAR RAPIDS</td>
      <td>949882.50</td>
      <td>249.051521</td>
      <td>25.053566</td>
      <td>52402</td>
      <td>43095.22</td>
      <td>11.299219</td>
      <td>3628</td>
      <td>7.502950</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>27.00</td>
      <td>DECORAH</td>
      <td>718686.00</td>
      <td>387.431806</td>
      <td>16.005237</td>
      <td>52101</td>
      <td>46340.50</td>
      <td>24.981402</td>
      <td>3819</td>
      <td>9.005391</td>
    </tr>
    <tr>
      <th>781</th>
      <td>17.63</td>
      <td>EAGLE GROVE</td>
      <td>714896.50</td>
      <td>507.018794</td>
      <td>17.630000</td>
      <td>50533</td>
      <td>40550.00</td>
      <td>28.758865</td>
      <td>4236</td>
      <td>5.880000</td>
    </tr>
    <tr>
      <th>569</th>
      <td>13.50</td>
      <td>IOWA CITY</td>
      <td>632245.50</td>
      <td>143.203964</td>
      <td>18.842637</td>
      <td>52240</td>
      <td>36714.97</td>
      <td>8.315961</td>
      <td>2512</td>
      <td>4.502661</td>
    </tr>
    <tr>
      <th>519</th>
      <td>12.38</td>
      <td>JOHNSTON</td>
      <td>551813.74</td>
      <td>91.877080</td>
      <td>16.641051</td>
      <td>50131</td>
      <td>33368.24</td>
      <td>5.555818</td>
      <td>2587</td>
      <td>4.130000</td>
    </tr>
    <tr>
      <th>704</th>
      <td>15.74</td>
      <td>MASON CITY</td>
      <td>545878.94</td>
      <td>143.388216</td>
      <td>18.656980</td>
      <td>50401</td>
      <td>31052.50</td>
      <td>8.156685</td>
      <td>4376</td>
      <td>5.250000</td>
    </tr>
    <tr>
      <th>650</th>
      <td>14.93</td>
      <td>WILLIAMSBURG</td>
      <td>463262.97</td>
      <td>506.298328</td>
      <td>8.552407</td>
      <td>52361</td>
      <td>54292.50</td>
      <td>59.336066</td>
      <td>4807</td>
      <td>5.229180</td>
    </tr>
    <tr>
      <th>1065</th>
      <td>27.74</td>
      <td>CARROLL</td>
      <td>428194.64</td>
      <td>211.977545</td>
      <td>35.289537</td>
      <td>51401</td>
      <td>13036.45</td>
      <td>6.453688</td>
      <td>4158</td>
      <td>9.250000</td>
    </tr>
    <tr>
      <th>439</th>
      <td>10.76</td>
      <td>ALTOONA</td>
      <td>424589.60</td>
      <td>142.050719</td>
      <td>6.186970</td>
      <td>50009</td>
      <td>68997.00</td>
      <td>23.083640</td>
      <td>4919</td>
      <td>3.590000</td>
    </tr>
    <tr>
      <th>419</th>
      <td>10.38</td>
      <td>WINDSOR HEIGHTS</td>
      <td>411317.88</td>
      <td>191.044069</td>
      <td>7.179196</td>
      <td>50311</td>
      <td>56210.75</td>
      <td>26.108105</td>
      <td>2620</td>
      <td>3.460000</td>
    </tr>
    <tr>
      <th>828</th>
      <td>18.75</td>
      <td>STORM LAKE</td>
      <td>407662.50</td>
      <td>192.657136</td>
      <td>22.797816</td>
      <td>50588</td>
      <td>18537.75</td>
      <td>8.760751</td>
      <td>2290</td>
      <td>6.254253</td>
    </tr>
    <tr>
      <th>661</th>
      <td>15.00</td>
      <td>WEST POINT</td>
      <td>398880.00</td>
      <td>94.476551</td>
      <td>18.213839</td>
      <td>52656</td>
      <td>25433.50</td>
      <td>6.024041</td>
      <td>4673</td>
      <td>5.000533</td>
    </tr>
    <tr>
      <th>1067</th>
      <td>27.75</td>
      <td>WAUKON</td>
      <td>396908.25</td>
      <td>485.811812</td>
      <td>27.817931</td>
      <td>52172</td>
      <td>14299.00</td>
      <td>17.501836</td>
      <td>3857</td>
      <td>9.250000</td>
    </tr>
    <tr>
      <th>948</th>
      <td>22.49</td>
      <td>ELDRIDGE</td>
      <td>376100.27</td>
      <td>248.908187</td>
      <td>25.210970</td>
      <td>52748</td>
      <td>15663.70</td>
      <td>10.366446</td>
      <td>3732</td>
      <td>7.509265</td>
    </tr>
    <tr>
      <th>697</th>
      <td>15.68</td>
      <td>CLIVE</td>
      <td>366849.28</td>
      <td>229.855439</td>
      <td>8.960000</td>
      <td>50325</td>
      <td>40943.00</td>
      <td>25.653509</td>
      <td>4731</td>
      <td>5.230000</td>
    </tr>
    <tr>
      <th>952</th>
      <td>22.61</td>
      <td>MARION</td>
      <td>346995.67</td>
      <td>235.091917</td>
      <td>30.146667</td>
      <td>50129</td>
      <td>11510.25</td>
      <td>7.798272</td>
      <td>2514</td>
      <td>7.540000</td>
    </tr>
    <tr>
      <th>933</th>
      <td>22.13</td>
      <td>AMES</td>
      <td>333366.32</td>
      <td>215.352920</td>
      <td>12.645714</td>
      <td>50010</td>
      <td>26362.00</td>
      <td>17.029716</td>
      <td>2501</td>
      <td>7.380000</td>
    </tr>
    <tr>
      <th>765</th>
      <td>17.24</td>
      <td>BONDURANT</td>
      <td>323318.96</td>
      <td>131.483920</td>
      <td>23.036160</td>
      <td>50035</td>
      <td>14064.99</td>
      <td>5.719801</td>
      <td>4757</td>
      <td>5.750000</td>
    </tr>
    <tr>
      <th>379</th>
      <td>9.75</td>
      <td>OSKALOOSA</td>
      <td>314886.00</td>
      <td>65.834414</td>
      <td>11.949415</td>
      <td>52577</td>
      <td>29264.13</td>
      <td>6.118363</td>
      <td>2600</td>
      <td>3.251528</td>
    </tr>
    <tr>
      <th>566</th>
      <td>13.47</td>
      <td>SPENCER</td>
      <td>312517.47</td>
      <td>172.566245</td>
      <td>17.330968</td>
      <td>51301</td>
      <td>18081.75</td>
      <td>9.984401</td>
      <td>2565</td>
      <td>4.490000</td>
    </tr>
    <tr>
      <th>755</th>
      <td>17.01</td>
      <td>DAVENPORT</td>
      <td>308255.22</td>
      <td>421.113689</td>
      <td>17.250123</td>
      <td>52804</td>
      <td>18098.00</td>
      <td>24.724044</td>
      <td>3917</td>
      <td>5.670000</td>
    </tr>
    <tr>
      <th>737</th>
      <td>16.50</td>
      <td>CEDAR RAPIDS</td>
      <td>291901.50</td>
      <td>141.493698</td>
      <td>15.486912</td>
      <td>52402</td>
      <td>19829.75</td>
      <td>9.612094</td>
      <td>2568</td>
      <td>5.500000</td>
    </tr>
    <tr>
      <th>1254</th>
      <td>40.50</td>
      <td>DUBUQUE</td>
      <td>279693.00</td>
      <td>384.193681</td>
      <td>53.796016</td>
      <td>52001</td>
      <td>5186.00</td>
      <td>7.123626</td>
      <td>2649</td>
      <td>13.500000</td>
    </tr>
    <tr>
      <th>478</th>
      <td>11.43</td>
      <td>SHELLSBURG</td>
      <td>265233.15</td>
      <td>116.330329</td>
      <td>10.411613</td>
      <td>52332</td>
      <td>25443.00</td>
      <td>11.159211</td>
      <td>4346</td>
      <td>3.810000</td>
    </tr>
    <tr>
      <th>830</th>
      <td>18.89</td>
      <td>CEDAR FALLS</td>
      <td>265177.82</td>
      <td>280.314820</td>
      <td>15.231049</td>
      <td>50613</td>
      <td>19539.50</td>
      <td>20.654863</td>
      <td>2106</td>
      <td>6.300000</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>27.57</td>
      <td>SCHLESWIG</td>
      <td>251024.85</td>
      <td>1050.313180</td>
      <td>27.570000</td>
      <td>51461</td>
      <td>9105.00</td>
      <td>38.096234</td>
      <td>4097</td>
      <td>9.190000</td>
    </tr>
    <tr>
      <th>283</th>
      <td>7.85</td>
      <td>TOLEDO</td>
      <td>248460.35</td>
      <td>102.500144</td>
      <td>10.574615</td>
      <td>52342</td>
      <td>23689.05</td>
      <td>9.772710</td>
      <td>3942</td>
      <td>2.620000</td>
    </tr>
    <tr>
      <th>964</th>
      <td>23.00</td>
      <td>BURLINGTON</td>
      <td>245341.00</td>
      <td>384.547022</td>
      <td>13.142857</td>
      <td>52601</td>
      <td>18667.25</td>
      <td>29.259013</td>
      <td>2506</td>
      <td>7.670000</td>
    </tr>
    <tr>
      <th>416</th>
      <td>10.35</td>
      <td>AMES</td>
      <td>242821.35</td>
      <td>99.272833</td>
      <td>13.800000</td>
      <td>50010</td>
      <td>17595.75</td>
      <td>7.193684</td>
      <td>4674</td>
      <td>3.450000</td>
    </tr>
    <tr>
      <th>611</th>
      <td>14.25</td>
      <td>AMES</td>
      <td>242221.50</td>
      <td>158.108029</td>
      <td>13.995314</td>
      <td>50010</td>
      <td>17551.50</td>
      <td>11.456593</td>
      <td>2501</td>
      <td>4.750000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>218</th>
      <td>6.81</td>
      <td>DES MOINES</td>
      <td>47.67</td>
      <td>47.670000</td>
      <td>18.194656</td>
      <td>50317</td>
      <td>2.62</td>
      <td>2.620000</td>
      <td>4599</td>
      <td>2.270000</td>
    </tr>
    <tr>
      <th>482</th>
      <td>11.53</td>
      <td>ANKENY</td>
      <td>46.12</td>
      <td>23.060000</td>
      <td>15.373333</td>
      <td>50023</td>
      <td>3.00</td>
      <td>1.500000</td>
      <td>4165</td>
      <td>3.840000</td>
    </tr>
    <tr>
      <th>642</th>
      <td>14.80</td>
      <td>CEDAR RAPIDS</td>
      <td>44.40</td>
      <td>22.200000</td>
      <td>8.457143</td>
      <td>52404</td>
      <td>5.25</td>
      <td>2.625000</td>
      <td>2552</td>
      <td>5.270000</td>
    </tr>
    <tr>
      <th>445</th>
      <td>10.81</td>
      <td>CEDAR FALLS</td>
      <td>43.24</td>
      <td>43.240000</td>
      <td>14.413333</td>
      <td>50613</td>
      <td>3.00</td>
      <td>3.000000</td>
      <td>4959</td>
      <td>3.940000</td>
    </tr>
    <tr>
      <th>597</th>
      <td>14.03</td>
      <td>SIOUX CITY</td>
      <td>42.09</td>
      <td>42.090000</td>
      <td>18.706667</td>
      <td>51105</td>
      <td>2.25</td>
      <td>2.250000</td>
      <td>2621</td>
      <td>4.680000</td>
    </tr>
    <tr>
      <th>226</th>
      <td>6.99</td>
      <td>CORNING</td>
      <td>41.94</td>
      <td>10.485000</td>
      <td>6.990000</td>
      <td>51632</td>
      <td>6.00</td>
      <td>1.500000</td>
      <td>2656</td>
      <td>2.330000</td>
    </tr>
    <tr>
      <th>223</th>
      <td>6.90</td>
      <td>DES MOINES</td>
      <td>41.40</td>
      <td>41.400000</td>
      <td>138.000000</td>
      <td>50312</td>
      <td>0.30</td>
      <td>0.300000</td>
      <td>4669</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>541</th>
      <td>12.84</td>
      <td>DES MOINES</td>
      <td>38.52</td>
      <td>38.520000</td>
      <td>21.400000</td>
      <td>50317</td>
      <td>1.80</td>
      <td>1.800000</td>
      <td>4974</td>
      <td>4.280000</td>
    </tr>
    <tr>
      <th>1222</th>
      <td>37.62</td>
      <td>DES MOINES</td>
      <td>37.62</td>
      <td>37.620000</td>
      <td>37.620000</td>
      <td>50312</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>4669</td>
      <td>12.540000</td>
    </tr>
    <tr>
      <th>271</th>
      <td>7.52</td>
      <td>MAPLETON</td>
      <td>37.60</td>
      <td>37.600000</td>
      <td>7.520000</td>
      <td>51034</td>
      <td>5.00</td>
      <td>5.000000</td>
      <td>3932</td>
      <td>2.510000</td>
    </tr>
    <tr>
      <th>1206</th>
      <td>36.80</td>
      <td>DES MOINES</td>
      <td>36.80</td>
      <td>36.800000</td>
      <td>21.028571</td>
      <td>50312</td>
      <td>1.75</td>
      <td>1.750000</td>
      <td>4669</td>
      <td>12.270000</td>
    </tr>
    <tr>
      <th>808</th>
      <td>18.14</td>
      <td>DES MOINES</td>
      <td>36.28</td>
      <td>36.280000</td>
      <td>24.186667</td>
      <td>50314</td>
      <td>1.50</td>
      <td>1.500000</td>
      <td>4829</td>
      <td>6.050000</td>
    </tr>
    <tr>
      <th>152</th>
      <td>5.85</td>
      <td>MOUNT VERNON</td>
      <td>35.10</td>
      <td>17.550000</td>
      <td>7.800000</td>
      <td>52314</td>
      <td>4.50</td>
      <td>2.250000</td>
      <td>5102</td>
      <td>1.950000</td>
    </tr>
    <tr>
      <th>101</th>
      <td>4.91</td>
      <td>SPENCER</td>
      <td>34.37</td>
      <td>34.370000</td>
      <td>6.546667</td>
      <td>51301</td>
      <td>5.25</td>
      <td>5.250000</td>
      <td>2565</td>
      <td>1.640000</td>
    </tr>
    <tr>
      <th>198</th>
      <td>6.54</td>
      <td>SPENCER</td>
      <td>32.70</td>
      <td>32.700000</td>
      <td>6.540000</td>
      <td>51301</td>
      <td>5.00</td>
      <td>5.000000</td>
      <td>2565</td>
      <td>2.180000</td>
    </tr>
    <tr>
      <th>711</th>
      <td>15.80</td>
      <td>CLEAR LAKE</td>
      <td>31.60</td>
      <td>31.600000</td>
      <td>9.028571</td>
      <td>50428</td>
      <td>3.50</td>
      <td>3.500000</td>
      <td>3456</td>
      <td>5.270000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3.06</td>
      <td>NORTH LIBERTY</td>
      <td>30.60</td>
      <td>15.300000</td>
      <td>15.300000</td>
      <td>52317</td>
      <td>2.00</td>
      <td>1.000000</td>
      <td>3925</td>
      <td>1.020000</td>
    </tr>
    <tr>
      <th>1119</th>
      <td>30.32</td>
      <td>WEST DES MOINES</td>
      <td>30.32</td>
      <td>30.320000</td>
      <td>40.426667</td>
      <td>50265</td>
      <td>0.75</td>
      <td>0.750000</td>
      <td>2648</td>
      <td>10.110000</td>
    </tr>
    <tr>
      <th>145</th>
      <td>5.73</td>
      <td>MAPLETON</td>
      <td>28.65</td>
      <td>28.650000</td>
      <td>5.730000</td>
      <td>51034</td>
      <td>5.00</td>
      <td>5.000000</td>
      <td>3932</td>
      <td>2.160000</td>
    </tr>
    <tr>
      <th>329</th>
      <td>8.70</td>
      <td>SHELDON</td>
      <td>26.10</td>
      <td>13.050000</td>
      <td>14.500000</td>
      <td>51201</td>
      <td>1.80</td>
      <td>0.900000</td>
      <td>3621</td>
      <td>2.900000</td>
    </tr>
    <tr>
      <th>222</th>
      <td>6.89</td>
      <td>CEDAR FALLS</td>
      <td>20.67</td>
      <td>20.670000</td>
      <td>18.455357</td>
      <td>50613</td>
      <td>1.12</td>
      <td>1.120000</td>
      <td>4932</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>869</th>
      <td>20.01</td>
      <td>DES MOINES</td>
      <td>20.01</td>
      <td>20.010000</td>
      <td>26.680000</td>
      <td>50320</td>
      <td>0.75</td>
      <td>0.750000</td>
      <td>2633</td>
      <td>6.670000</td>
    </tr>
    <tr>
      <th>365</th>
      <td>9.51</td>
      <td>AMES</td>
      <td>19.02</td>
      <td>19.020000</td>
      <td>9.510000</td>
      <td>50010</td>
      <td>2.00</td>
      <td>2.000000</td>
      <td>4509</td>
      <td>3.170000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3.33</td>
      <td>ALGONA</td>
      <td>16.65</td>
      <td>16.650000</td>
      <td>3.330000</td>
      <td>50511</td>
      <td>5.00</td>
      <td>5.000000</td>
      <td>2585</td>
      <td>1.110000</td>
    </tr>
    <tr>
      <th>266</th>
      <td>7.48</td>
      <td>MAPLETON</td>
      <td>14.96</td>
      <td>14.960000</td>
      <td>19.946667</td>
      <td>51034</td>
      <td>0.75</td>
      <td>0.750000</td>
      <td>3932</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <th>596</th>
      <td>14.03</td>
      <td>MONTICELLO</td>
      <td>14.03</td>
      <td>14.030000</td>
      <td>18.706667</td>
      <td>52310</td>
      <td>0.75</td>
      <td>0.750000</td>
      <td>4252</td>
      <td>4.680000</td>
    </tr>
    <tr>
      <th>73</th>
      <td>3.99</td>
      <td>WAVERLY</td>
      <td>11.97</td>
      <td>11.970000</td>
      <td>19.950000</td>
      <td>50677</td>
      <td>0.60</td>
      <td>0.600000</td>
      <td>2651</td>
      <td>1.330000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.76</td>
      <td>DES MOINES</td>
      <td>8.80</td>
      <td>8.800000</td>
      <td>8.800000</td>
      <td>50315</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>4889</td>
      <td>0.590000</td>
    </tr>
    <tr>
      <th>221</th>
      <td>6.89</td>
      <td>BURLINGTON</td>
      <td>6.89</td>
      <td>6.890000</td>
      <td>18.131579</td>
      <td>52601</td>
      <td>0.38</td>
      <td>0.380000</td>
      <td>4794</td>
      <td>2.300000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2.96</td>
      <td>WATERLOO</td>
      <td>5.92</td>
      <td>5.920000</td>
      <td>7.893333</td>
      <td>50703</td>
      <td>0.75</td>
      <td>0.750000</td>
      <td>4077</td>
      <td>0.990000</td>
    </tr>
  </tbody>
</table>
<p>1488 rows × 10 columns</p>
</div>


#### Summary

Challenges and highlights of project 3.
