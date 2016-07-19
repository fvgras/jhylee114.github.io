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

#### Summary & Recommendations
To build the models of total sales based on location, price per bottle, and bottles sold, I first defined my feature columns and my y and split them into training and testing sets. The results when including all three variables in my feature columns were as follows:
Intercept = -105.57
Coefficients: [('County Number', 0.0081),  ('Price per Bottle', 6.78),  ('Bottles Sold', 13.62)]
The intercept is less relevant when evaluating the information than the coefficients. I zipped the feature columns with the resulting coefficients, where the lower the coefficient, the less of a correlation between the specific X variable and the y variable. These results demonstrate that the quantity of bottles sold affected sales the most, then price per bottle, then county number.

	I also found the root mean squared errors to evaluate the model by deriving it from all 3 variables, 2 variables, and also single variables, which gave me this:

RMSE:
All 3 variables:
RMSE: 212.11562256 

2 variables:
Price per bottle and bottles sold RMSE: 212.12
County number and bottles sold RMSE: 223.71
Price per bottle and county number RMSE: 357.66

1 variable:
Bottles sold RMSE: 223.72
Price per bottle RMSE: 357.70
County number RMSE: 361.40

	When all 3 variables were included in the feature columns, the RMSE was the lowest, which is a good indicator because we are trying to minimize with a loss function. Then price per bottle and bottles sold led to the second lowest RMSE. County number and bottles sold as well as bottles sold on its own resulted in around the same RMSE value. The rest of the tests gave much higher RMSEs.

I used 10-fold cross-validation with various features to calculate the score of the MSE. The results were very similar to that of calculating the RMSE. 

MSE Scores:
All 3 variables: 201.87

2 variables:
Price per bottle and bottles sold: 201.86
County number and bottles sold: 214.54
Price per bottle and county number: 377.28

1 variable:
Bottles sold: 214.54
Price per bottle: 377.34
County number: 380.86

I plotted some of the models to visualize clearer and obviously, combing bottles sold and price per bottle against sales as well as bottles sold and sales had strong correlations. 

I built a few pivot tables to make recommendations on the best locations to open a store. Here are the top 10 lists by zip code, city, and county number, and also the bonus two variables of the bottle volume (ml) and the price per bottle. What could have been done better is sorting these results by more specific categories.

Top 10 Zip Codes:
50314
50320
52402
52240
50010
52807
51501
50311
50266
52722

Top 10 Cities:
Des Moines
Cedar Rapids
Davenport
Iowa City
Waterloo
Sioux City
Council Bluffs
West Des Moines
Ames
Dubuque

Top 10 County Numbers:
77
57
82
52
7
78
97
31
85
17

Bonus
Top 10 Bottle Volumes (ml):
750
1750
1000
375
200
500
3000
600
100
300

Top 10 Prices per Bottle:
22.5
27
17.63
13.5
12.38
15.74
14.93
27.74
10.76
10.38

I played around with RidgeCV and LassoCV but did not have enough time to fully dissect the information. In addition, I also made a few visualizations through Tableau to get a better understanding of the data.








#### Python Code

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

## Load the data into a DataFrame
df = pd.read_csv('/Users/JHYL/DSI-HK-1/projects/project-03/assets/Iowa_Liquor_sales_sample_10pct.csv')

df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Store Number</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>County Number</th>
      <th>County</th>
      <th>Category</th>
      <th>Category Name</th>
      <th>Vendor Number</th>
      <th>Item Number</th>
      <th>Item Description</th>
      <th>Bottle Volume (ml)</th>
      <th>State Bottle Cost</th>
      <th>State Bottle Retail</th>
      <th>Bottles Sold</th>
      <th>Sale (Dollars)</th>
      <th>Volume Sold (Liters)</th>
      <th>Volume Sold (Gallons)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11/04/2015</td>
      <td>3717</td>
      <td>SUMNER</td>
      <td>50674</td>
      <td>9.0</td>
      <td>Bremer</td>
      <td>1051100.0</td>
      <td>APRICOT BRANDIES</td>
      <td>55</td>
      <td>54436</td>
      <td>Mr. Boston Apricot Brandy</td>
      <td>750</td>
      <td>$4.50</td>
      <td>$6.75</td>
      <td>12</td>
      <td>$81.00</td>
      <td>9.0</td>
      <td>2.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03/02/2016</td>
      <td>2614</td>
      <td>DAVENPORT</td>
      <td>52807</td>
      <td>82.0</td>
      <td>Scott</td>
      <td>1011100.0</td>
      <td>BLENDED WHISKIES</td>
      <td>395</td>
      <td>27605</td>
      <td>Tin Cup</td>
      <td>750</td>
      <td>$13.75</td>
      <td>$20.63</td>
      <td>2</td>
      <td>$41.26</td>
      <td>1.5</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02/11/2016</td>
      <td>2106</td>
      <td>CEDAR FALLS</td>
      <td>50613</td>
      <td>7.0</td>
      <td>Black Hawk</td>
      <td>1011200.0</td>
      <td>STRAIGHT BOURBON WHISKIES</td>
      <td>65</td>
      <td>19067</td>
      <td>Jim Beam</td>
      <td>1000</td>
      <td>$12.59</td>
      <td>$18.89</td>
      <td>24</td>
      <td>$453.36</td>
      <td>24.0</td>
      <td>6.34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>02/03/2016</td>
      <td>2501</td>
      <td>AMES</td>
      <td>50010</td>
      <td>85.0</td>
      <td>Story</td>
      <td>1071100.0</td>
      <td>AMERICAN COCKTAILS</td>
      <td>395</td>
      <td>59154</td>
      <td>1800 Ultimate Margarita</td>
      <td>1750</td>
      <td>$9.50</td>
      <td>$14.25</td>
      <td>6</td>
      <td>$85.50</td>
      <td>10.5</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>08/18/2015</td>
      <td>3654</td>
      <td>BELMOND</td>
      <td>50421</td>
      <td>99.0</td>
      <td>Wright</td>
      <td>1031080.0</td>
      <td>VODKA 80 PROOF</td>
      <td>297</td>
      <td>35918</td>
      <td>Five O'clock Vodka</td>
      <td>1750</td>
      <td>$7.20</td>
      <td>$10.80</td>
      <td>12</td>
      <td>$129.60</td>
      <td>21.0</td>
      <td>5.55</td>
    </tr>
  </tbody>
</table>
</div>



# Cleaning


```python
# Remove redundant columns
df_no_dup = df.drop_duplicates()
print df_no_dup.shape
print df.shape
```

    (270920, 18)
    (270955, 18)



```python
# Remove $ from certain columns
df['State Bottle Cost'] = df['State Bottle Cost'].map(lambda x: x.lstrip('$'))
df['State Bottle Retail'] = df['State Bottle Retail'].map(lambda x: x.lstrip('$'))
df['Sale (Dollars)'] = df['Sale (Dollars)'].map(lambda x: x.lstrip('$'))
```


```python
# Convert dates
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
```


```python
# Drop or replace bad values
df.fillna(0)

df.drop(['Volume Sold (Gallons)'], axis=1, inplace=True)
df.head()

# Convert integers
df['Sale (Dollars)'] = df['Sale (Dollars)'].astype(float) 
df['Volume Sold (Liters)'] = df['Volume Sold (Liters)'].astype(float) 
```

# Explore the data

Perform some exploratory statistical analysis and make some plots, such as histograms of transaction totals, bottles sold, etc.


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Store Number</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>County Number</th>
      <th>County</th>
      <th>Category</th>
      <th>Category Name</th>
      <th>Vendor Number</th>
      <th>Item Number</th>
      <th>Item Description</th>
      <th>Bottle Volume (ml)</th>
      <th>State Bottle Cost</th>
      <th>State Bottle Retail</th>
      <th>Bottles Sold</th>
      <th>Sale (Dollars)</th>
      <th>Volume Sold (Liters)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-11-04</td>
      <td>3717</td>
      <td>SUMNER</td>
      <td>50674</td>
      <td>9.0</td>
      <td>Bremer</td>
      <td>1051100.0</td>
      <td>APRICOT BRANDIES</td>
      <td>55</td>
      <td>54436</td>
      <td>Mr. Boston Apricot Brandy</td>
      <td>750</td>
      <td>4.50</td>
      <td>6.75</td>
      <td>12</td>
      <td>81.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>2614</td>
      <td>DAVENPORT</td>
      <td>52807</td>
      <td>82.0</td>
      <td>Scott</td>
      <td>1011100.0</td>
      <td>BLENDED WHISKIES</td>
      <td>395</td>
      <td>27605</td>
      <td>Tin Cup</td>
      <td>750</td>
      <td>13.75</td>
      <td>20.63</td>
      <td>2</td>
      <td>41.26</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-02-11</td>
      <td>2106</td>
      <td>CEDAR FALLS</td>
      <td>50613</td>
      <td>7.0</td>
      <td>Black Hawk</td>
      <td>1011200.0</td>
      <td>STRAIGHT BOURBON WHISKIES</td>
      <td>65</td>
      <td>19067</td>
      <td>Jim Beam</td>
      <td>1000</td>
      <td>12.59</td>
      <td>18.89</td>
      <td>24</td>
      <td>453.36</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-02-03</td>
      <td>2501</td>
      <td>AMES</td>
      <td>50010</td>
      <td>85.0</td>
      <td>Story</td>
      <td>1071100.0</td>
      <td>AMERICAN COCKTAILS</td>
      <td>395</td>
      <td>59154</td>
      <td>1800 Ultimate Margarita</td>
      <td>1750</td>
      <td>9.50</td>
      <td>14.25</td>
      <td>6</td>
      <td>85.50</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-08-18</td>
      <td>3654</td>
      <td>BELMOND</td>
      <td>50421</td>
      <td>99.0</td>
      <td>Wright</td>
      <td>1031080.0</td>
      <td>VODKA 80 PROOF</td>
      <td>297</td>
      <td>35918</td>
      <td>Five O'clock Vodka</td>
      <td>1750</td>
      <td>7.20</td>
      <td>10.80</td>
      <td>12</td>
      <td>129.60</td>
      <td>21.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Store Number</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>County Number</th>
      <th>County</th>
      <th>Category</th>
      <th>Category Name</th>
      <th>Vendor Number</th>
      <th>Item Number</th>
      <th>Item Description</th>
      <th>Bottle Volume (ml)</th>
      <th>State Bottle Cost</th>
      <th>State Bottle Retail</th>
      <th>Bottles Sold</th>
      <th>Sale (Dollars)</th>
      <th>Volume Sold (Liters)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>270950</th>
      <td>2015-12-22</td>
      <td>4057</td>
      <td>DES MOINES</td>
      <td>50316</td>
      <td>77.0</td>
      <td>Polk</td>
      <td>1022100.0</td>
      <td>TEQUILA</td>
      <td>410</td>
      <td>88291</td>
      <td>Patron Tequila Silver Mini</td>
      <td>300</td>
      <td>20.30</td>
      <td>30.45</td>
      <td>4</td>
      <td>121.80</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>270951</th>
      <td>2015-11-04</td>
      <td>5151</td>
      <td>IDA GROVE</td>
      <td>51445</td>
      <td>47.0</td>
      <td>Ida</td>
      <td>1011200.0</td>
      <td>STRAIGHT BOURBON WHISKIES</td>
      <td>259</td>
      <td>17956</td>
      <td>Evan Williams Str Bourbon</td>
      <td>750</td>
      <td>7.47</td>
      <td>11.21</td>
      <td>3</td>
      <td>33.63</td>
      <td>2.25</td>
    </tr>
    <tr>
      <th>270952</th>
      <td>2015-10-20</td>
      <td>5152</td>
      <td>WATERLOO</td>
      <td>50702</td>
      <td>7.0</td>
      <td>Black Hawk</td>
      <td>1011300.0</td>
      <td>TENNESSEE WHISKIES</td>
      <td>85</td>
      <td>26826</td>
      <td>Jack Daniels Old #7 Black Lbl</td>
      <td>750</td>
      <td>15.07</td>
      <td>22.61</td>
      <td>6</td>
      <td>135.66</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>270953</th>
      <td>2015-11-20</td>
      <td>3562</td>
      <td>WEST BURLINGTON</td>
      <td>52655</td>
      <td>29.0</td>
      <td>Des Moines</td>
      <td>1082900.0</td>
      <td>MISC. IMPORTED CORDIALS &amp; LIQUEURS</td>
      <td>192</td>
      <td>65258</td>
      <td>Jagermeister Liqueur</td>
      <td>1750</td>
      <td>26.05</td>
      <td>39.08</td>
      <td>6</td>
      <td>234.48</td>
      <td>10.50</td>
    </tr>
    <tr>
      <th>270954</th>
      <td>2015-01-27</td>
      <td>4446</td>
      <td>URBANDALE</td>
      <td>50322</td>
      <td>77.0</td>
      <td>Polk</td>
      <td>1031080.0</td>
      <td>VODKA 80 PROOF</td>
      <td>260</td>
      <td>37993</td>
      <td>Smirnoff Vodka 80 Prf</td>
      <td>200</td>
      <td>2.75</td>
      <td>4.13</td>
      <td>8</td>
      <td>33.04</td>
      <td>1.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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




```python
df.shape
```




    (270955, 17)




```python
df.dtypes
```




    Date                    datetime64[ns]
    Store Number                     int64
    City                            object
    Zip Code                        object
    County Number                  float64
    County                          object
    Category                       float64
    Category Name                   object
    Vendor Number                    int64
    Item Number                      int64
    Item Description                object
    Bottle Volume (ml)               int64
    State Bottle Cost               object
    State Bottle Retail             object
    Bottles Sold                     int64
    Sale (Dollars)                 float64
    Volume Sold (Liters)           float64
    dtype: object




```python
df['City'].value_counts()
df['County'].value_counts()
df['Category Name'].value_counts()
df['Vendor Number'].value_counts()
df['Bottle Volume (ml)'].value_counts()
df['State Bottle Cost'].value_counts()
df['State Bottle Retail'].value_counts()
df['Bottles Sold'].value_counts()
df['Sale (Dollars)'].value_counts()
df['Volume Sold (Liters)'].value_counts().head()
```




    9.00     53048
    10.50    35789
    1.50     24640
    2.25     19608
    0.75     17221
    Name: Volume Sold (Liters), dtype: int64




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


```python
df.dtypes
```




    Date                    datetime64[ns]
    Store Number                     int64
    City                            object
    Zip Code                        object
    County Number                  float64
    County                          object
    Category                       float64
    Category Name                   object
    Vendor Number                    int64
    Item Number                      int64
    Item Description                object
    Bottle Volume (ml)               int64
    State Bottle Cost               object
    State Bottle Retail             object
    Bottles Sold                     int64
    Sale (Dollars)                 float64
    Volume Sold (Liters)           float64
    Margin                         float64
    Price per Liter                float64
    Price per Bottle               float64
    dtype: object




```python
#sns.pairplot(df, x_vars=['County Number', 'Price per Bottle', 'Bottles Sold'], y_vars = 'Sale (Dollars)', kind='reg')
```

#  Build models of total sales based on location, price per bottle, total bottles sold. You may find it useful to build models for each county, zip code, or city.


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


# 2) Which are the best performing stores by location type?

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
sales_zip.sort_values(by='Sale (Dollars) sum', ascending=False).head(10)
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
  </tbody>
</table>
</div>




```python
# List of best stores per Zip Code according to value counts/total number of transactions
group_zip = df.groupby(by='Zip Code')
group_zip['Store Number'].agg(lambda x: x.value_counts().index[0]).head(10)
```




    Zip Code
    50002    4417
    50003    4678
    50006    4172
    50009    2548
    50010    2501
    50014    4251
    50020    4267
    50021    2502
    50022    2591
    50023    2666
    Name: Store Number, dtype: int64




```python
df_pivot = pd.pivot_table(df,index=["Zip Code","Store Number"], values=["Sale (Dollars)"], aggfunc=np.sum)
df_pivot.head(10)
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
      <th rowspan="3" valign="top">50009</th>
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
  </tbody>
</table>
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
sales_city.sort_values(by='Sale (Dollars) sum', ascending=False).head()
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
  </tbody>
</table>
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
sales_county.sort_values(by='Sale (Dollars) sum', ascending=False).head(10)
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
  </tbody>
</table>
</div>



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



![png](output_57_1.png)


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



![png](output_58_1.png)


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



![png](output_60_1.png)


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



![png](output_61_1.png)


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



![png](output_63_1.png)


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



![png](output_64_1.png)


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



![png](output_65_1.png)


    MSE: 41541.3227776


# Bonus


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
sales_volume.sort_values(by='Sale (Dollars) sum', ascending=False).head(10)
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
sales_bp.sort_values(by='Sale (Dollars) sum', ascending=False).head(10)
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
  </tbody>
</table>
</div>

