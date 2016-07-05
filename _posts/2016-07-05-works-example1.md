---
priority: 0.6
title: Project 1
excerpt: A brief analysis of SAT Scores
categories: works
background-image: complex-data.jpg
tags:
  - csv
  - matplotlib
  - numpy
  - scipy
---

#### Workbook


## Step 1: Open the `sat_scores.csv` file. Investigate the data, and answer the questions below.


##### 1. What does the data describe?

The data describes a collection of SAT scores in America by State and also describes the SAT's Verbal, Math, and each State's participation rate. 

##### 2. Does the data look complete? Are there any obvious issues with the observations?

The Writing section of the SAT scores, the number of participants per State, and also the total SAT scores per State are missing. 

It's unclear whether or not what the actual data means, whether it's average scores per State, min, or max...
I assume that these are the mean Math and mean Verbal scores aggregated per State to continue with this exercise.

##### 3. Create a data dictionary for the dataset.

See below.

## Step 2: Load the data.

##### 4. Load the data into a list of lists


```python
sat_csv_path = '/Users/JHYL/DSI-HK-1/projects/project-01/assets/sat_scores.csv'
```


```python
import csv
satdata = []
with open(sat_csv_path, 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        satdata.append(i)

f.close()
```

##### 5. Print the data


```python
import csv
satdata = []
with open(sat_csv_path, 'r') as f:
    reader = csv.reader(f)
    for i in reader:
        satdata.append(i)
        print i
```

    ['State', 'Rate', 'Verbal', 'Math']
    ['CT', '82', '509', '510']
    ['NJ', '81', '499', '513']
    ['MA', '79', '511', '515']
    ['NY', '77', '495', '505']
    ['NH', '72', '520', '516']
    ['RI', '71', '501', '499']
    ['PA', '71', '500', '499']
    ['VT', '69', '511', '506']
    ['ME', '69', '506', '500']
    ['VA', '68', '510', '501']
    ['DE', '67', '501', '499']
    ['MD', '65', '508', '510']
    ['NC', '65', '493', '499']
    ['GA', '63', '491', '489']
    ['IN', '60', '499', '501']
    ['SC', '57', '486', '488']
    ['DC', '56', '482', '474']
    ['OR', '55', '526', '526']
    ['FL', '54', '498', '499']
    ['WA', '53', '527', '527']
    ['TX', '53', '493', '499']
    ['HI', '52', '485', '515']
    ['AK', '51', '514', '510']
    ['CA', '51', '498', '517']
    ['AZ', '34', '523', '525']
    ['NV', '33', '509', '515']
    ['CO', '31', '539', '542']
    ['OH', '26', '534', '439']
    ['MT', '23', '539', '539']
    ['WV', '18', '527', '512']
    ['ID', '17', '543', '542']
    ['TN', '13', '562', '553']
    ['NM', '13', '551', '542']
    ['IL', '12', '576', '589']
    ['KY', '12', '550', '550']
    ['WY', '11', '547', '545']
    ['MI', '11', '561', '572']
    ['MN', '9', '580', '589']
    ['KS', '9', '577', '580']
    ['AL', '9', '559', '554']
    ['NE', '8', '562', '568']
    ['OK', '8', '567', '561']
    ['MO', '8', '577', '577']
    ['LA', '7', '564', '562']
    ['WI', '6', '584', '596']
    ['AR', '6', '562', '550']
    ['UT', '5', '575', '570']
    ['IA', '5', '593', '603']
    ['SD', '4', '577', '582']
    ['ND', '4', '592', '599']
    ['MS', '4', '566', '551']
    ['All', '45', '506', '514']


##### 6. Extract a list of the labels from the data, and remove them from the data.


```python
import numpy as np
satdata_array = np.array(satdata)

#pure_data stands for the data without its labels/headers
pure_data = satdata[1:len(satdata)]
print pure_data
#print satdata[0]

#I used this space here to organise my columns, then changing the strings into integers
state_data = [item[0] for item in satdata_array]
rate_data = [item[1] for item in satdata_array]
verbal_data = [item[2] for item in satdata_array]
math_data = [item[3] for item in satdata_array]

#pure data for states
s_data = [item[0] for item in satdata_array]
del s_data[0]
#print s_data

#pure data for rates
r_data = [item[1] for item in satdata_array]
del r_data[0]
#print r_data

#pure data for verbal
v_data = [item[2] for item in satdata_array]
del v_data[0]
#print v_data

#pure data for math
m_data = [item[3] for item in satdata_array]
del m_data[0]
#print m_data


#no more strings
r_data = [int(x) for x in r_data]
v_data = [int(x) for x in v_data]
m_data = [int(x) for x in m_data]
#print r_data
#print v_data
#print m_data
```

    [['CT', '82', '509', '510'], ['NJ', '81', '499', '513'], ['MA', '79', '511', '515'], ['NY', '77', '495', '505'], ['NH', '72', '520', '516'], ['RI', '71', '501', '499'], ['PA', '71', '500', '499'], ['VT', '69', '511', '506'], ['ME', '69', '506', '500'], ['VA', '68', '510', '501'], ['DE', '67', '501', '499'], ['MD', '65', '508', '510'], ['NC', '65', '493', '499'], ['GA', '63', '491', '489'], ['IN', '60', '499', '501'], ['SC', '57', '486', '488'], ['DC', '56', '482', '474'], ['OR', '55', '526', '526'], ['FL', '54', '498', '499'], ['WA', '53', '527', '527'], ['TX', '53', '493', '499'], ['HI', '52', '485', '515'], ['AK', '51', '514', '510'], ['CA', '51', '498', '517'], ['AZ', '34', '523', '525'], ['NV', '33', '509', '515'], ['CO', '31', '539', '542'], ['OH', '26', '534', '439'], ['MT', '23', '539', '539'], ['WV', '18', '527', '512'], ['ID', '17', '543', '542'], ['TN', '13', '562', '553'], ['NM', '13', '551', '542'], ['IL', '12', '576', '589'], ['KY', '12', '550', '550'], ['WY', '11', '547', '545'], ['MI', '11', '561', '572'], ['MN', '9', '580', '589'], ['KS', '9', '577', '580'], ['AL', '9', '559', '554'], ['NE', '8', '562', '568'], ['OK', '8', '567', '561'], ['MO', '8', '577', '577'], ['LA', '7', '564', '562'], ['WI', '6', '584', '596'], ['AR', '6', '562', '550'], ['UT', '5', '575', '570'], ['IA', '5', '593', '603'], ['SD', '4', '577', '582'], ['ND', '4', '592', '599'], ['MS', '4', '566', '551'], ['All', '45', '506', '514']]


##### 7. Create a list of State names extracted from the data. (Hint: use the list of labels to index on the State column)


```python
print s_data
```

    ['CT', 'NJ', 'MA', 'NY', 'NH', 'RI', 'PA', 'VT', 'ME', 'VA', 'DE', 'MD', 'NC', 'GA', 'IN', 'SC', 'DC', 'OR', 'FL', 'WA', 'TX', 'HI', 'AK', 'CA', 'AZ', 'NV', 'CO', 'OH', 'MT', 'WV', 'ID', 'TN', 'NM', 'IL', 'KY', 'WY', 'MI', 'MN', 'KS', 'AL', 'NE', 'OK', 'MO', 'LA', 'WI', 'AR', 'UT', 'IA', 'SD', 'ND', 'MS', 'All']


##### 8. Print the types of each column


```python
state_ie = satdata[1][0]
rate_ie = satdata[1][1]
verbal_ie = satdata[1][2]
math_ie = satdata[1][3]

print type(state_data)
print type(rate_data)
print type(verbal_data)
print type(math_data)
```

    <type 'list'>
    <type 'list'>
    <type 'list'>
    <type 'list'>


##### 9. Do any types need to be reassigned? If so, go ahead and do it.


```python
print type(state_ie)
print type(r_data[0])
print type(v_data[0])
print type(m_data[0])
```

    <type 'str'>
    <type 'int'>
    <type 'int'>
    <type 'int'>


##### 10. Create a dictionary for each column mapping the State to its respective value for that column.  (3 dictionaries: rates, verbal, math)

* Dictionary for the Rates column


```python
r_dict = dict(zip(s_data, r_data))
print r_dict
```

    {'WA': 53, 'DE': 67, 'DC': 56, 'WI': 6, 'WV': 18, 'HI': 52, 'FL': 54, 'WY': 11, 'NH': 72, 'NJ': 81, 'NM': 13, 'TX': 53, 'LA': 7, 'NC': 65, 'ND': 4, 'NE': 8, 'TN': 13, 'NY': 77, 'PA': 71, 'RI': 71, 'NV': 33, 'VA': 68, 'CO': 31, 'AK': 51, 'AL': 9, 'AR': 6, 'VT': 69, 'IL': 12, 'GA': 63, 'IN': 60, 'IA': 5, 'OK': 8, 'AZ': 34, 'CA': 51, 'ID': 17, 'CT': 82, 'ME': 69, 'MD': 65, 'All': 45, 'MA': 79, 'OH': 26, 'UT': 5, 'MO': 8, 'MN': 9, 'MI': 11, 'KS': 9, 'MT': 23, 'MS': 4, 'SC': 57, 'KY': 12, 'OR': 55, 'SD': 4}


* Dictionary for the Verbal column


```python
v_dict = dict(zip(s_data, v_data))
print v_dict
```

    {'WA': 527, 'DE': 501, 'DC': 482, 'WI': 584, 'WV': 527, 'HI': 485, 'FL': 498, 'WY': 547, 'NH': 520, 'NJ': 499, 'NM': 551, 'TX': 493, 'LA': 564, 'NC': 493, 'ND': 592, 'NE': 562, 'TN': 562, 'NY': 495, 'PA': 500, 'RI': 501, 'NV': 509, 'VA': 510, 'CO': 539, 'AK': 514, 'AL': 559, 'AR': 562, 'VT': 511, 'IL': 576, 'GA': 491, 'IN': 499, 'IA': 593, 'OK': 567, 'AZ': 523, 'CA': 498, 'ID': 543, 'CT': 509, 'ME': 506, 'MD': 508, 'All': 506, 'MA': 511, 'OH': 534, 'UT': 575, 'MO': 577, 'MN': 580, 'MI': 561, 'KS': 577, 'MT': 539, 'MS': 566, 'SC': 486, 'KY': 550, 'OR': 526, 'SD': 577}


* Dictionary for the Math column


```python
m_dict = dict(zip(s_data, m_data))
print m_dict

```

    {'WA': 527, 'DE': 499, 'DC': 474, 'WI': 596, 'WV': 512, 'HI': 515, 'FL': 499, 'WY': 545, 'NH': 516, 'NJ': 513, 'NM': 542, 'TX': 499, 'LA': 562, 'NC': 499, 'ND': 599, 'NE': 568, 'TN': 553, 'NY': 505, 'PA': 499, 'RI': 499, 'NV': 515, 'VA': 501, 'CO': 542, 'AK': 510, 'AL': 554, 'AR': 550, 'VT': 506, 'IL': 589, 'GA': 489, 'IN': 501, 'IA': 603, 'OK': 561, 'AZ': 525, 'CA': 517, 'ID': 542, 'CT': 510, 'ME': 500, 'MD': 510, 'All': 514, 'MA': 515, 'OH': 439, 'UT': 570, 'MO': 577, 'MN': 589, 'MI': 572, 'KS': 580, 'MT': 539, 'MS': 551, 'SC': 488, 'KY': 550, 'OR': 526, 'SD': 582}


##### 11. Create a dictionary with the values for each of the numeric columns
* one dictionary


```python
num_dict = dict([('Rate', r_data), ('Verbal', v_data), ('Math', m_data)])
print num_dict
```

    {'Rate': [82, 81, 79, 77, 72, 71, 71, 69, 69, 68, 67, 65, 65, 63, 60, 57, 56, 55, 54, 53, 53, 52, 51, 51, 34, 33, 31, 26, 23, 18, 17, 13, 13, 12, 12, 11, 11, 9, 9, 9, 8, 8, 8, 7, 6, 6, 5, 5, 4, 4, 4, 45], 'Math': [510, 513, 515, 505, 516, 499, 499, 506, 500, 501, 499, 510, 499, 489, 501, 488, 474, 526, 499, 527, 499, 515, 510, 517, 525, 515, 542, 439, 539, 512, 542, 553, 542, 589, 550, 545, 572, 589, 580, 554, 568, 561, 577, 562, 596, 550, 570, 603, 582, 599, 551, 514], 'Verbal': [509, 499, 511, 495, 520, 501, 500, 511, 506, 510, 501, 508, 493, 491, 499, 486, 482, 526, 498, 527, 493, 485, 514, 498, 523, 509, 539, 534, 539, 527, 543, 562, 551, 576, 550, 547, 561, 580, 577, 559, 562, 567, 577, 564, 584, 562, 575, 593, 577, 592, 566, 506]}


## Step 3: Describe the data

##### 12. Print the min and max of each column


```python
r = sorted(r_data)
v = sorted(v_data)
m = sorted(m_data)

print "Rates min:", r[0]
print "Rates max:", r[len(r)-1]

print "Verbal min:", v[0]
print "Verbal max:", v[len(v)-1]

print "Math min:", m[0]
print "Math max:", m[len(m)-1]
```

    Rates min: 4
    Rates max: 82
    Verbal min: 482
    Verbal max: 593
    Math min: 439
    Math max: 603


##### 13. Write a function using only list comprehensions, no loops, to compute Standard Deviation. Print the Standard Deviation of each numeric column.


```python
#std for rates
r_std = np.std(r_data)
print "Std for rates:", r_std

#std for verbal
v_std = np.std(v_data)
print "Std for verbal:", v_std

#std for math
m_std = np.std(m_data)
print "Std for math:", m_std
```

    Std for rates: 27.0379964945
    Std for verbal: 32.9150949616
    Std for math: 35.6669961643


## Step 4: Visualize the data

##### 14. Using MatPlotLib and PyPlot, plot the distribution of the Rate using histograms.


```python
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

plt.hist(r_data)
plt.title("Rates Histogram")
plt.xlabel("Rates")
plt.ylabel("Frequency")

plt.show()
```


![png](output_37_0.png)


##### 15. Plot the Math distribution


```python
plt.hist(m_data)
plt.title("Math Histogram")
plt.xlabel("Math Scores")
plt.ylabel("Frequency")

plt.show()
```


![png](output_39_0.png)


##### 16. Plot the Verbal distribution


```python
plt.hist(v_data)
plt.title("Verbal Histogram")
plt.xlabel("Verbal Scores")
plt.ylabel("Frequency")

plt.show()
```


![png](output_41_0.png)


##### 17. What is the typical assumption for data distribution?

The normal distribution, or Gaussian distribution is a bell curve, where the function is symmetric around x=0.

##### 18. Does that distribution hold true for our data?

The distribution does not hold true for our data. The distribution of data that is the most similar to a normal distribution is the Math scores' distribution.

##### 19. Plot some scatterplots. **BONUS**: Use a PyPlot `figure` to present multiple plots at once.


```python
#rates and verbal
plt.title("Rates and Verbal Scores Scatterplot")
plt.xlabel("Rates")
plt.ylabel("Verbal Scores")
plt.scatter(r_data, v_data)
plt.show()

#rates and math
plt.title("Rates and Math Scores Scatterplot")
plt.xlabel("Rates")
plt.ylabel("Math Scores")
plt.scatter(r_data, m_data)
plt.show()

#verbal and math
plt.title("Verbal and Math Scores Scatterplot")
plt.xlabel("Verbal Scores")
plt.ylabel("Math Scores")
plt.scatter(v_data, m_data)
plt.show()
```


![png](output_47_0.png)



![png](output_47_1.png)



![png](output_47_2.png)


##### 20. Are there any interesting relationships to note?

For the Rates & Verbal scatter plot and the Rates and Math scatter plot, there is a negative correlation between the two variables in each. The higher the participation rates, the lower the Verbal and Math scores. 
This could mean that although some states have lower participation rates, but their participants score higher on average than the participants of other states with higher participation rates.

Regarding the Verbal & Math scatterplot, there is a positive correlation between the two variables, where the higher the Verbal scores of a state, the higher the Math scores.
This suggests that if a participant scores highly in either one of the categories, then it is likely that they will score highly in the other.

##### 21. Create box plots for each variable. 


```python
#Rates
plt.title("Rates Boxplot")
plt.xlabel("Rates")
plt.ylabel("Rates")
plt.boxplot(r_data, notch=True, patch_artist=True)
plt.show()

#Verbal Scores
plt.title("Verbal Scores Boxplot")
plt.xlabel("Verbal Scores")
plt.ylabel("Verbal Scores")
plt.boxplot(v_data, notch=True, patch_artist=True)
plt.show()

#Math Scores
plt.title("Math Scores Boxplot")
plt.xlabel("Math Scores")
plt.ylabel("Math Scores")
plt.boxplot(m_data, notch=True, patch_artist=True)
plt.show()

#Verbal and Math Scores Compared
scores = [m_data, v_data]

plt.title("Verbal Scores and Math Scores Boxplot")
plt.xlabel("Verbal Scores and Math Scores")
plt.ylabel("Scores")
plt.boxplot(scores, positions = [1, 2], notch=True, patch_artist=True)
plt.xticks([1, 2], ['Verbal', 'Math'])
plt.show()
```


![png](output_51_0.png)



![png](output_51_1.png)



![png](output_51_2.png)



![png](output_51_3.png)


##### BONUS: Using Tableau, create a heat map for each variable using a map of the US. 


```python
from IPython.display import Image
print"Rates by State"
Image(filename='/Users/JHYL/DSI Work/Week_1/Project_1/sat_rate.png')

```

    Rates by State





![png](output_53_1.png)




```python
print "Verbal Scores by State"
Image(filename='/Users/JHYL/DSI Work/Week_1/Project_1/sat_verbal.png')
```

    Verbal Scores by State





![png](output_54_1.png)




```python
print "Math Scores by State"
Image(filename='/Users/JHYL/DSI Work/Week_1/Project_1/sat_math.png')
```

    Math Scores by State





![png](output_55_1.png)




```python

```



#### Summary

Project 1 involved blah blah blah
