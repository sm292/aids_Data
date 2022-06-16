# aids_Data
Data Visualization is done based on the Aids Dataset on with data from 1990 to 2020
## Libraries

import pandas as pd
import numpy as np

## Data Collection

df=pd.read_csv("aids.csv")

df

## Data Preparation for years

x1=df['Year']
y1=df['Data.HIV Prevalence.Adults']

x2=df['Country']
y2=df['Data.HIV Prevalence.Adults']

## Plotting Libraries

import matplotlib.pyplot as plt

## Plotting Year-based data

plt.rcParams['figure.figsize'] = [25,25]

fig, axes = plt.subplots(nrows=2,ncols=1)

axes[0].scatter(x1,y1)
axes[0].set_title("Year Based Data",size=50)
axes[0].set_xlabel("Year",size=25)
axes[0].set_ylabel("HIV in ADULTS",size=25)


axes[1].bar(x2.head(10),y2.head(10), color='g')
axes[1].set_title("Country based data",size=50)
axes[1].set_xlabel("Country",size=25)
axes[1].set_ylabel("HIV in ADULTS",size=25)

plt.tight_layout()


## Data Preparation for Country-based data

x=df['Country']
y=df['Data.HIV Prevalence.Adults']

## Plotting country-based data 

plt.rcParams['figure.figsize'] = [20,20]

plt.figure()

plt.bar(x2.head(10),y2.head(10), color='g')
plt.title("Country based data",size=25)
plt.xlabel("Country",size=25)
plt.ylabel("HIV in ADULTS",size=25)
plt.show()

## Average of HIV Infections in Young Adults

summ = sum(df['Data.New HIV Infections.Young Adults'])
print("Summation of HIV infection data in young adults:     " + str(summ))
count = len(df['Data.New HIV Infections.Young Adults'])
print("Count of HIV infection data in young adults:         " + str(count))
avg = summ/count
print("Average of HIV infection data in young adults:       " + str(avg))

## Average of HIV Infections in Children

summ1 = sum(df['Data.New HIV Infections.Children'])
print("Summation of HIV infection data in children:     " + str(summ1))
count1 = len(df['Data.New HIV Infections.Children'])
print("Count of HIV infection data in children:         " + str(count1))
avg1 = summ1/count1
print("Average of HIV infection data in children:       " + str(avg1))

## Average of Infections in  Adults

summ2 = sum(df['Data.New HIV Infections.Adults'])
print("Summation of HIV infection data in Adults:     " + str(summ2))
count2 = len(df['Data.New HIV Infections.Adults'])
print("Count of HIV infection data in Adults:         " + str(count2))
avg2 = summ2/count2
print("Average of HIV infection data in Adults:       " + str(avg2))

dfu = pd.DataFrame(index=[0,1,2],columns=['Value'])
dfu['Data']=['Summation of HIV infection data in Adults:',
         'Count of HIV infection data in Adults:',
         'Average of HIV infection data in Adults:']
dfu['Value'] =[str(summ2),str(count2),str(avg2)]
dfu.set_index('Data')

## preparing average dataset

df1=pd.DataFrame()
x=df1['Type']=['Young Adults','Children','Adults']
y=df1['Avg Value']=[avg,avg1,avg2]
df1

## Plotting average dataset 

plt.rcParams['figure.figsize'] = [10, 10]

plt.figure()

## HIV in Adults vs. Children Average

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()
x=df['Country']
y=df['Data.HIV Prevalence.Adults']
ax.scatter(x.head(10),y.head(10), color='r', linewidths=25.0)
plt.title("HIV in Young Adults vs. Children Average vs. Adults",size=10)
plt.xlabel("Type",size=10)
plt.ylabel("Values",size=10)
line = mlines.Line2D([0, 10], [0, 10], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()

## Average of Male Adults infection

summ3 = sum(df['Data.New HIV Infections.Male Adults'])
print("Summation of HIV infection data in Male Adults:     " + str(summ3))
count3 = len(df['Data.New HIV Infections.Male Adults'])
print("Count of HIV infection data in Male Adults:         " + str(count3))
avg3 = summ3/count3
print("Average of HIV infection data in Male Adults:       " + str(avg3))

## Average of Female Adults infection

summ4 = sum(df['Data.New HIV Infections.Female Adults'])
print("Summation of HIV infection data in Female Adults:     " + str(summ4))
count4 = len(df['Data.New HIV Infections.Female Adults'])
print("Count of HIV infection data in Female Adults:         " + str(count4))
avg4 = summ4/count4
print("Average of HIV infection data in Female Adults:       " + str(avg4))

## Male vs Female Dataset Preparation

df2=pd.DataFrame()
x=df2['Type']=['Male Adults','Female Adults']
y=df2['Avg Value']=[avg3,avg4]
df2

## Plotting Male vs. Female Adults infection

plt.rcParams['figure.figsize'] = [10, 10]

plt.figure()

plt.bar(x,y, color='b')
plt.title("Male Adults vs. Female Adults",size=10)
plt.xlabel("Types",size=10)
plt.ylabel("Values",size=10)
plt.show()

## Developing SNS Plots

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

x=df['Data.HIV Prevalence.Young Men'].head(10)
y=df['Data.HIV Prevalence.Adults'].head(10)

plt.figure(figsize=(25,25))
sns.catplot(x, y, data = df, kind='bar')

x=df['Data.New HIV Infections.Children'].head(10)
y=df['Data.New HIV Infections.Adults'].head(10)

plt.figure(figsize=(10,10))
sns.countplot(x, data = df)

plt.figure(figsize=(10,10))
sns.countplot(y, data = df)

plt.figure(figsize=(25,25))
x = df['Year']
y = df['Data.People Living with HIV.Adults']
sns.barplot(x,y,data=df)

## Linear Regression 

x = df['Data.AIDS-Related Deaths.All Ages']
x = df['Data.AIDS-Related Deaths.Children']

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)

train_score = regr.score(X_train, y_train)
test_score = regr.score(X_test, y_test)

print("The Train Score is :", train_score)
print("The Test Score is :", test_score)

y_pred = regr.predict(X_test)

plt.figure(figsize=(15,15))
plt.xlabel('Data.AIDS-Related Deaths.Female Adults')
plt.ylabel('Data.AIDS-Related Deaths.Male Adults')
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_test, y_pred, color = 'k')
plt.show()

## Continued





