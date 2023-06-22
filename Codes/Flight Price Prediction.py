#!/usr/bin/env python
# coding: utf-8

# # Flight Price Prediction 

# ***
# _**Importing the required libraries & packages**_
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chart_studio.plotly import plot,iplot
import cufflinks as cf
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pickle
import ydata_profiling as pf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# _**Changing The Default Working Directory Path & Reading the Dataset using Pandas Command**_

# In[2]:


os.chdir('C:\\Users\\Shridhar\\OneDrive\\Desktop\\Top Mentor\\Batch 74 Day 38\\Project 12 Flight Price Predict Heroku')
df=pd.read_excel('Data_Train.xlsx')


# _**Automated Exploratory Data Analysis (EDA) with ydata_profiling(pandas_profiling)**_

# In[3]:


display(pf.ProfileReport(df))


# ## Data Cleaning:

# _**Dropping all the null values from the dataset**_

# In[4]:


df.dropna(how='any',inplace=True)


# _**Checking for the duplicate values after dropping it**_

# In[5]:


df.isna().sum()


# _**Changing the Data Types to DataTime format using Pandas command**_

# In[6]:


df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'])


# _**Adding two new columns by seperating the day and month from the <span style= "color:blue">Date of Journey</span> column**_

# In[7]:


df['Day_of_Journey']=(df['Date_of_Journey']).dt.day
df['Month_of_Journey']=(df['Date_of_Journey']).dt.month


# _**Dropping the <span style="color:blue">Date of Journey</span> column since we extracted both day and month sepeartely to new column**_

# In[8]:


df.drop(['Date_of_Journey'],axis=1,inplace=True)


# _**Changing the Data Types to DataTime format using Pandas command and Adding two new columns by seperating the hour and minutes from the <span style= "color:blue">Departure Time</span> column**_

# In[9]:


df['Dep_hr']=pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min']=pd.to_datetime(df['Dep_Time']).dt.minute


# _**Dropping the <span style="color:blue">Departure Time</span> column since we extracted both hour and minutes sepeartely to new column**_

# In[10]:


df.drop(['Dep_Time'],axis=1,inplace=True)


# _**Changing the Data Types to DataTime format using Pandas command and Adding two new columns by seperating the hour and minutes from the <span style= "color:blue">Arrival Time </span> column**_

# In[11]:


df['Arrival_hr']=pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min']=pd.to_datetime(df['Arrival_Time']).dt.minute


# _**Dropping the <span style="color:blue">Arrival Time</span> column since we extracted both hour and minutes sepeartely to new column**_

# In[12]:


df.drop(['Arrival_Time'],axis=1,inplace=True)


# _**Assigning the new variable and splitting hour and minutes and extracting it from the <span style="color:blue">Duration</span> column**_

# In[13]:


duration=df['Duration'].str.split(' ',expand=True)


# _**Filling out the null values to 00 minutes to the minutes column of the assigned new variable**_ 

# In[14]:


duration[1].fillna('00m',inplace=True)


# _**Adding two new columns with the duration hour and duration minutes from the assigned new variable columns**_

# In[15]:


df['duration_hr']=duration[0].apply(lambda x:x[:-1])
df['duration_min']=duration[1].apply(lambda x:x[:-1])


# _**Dropping the <span style="color:blue">Duration</span> column since we extracted both hour and minutes sepeartely to new column**_

# In[16]:


df.drop(['Duration'],axis=1,inplace=True)


# _**Grouping by the unique values from the <span style= "color:blue"> Total Stops </span> column**_ 

# In[17]:


df.groupby('Total_Stops').size()


# ## Data Visualization:
# _**Setting the configuration file and plotting the bar graph with Airlines and its average price; and saving the PNG file**_

# In[18]:


cf.set_config_file(theme='ggplot',sharing='public',offline=True)
Airprices=df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(15,10))
sns.barplot(x=Airprices.index,y=Airprices.values)
plt.xticks(rotation=90)
plt.title('Airline with its average price')
plt.savefig('Airline with its average price.png')
plt.show()


# _**Plotting the box plot with Airlines and its price; and saving the PNG file**_

# In[19]:


plt.figure(figsize=(20,10))
sns.boxplot(y='Price',x='Airline',data= df.sort_values('Price',ascending=False))
plt.title('Airline and its price')
plt.savefig('Airline and its price.png')
plt.show()


# _**Plotting the bar graph with Airlines its price and Total Stops; and saving the PNG file**_

# In[20]:


plt.figure(figsize=(18,10))
ax=sns.barplot(x=df['Airline'],y=df['Price'],hue=df['Total_Stops'],palette="Set1")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Airline,Price and Total Stops')
plt.savefig('Airline,Price and Total Stops.png')
plt.show()


# _**Plotting the bar graph with Source and Price and saving the png file**_

# In[21]:


plt.figure(figsize=(15,10))
sns.barplot(y='Price',x='Source',ci=None,data=df.sort_values('Price',ascending=False))
plt.title('Source and Price')
plt.savefig('Source and Price.png')
plt.show()


# _**Plotting the bar graph with Destination and Price and saving the png file**_

# In[22]:


plt.figure(figsize=(15,10))
sns.barplot(y='Price',x='Destination',ci=None,data=df.sort_values('Price',ascending=False))
plt.title('Destination and Price')
plt.savefig('Destination and Price.png')
plt.show()


# _**Renaming the same cities under single name from the <span style = "color:blue"> Destination</span> column**_

# In[23]:


df['Destination']=df['Destination'].apply(lambda x:x.replace('New Delhi','Delhi'))


# _**Checking the city names of the  <span style = "color:blue"> Destination</span> column after renaming it**_

# In[24]:


df['Destination'].unique()


# _**Plotting the bar graph again with Renamed Destination and Price and saving the png file**_

# In[25]:


plt.figure(figsize=(15,10))
sns.barplot(y='Price',x='Destination',ci=None,data=df.sort_values('Price',ascending=False))
plt.title('Renamed Destination and Price')
plt.savefig('Renamed Destination and Price.png')
plt.show()


# _**Getting the Correlation Values from all the numeric columns from the dataset using Seaborn Heatmap & saving the PNG File**_

# In[26]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# _**Label Encoding the <span style = "color:blue"> Total stops </span> columns using mapping function**_

# In[27]:


df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})


# _**One Hot Encoding the <span style = "color:blue"> Airline </span>,<span style = "color:blue"> Source </span>,<span style = "color:blue"> Destination </span> columns using pandas get dummies command**_

# In[28]:


air_dum=pd.get_dummies(df['Airline'],drop_first=True)
sour_dest=pd.get_dummies(df[['Source','Destination']],drop_first=True)
df=pd.concat([air_dum,sour_dest,df],axis=1)


# _**Dropping the columns <span style = "color:blue"> Airline </span>,<span style = "color:blue"> Source </span>,<span style = "color:blue"> Destination </span>,<span style = "color:blue"> Additional Info </span>,<span style = "color:blue"> Route </span> from the dataset which is not needed**_

# In[29]:


df.drop(['Airline','Source','Destination','Additional_Info','Route'],axis=1,inplace=True)


# ## Loading Test Data:
# _**Reading the Test Dataset using Pandas Command**_

# In[30]:


df_test=pd.read_excel('Test_set.xlsx')


# ## Data Cleaning of the Test Data:
# _**Changing the Data Types to DataTime format using Pandas command and Adding two new columns by seperating the day and month from the <span style= "color:blue">Date of Journey</span> column in the test data**_

# In[31]:


df_test['Day_of_Journey']=pd.to_datetime(df_test['Date_of_Journey']).dt.day
df_test['Month_of_Journey']=pd.to_datetime(df_test['Date_of_Journey']).dt.month


# _**Changing the Data Types to DataTime format using Pandas command and Adding two new columns by seperating the hour and minutes from the <span style= "color:blue">Depature Time</span> column in the test data**_

# In[32]:


df_test['Dep_hr']=pd.to_datetime(df_test['Dep_Time']).dt.hour
df_test['Dep_min']=pd.to_datetime(df_test['Dep_Time']).dt.minute


# _**Changing the Data Types to DataTime format using Pandas command and Adding two new columns by seperating the hour and minutes from the <span style= "color:blue">Arrival Time</span> column in the test data**_

# In[33]:


df_test['Arrival_hr']=pd.to_datetime(df_test['Arrival_Time']).dt.hour
df_test['Arrival_min']=pd.to_datetime(df_test['Arrival_Time']).dt.minute


# _**Assigning the new variable and splitting hour and minutes and extracting it from the <span style="color:blue">Duration</span> column and Filling out the null values to 00 minutes to the minutes column of the assigned new variable**_

# In[34]:


dur=df_test['Duration'].str.split(' ',expand=True)
dur[1].fillna('00m',inplace=True)


# _**Adding two new columns with the duration hour and duration minutes from the assigned new variable columns in the test data**_

# In[35]:


df_test['duration_hr']=dur[0].apply(lambda x:x[:-1])
df_test['duration_min']=dur[1].apply(lambda x:x[:-1])


# _**Dropping the columns <span style = "color:blue"> Date of Journey  </span>,<span style = "color:blue"> Departure Time </span>,<span style = "color:blue"> Arrival Time </span>,<span style = "color:blue"> Duration </span> from the test data since we extracted all the necessary datas from these columns**_

# In[36]:


df_test.drop(['Date_of_Journey','Dep_Time', 'Arrival_Time', 'Duration'],axis=1,inplace=True)


# _**Label Encoding the <span style = "color:blue"> Total stops </span> columns of test data using mapping function**_

# In[37]:


df_test['Total_Stops']=df_test['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4})


# _**One Hot Encoding the <span style = "color:blue"> Airline </span>,<span style = "color:blue"> Source </span>,<span style = "color:blue"> Destination </span> columns of test data using pandas get dummies command**_

# In[38]:


air_dummy=pd.get_dummies(df_test['Airline'],drop_first=True)
source_dest=pd.get_dummies(df_test[['Source','Destination']],drop_first=True)
df_test=pd.concat([air_dummy,source_dest,df_test],axis=1)


# _**Dropping the columns <span style = "color:blue"> Airline </span>,<span style = "color:blue"> Source </span>,<span style = "color:blue"> Destination </span>,<span style = "color:blue"> Additional Info </span>,<span style = "color:blue"> Route </span> from the test data which is not needed**_

# In[39]:


df_test.drop(['Airline','Source','Destination','Additional_Info','Route'],axis=1,inplace=True)


# _**Exporting the Clean Train Data and Clean Test Data to a CSV file**_

# In[40]:


df.to_csv('Cleaned_Train_Data.csv',index=False)
df_test.to_csv('Cleaned_Test_Data.csv',index=False)


# _**Assigning the dependent and independent variable**_

# In[41]:


x=df.drop(['Price'],axis=1)
y=df['Price']


# ## Model Fitting:
# _**Fitting the Extra Tree Regression model with the dependent and independent variable and getting the r2 Score between the predicted value and dependent variable**_

# In[42]:


Et=ExtraTreesRegressor()
Et.fit(x,y)
y_pred=Et.predict(x)
r2_score(y,y_pred)


# _**Plotting the Bar Graph to represent the Feature Importances of the Independent variable column and saving the PNG file.**_

# In[43]:


pd.Series(Et.feature_importances_,index=x.columns).sort_values(ascending=False).plot(kind='bar',figsize=(18,10))
plt.title('Feature Importance of the data')
plt.savefig('Feature Importance of the data.png')
plt.show()


# _**Splitting the dependent variable & independent variable into training and test dataset using train test split**_

# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=44)


# _**Fitting the Extra Tree Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[45]:


ET_Model=ExtraTreesRegressor(n_estimators=120)
ET_Model.fit(x_train,y_train)
y_predict=ET_Model.predict(x_test)
r2_score(y_test,y_predict)


# _**Fitting the Random Forest Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[46]:


RF_Model=RandomForestRegressor()
RF_Model.fit(x_train,y_train)
y_predict=RF_Model.predict(x_test)
r2_score(y_test,y_predict)


# _**Fitting The Random Forest Regressor Model with the list of parameters in the RandomizedSearchCV Algorithm and getting the r2 Score between the predicted value and dependent test dataset**_

# In[47]:


n_estimators = [int(x) for x in np.linspace(start = 80, stop = 1500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(6, 45, num = 5)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

rand_grid={'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf=RandomForestRegressor()

rCV=RandomizedSearchCV(estimator=rf,param_distributions=rand_grid,scoring='neg_mean_squared_error',
                       n_iter=10,cv=3,random_state=42, n_jobs = 1)

rCV.fit(x_train,y_train)
rf_pred=rCV.predict(x_test)
r2_score(y_test,rf_pred)


# _**Getting the Mean Absolute Error and Mean Squared Error values between the predicted value from the RandomizedSearchCV and dependent test dataset.**_

# In[48]:


print('MAE',mean_absolute_error(y_test,rf_pred))
print('MSE',mean_squared_error(y_test,rf_pred))


# _**Fitting the CatBoost Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[49]:


cat=CatBoostRegressor()
cat.fit(x_train,y_train)
cat_pred=cat.predict(x_test)
r2_score(y_test,cat_pred)


# _**Changing the data type of the <span style = "color:blue"> duration_hr </span>,<span style = "color:blue"> duration_min </span> column from "object" to "int"**_

# In[50]:


x_train[['duration_hr','duration_min']]=x_train[['duration_hr','duration_min']].astype(int)
x_test[['duration_hr','duration_min']]=x_test[['duration_hr','duration_min']].astype(int)


# _**Fitting the LGBM Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[51]:


lgb=LGBMRegressor()
lgb.fit(x_train,y_train)
lgb_pred=lgb.predict(x_test)
r2_score(y_test,lgb_pred)


# _**Fitting the XGB Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset**_

# In[52]:


xgb=XGBRegressor()
xgb.fit(x_train,y_train)
xgb_pred=xgb.predict(x_test)
r2_score(y_test,xgb_pred)


# **_Fitting the EXtra Tree Regressor model with the train dependent and train independent variable and getting the r2 Score between the predicted value and dependent test dataset_**

# In[53]:


Et2=ExtraTreesRegressor()
Et2.fit(x_train,y_train)
Et2_pred=Et2.predict(x_test)
r2_score(y_test,Et2_pred)


# _**Create the pickle file of the model with the highest r2 score with the model name**_

# In[54]:


pickle.dump(cat,open('Predictive model.pkl','wb'))


# _**Loading the pickle file**_

# In[55]:


model=pickle.load(open('Predictive model.pkl','rb'))


# _**Predicting the <span style="color:blue"> Price</span> of test data using the loaded pickle file**_

# In[56]:


prediction=model.predict(df_test)


# _**Making the Predicted value as a new dataframe and concating it with test data**_

# In[57]:


prediction_df=pd.DataFrame(prediction,columns=['Predicted Price(Approx.)'])
pred_df=pd.concat([df_test,prediction_df],axis=1)


# _**Exporting the Test Data With Price to a csv file**_

# In[58]:


pred_df.to_csv('Prediction Flight Price.csv',index=False)

