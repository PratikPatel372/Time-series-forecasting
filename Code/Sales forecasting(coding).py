#LIBRARY IMPORT
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

#IMPORT THE DATA
data = pd.read_csv(r"C:\Users\Pratik\Desktop\IIT KGP\Seminar-I (IM69001)\daily demand dataset\Retail_daily_sales.csv")

data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
data.index = data['Date']

#DATA PLOTTING
plt.plot(data.index, data['Sales'], label='Retail Sales')
plt.legend(loc='best')
plt.show()

#BOX PLOTTING OF DATA
sns.boxplot(data['Sales'])

#REMOVING THE OUTLIERS
IQR = (np.percentile(data['Sales'], 75)) - (np.percentile(data['Sales'], 25))
whisker_val = (np.percentile(data['Sales'], 75)) + (1.5*(IQR))

whisker_val

data.loc[data['Sales']>whisker_val].shape
data.loc[data['Sales']>whisker_val]

data_original = data['Sales']#................data_original file having outliers & No missing values.....................
data['Sales'] = data['Sales'].apply(lambda x: np.nan if x > whisker_val else x)

No_outlier= data['Sales'].isnull().sum()

# removing outliers using ffill
data['Sales'] = data['Sales'].fillna(method ='ffill')

fig, axs = plt.subplots(2, 1,  sharex=True)
axs[0].plot(data_original) 
axs[1].plot(data['Sales'])
plt.show() 

#MISSING DATA VALUES IN THE DATASET and PUT THE ZERO

data['Date'].min(), data['Date'].max()
print('Total days between 01-jan-07 to 24-Dec-08:', (data['Date'].max() - data['Date'].min()).days)
print('Number of rows present in the data are:', data.shape[0])

missing_dates = pd.DataFrame(data = pd.date_range(start = '2007-01-01', end = '2008-12-24').difference(data.index),columns= ['Date'])

# add missing dates to the data
data2 = pd.DataFrame(data['Sales'])
data2.index = pd.DatetimeIndex(data.Date)

idx = pd.date_range('2007-01-01', '2008-12-24')
data2 = data2.reindex(idx, fill_value=0)

# extract WEEKDAY,DATE, MONTH, YEAR from the dates
data2['Date'] = data2.index
data2['weekday_name'] = data2['Date'].dt.strftime("%A")
data2['year'] = pd.DatetimeIndex(data2['Date']).year
data2['month'] = pd.DatetimeIndex(data2['Date']).month
data2['day'] = pd.DatetimeIndex(data2['Date']).day

data_feat = pd.DataFrame({"year": data2['Date'].dt.year,
                          "month": data2['Date'].dt.month,
                          "day": data2['Date'].dt.day,
                          "weekday": data2['Date'].dt.dayofweek,
                          "weekday_name":data2['Date'].dt.strftime("%A"),
                          "dayofyear": data2['Date'].dt.dayofyear,
                          "week": data2['Date'].dt.week,
                          "quarter": data2['Date'].dt.quarter,
                         })
complete_data = pd.concat([data_feat, data2['Sales']], axis=1)

#Weekly Sales observation
sns.barplot(x='weekday_name',y='Sales',data = complete_data)

plt.figure(figsize=(10,6))
sns.boxplot(x=complete_data['weekday_name'], y=complete_data['Sales'],)
plt.title('Weekly Sales Trend')
plt.show()

#Monthly Sales observation
sns.barplot(x='month',y='Sales',data = complete_data)

plt.figure(figsize=(10,6))
sns.boxplot(x=complete_data['month'], y=complete_data['Sales'])
plt.title('Montly Sales Trend')
plt.show()

#Yearly Sales observation
sns.barplot(x='year',y='Sales',data = complete_data)

plt.figure(figsize=(10,6))
sns.boxplot(x=complete_data['year'], y=complete_data['Sales'], )
plt.title('Yearly Sales')
plt.show()


# Exponential Moving Average chart of Weekly Sales figure
complete_data_weekly = complete_data.groupby(['year','week'],as_index = False)
complete_data_weekly.groups
complete_data_weekly = complete_data_weekly.agg({'Sales':np.mean})
complete_data_weekly = complete_data_weekly.sort_values(by = ['year','week'])

complete_data_weekly['weekrow']= complete_data_weekly.reset_index().index
complete_data_weekly['10weeks_ema'] = complete_data_weekly.Sales.ewm(span=10).mean()
complete_data_weekly['20weeks_ema'] = complete_data_weekly.Sales.ewm(span=20).mean()
complete_data_weekly['50weeks_ema'] = complete_data_weekly.Sales.ewm(span=50).mean()

complete_data_weekly.plot('weekrow',['10weeks_ema','20weeks_ema','50weeks_ema'])
plt.title('Trend & Seasonality in data')
plt.show()


# Exponential Moving Average chart of Daily Sales figure
complete_data_daily = complete_data.groupby(['year','dayofyear'],as_index = False)
complete_data_daily.groups
complete_data_daily = complete_data_daily.agg({'Sales':np.mean})
complete_data_daily = complete_data_daily.sort_values(by = ['year','dayofyear'])

complete_data_daily['dayrow']= complete_data_daily.reset_index().index
complete_data_daily['15day_ema'] = complete_data_daily.Sales.ewm(span=15).mean()
complete_data_daily['30day_ema'] = complete_data_daily.Sales.ewm(span=30).mean()

complete_data_daily.plot('dayrow',['15day_ema','30day_ema'])
plt.title('Trend & Seasonality in data')    
plt.show()

#.......Removing the Sunday from Data2
data_ = complete_data.copy()
data_ = data_.loc[data_['weekday_name']!= 'Sunday']
data_.shape
#..........fill "0" values with "ffill method"
data_['Sales'] = data_['Sales'].apply(lambda x: np.nan if x == 0.0 else x)
print("No. of sales with still 0:", data_.isnull().sum())
data_['Sales'] = data_['Sales'].fillna(method ='ffill')
print("Now, No. of sales with 0",data_.isnull().sum())

#Training And Testing(Validation) of data
train_data = data_[:469]
valid_data = data_[469:]

plt.figure(figsize=(15,10))

plt.plot(train_data.index, train_data['Sales'], label='Train')
plt.plot(valid_data.index, valid_data['Sales'], label='Validation')
plt.legend(loc='best')
plt.show()
   
#RMSLE function
def rmsle(actual, preds):
    for i in range(0,len(preds)):
        if preds[i]<0:
            preds[i] = 0
        else:
            pass
    
    error = (sqrt(mean_squared_log_error(actual, preds)))*100
    return error

# ..........................................................1) Holt-Winter Method.....................................................
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
from statistics import mean, stdev

#training the model
model = ExponentialSmoothing(np.asarray(train_data['Sales']), seasonal_periods=6, trend='add', seasonal='add')
model = model.fit(smoothing_level=0.2, smoothing_slope=0.001, smoothing_seasonal=0.2)
    
# predictions and evaluation
preds = model.forecast(len(valid_data))

# results
RMSLEscore = rmsle(valid_data['Sales'], preds)
print('RMSLE for Holt Winter is:', RMSLEscore)

plt.figure(figsize = (12,8))
plt.plot(train_data.index , train_data['Sales'], label = 'train')
plt.plot(valid_data.index , valid_data['Sales'], label = 'valid')
plt.plot(valid_data.index , preds, label = 'forecast')
plt.legend(loc='best')
plt.show()

#.....................................................Modified alpha,beta,gamma value combination...........................
from itertools import product

level = [0.1, 0.3, 0.5, 0.8]
smoothing_slope = [0.0001, 0.001, 0.05] 
smoothing_seasonal = [0.2, 0.4, 0.6]

# creating list with all the possible combinations of parameters
parameters = product(level, smoothing_slope, smoothing_seasonal)
parameters_list = list(parameters)
len(parameters_list)

best_score = float('inf')
result = []

for param in parameters_list:
    #print(param)
    #training the model
    model = ExponentialSmoothing(np.asarray(train_data['Sales']), seasonal_periods=6, trend='add', seasonal='add')
    model = model.fit(smoothing_level=param[0], smoothing_slope=param[1], smoothing_seasonal=param[2])
    # predictions and evaluation
    preds = model.forecast(len(valid_data))
    # result
    RMSLEscore = rmsle(valid_data['Sales'], preds)
    #print('RMSLE for Holt Winter is:', RMSLEscore)
    result.append([param,RMSLEscore])
    
    if RMSLEscore < best_score:
        best_score = RMSLEscore
        best_model = model
        best_param = param

result = pd.DataFrame(result)
result.columns = ['parameters','RMSLEScore']
result = result.sort_values(by='RMSLEScore', ascending=True).reset_index(drop=True)
print("Best RMSLE valueof Holt winter method : ", result['RMSLEScore'][0])

#.................................................2) SARIMA MODEL.............................................

plt.figure(figsize=(12,8))
plt.plot(train_data.index, train_data['Sales'])
plt.legend(loc='best')
plt.title("Non-Stationary Series")
plt.show()

#[BOXCOX METHOD] = Non-Stationary series to stationary series 
from scipy import stats
train_data['Sales_log'], lambda_val = stats.boxcox(train_data['Sales'])
print(lambda_val)

#Taking a 1st seasonal difference of Sales_log column over 6-period.(S=6,D=1)
train_data['Sales_log_diff']=train_data['Sales_log']-train_data['Sales_log'].shift(6)
plt.figure(figsize=(12,8))
plt.plot(train_data.index, train_data['Sales_log_diff'])
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

#1st order difference of Sales_log_diff column over 1-period(d=1)
train_data['Sales_log_diff_diff']=train_data['Sales_log_diff']-train_data['Sales_log_diff'].shift(1)
plt.figure(figsize=(12,8))
plt.plot(train_data.index, train_data['Sales_log_diff_diff'])
plt.legend(loc='best')
plt.title("Stationary Series")
plt.show()

#Building the Model
from statsmodels.tsa.statespace import sarimax
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(train_data['Sales_log_diff_diff'].dropna(), lags=25)
plot_pacf(train_data['Sales_log_diff_diff'].dropna(), lags=25)
plt.show()

#Training the model : model = sarimax.SARIMAX(train_data['Sale_log'], seasonal_order=(P,D,Q,s), order=(p,d,q))
model = sarimax.SARIMAX(train_data['Sales_log'], seasonal_order=(1,1,2,6), order=(1,1,2))
model = model.fit()

# predictions 
preds = model.predict(start=len(train_data)+1, end=len(train_data) + len(valid_data))

#inverse BOXCOX convert to get the main data
def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)

preds = inverse_boxcox(preds, lambda_val)
preds = preds.reset_index().drop(['index'],axis='columns') # storing validation predictions

# results
RMSLEscore = rmsle(valid_data['Sales'], preds['predicted_mean'])
print('RMSLE for SARIMA model Forecasts is', RMSLEscore)

plt.figure(figsize = (12,8))
plt.plot(train_data.index, train_data['Sales'], label = 'train')
plt.plot(valid_data.index, valid_data['Sales'], label = 'valid')
plt.plot(valid_data.index, preds['predicted_mean'], label = 'preds')
plt.show()


#.................................................3) Linear regression model : Machine learning MODEL.............................................
from sklearn.linear_model import LinearRegression

train_data = data_[:469]
valid_data = data_[469:]

x_train = train_data.drop(['Sales','weekday_name'], axis=1)
y_train = train_data['Sales']
x_valid = valid_data.drop(['Sales', 'weekday_name'], axis=1)
y_valid = valid_data['Sales']

#training the model
model = LinearRegression(normalize=True)
model.fit(x_train, y_train)
preds = model.predict(x_valid)

# results
RMSLEscore = rmsle(y_valid, preds)
print('RMSLE for Linear Regression is', RMSLEscore)
plt.bar(x_train.columns, model.coef_)
#................................................. Regularization for Linear Regression

from sklearn.linear_model import Ridge

# month level encoding
monthly_average = pd.DataFrame(train_data.groupby('month')['Sales'].mean())
train_data['monthly_average'] = train_data['month'].map(monthly_average.Sales)
valid_data['monthly_average'] = valid_data['month'].map(monthly_average.Sales)

# week target encoding
week_average = pd.DataFrame(train_data.groupby('weekday')['Sales'].mean())
train_data['week_average'] = train_data['weekday'].map(week_average.Sales)
valid_data['week_average'] = valid_data['weekday'].map(week_average.Sales)

x_train = train_data.drop(['Sales','weekday_name'], axis=1)
y_train = train_data['Sales']
x_valid = valid_data.drop(['Sales', 'weekday_name'], axis=1)
y_valid = valid_data['Sales']

#training the model
for alpha in [0.01, 0.05, 0.1, 0.5, 1, 5]:
    print('----- ----- ----- ----- -----')
    print('At alpha value:', alpha)

    #training the model
    model = Ridge(alpha = alpha, normalize=True)
    model.fit(x_train, y_train)

    # predictions 
    preds = model.predict(x_valid)

    # results
    score = rmsle(y_valid, preds)
    print('RMSLE is', score)

#:::::::BEST RMSLE VALUE IS FOR ALPHA = 0.5
#training the model
model = Ridge(alpha = 0.5, normalize=True)
model.fit(x_train, y_train)

# predictions 
preds = model.predict(x_valid)
RMSLEscore = rmsle(y_valid, preds)

# results
print('Best RMSLE for Linear Regression is', RMSLEscore)
plt.bar(x_train.columns, model.coef_)

plt.figure(figsize=(12,8))
plt.plot(train_data.index, train_data['Sales'], label = 'train')
plt.plot(valid_data.index, valid_data['Sales'], label = 'valid')
plt.plot(valid_data.index, preds, label = 'preds')
plt.show()


#.................................................4) Random Forest model : Machine learning MODEL.............................................

from sklearn.ensemble import RandomForestRegressor

#training the model
model = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split = 25, random_state=0)
model.fit(x_train, y_train)
    
# predictions 
preds = model.predict(x_valid)
RMSLEscore = rmsle(y_valid, preds)
   
# results
print('Average Error is',RMSLEscore)

plt.figure(figsize = (12,8))
plt.plot(train_data.index, train_data['Sales'], label = 'train')
plt.plot(valid_data.index, valid_data['Sales'], label = 'valid')
plt.plot(valid_data.index, preds, label = 'preds')
plt.show()

#Features Importants
features = x_train.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

#...........................Hyperparameter tuning.......................

n_estimator = [70, 100, 130, 150]
max_depth =  [5, 6, 7, 8, 9]
min_samples_split = [20, 30, 50]

parameters = product(n_estimator,max_depth,min_samples_split)
parameters_list = list(parameters)
len(parameters_list)

results = []
best_error_ = float("inf")

for param in parameters_list:
    #training the model
    model = RandomForestRegressor(n_estimators=param[0], max_depth=param[1], min_samples_split = param[2], random_state=0)
    model.fit(x_train, y_train)
    
    # predictions 
    preds = model.predict(x_valid)

    # predictions and evaluation
    score = rmsle(y_valid, preds)
        
    # saving best model, rmsle and parameters
    if score < best_error_:
        best_model = model
        best_error_ = score
        best_param = param
    results.append([param, score])    
    
result_table = pd.DataFrame(results)
result_table.columns = ['parameters','RMSLEScore']
result_table = result_table.sort_values(by='RMSLEScore', ascending=True).reset_index(drop=True)
print("Best RMSLE value of Random Forest model : ", result_table['RMSLEScore'][0])

plt.figure(figsize = (12,8))

plt.plot(train_data.index, train_data['Sales'], label = 'train')
plt.plot(valid_data.index, valid_data['Sales'], label = 'valid')
plt.plot(valid_data.index, preds, label = 'preds')
plt.show()

#......................................FORECAST FOR NEXT 6 MONTHS.................................

# DATA PREPARATION
test_data = pd.DataFrame(data=pd.date_range(start='2009-01-01',end='2009-06-30',freq='D'),columns=['Date'])
test_data.head()

test_feat = pd.DataFrame({"Date":test_data['Date'],
                          "year": test_data['Date'].dt.year,
                          "month": test_data['Date'].dt.month,
                          "day": test_data['Date'].dt.day,
                          "weekday": test_data['Date'].dt.dayofweek,
                          "weekday_name":test_data['Date'].dt.strftime("%A"),
                          "dayofyear": test_data['Date'].dt.dayofyear,
                          "week": test_data['Date'].dt.week,
                          "quarter": test_data['Date'].dt.quarter,
                         })

# month level encoding
monthly_average = pd.DataFrame(data_.groupby('month')['Sales'].mean())
data_['monthly_average'] = data_['month'].map(monthly_average.Sales)
test_feat['monthly_average'] = test_feat['month'].map(monthly_average.Sales)

# week target encoding
week_average = pd.DataFrame(data_.groupby('weekday')['Sales'].mean())
data_['week_average'] = data_['weekday'].map(week_average.Sales)
test_feat['week_average'] = test_feat['weekday'].map(week_average.Sales)

#Remove the NULL value from test_feat
test_feat.isnull().sum()
test_feat.shape
test_feat.dropna(axis=0, inplace = True)
test_feat.shape

x_train = data_.drop(['Sales', 'weekday_name'], axis=1)
y_train = data_['Sales']
x_valid = test_feat.drop(['weekday_name', 'Date'], axis=1)

#training the model using RANDOM FOREST
model = RandomForestRegressor(n_estimators=150, max_depth=9, min_samples_split = 50, random_state=0)
model.fit(x_train, y_train)
    
# predictions 
preds = model.predict(x_valid)
test_feat['Sales'] = preds

plt.figure(figsize = (12,8))
plt.plot(data_.index, data_['Sales'], label = 'train')
plt.plot(test_feat.Date, preds, label = 'preds')
plt.legend(loc='best')
plt.show()

forecast = test_data.merge(test_feat[['Date','Sales']],how='left',on='Date')












