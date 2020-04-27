#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading neccesary packages
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from datetime import timedelta


# In[2]:


airbnb_raw = pd.read_csv('data/Jan19-Feb20_listings_with_acessibility.csv')



# In[3]:


print(len(airbnb_raw.index))
print(airbnb_raw.columns)
airbnb_raw['last_scraped'] = airbnb_raw['last_scraped_x']
del airbnb_raw['last_scraped_y']
del airbnb_raw['last_scraped_x']


# In[4]:


print('The initial size of the dataset: ',airbnb_raw.size)



# In[5]:


#Drop rows with price equal to 0
airbnb_raw = airbnb_raw[airbnb_raw['price']!=0]


# In[6]:


#Function to convert dates efficiently
def lookup(s):
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


# In[7]:


airbnb_raw['last_scraped'] = lookup(airbnb_raw['last_scraped'])


# In[8]:


print(airbnb_raw['last_scraped'].head())


# In[9]:


airbnb_raw['year'] = airbnb_raw['last_scraped'].dt.year
airbnb_raw['month'] = airbnb_raw['last_scraped'].dt.month
airbnb_raw['day'] = airbnb_raw['last_scraped'].dt.day
airbnb_raw['month_pad'] = airbnb_raw['month'].apply(lambda x : format(x, '02'))
airbnb_raw['yearmonth'] = airbnb_raw['year'].astype(str) + "-" + airbnb_raw['month_pad'].astype(str)


# In[10]:


#Number of unique room ids per month to check for homogeneity across observed months



# ## Adding additional features
# 1. Weather data
# 2. Holidays data 
#   1. Function *holidays_soon_n_week* can be used to calculate how many holidays from holidays_time dataset will be in coming N weeks for specific date
# 3. Seasonalisy as dummy variables 

# In[11]:


#Add weather data to airbnb_raw dataset
#TAV - average daily temp, TAV_Month - avarage monthly temp
#TDIFF - max difference daily, TDIFF_Month - max difference monthly
#Weather_type - cold/hot/mild and etc.
#TSNOW_Month - total amoint of snow this month

weather = pd.read_excel("data/weather_data/weather_output.xlsx")
weather = weather.rename(columns={"DATE": "last_scraped"}) 
weather['last_scraped'] = lookup(weather['last_scraped'])

weather_use = weather[['last_scraped','TAV','TAV_Month','TDIFF','TDIFF_Month', 'Weather_type', 'TSNOW_Month']]
airbnb_raw = pd.merge(airbnb_raw,weather_use,on=['last_scraped'])


# In[12]:


#import holidays set 

us_holidays = holidays.UnitedStates(state='NY', years=[2019, 2020])
holiday_names = []
holiday_dates = []
for holiday in list(us_holidays.items()):
    holiday_dates.append(holiday[0])
    holiday_names.append(holiday[1])
    
holidays_time = pd.DataFrame({
    'holiday': holiday_names,
    'ds': holiday_dates,
})

holidays_time['ds'] = lookup(holidays_time['ds'])


# In[13]:


#function that defines how many holidays are in coming N weeks for specific date
def holidays_soon_n_week(date, n):
    s = 0
    delta = timedelta(days = n*7)
    for j in range(0, len(holidays_time)):
        if (holidays_time['ds'].iloc[j] > date) & (holidays_time['ds'].iloc[j] <= (date + delta)):
            s = s + 1
        j = j + 1
    return s


# In[14]:


#save unique dates from airbnb listings to separate dataset
dates_raw = pd.DataFrame()
dates_raw['last_scraped'] = airbnb_raw['last_scraped'].unique()

#loop to create column with Number of holidays for each date in column 
dates_raw['holidays'] = ''
for i in range(0,len(dates_raw)):
    dates_raw['holidays'].iloc[i] = holidays_soon_n_week(dates_raw['last_scraped'].iloc[i],4)
    i = i + 1


# In[15]:


airbnb_raw = pd.merge(airbnb_raw,dates_raw,on=['last_scraped'])


# In[16]:


#seasonality

months = airbnb_raw['month'].unique()
seasons = ['Winter','Winter','Spring','Spring','Spring','Summer','Summer','Summer','Autumn','Autumn','Autumn','Winter']
month_season = pd.DataFrame({
    'month': months,
    'season': seasons
})

month_season = pd.get_dummies(month_season)


# In[17]:


airbnb_raw = pd.merge(airbnb_raw,month_season,on=['month'])


# In[18]:


# In[19]:



# In[20]:


#Convert boolean values into 0/1
def get_binary(x):
    if x is True:
        return 1
    else: 
        return 0

airbnb_raw['has_availability'] = airbnb_raw['has_availability'].apply(lambda x: get_binary(x))
airbnb_raw['is_instant_bookable'] = airbnb_raw['is_instant_bookable'].apply(lambda x: get_binary(x))
airbnb_raw['is_business_travel_ready'] = airbnb_raw['is_business_travel_ready'].apply(lambda x: get_binary(x))
airbnb_raw['is_wifi'] = airbnb_raw['is_wifi'].apply(lambda x: get_binary(x))
airbnb_raw['is_kitchen'] = airbnb_raw['is_kitchen'].apply(lambda x: get_binary(x))
airbnb_raw['is_heating'] = airbnb_raw['is_heating'].apply(lambda x: get_binary(x))
airbnb_raw['is_smoke_detector'] = airbnb_raw['is_smoke_detector'].apply(lambda x: get_binary(x))
airbnb_raw['is_aircon'] = airbnb_raw['is_aircon'].apply(lambda x: get_binary(x))
airbnb_raw['host_is_superhost'] = airbnb_raw['host_is_superhost'].apply(lambda x: get_binary(x))


# In[21]:


#converting neighborhood and borough into values
# print(airbnb_raw['neighbourhood'].unique())
# print(airbnb_raw['borough'].unique())
airbnb_raw['cat_neighbourhood'] = airbnb_raw['neighbourhood'].astype('category').cat.codes
airbnb_raw['cat_property_type'] = airbnb_raw['property_type'].astype('category').cat.codes
airbnb_raw['cat_weather_type'] = airbnb_raw['Weather_type'].astype('category').cat.codes

airbnb_raw = pd.concat((airbnb_raw, pd.get_dummies(airbnb_raw['borough'], drop_first=True)), axis=1)
airbnb_raw = pd.concat((airbnb_raw, pd.get_dummies(airbnb_raw['cancellation_policy'], drop_first=True)), axis=1)
airbnb_raw = pd.concat((airbnb_raw, pd.get_dummies(airbnb_raw['host_response_time'], drop_first=True)), axis=1)
airbnb_raw = pd.concat((airbnb_raw, pd.get_dummies(airbnb_raw['room_type'], drop_first=True)), axis=1)
airbnb_raw = pd.concat((airbnb_raw, pd.get_dummies(airbnb_raw['Weather_type'], drop_first=True)), axis=1)

airbnb_raw = airbnb_raw.drop(['borough', 'cancellation_policy', 'host_response_time', 'room_type'], axis=1)


# In[22]:


#Correlation matrix for relevant columns 
imp_cols = "price, TAV_Month, TDIFF, TSNOW_Month, host_response_rate,host_total_listings_count,bathrooms,bedrooms,beds, amenities_count, security_deposit, cleaning_fee, price_for_extra_people, guests_included, minimum_nights, maximum_nights, number_of_reviews_ltm, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin,review_scores_communication, review_scores_location, review_scores_value "
imp_cols = imp_cols.replace(" ","")
cols = imp_cols.split(",")

accessibility_cols = ['Subway_Count_Within_200m', 'Subway_Count_Within_1000m',
       'Bus_Count_Within_200m', 'Bus_Count_Within_1000m',
       'Restaurants_Count_Within_200m', 'Restaurants_Count_Within_1000m',
       'Shops_Count_Within_200m', 'Shops_Count_Within_1000m',
       'Attractions_Count_Within_200m', 'Attractions_Count_Within_1000m']

cols = cols + accessibility_cols



# In[23]:





# # Exploring outliers in price variable

# In[24]:




# In[25]:


print(airbnb_raw['price'].count())
print(airbnb_raw['price'].skew())
print(airbnb_raw['price'].mean())


# ## Remove price outliers - IQR approach
# All rows with prices exceeding  75th percentile + 3*IQR are excluded
# IQR = Interquartile range

# In[26]:


q1_price = np.percentile(airbnb_raw['price'], 25, interpolation='midpoint')
q3_price = np.percentile(airbnb_raw['price'], 75, interpolation='midpoint')
iqr = q3_price - q1_price
print(iqr)


# In[27]:


print("Number of outliers:", airbnb_raw[airbnb_raw['price']>(3*iqr)]['price'].count())
print('New size of the dataset: ',airbnb_raw[airbnb_raw['price']<=(3*iqr)]['price'].count())
print("Old mean:", round(airbnb_raw['price'].mean(),2))
print("New mean:", round(airbnb_raw[airbnb_raw['price']<(3*iqr)]['price'].mean(),2))


# In[28]:

# ## Price outliers Log transformation approach
# All rows are included, price value is log transformed (used to move skewed data towards normality)
# 
# (Scientifically correct way of handling right-skewed data)


# #### It is decided to use the first method of simply removing outliers, and thus focus on the remaining listings (just abov 90% of the entire dataset)
# This is because working with linear regressions on log-transformed data cannot be anti-logged

# In[32]:


airbnb_final = airbnb_raw[airbnb_raw['price']<=(3*iqr)]
airbnb_final.info()



# In[34]:


print(airbnb_final['price'].count())
print(airbnb_final['price'].skew())
print(airbnb_final['price'].mean())


# ## Principle Component Analysis (PCA) for accessibility measures
# Since the accessibility measures are highly correlated, we are going to try to reduce the number of featues by conducting PCA (Principle Component Analysis). These all have the same unit, thus we do not necessarily have to normalize the data, which would be the traditional practice before PCA. The aim is to reduce the number of features, while maintaining the same level of information (98% of the initial variance).
# 
# First, we extract the relevant features and transpose them to fit the format for PCA. 



# # Preparing data for prediction

# In[35]:


from sklearn.model_selection import train_test_split

#Splitting and shuffling data
prediction_data = airbnb_final.drop(['listing_id', 'host_id', 'last_scraped', 'experience', 'host_identity_verified', 'neighbourhood', 
                                     'property_type', 'bed_type', 'require_guest_profile_picture', 'require_guest_phone_verification', 
                                     'year', 'month', 'day', 'month_pad', 'yearmonth','latitude','longitude', 'Weather_type'], axis=1)
features_using = prediction_data.drop(['price'],axis=1).columns
predicting = ['price']

train, test = train_test_split(prediction_data, train_size=0.8, shuffle=True) #Test set is approx 20% of total
train, val = train_test_split(train, train_size=0.8, shuffle=True) #Validation set is approx 16% of total, train set is approx 64% of total

X_train, y_train = train[features_using], train[predicting]
X_val, y_val = val[features_using], val[predicting]
X_test, y_test = test[features_using], test[predicting]




# ### Function to evaluate predictions

# In[36]:


# Reference: https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from statistics import mean

def evaluate(y_true, y_pred):
    y_true = y_true['price'].tolist()
    print("MSE: ", round(mean_squared_error(y_true, y_pred), 4))
    print("RMSE: ", round(math.sqrt(mean_squared_error(y_true, y_pred)), 4))
    RMSE = math.sqrt(mean_squared_error(y_true, y_pred))    
    mean_y_true = mean(y_true)
    print("RRMSE: ",round(RMSE*(100/mean_y_true), 4))    
    #print("MAE: ", round(mean_absolute_error(y_true, y_pred), 4))
    print("R2: ", round(r2_score(y_true, y_pred), 4))


# # Building the model

# import numpy as np
# from sklearn.svm import LinearSVR
# from sklearn.svm import SVR
# import matplotlib.pyplot as plt


# import statsmodels.api as sm

# cols = list(X_train.drop(['cat_neighbourhood', 'cat_property_type'],axis=1).columns)

# pmax = 1
# while (len(cols)>0):
#     p= []
#     X_1 = X_train[cols]
#     X_1 = sm.add_constant(X_1)
#     model = sm.OLS(y_train,X_1).fit()
#     p = pd.Series(model.pvalues.values, index = cols)      
#     pmax = max(p)
#     feature_with_p_max = p.idxmax()
#     if(pmax>0.05):
#         cols.remove(feature_with_p_max)
#     else:
#         break
# significant_features = cols
# print(significant_features)
# print(len(significant_features))



# with open("significant_features.txt", "w") as output:
#     output.write(str(significant_features))

with open("significant_features.txt", "r") as output:
    significant_features = output.readlines()

import pdb 


significant_features_ = significant_features[0]
significant_features = []
import re
for feat in significant_features_.split(','):
    feat = re.sub("[^A-Za-z_0-9\-\ ]+",'', feat)
    feat = feat.strip()
    significant_features.append(feat)
    

    
# ## polynomial
# svr_poly = LinearSVR()
# svr_poly.fit(X_train, y_train)
# poly_pred_train = svr_poly.predict(X_train)
# poly_pred_test = svr_poly.predict(X_test)

# # Printing the results
# #print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
# print("- Train set results:")
# evaluate(y_train, poly_pred_train)
# print('\n')
# print("- Test set results:")
# evaluate(y_test, poly_pred_test)


# svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
#                coef0=1)
# svr_poly.fit(X_train, y_train)
# poly_pred_train = svr_poly.predict(X_train)
# poly_pred_test = svr_poly.predict(X_test)

# # Printing the results
# #print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
# print("- Train set results:")
# evaluate(y_train, poly_pred_train)
# print('\n')
# print("- Test set results:")
# evaluate(y_test, poly_pred_test)

# import sys
# sys.exit()


from keras.callbacks import ModelCheckpoint
from keras import models, layers, optimizers, regularizers
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG


nn2 = models.Sequential()
nn2.add(layers.Dense(512, input_shape=(X_train[significant_features].shape[1],), activation='relu'))
nn2.add(layers.Dense(512, activation='relu'))
nn2.add(layers.Dense(256, activation='relu'))
nn2.add(layers.Dense(64, activation='relu'))
nn2.add(layers.Dense(16, activation='relu'))
nn2.add(layers.Dense(1, activation='relu'))

# Compiling the model
nn2.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['mean_squared_error'])

# Printing the model summary
print(nn2.summary())

# Visualising the neural network
#SVG(model_to_dot(nn2, show_layer_names=False, show_shapes=True).create(prog='dot', format='svg'))

#Training the model

# checkpoint = ModelCheckpoint('models/model_-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')  
# nn2_history = nn2.fit(X_train[significant_features],
#                   y_train,
#                   epochs=500,
#                   batch_size=256,
#                   validation_data=(X_val[significant_features],y_val),
#                       verbose=1, callbacks=[checkpoint])

# pdb.set_trace()
# ## save model
# import pickle
# with open('trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(nn2_history.history, file_pi)

# # serialize model to JSON
# model_json = nn2.to_json()
# with open("models/model.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to HDF5
# nn2.save_weights("models/model.h5")
# print("Model saved")


def nn_model_evaluation(model,skip_epochs=0, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    """
    For a given neural network model that has already been fit, prints for the train and tests sets the MSE and r squared
    values, a line graph of the loss in each epoch, and a scatterplot of predicted vs. actual values with a line
    representing where predicted = actual values. Optionally, a value for skip_epoch can be provided, which skips that
    number of epochs in the line graph of losses (useful in cases where the loss in the first epoch is orders of magnitude
    larger than subsequent epochs). Training and test sets can also optionally be specified.
    """

    # MSE and r squared values
    y_test_pred = model.predict(X_test[significant_features])
    y_train_pred = model.predict(X_train[significant_features])
    
    print("- Train set results:")
    evaluate(y_train, y_train_pred)
    print('\n')
    print("- Test set results:")
    evaluate(y_test, y_test_pred)
    
    # Line graph of losses
    #model_results = model.history.history
    model_results = nn2_history
    plt.plot(list(range((skip_epochs+1),len(model_results['loss'])+1)), model_results['loss'][skip_epochs:], label='Train')
    plt.plot(list(range((skip_epochs+1),len(model_results['val_loss'])+1)), model_results['val_loss'][skip_epochs:], label='Test', color='green')
    plt.legend()
    plt.title('Training and test loss at each epoch', fontsize=14)
    plt.savefig("training_vs_testing_loss_sigf.png",dpi=300)

    
    # Scatterplot of predicted vs. actual values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Predicted vs. actual values', fontsize=14, y=1)
    plt.subplots_adjust(top=0.93, wspace=0)
    
    ax1.scatter(y_test, y_test_pred, s=2, alpha=0.7)
    ax1.plot(list(range(2,8)), list(range(2,8)), color='black', linestyle='--')
    ax1.set_title('Test set')
    ax1.set_xlabel('Actual values')
    ax1.set_ylabel('Predicted values')
    
    ax2.scatter(y_train, y_train_pred, s=2, alpha=0.7)
    ax2.plot(list(range(2,8)), list(range(2,8)), color='black', linestyle='--')
    ax2.set_title('Train set')
    ax2.set_xlabel('Actual values')
    ax2.set_ylabel('')
    ax2.set_yticklabels(labels='')
    plt.savefig("nn_train_and_pred_values_sigf.png",dpi=300)
    
import pickle
nn2_history = pickle.load(open('trainHistoryDict', "rb"))
nn2.load_weights('models/model.h5')
nn_model_evaluation(nn2)

import sys
sys.exit()
# ### SVM Regressor

# In[ ]:


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Fit regression model

## radial basis function
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_rbf.fit(X_train, y_train)
rbf_pred_train = svr_rbf.predict(X_train)
rbf_pred_test = svr_rbf.predict(X_test)

# Printing the results
print("- Train set results:")
evaluate(y_train, rbf_pred_train)
print('\n')
print("- Test set results:")
evaluate(y_test, rbf_pred_test)



## linear
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_lin.fit(X_train, y_train)
lin_pred_train = svr_lin.predict(X_train)
lin_pred_test = svr_lin.predict(X_test)

# Printing the results
print("- Train set results:")
evaluate(y_train, lin_pred_train)
print('\n')
print("- Test set results:")
evaluate(y_test, lin_pred_test)



## polynomial
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
svr_poly.fit(X_train, y_train)
poly_pred_train = svr_poly.predict(X_train)
poly_pred_test = svr_poly.predict(X_test)

# Printing the results
#print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
print("- Train set results:")
evaluate(y_train, poly_pred_train)
print('\n')
print("- Test set results:")
evaluate(y_test, poly_pred_test)


# In[ ]:




