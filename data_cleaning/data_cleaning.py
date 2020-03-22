#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datetime import datetime
import time
import math
from collections import Counter


# In[46]:


imp_cols = "id,last_scraped,host_id,host_since,host_location,host_response_time,host_response_rate,host_acceptance_rate,host_is_superhost,host_neighbourhood,instant_bookable,review_scores_checkin,review_scores_value,reviews_per_month,review_scores_communication,review_scores_location,calculated_host_listings_count,calculated_host_listings_count_shared_rooms,calculated_host_listings_count_private_rooms,calculated_host_listings_count_entire_homes,require_guest_profile_picture,require_guest_phone_verification,cancellation_policy,is_business_travel_ready,host_listings_count,host_total_listings_count,host_identity_verified,neighbourhood,neighbourhood_cleansed,neighbourhood_group_cleansed,city,zipcode,market,smart_location,latitude,longitude,property_type,room_type,accommodates,bathrooms,bedrooms,beds,bed_type,amenities,price,weekly_price,monthly_price,security_deposit,cleaning_fee,guests_included,extra_people,minimum_nights,maximum_nights,minimum_minimum_nights,maximum_minimum_nights,minimum_maximum_nights,maximum_maximum_nights,minimum_nights_avg_ntm,calendar_updated,has_availability,availability_30,availability_60,availability_90,availability_365,calendar_last_scraped,number_of_reviews,number_of_reviews_ltm,first_review,last_review,review_scores_rating,review_scores_accuracy,review_scores_cleanliness"
imp_cols_list = imp_cols.split(",")
print(imp_cols_list)

data = pd.read_csv("/Users/toshalpatel/is5152_project/data_cleaning/concatenated_listings.csv", usecols=imp_cols_list)


# ## Exploring the data stats

# In[4]:


print("Number of rows: ",len(data.index))
print("Number of cols: ",len(data.columns))


# In[5]:


data.head()


# #### Number of null values in each of the columns

# In[6]:


data.isna().sum()


# # Data Cleaning

# ### Dropping null values and retaining imp rows and cols

# In[47]:


data.drop(['host_acceptance_rate','weekly_price','monthly_price'], axis=1, inplace=True)

data = data.dropna()

print("Number of rows: ",len(data.index))
print("Number of cols: ",len(data.columns))


# ### Defining new DataFrame

# In[8]:


df = pd.DataFrame()
df['listing_id'] = data['id']
df['host_id'] = data['host_id']
df['last_scraped'] = data['last_scraped']


# ## Defining some cleaning functions

# In[9]:


#calculate day difference    
def day_differ(time1,time2):
    try:
        if time1<=time2:
            return (time2-time1).days
        else:
            return None
    except Exception:pass
    
def get_bool(x):
    if x == 't':
        return True
    elif x == 'f':
        return False
    else:
        return None
    
def get_int(x,s):
    x = str(x)
    n_x = x.strip(s)
    return n_x


# ## Calculating the experience of the host
# 
# We calculate the experience of the hosts based on how long he has been on the AirBnb platform.

# In[10]:


data['last_scraped_datetime']=data['last_scraped'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
data['host_since_datetime']=data['host_since'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
df['experience'] = data.apply(lambda x: day_differ( x['host_since_datetime'], x['last_scraped_datetime']),axis=1)
print(df['experience'].head())


# ## Host descriptors
# 
# host_response_time \
# host_response_rate \
# host_acceptance_rate \
# host_is_superhost \
# host_total_listings_count \
# is_host_verified 

# Unique values in host response time

# In[11]:


print(data['host_response_time'].unique())
df['host_response_time'] = data['host_response_time']


# In[12]:


print(data['host_response_rate'].unique())
    
df['host_response_rate'] = data['host_response_rate'].apply(lambda x : get_int(x,'%'))
print(df['host_response_rate'].head())


# In[13]:


print(data['host_is_superhost'].unique())
df['host_is_superhost'] = data['host_is_superhost'].apply(lambda x: get_bool(x))
print(df['host_is_superhost'].head(),"\n",data['host_is_superhost'].head())


# In[14]:


print(data['host_total_listings_count'].head())
df['host_total_listings_count'] = data['host_total_listings_count']


# In[15]:


print(data['host_identity_verified'].head())
df['host_identity_verified'] = data['host_identity_verified'].apply(lambda x: get_bool(x))
print(df['host_identity_verified'].head())


# In[16]:


df['calculated_host_listings_count'] = data['calculated_host_listings_count']
df['calculated_host_listings_count_entire_homes'] = data['calculated_host_listings_count_entire_homes']
df['calculated_host_listings_count_private_rooms'] = data['calculated_host_listings_count_private_rooms']
df['calculated_host_listings_count_shared_rooms'] = data['calculated_host_listings_count_shared_rooms']


# ## Neighbourhood and Listing
# 
# neighbourhood \
# Market (check values) \
# property_type \
# room_type \
# accommodates \
# bathrooms \
# bedrooms \
# beds \
# bed_type \
# amenities
# 

# #### Neighbourhood

# In[17]:


print(data['neighbourhood'].unique())
print(data['neighbourhood_cleansed'].unique())
print(data['neighbourhood_group_cleansed'].unique())

df['neighbourhood'] = data['neighbourhood_cleansed']
df['borough'] = data['neighbourhood_group_cleansed']


# #### Property

# In[18]:


print("property_type: ",data['property_type'].unique())
print("room_type: ",data['room_type'].unique())
print("bathrooms: ",data['bathrooms'].unique())
print("bedrooms: ", data['bedrooms'].unique())
print("beds: ", data['beds'].unique())
print("bed_type: ", data['bed_type'].unique())

df['property_type'] = data['property_type']
df['room_type'] = data['property_type']
df['bathrooms'] = data['bathrooms']
df['bedrooms'] = data['bedrooms']
df['beds'] = data['beds']
df['bed_type'] = data['bed_type']


# #### Amentities provided in the bnb

# In[48]:


print(data['amenities'].unique())


# In[57]:


print(len(data.index))
data['amenities_n'] = data['amenities'].apply(lambda x : str(x).replace('{','').replace('}','')
                        .replace('"','').lower().replace('nan','').split(','))


# In[59]:


amenities = []

def get_list_data(x):
    if x is not None:
        for i in x:
            amenities.append(i)

def convert_to_none(x):
    a = str(x)
    if a == 'nan':
        return None
    else:
        return x
        
data['amenities_n'] = data['amenities_n'].apply(lambda x: convert_to_none(x))
data['amenities_n'].apply(lambda x: get_list_data(x))


# In[61]:


#print(dict((x,amenities.count(x)) for x in set(amenities)))
print(dict(Counter(amenities).most_common(6)))


# Hence, {'wifi': 242492, 'heating': 237638, 'essentials': 236237, 'smoke detector': 223918, 'kitchen': 223320} are the TOP 5 amentities 

# In[63]:


def if_amenities(x, a):
    if x is not None:
        if a in x:
            return True
        else:
            return False
    else:
        return None

def get_count(x):
    if x is not None:
        return len(x)
    else:
        return None
    
df['amenities_count'] = data['amenities_n'].apply(lambda x: get_count(x))
df['is_wifi'] = data['amenities_n'].apply(lambda x : if_amenities(x,'wifi'))
df['is_kitchen'] = data['amenities_n'].apply(lambda x : if_amenities(x,'kitchen'))
df['is_heating'] = data['amenities_n'].apply(lambda x : if_amenities(x,'heating'))
df['is_smoke_detector'] = data['amenities_n'].apply(lambda x : if_amenities(x,'smoke detector'))
df['is_aircon'] = data['amenities_n'].apply(lambda x : if_amenities(x,'air conditioning'))


# ### Price

# In[24]:


print(data['price'].unique())


# In[25]:


df['price'] = data['price'].apply(lambda x: float(str(x).replace(',','').strip('$')))
df['security_deposit'] = data['security_deposit'].apply(lambda x: float(str(x).replace(',','').strip('$')))
df['cleaning_fee'] = data['cleaning_fee'].apply(lambda x: float(str(x).replace(',','').strip('$')))
df['price_for_extra_people'] = data['extra_people'].apply(lambda x: float(str(x).replace(',','').strip('$')))


# ### Stay

# In[26]:


df['guests_included'] = data['guests_included']
df['minimum_nights'] = data['minimum_nights']
df['maximum_nights'] = data['maximum_nights']
df['has_availability'] = data['has_availability'].apply(lambda x: get_bool(x))
df['availability_30'] = data['availability_30']
df['availability_60'] = data['availability_60']
df['availability_90'] = data['availability_90']
df['availability_365'] = data['availability_365']
df['is_instant_bookable'] = data['instant_bookable'].apply(lambda x: get_bool(x))
df['is_business_travel_ready'] = data['is_business_travel_ready'].apply(lambda x: get_bool(x))


# In[27]:


print(data['cancellation_policy'].unique())
df['cancellation_policy'] = data['cancellation_policy']


# #### steps to be taken for the guests

# In[28]:


df['require_guest_profile_picture'] = data['require_guest_profile_picture'].apply(lambda x: get_bool(x))
df['require_guest_phone_verification'] = data['require_guest_phone_verification'].apply(lambda x: get_bool(x))


# ### Reviews of the listing

# In[29]:


df['number_of_reviews'] = data['number_of_reviews']
df['reviews_per_month'] = data['reviews_per_month']
df['number_of_reviews_ltm'] = data['number_of_reviews_ltm']
df['review_scores_rating'] = data['review_scores_rating']
df['review_scores_accuracy'] = data['review_scores_accuracy']
df['review_scores_cleanliness'] = data['review_scores_cleanliness']
df['review_scores_checkin'] = data['review_scores_checkin']
df['review_scores_communication'] = data['review_scores_communication']
df['review_scores_location'] = data['review_scores_location']
df['review_scores_value'] = data['review_scores_value']


# In[70]:


df.to_csv("/Users/toshalpatel/is5152_project/Jan2019-Feb2020_cleaned_data.csv", index=None)


