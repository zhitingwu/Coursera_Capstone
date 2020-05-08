#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install BeautifulSoup4')


# In[2]:


get_ipython().system('conda install -c conda-forge geopy --yes')


# In[3]:


get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')


# In[4]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

from bs4 import BeautifulSoup

print('Libraries imported.')


# # Function for getting lat and long from geocode

# In[5]:


def geo(dataframe): 
    geolocator = Nominatim(user_agent="singapore_explorer")
    from geopy.extra.rate_limiter import RateLimiter
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
#RateLimiter allows to perform bulk operations while gracefully handling error responses and adding delays when needed.

#In the example below a delay of 1 second (min_delay_seconds=1) will be added between each pair of geolocator.geocode calls;
#all geopy.exc.GeocoderServiceError exceptions will be retried (up to max_retries times):
    dataframe['location'] = dataframe['location_of_centre'].apply(geocode)

    dataframe['point'] = dataframe['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    return dataframe.head()


# # Function for getting lat and long separately 

# In[6]:


def split(data):
    long = []
    lat = []
    for x,y,z in data.point:
            long.append(x)
            lat.append(y)
    frame = pd.DataFrame({'Latitude':long , 'Longitude':lat})
    
    return frame


# # Function for merging dataframes 

# In[7]:


def together(data1, data2): 
    data1 = data1.merge(data2, on=data1.location_of_centre)
    data1 = data1.drop('key_0', axis=1)
    return data1


# # Get Postalcodes from wikipedia

# In[9]:


url = 'https://en.wikipedia.org/wiki/Postal_codes_in_Singapore'
results = requests.get(url).text
re = BeautifulSoup(results, 'html.parser')
table = re.find('table')


# # Create empty dataframe 

# In[10]:


column_names = ['Postal_District', 'Postalcode', 'Neighborhood']
frame = pd.DataFrame(columns = column_names)
frame


# In[11]:


for tr in table.findAll("tr"):
    row = []
    for td in tr.findAll('td'):
        row.append(td.text.strip())
    if len(row) == 3:
        frame.loc[len(frame)] = row 


# In[12]:


frame.isnull().sum() #check for bad rows


# In[13]:


new_df = pd.DataFrame(frame.Neighborhood.str.split(',').tolist()).stack().reset_index()
new_df.columns = ['one', 'two', 'location_of_centre']
new = new_df.drop(['one','two'], axis=1)
new


# In[14]:


new_1 = new[['location_of_centre']] + ', Singapore' #adding 'Singapore' so as to locate on geolocator
new_1.head()


# In[15]:


geo(new_1)


# In[16]:


new_1.tail()


# In[17]:


new_1.shape


# In[18]:


#new_1.isnull()


# In[19]:


new_1.iloc[12:16, :]


# In[20]:


new_1 = new_1.dropna(subset = ['location', 'point'])


# In[21]:


new_1.reset_index()


# In[22]:


new_2 = split(new_1)


# In[52]:


new_lst = together(new_1,new_2)


# In[53]:


city = 'Singapore'

geolocator = Nominatim(user_agent="sin_explorer")
location = geolocator.geocode(city)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Singapore are {}, {}.'.format(latitude, longitude))


# In[54]:


CLIENT_ID = 'WFK1SF3GTUYLQUHZ15BDG1XOEBSVEAMLQ3DMUWKTJAZOCFGH' # your Foursquare ID
CLIENT_SECRET = 'G1ZQ50ZJ5XIOFMYYCOD1OWG5BCMGR4GDA0ZYTIJLB1XKLD2A' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 500


# In[55]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[56]:


sin_venues = getNearbyVenues(names=new_lst['location'],
                                   latitudes=new_lst['Latitude'],
                                   longitudes=new_lst['Longitude']
                                  )


# In[57]:


sin_venues.head()


# In[58]:


sin_venues.Neighborhood = sin_venues.Neighborhood.astype(str)


# In[59]:


sin_venues.groupby('Neighborhood').count()


# # Check how many venues were returned for each neighborhood

# In[60]:


# one hot encoding
sin = pd.get_dummies(sin_venues[['Venue Category']], prefix="", prefix_sep="")
#separates the different venues into columns

# add neighborhood column back to dataframe
sin['Neighborhood'] =sin_venues['Neighborhood'] 
sin['Neighborhood']
# move neighborhood column to the first column
fixed_columns = [sin.columns[-1]] + list(sin.columns[:-1])
sin = sin[fixed_columns]


# In[61]:


sin['Neighborhood'] = sin[['Neighborhood']].astype(str) #original data was in geopy.location.Location


# In[62]:


sin_grouped = sin.groupby(sin['Neighborhood']).mean().reset_index()


# In[63]:


sin_grouped.head()


# # Let's print each neighborhood along with the top 10 most common venues¶

# In[64]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[65]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = sin_grouped['Neighborhood']

for ind in np.arange(sin_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(sin_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[66]:


# set number of clusters
kclusters = 8

sin_grouped_clustering = sin_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(sin_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[67]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_.astype(int))


# In[68]:


for x in neighborhoods_venues_sorted['Neighborhood']:
    print (x)
    break


# In[69]:


#new_lst.head()


# In[70]:


neighborhoods_venues_sorted.Neighborhood = neighborhoods_venues_sorted.Neighborhood.astype(str)


# In[71]:


new_again = new_lst


# In[72]:


new_again['location'] = new_again['location'].astype(str)


# In[73]:


new_again.sort_values(by=['location'], ascending=True)


# In[74]:


new_again = new_again.rename(columns={'location':'Neighborhood'})


# In[75]:


new_merge = new_again.join(neighborhoods_venues_sorted.set_index('Neighborhood'),on=new_again['Neighborhood'])


# In[76]:


new_merge.head()


# In[77]:


new_merge = new_merge.dropna(subset = ['Cluster Labels'])


# In[78]:


new_merge[['Cluster Labels']] = new_merge[['Cluster Labels']].astype(int)


# In[79]:


from folium import plugins


# In[80]:



map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)
# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(map_clusters)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(new_merge['Latitude'], new_merge['Longitude'], new_merge['Neighborhood'], new_merge['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# # examine cluster 0 

# In[81]:


new_merge.loc[new_merge['Cluster Labels'] == 0, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 1

# In[82]:


new_merge.loc[new_merge['Cluster Labels'] == 1, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 2

# In[83]:


new_merge.loc[new_merge['Cluster Labels'] == 2, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 3

# In[84]:


new_merge.loc[new_merge['Cluster Labels'] == 3, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 4

# In[85]:


new_merge.loc[new_merge['Cluster Labels'] == 4, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 5

# In[86]:


new_merge.loc[new_merge['Cluster Labels'] == 5, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 6

# In[87]:


new_merge.loc[new_merge['Cluster Labels'] == 6, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# In[88]:


# Examine cluster 7 


# In[89]:


new_merge.loc[new_merge['Cluster Labels'] == 7, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Examine cluster 8 

# In[90]:


new_merge.loc[new_merge['Cluster Labels'] == 8, new_merge.columns[[1] + list(range(5, new_merge.shape[1]))]]


# # Import data for food centres not found in foursquare 

# In[91]:


path = r'list-of-government-markets-hawker-centres.csv'
df = pd.read_csv(path)
df.head()


# In[92]:


hawker = df['type_of_centre'] == 'MHC'


# In[93]:


df_MHC = df[hawker]
df_MHC.shape


# In[94]:


geo(df_MHC)


# In[95]:


df_MHC = df_MHC.drop(['owner', 'no_of_cooked_food_stalls','no_of_mkt_produce_stalls', 'no_of_stalls'],axis=1)


# In[96]:


df_MHC = df_MHC.dropna(subset =['location', 'point'])


# In[97]:


data2 = split(df_MHC)


# In[98]:


df_MHC_1 = together(df_MHC, data2)


# In[99]:


# instantiate a feature group for the incidents in the dataframe
hawker_map = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the incidents feature group
for lat, lng, in zip(df_MHC_1.Latitude, df_MHC_1.Longitude):
    hawker_map.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add incidents to map
new_map = map_clusters.add_child(hawker_map)
new_map


# # Import data for no of professionals

# In[100]:


path = r'resident-working-persons-aged-15-years-and-over-by-planning-area-and-occupation.csv'
jobs = pd.read_csv(path)
jobs.head()


# In[101]:


total = jobs['level_1'] == 'Total'
total_occupation = jobs[total]
total_occupation.head()


# # Change the value in total_occupation to actual value (in thousands)

# In[102]:


total_occupation['value'] = total_occupation['value'] * 1000


# In[103]:


total_occupation_1 = total_occupation.drop(['year','level_2'], axis=1)


# In[104]:


total_occupation_1.columns = ['Jobs', 'location_of_centre', 'No of Population']
total_occupation_1['location_of_centre'] = total_occupation_1['location_of_centre'] + ' , Singapore'


# In[105]:


total_occupation_1


# In[106]:


geo(total_occupation_1)


# In[107]:


total_occupation_1 = total_occupation_1.dropna(subset=['point', 'location'])


# In[108]:


new = split(total_occupation_1)


# In[109]:


total_occupation_1 = total_occupation_1.merge(new, on=total_occupation_1.location_of_centre)
total_occupation_1.head()


# In[132]:


total_occupation_1['No of Population'].min()


# In[133]:


total_occupation_1['No of Population'].max()


# In[110]:


#import csv
#import json
#from collections import OrderedDict


# In[112]:


#!pip install geojson


# In[113]:


#import geojson
#from geojson import Feature, FeatureCollection, Point


# In[125]:


total_occupation_1['location'] = total_occupation_1['location'].astype(str)


# In[126]:


# columns used for constructing geojson object
#features = total_occupation_1.apply(
#    lambda row: Feature(geometry=Point((float(row['Longitude']), float(row['Latitude'])))),
#    axis=1).tolist()

# all the other columns used as properties
#properties = total_occupation_1.drop(['Latitude', 'Longitude'], axis=1).to_dict('records')

# whole geojson object
#feature_collection = FeatureCollection(features=features, properties=properties)


# In[127]:


#with open('wl.geojson', 'w', encoding='utf-8') as f:
#    json.dump(feature_collection, f, ensure_ascii=False)


# In[ ]:


#total_occupation_1.location = total_occupation_1.location.astype(str)


# In[120]:


total_occupation_1['No of Population'] = total_occupation_1['No of Population'].astype(str)


# In[121]:


# instantiate a feature group for the incidents in the dataframe


real_map = new_map
tired = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the incidents feature group
for lat, lng, in zip(total_occupation_1.Latitude, total_occupation_1.Longitude):
    tired.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add pop-up text to each marker on the map
latitudes = list(total_occupation_1.Latitude)
longitudes = list(total_occupation_1.Longitude)
label1 = list(total_occupation_1['No of Population'])
#label2 = list(total_occupation_1['location_of_centre'])


# In[122]:


for lat, lng, label1 in zip(latitudes, longitudes, label1):
              folium.Marker([lat, lng], popup=label1).add_to(real_map)    
    
# add incidents to map
real_map.add_child(tired)


# In[ ]:




