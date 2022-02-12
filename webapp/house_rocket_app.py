
from email.policy import default
import folium
import geopandas

#import plotly.express as px
import streamlit      as st
import pandas         as pd
import numpy          as np
import plotly.express as px

from functools        import reduce
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster
from datetime         import datetime


st.set_page_config( layout='wide' )

# read data
@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )
    data['date'] = pd.to_datetime( data['date'] )

    return data


@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile

# load data
path = 'data/kc_house_data.csv'
data = get_data( path )


# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile( url )



# add new features
data['price_m2'] = data['price'] / ( data['sqft_lot'] * 0.092903 )


st.title( 'House Rocket Company' )

st.markdown( 'Welcome to House Rocket Data Analysis' )


###################
# Data Overview
###################

# filters
f_attributes = st.sidebar.multiselect(
    label='Enter attributes',
    options=data.columns
)

f_zipcode = st.sidebar.multiselect(
    label='Enter zipcode',
    options=data['zipcode'].unique()
)


st.title( 'Data Overview' )


# filter dataset rows and columns
if ( f_zipcode != [] ) & ( f_attributes != [] ):
    data = data.loc[ data['zipcode'].isin( f_zipcode ), f_attributes ]

elif ( f_zipcode != [] ) & ( f_attributes == [] ):    
    data = data.loc[ data['zipcode'].isin( f_zipcode ), : ]

elif ( f_zipcode == [] ) & ( f_attributes != [] ):    
    data = data.loc[ :, f_attributes ]

else:
    data = data.copy()

# Filtered data
st.dataframe( data )

# Columns for the average metrics and descriptive statistcs
c1, c2 = st.columns((1,1))

# Average metrics
df1 = data[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()
df2 = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df3 = data[['sqft_living', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df4 = data[['price_m2', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()

# merge
data_frames = [df1, df2, df3, df4]
df = reduce( lambda left, right: pd.merge( left, right, on=['zipcode'], how='inner'), data_frames)
df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

c1.header( "Average Values" )
c1.dataframe( df, height=600 )

# Descriptive Statistcs
# dataset with only numerical attributes
num_attributes = data.select_dtypes( include=[ int, 'float64' ] )

# central tendency
_mean   = pd.DataFrame( num_attributes.apply( np.mean, axis=0) )
_median = pd.DataFrame( num_attributes.apply( np.median, axis=0 ) ) 

# dispersion
_std = pd.DataFrame( num_attributes.apply( np.std, axis=0 ) )
_min = pd.DataFrame( num_attributes.apply( np.min, axis=0 ) )
_max = pd.DataFrame( num_attributes.apply( np.max, axis=0 ) )

# concatenate
df1 = pd.concat( [_max, _min, _mean, _median, _std], axis=1 ).reset_index()
df1.columns = ['ATTRIBUTES', 'MAX','MIN', 'MEAN', 'MEDIAN', 'STD']

c2.header( "Descriptive Statistcs" )
c2.dataframe( df1, height=600 )


###################
# Portfolio density
###################
st.title( 'Region Overview' )

c1, c2 = st.columns((1,1))
 
c1.header( 'Portfolio Density' )
df = data.sample( 10 )

# Base Map - Folium 
density_map = folium.Map( 
    location=[data['lat'].mean(), data['long'].mean() ],
    default_zoom_start=15 
) 

marker_cluster = MarkerCluster().add_to( density_map )
for name, row in df.iterrows():
    folium.Marker( 
        [row['lat'], row['long'] ], 
        popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'],
                row['date'],
                row['sqft_living'],
                row['bedrooms'],
                row['bathrooms'],
                row['yr_built'] ) ).add_to( marker_cluster )

with c1:
    folium_static( density_map )


# Region Price Map
c2.header( 'Price Density' )

df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df.columns = ['ZIP', 'PRICE']

geofile = geofile[ geofile['ZIP'].isin( df['ZIP'].tolist() ) ]

region_price_map = folium.Map( 
    location=[ data['lat'].mean(), data['long'].mean() ],
    default_zoom_start=15 
) 


region_price_map.choropleth( 
    data = df,
    geo_data = geofile,
    columns=['ZIP', 'PRICE'],
    key_on='feature.properties.ZIP',
    fill_color='YlOrRd',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    legend_name='AVG PRICE' 
)

with c2:
    folium_static( region_price_map )


###################
# Properties distribution by commercial categories
###################
st.sidebar.title( 'Comercial Options' )
st.title( 'Comercial Attributes' )

# ---- Average Price per Year

# filters
min_year_built = int( data['yr_built'].min() )
max_year_built = int( data['yr_built'].max() )

st.sidebar.subheader( 'Select Max Year Built' )
f_year_built = st.sidebar.slider( 
    'Year Built',
    min_year_built,
    max_year_built,
    max_year_built
)

# data filtering
df = data.loc[ data['yr_built'] <= f_year_built]
df = df[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

# plot
st.header( 'Average Price per Year Built' )
fig = px.line( df, x='yr_built', y='price' )
st.plotly_chart( fig, use_container_width=True )




# ---- Average Price per Day

# filters
min_date = data['date'].min().to_pydatetime()

max_date = data['date'].max().to_pydatetime()

st.sidebar.subheader( 'Select Max Date' )
f_date = st.sidebar.slider( 
    'Date',
    min_date,
    max_date,
    max_date
)

# data filtering
df = data.loc[ data['date'] <= f_date ]
df = df[['date', 'price']].groupby( 'date' ).mean().reset_index()

# plot
fig = px.line( df, x='date', y='price' )
st.plotly_chart( fig, use_container_width=True )