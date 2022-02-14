from this import d
import folium
import geopandas

import streamlit      as st
import pandas         as pd
import numpy          as np
import plotly.express as px

from functools        import reduce
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster
from datetime         import datetime


def script_settings():
    st.set_page_config( layout='wide' )

    return None


@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )
    data['date'] = pd.to_datetime( data['date'] )

    return data


@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile


def set_feature( data ):
    # add new features
    data['price_m2'] = data['price'] / ( data['sqft_lot'] * 0.092903 )

    return data


def overview_data( data ):
    # copy data
    df = data.copy()

    # filters
    f_attributes = st.sidebar.multiselect(
        label='Enter attributes',
        options=df.columns
    )

    f_zipcode = st.sidebar.multiselect(
        label='Enter zipcode',
        options=df['zipcode'].unique()
    )

    # filter dataset rows and columns
    if ( f_zipcode != [] ) & ( f_attributes != [] ):
        df = df.loc[ df['zipcode'].isin( f_zipcode ), f_attributes ]

    elif ( f_zipcode != [] ) & ( f_attributes == [] ):    
        df = df.loc[ df['zipcode'].isin( f_zipcode ), : ]

    elif ( f_zipcode == [] ) & ( f_attributes != [] ):    
        df = df.loc[ :, f_attributes ]

    else:
        df = df.copy()

    # Page title and description
    st.title( 'House Rocket Company' )
    st.markdown( 'Welcome to House Rocket Data Analysis!' )  

    # Section title
    st.title( 'Data Overview' )

    # Show filtered data
    st.dataframe( df )

    # Columns for the average metrics and descriptive statistcs
    c1, c2 = st.columns((1,1))

    # copy dataset
    df = data.copy() if f_zipcode == [] else data[ data['zipcode'].isin( f_zipcode ) ]

    # Average metrics
    df1 = df[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()
    df2 = df[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df3 = df[['sqft_living', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df4 = df[['price_m2', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()

    # merge
    data_frames = [df1, df2, df3, df4]
    df5 = reduce( lambda left, right: pd.merge( left, right, on=['zipcode'], how='inner'), data_frames)
    df5.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

    # plot 
    c1.header( "Average Values" )
    c1.dataframe( df5, height=600 )

    # Descriptive Statistcs
    # dataset with only numerical attributes
    num_attributes = df.select_dtypes( include=[ int, 'float64' ] )

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

    # plot 
    c2.header( "Descriptive Statistcs" )
    c2.dataframe( df1, height=600 )

    return None


def portfolio_density( data, geofile ):
    # Section title
    st.title( 'Region Overview' )

    # columns for plot
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

    return None


def commercial_distribution( data ):
    # titles
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
    df = data[ data['yr_built'] <= f_year_built]
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
    df = data[ data['date'] <= f_date ]
    df = df[['date', 'price']].groupby( 'date' ).mean().reset_index()

    # plot
    fig = px.line( df, x='date', y='price' )
    st.plotly_chart( fig, use_container_width=True )

    # ---- Histogram
    st.header( 'Price Distributinon' )
    st.sidebar.subheader( 'Select Max Price' )

    # filter
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )

    f_price = st.sidebar.slider(  
        'Price',
        min_price, 
        max_price,
        avg_price
    )

    # data filtering
    df = data[ data['price'] <=  f_price ]

    # plot
    fig = px.histogram( df, x='price', nbins=50 )
    st.plotly_chart( fig, use_container_width=True )

    return None


def attributes_distribution( data ):
    # titles
    st.sidebar.title( 'Attributes Options' )
    st.title( 'House Attributes' )

    # filters
    f_bedrooms = st.sidebar.selectbox( 
        'Max Number of Bedrooms',
        sorted( set( data['bedrooms'].unique() ) ),
        len( data['bedrooms'].unique() ) - 1
    )

    f_bathrooms = st.sidebar.selectbox( 
        'Max Number of Bathrooms',
        sorted( set( data['bathrooms'].unique() ) ),
        len( data['bathrooms'].unique() ) - 1 
    )

    f_floors = st.sidebar.selectbox( 
        'Max Number of Floors',
        sorted( set( data['floors'].unique() ) ),
        len( data['floors'].unique() ) - 1
    )

    f_waterview = st.sidebar.checkbox(
        'Only Houses with Water View'
    )

    # columns for plot
    c1, c2 = st.columns( 2 )

    # House per bedrooms
    c1.header( 'Houses per Bedrooms' )
    df = data[ data['bedrooms'] <= f_bedrooms ]
    fig = px.histogram( df, x='bedrooms', nbins=19 )
    c1.plotly_chart( fig, use_container_width=True )

    # House per bathrooms
    c2.header( 'Houses per Bathrooms' )
    df = data[ data['bathrooms'] <= f_bathrooms ]
    fig = px.histogram( df, x='bathrooms', nbins=19 )
    c2.plotly_chart( fig, use_container_width=True )

    # columns for plot
    c1, c2 = st.columns( 2 )

    # House per floors
    c1.header( 'Houses per Floors' )
    df = data[ data['floors'] <= f_floors ]
    fig = px.histogram( df, x='floors', nbins=19 )
    c1.plotly_chart( fig, use_container_width=True )

    # House per water view
    df = data[ data['waterfront'] == 1 ] if f_waterview else data

    c2.header( 'Houses with Water View ' )
    fig = fig = px.histogram( df, x='waterfront', nbins=2 )
    c2.plotly_chart( fig, use_container_width=True )

    return None


if __name__ == '__main__':
    script_settings()

    # ---- data extraction
    path = 'data/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data( path )
    geofile = get_geofile( url )

    # ---- data transformation  
    data = set_feature( data )

    overview_data( data )
    portfolio_density( data, geofile )
    commercial_distribution( data )
    attributes_distribution( data ) 