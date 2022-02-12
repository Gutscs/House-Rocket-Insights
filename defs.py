from geopy.geocoders import Nominatim
import time

geolocator = Nominatim( user_agent='geopy' )

def get_data( x ):
    index, row = x
    time.sleep( 1 )
    
    # call API
    response = geolocator.reverse( row['query'], timeout=20 )
    address = response.raw['address']
    
    # parse response
    place_id = response.raw['place_id'] if 'place_id' in response.raw else 'NA'
    osm_type = response.raw['osm_type'] if 'osm_type' in response.raw else 'NA'
    country = address['country'] if 'country' in address else 'NA'
    country_code = address['country_code'] if 'country_code' in address else 'NA'
    
    return place_id, osm_type, country, country_code