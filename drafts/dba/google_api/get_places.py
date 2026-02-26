#%%
import json
from shapely.geometry import Point,Polygon
import csv
import random
import os
import googlemaps
import logging
from pipeline.api_runner import GooglePlacesCollector

api_key = os.getenv("GOOGLE_MAPS_API")
gmaps = googlemaps.Client(key=api_key)
polygon_file = './polygons/sao_paulo_polygon.json'

polygon = json.load(open(polygon_file))

polygon_points = [(lng, lat) for lat, lng in polygon['coordinates']]

city_polygon = Polygon(polygon_points)

# Termos de busca
search_terms = ['bar', 'restaurant', 'pub', 'cafe']

logging.basicConfig(level=logging.INFO)

collector = GooglePlacesCollector(
    gmaps_client=gmaps,
    polygon=city_polygon,
    output_csv='google_places.csv',
    search_terms=search_terms
)
# Ponto inicial (minha casa)
location = (-23.585312763771512, -46.637254151287834)

collector.run(start_location=location)