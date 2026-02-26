#%%
import json
from shapely.geometry import Point,Polygon
import csv
import random
import os
import googlemaps
import logging
from pipeline.api_runner import GooglePlacesCollector


gmaps = googlemaps.Client(key='AIzaSyA9dYrpkWoIBubwT92QpatvnAT-1b8suU4')
# Polígono final da cidade de São Paulo obtido no script get_polygons.py

print(gmaps.places(query='bar', location=(40.7128, -74.0060), radius=1000))

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