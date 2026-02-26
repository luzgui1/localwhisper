#%%

# Este script é responsável por coletar os dados dos locais da API do Google Places.
# Ele utiliza a API do Google Places para coletar os dados dos locais e salvar em um arquivo CSV.
# Ele também salva um mapa com todos os locais em um arquivo HTML.

import csv
import time
import random
import logging
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point

class GooglePlacesCollector:
    def __init__(self, gmaps_client, polygon, output_csv, search_terms, radius=1000, max_results=1000000):
        self.gmaps = gmaps_client
        self.polygon = polygon
        self.output_csv = output_csv
        self.search_terms = search_terms
        self.radius = radius
        self.max_results = max_results
        self.visited_places = set()
        self.points_inside_polygon = []
        self.logger = logging.getLogger("GooglePlacesCollector")

    def generate_random_point_within_polygon(self, max_attempts=1000):
        minx, miny, maxx, maxy = self.polygon.bounds
        for attempt in range(max_attempts):
            random_point = Point(
                np.random.uniform(minx, maxx),
                np.random.uniform(miny, maxy)
            )
            if self.polygon.contains(random_point):
                lat, lng = random_point.y, random_point.x
                self.points_inside_polygon.append((lat, lng))
                self.logger.debug(f"[Válido] Ponto dentro do polígono: ({lat}, {lng}) [Tentativa {attempt+1}]")
                return lat, lng
            else:
                self.logger.debug(f"[Inválido] Fora do polígono: ({random_point.y}, {random_point.x}) [Tentativa {attempt+1}]")
        raise RuntimeError("Não foi possível gerar um ponto válido dentro do polígono após várias tentativas.")

    def write_results(self, writer, results):
        for result in results:
            place_id = result.get('place_id', '')
            if place_id in self.visited_places:
                continue
            writer.writerow({
                'business_status': result.get('business_status', ''),
                'formatted_address': result.get('formatted_address', ''),
                'geometry': result.get('geometry', ''),
                'icon': result.get('icon', ''),
                'icon_background_color': result.get('icon_background_color', ''),
                'icon_mask_base_uri': result.get('icon_mask_base_uri', ''),
                'name': result.get('name', ''),
                'opening_hours': result.get('opening_hours', ''),
                'photos': result.get('photos', ''),
                'place_id': place_id,
                'plus_code': result.get('plus_code', ''),
                'price_level': result.get('price_level', ''),
                'rating': result.get('rating', ''),
                'reference': result.get('reference', ''),
                'types': result.get('types', ''),
                'user_ratings_total': result.get('user_ratings_total', '')
            })
            self.visited_places.add(place_id)

    def run(self, start_location=None):
        location = start_location or self.generate_random_point_within_polygon()

        fieldnames = [
            'business_status', 'formatted_address', 'geometry', 'icon',
            'icon_background_color', 'icon_mask_base_uri', 'name',
            'opening_hours', 'photos', 'place_id', 'plus_code',
            'price_level', 'rating', 'reference', 'types', 'user_ratings_total'
        ]

        with open(self.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            pbar = tqdm(total=self.max_results)
            while pbar.n < self.max_results:
                params = {
                    'query': random.choice(self.search_terms),
                    'location': location,
                    'radius': self.radius
                }
                data = self.gmaps.places(**params)
                self.write_results(writer, data['results'])
                pbar.update(len(data['results']))

                while 'next_page_token' in data:
                    time.sleep(2)
                    params['page_token'] = data['next_page_token']
                    data = self.gmaps.places(**params)
                    self.write_results(writer, data['results'])
                    pbar.update(len(data['results']))

                location = self.generate_random_point_within_polygon()
                self.logger.info(f"Nova localização: {location}")

        # Geração do mapa com pontos usados
        self.logger.info("Processo concluído. Gerando mapa com os pontos visitados...")

        map_center = self.points_inside_polygon[0] if self.points_inside_polygon else location
        city_map = folium.Map(location=map_center, zoom_start=14)

        for lat, lng in self.points_inside_polygon:
            folium.Marker(location=[lat, lng]).add_to(city_map)

        city_map.save("city_points_map.html")
        self.logger.info("Mapa salvo como 'city_points_map.html'")


#%%