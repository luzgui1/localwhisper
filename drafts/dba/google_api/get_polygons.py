#%%

#Este script é responsável por obter os polígonos dos bairros de São Paulo e salvar em um arquivo JSON.
#Ele utiliza o Nominatim para obter os polígonos e o Overpass para obter os polígonos dos bairros que não foram encontrados no Nominatim.
#Ele também salva um mapa com todos os polígonos em um arquivo HTML.

import requests
import json
import folium
import os
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

# Inputs
neighborhoods = ["Vila Mariana", "Bela Vista", "Bairro de Pinheiros", "Moema", "Jardim Paulista","Itaim Bibi", "Liberdade"]

# Functions
def get_neighborhood_polygon(name, city="São Paulo"):
    """
    Abordagem de obtenção de polígonos pelo Nominatim.
    Geração das consultas e armazenamento no diretório de Polygons.

    Inputs:
        - name: Nome do bairro
        - city: Nome da cidade
    """

    query = f"{name}, {city}, Brazil"
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': query,
        'format': 'json',
        'polygon_geojson': 1,
        'addressdetails': 0,
        'limit': 1
    }
    headers = {"User-Agent": "MyApp/1.0"}
    try:
        res = requests.get(url, params=params, headers=headers)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"✗ Erro ao buscar {name} no Nominatim: {e}")
        return None, None

    if not data or 'geojson' not in data[0]:
        return None, None

    geom = shape(data[0]['geojson'])

    if isinstance(geom, Polygon):
        coords = [(lat, lon) for lon, lat in geom.exterior.coords]
    elif isinstance(geom, MultiPolygon):
        coords = []
        for poly in geom.geoms:
            coords.extend([(lat, lon) for lon, lat in poly.exterior.coords])
    else:
        print(f"✗ {name} retornou tipo {geom.geom_type}, ignorado.")
        return None, None

    os.makedirs("polygons", exist_ok=True)
    path = f"polygons/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_polygon.json"
    with open(path, "w") as f:
        json.dump({"name": name, "coordinates": coords}, f, indent=2)

    return coords, path

def get_polygon_overpass(name, city="São Paulo"):
    """
    Abordagem de redundância para a obtenção dos polígonos pelo Overpass.

    Inputs:
        - name: Nome do bairro
        - city: cidade
    """

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    area["name"="{city}"]["boundary"="administrative"]["admin_level"="8"]->.searchArea;
    rel(area.searchArea)["name"="{name}"]["boundary"="administrative"];
    out geom;
    """
    try:
        res = requests.get(overpass_url, params={'data': query})
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"✗ Erro no Overpass para {name}: {e}")
        return None, None

    if not data.get("elements"):
        print(f"✗ Overpass não retornou resultados para {name}")
        return None, None

    coords = []
    for member in data['elements'][0].get('members', []):
        if member['role'] == 'outer' and 'geometry' in member:
            coords.extend([(pt['lat'], pt['lon']) for pt in member['geometry']])

    if coords:
        os.makedirs("polygons", exist_ok=True)
        path = f"polygons/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_polygon.json"
        with open(path, "w") as f:
            json.dump({"name": name, "coordinates": coords}, f, indent=2)
        return coords, path

    return None, None

# Criar mapa base
m = folium.Map(location=[-23.58, -46.63], zoom_start=13)

shapely_polygons = []

# Iterar e adicionar polígonos
for bairro in neighborhoods:
    coords, path = get_neighborhood_polygon(bairro)
    if not coords:
        print(f"↪️ Tentando Overpass para {bairro}")
        coords, path = get_polygon_overpass(bairro)

    if coords:
        shapely_polygons.append(Polygon(coords))
        print(f"✓ {bairro} salvo em {path}")
    else:
        print(f"✗ Falha total ao obter polígono para {bairro}")

# Unindo todos os polígonos em um único polígono
merged = unary_union(shapely_polygons)

if merged.geom_type == 'Polygon':
    merged_coords = list(merged.exterior.coords)
else:
    merged_coords = []
    for poly in merged.geoms:
        merged_coords.extend(list(poly.exterior.coords))

with open(os.path.join("polygons", "sao_paulo_polygon.json"), "w") as f:
    json.dump({"name": "Sao Paulo", "coordinates": merged_coords}, f, indent=2)
print("✓ Polígono final salvo em polygons/sao_paulo_polygon.json")

folium.Polygon(
    locations=[(lat, lon) for lat, lon in merged_coords],
    color="red",
    fill=True,
    fill_opacity=0.2,
    tooltip="São Paulo (merged)"
).add_to(m)

# Salvar mapa final com todos os bairros
m.save(os.path.join("polygons", "sao_paulo_bairros_map.html"))
print("🗺️ Mapa final (com polígono unido) salvo em polygons/sao_paulo_bairros_map.html")


#%%