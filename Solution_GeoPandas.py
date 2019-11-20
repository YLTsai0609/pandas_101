# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Pratical GeoPandas usage
#

# # Geo-Encoding
# * coordinate <---> landmark, adress
#
# `from geopandas.tools import geocode`
#
# > Open
#
# *[OpenStreetMap Nominatim geocoder](https://nominatim.openstreetmap.org/)
#
# > Payment needed
#
# * [google map](https://www.google.com/maps/@25.0539054,121.5754524,13z)
# * [Bing map](https://www.bing.com/maps)
# * [Baidu](https://map.baidu.com/)

# # GeoPandas 101

import geopandas as gpd
from IPython.core.display import display

# 1. How to import geopandas and check the version? 
print(gpd.__version__)

# 2. What class / method / arrtibutes in geopandas?
print(dir(gpd))


# +
# 3 how to create a GeoDataFrame from FataFrame

# +
# 4 how to read file with shp file

# +
# 5 use geocoder to get coordinate from adress

def my_geocoder(row):
    try:
        point = geocode(row, provider='nominatim').geometry.iloc[0]
        return pd.Series({'Latitude': point.y, 'Longitude': point.x, 'geometry': point})
    except:
        return None
# -


