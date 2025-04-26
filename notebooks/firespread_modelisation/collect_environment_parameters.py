#python collect_environment_parameters.py


import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import ee
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import datetime
import xarray as xr
import os
import geemap
from scipy.ndimage import gaussian_filter
from datetime import datetime
import seaborn as sns



# Constants
RESOLUTION = 1000
ROI = 32000
LATITUDE = 39.8180108
LONGITUDE = 27.3259523
TRESHOLD =2000
#Need at least 5 to 10 days 
#Extend if it is not working
STARTDATE = datetime(2017, 10, 13)
ENDDATE = datetime(2017, 10, 25)


def authenticate_earth_engine():
    """
    Authenticate Earth Engine API.
    """
    try:
        print('Beginning of the program')
        ee.Authenticate()
        ee.Initialize()
        print("Connection to Google Earth done")
    except Exception as e:
        print("Error initializing Earth Engine:", e)
        ee.Authenticate()
        ee.Initialize()


# 1-Get Density Population Data
def get_population_array(longitude, latitude,roi):
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(roi).bounds()

    image = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density') \
                .filterDate('2015-01-01', '2021-01-01') \
                .first() \
                .select('population_density') \
                .clip(region)

    array = geemap.ee_to_numpy(image, region=region, scale=1000)
    height, width, depth = array.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2

    # Effectuer un crop pour obtenir une image de 64x64
    array_cropped = array[start_y:start_y + 64, start_x:start_x + 64]
    array_density_population= array_cropped.squeeze()
    print("Density array done")
    return array_density_population



#2- Vegetation
def get_vegetation_array(longitude, latitude,roi,startDate, endDate):
    """
    Get vegetation data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    buffer_distance = roi

    region = point.buffer(buffer_distance).bounds()

    image = ee.ImageCollection('NASA/VIIRS/002/VNP13A1') \
                .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
                .first() \
                .select('NDVI') \
                .clip(region)

    image_resampled = image.reproject(
        crs='EPSG:4326', 
        scale=1000 
    )

    array = geemap.ee_to_numpy(image_resampled, region=region, scale=1000)
    height, width, depth = array.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2

    array_cropped = array[start_y:start_y + 64, start_x:start_x + 64]
    array_ndvi = array_cropped.squeeze()
    print('Vegetation array done')
    return array_ndvi

#3-Elevation

def get_elevation_array(longitude, latitude,roi,startDate, endDate):
    """
    Get elevation data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])

    side_length_m = roi
    region = point.buffer(side_length_m).bounds()

    image = ee.Image("USGS/SRTMGL1_003").select('elevation')

    image_reprojected = image \
        .reproject(crs='EPSG:4326', scale=1000) \
        .clip(region)

    array = geemap.ee_to_numpy(image_reprojected, region=region, scale=1000)

    height, width, depth = array.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2

    array_cropped = array[start_y:start_y + 64, start_x:start_x + 64]

    array = np.squeeze(array_cropped)  
    array = np.where(array == None, np.nan, array)
    print("Elevation array done")
    return array

#4-Wind direction
def get_wind_direction_array(longitude, latitude,roi, startDate, endDate):
    """
    Get wind direction data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(roi).bounds()  # 64 km

    image = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .first() \
        .select(['u_component_of_wind_10m', 'v_component_of_wind_10m']) \
        .clip(region)

    u = geemap.ee_to_numpy(image.select('u_component_of_wind_10m'), region=region, scale=1000 )
    v = geemap.ee_to_numpy(image.select('v_component_of_wind_10m'), region=region, scale=1000)

    # arctan2 gives the angle from north
    direction = (np.arctan2(u, v) * (180 / np.pi)) % 360

    #print("Shape before crop:", direction.shape)
    height, width, depth = direction.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2
    array_cropped = direction[start_y:start_y + 64, start_x:start_x + 64]

    array_smoothed = gaussian_filter(array_cropped, sigma=2)  # degree of smoothing
    array_wind_direction = array_smoothed.squeeze()
    print('Wind direction done')
    return array_wind_direction

#5-Wind speed

def get_wind_speed_array(longitude, latitude,roi,startDate, endDate):
    """
    Get wind speed data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(roi).bounds()  # 64 km

    image = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .first() \
        .select(['u_component_of_wind_10m', 'v_component_of_wind_10m']) \
        .clip(region)

    u = geemap.ee_to_numpy(image.select('u_component_of_wind_10m'), region=region, scale=1000)
    v = geemap.ee_to_numpy(image.select('v_component_of_wind_10m'), region=region, scale=1000)

    wind_speed = np.sqrt(u**2 + v**2)  # Calcul de la vitesse du vent

    height, width, depth = wind_speed.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2
    wind_speed_cropped = wind_speed[start_y:start_y + 64, start_x:start_x + 64]

    array_smoothed = gaussian_filter(wind_speed_cropped, sigma=2)  
    array_wind_speed = array_smoothed.squeeze()
    print("Wind velocity array done")
    return array_wind_speed

#6-Precipitation
def get_precipitation_array(longitude, latitude,roi,startDate, endDate):
    """
    Get precipitation data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(32000).bounds()  # 64 km

    image = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .first() \
        .select(['total_precipitation']) \
        .clip(region)

    precipitation = geemap.ee_to_numpy(image.select('total_precipitation'), region=region, scale=1000)
    precipitation = precipitation / 1000

    height, width, depth = precipitation.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2
    precipitation_cropped = precipitation[start_y:start_y + 64, start_x:start_x + 64]
    array_smoothed = gaussian_filter(precipitation_cropped, sigma=2)
    array_precipitation = array_smoothed.squeeze()
    print("Precipitation array done")
    return array_precipitation

#7-Temperature Max
def get_temperature_max_array(longitude, latitude,roi,startDate, endDate):
    """
    Get maximum temperature data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(roi).bounds()  # 64 km

    image = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .first() \
        .select(['temperature_2m']) \
        .clip(region)

    array = geemap.ee_to_numpy(image.select('temperature_2m'), region=region, scale=1000)
    height, width, depth = array.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2
    array_cropped = array[start_y:start_y + 64, start_x:start_x + 64]
    array_smoothed = gaussian_filter(array_cropped, sigma=2)  
    array_temperature = array_smoothed.squeeze()
    print('Max temperature array done')
    return array_temperature

#8-Specific humidity

def get_specific_humidity_array(longitude, latitude,roi,startDate, endDate):
    """
    Get specific humidity data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(roi).bounds()  # 64 km

    image = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .first() \
        .select(['dewpoint_temperature_2m']) \
        .clip(region)

    dewpoint_temperature = geemap.ee_to_numpy(image.select('dewpoint_temperature_2m'), region=region, scale=1000)
    Patm= 1013.25
    dewpoint_pressure = 6.112*np.exp(17.67*(dewpoint_temperature-273.15)/(dewpoint_temperature-29.65))
    humidity_specific = (0.622 * dewpoint_pressure) / (Patm - 0.378*dewpoint_pressure)  # Humidity spécific
    array = humidity_specific

    height, width, depth = array.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2
    array_cropped = array[start_y:start_y + 64, start_x:start_x + 64]

    array_smoothed = gaussian_filter(array_cropped, sigma=2)
    array_humidity = array_smoothed.squeeze()
    print('Humidity array recover')
    return array_humidity

#9-FireMask

def get_fire_mask_array(longitude, latitude,roi,startDate, endDate,cloud_filter=50):
    """
    Get fire mask data from Earth Engine.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(roi).bounds()  # 64 km

    SBands = ['B11']

    # Charger la collection de données Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)
    image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .filterBounds(region) \
        .select(SBands) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))   # Filtrer par date et région
    image_count = image_collection.size().getInfo()
    print("Number of fire image found ")
    print(image_count)

    mosaic = image_collection.max() # Get the maximum value of the pixel among the image collection
    
    return mosaic

def define_treshold(longitude, latitude,mosaic):
    """
    Refine the threshold of the fire mask.
    """
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(ROI).bounds()
    

    fire_mask_array = geemap.ee_to_numpy(mosaic, region=region, scale=1000)

    flattened = fire_mask_array.flatten()
    sns.histplot(flattened, bins=100)
    plt.title("Histogramme des valeurs B12")
    plt.xlabel("Réflectance B12")
    plt.ylabel("Nombre de pixels")
    plt.show()
    print("Graph Done")




def get_binary_fire_mask_array(longitude, latitude,mosaic, treshold):

    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(ROI).bounds()
    
    fire_mask = mosaic.gt(treshold)
    fire_mask_array = geemap.ee_to_numpy(fire_mask, region=region, scale=1000)

    array = fire_mask_array

    height, width, depth = array.shape
    start_x = (width - 64) // 2
    start_y = (height - 64) // 2
    array_cropped = array[start_y:start_y + 64, start_x:start_x + 64]
    array_fire_mask = array_cropped.squeeze()
    print("Binary Fire Mask array done")
    return array_fire_mask

def rgb_firemask(longitude, latitude,roi,startDate, endDate,cloud_filter=50):

    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(32000).bounds()  # 64 km

    SBands = ['B4','B3','B2']

    # Charger la collection de données Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)
    image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')) \
        .filterBounds(region) \
        .select(SBands) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))   
    image_count = image_collection.size().getInfo()
    print(image_count)

    #  R, G, B
    SBands = ['B4', 'B3', 'B2'] 

    # Image mosaic to cpver the whole zone
    mosaic = image_collection.max()

    rgb_array = geemap.ee_to_numpy(mosaic, region=region, scale=100)

    rgb_array = np.squeeze(rgb_array)

    # Verify the shape of the array
    print("RGB array shape:", rgb_array.shape)  # (H, W, 3)

    # Normalisation on each band
    def normalize(img):
        img = img.astype(np.float32)
        for i in range(3):  
            band = img[:, :, i]
            min_val = np.nanmin(band)
            max_val = np.nanmax(band)
            if max_val - min_val > 0:
                img[:, :, i] = (band - min_val) / (max_val - min_val)
            else:
                img[:, :, i] = 0
        return img

    rgb_norm = normalize(rgb_array)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_norm)
    plt.title("Image RGB - Sentinel-2")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    authenticate_earth_engine()

    array_pop_density = get_population_array(LONGITUDE, LATITUDE, ROI)
    array_ndvi = get_vegetation_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)
    array_elevation = get_elevation_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)
    array_wind_direction = get_wind_direction_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)
    array_wind_speed = get_wind_speed_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)
    array_precipitation = get_precipitation_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)
    array_temperature = get_temperature_max_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)
    array_humidity = get_specific_humidity_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE)

    mosaic = get_fire_mask_array(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE, 50)
    print('test')
    rgb_firemask(LONGITUDE, LATITUDE, ROI, STARTDATE, ENDDATE, 50)

    try:
        status = False
        while(status==False):
            print("tour")
            define_treshold(LONGITUDE, LATITUDE,mosaic)
            user_input = int(input("According to the graph what treshold do you choose ? int "))
            binary_firemask_array = get_binary_fire_mask_array(LONGITUDE, LATITUDE,mosaic,user_input)

            plt.figure(figsize=(6, 6))
            plt.imshow(binary_firemask_array, cmap='grey')  # ou 'binary' pour fond blanc et pixels noirs
            plt.title("Binary Map")
            plt.axis('off')  # Cache les axes
            plt.colorbar(label="Class")
            plt.savefig("example/binary_firemask.png", dpi=300)
            plt.show()

            user_input2 = int(input("Are you satisfied with the result ? (True=1, False=0) "))
            if(user_input2==1):
                status=True



        feature_names = ["Elevation", "Wind Direction", "Wind Speed", "Temperature","Precipitation","Humidity","NDVI","Population Density","PrevFireMask"]

        arrays = [array_elevation,array_wind_direction,array_wind_speed,array_temperature,array_precipitation,array_humidity,array_ndvi,array_pop_density,binary_firemask_array]


        assert all(arr.shape == (64, 64) for arr in arrays)
        stacked_array = np.stack(arrays, axis=-1)
        print("Final shape :", stacked_array.shape)  # (64, 64, 9)
        np.save("example/array_extracted.npy", stacked_array)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle("9 environmental parameters (64x64)")

        for i in range(9):
            ax = axes[i // 3, i % 3]  
            im = ax.imshow(stacked_array[:, :, i], cmap='viridis')  
            ax.set_title(feature_names[i])
            ax.axis('off')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show() 

    except ValueError:
        print("Error : you have to enter a valid int.")
    