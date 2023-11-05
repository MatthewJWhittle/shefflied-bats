import rasterio as rio
import geopandas as gpd
import rasterio as rio
from rasterio.features import geometry_mask
import numpy as np


def rasterise_gdf(gdf:gpd.GeoDataFrame, resolution:float, output_file:str, bbox=None):
    # Define the raster size and transform
    # Here, I'm assuming a 1x1 meter resolution and using the bounds of the GeoDataFrame
    if bbox is None:
        x_min, y_min, x_max, y_max = gdf.total_bounds
    else: 
        x_min, y_min, x_max, y_max = bbox

    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    transform = rio.transform.from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Create a mask: rasterize the GeoDataFrame. This gives a value of True where the geometry covers a square
    mask = geometry_mask(gdf.geometry, transform=transform, invert=True, out_shape=(height, width))
    # Convert the boolean mask to uint8 (or another supported data type)
    mask = mask.astype('uint8')

    # Write the mask to a raster file
    with rio.open(
        output_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mask.dtype,
        crs=gdf.crs,
        transform=transform,
        nodata=255,
    ) as dest:
        dest.write(mask.astype(rio.uint8), 1)

    return output_file


import rasterio as rio
from rasterio.enums import Resampling

def aggregate_raster(input_file, output_file, scale_factor):
    with rio.open(input_file) as src:
        # Read the data
        data = src.read(1)
        
        # Calculate the shape of the destination array
        dst_height = int(data.shape[0] // scale_factor)
        dst_width = int(data.shape[1] // scale_factor)
        
        # Aggregate data to new resolution
        aggregated_data = data.reshape((dst_height, scale_factor, dst_width, scale_factor)).sum(axis=(1, 3))
        
        # Create destination transform
        dst_transform = src.transform * src.transform.scale(
            (src.width / dst_width),
            (src.height / dst_height)
        )
        
        # Write the new raster
        with rio.open(
            output_file,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype=aggregated_data.dtype,
            crs=src.crs,
            transform=dst_transform,
        ) as dest:
            dest.write(aggregated_data, 1)




# Generate a grid of points
def generate_point_grid(bbox, resolution, crs) -> gpd.GeoDataFrame:
    """
    Generates a grid of points within a given bounding box and resolution.

    Args:
        bbox (tuple): A tuple of four floats representing the bounding box coordinates in the order (xmin, ymin, xmax, ymax).
        resolution (float): The resolution of the grid, in the same units as the bounding box coordinates.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the generated grid of points.

    """
    xmin, ymin, xmax, ymax = bbox
    # Pad the bounding box
    width = xmax - xmin
    height = ymax - ymin
    # Pad the bounding box to make it fit the resolution
    # This ensure that the grid is aligned with the raster
    xmin -= width % resolution
    ymin -= height % resolution
    xmax += width % resolution
    ymax += height % resolution

    # Create a grid of points
    x_coords = np.arange(xmin, xmax, resolution)
    y_coords = np.arange(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    xx = xx.flatten()
    yy = yy.flatten()

    # Create a geo dataframe
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx, yy))
    grid.crs = crs

    return grid

def load_boundary():
    # Load the boundary
    boundary = gpd.read_file("data/processed/boundary.geojson")
    return boundary

import xarray as xr
import rioxarray as rxr
def generate_model_raster():
    # Generate an empty xarray raster using a 7km buffer around the boundary as the bounding box
    # And a resolution of 100m. This will be used to tranform other rasters using reproject match
    # To make it easy to combine them in modelling

    # Get the boundary
    boundary = load_boundary()
    boundary_buffer = boundary.buffer(7000)
    boundary_buffer = boundary_buffer.unary_union
    bbox = boundary_buffer.bounds
    resolution = 100
    crs = boundary.crs
    xmin, ymin, xmax, ymax = bbox
    # Pad the bounding box
    width = xmax - xmin
    height = ymax - ymin
    # Pad the bounding box to make it fit the resolution
    # This ensure that the raster covers the boundary
    xmin -= width % resolution
    ymin -= height % resolution
    xmax += width % resolution
    ymax += height % resolution

    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)

    data = np.empty((height, width))
    coords = {'y': np.linspace(ymax-resolution/2, ymin+resolution/2, height),
              'x': np.linspace(xmin+resolution/2, xmax-resolution/2, width)}
    da = xr.DataArray(data, coords=coords, dims=('y', 'x'))

    da.rio.write_crs(crs, inplace=True)
    da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
    return da

def reproject_to_model_raster(raster):
    # Reprojects a raster to the model raster
    # This is useful for combining rasters in a model
    # Get the model raster
    model_raster = generate_model_raster()
    # Reproject the raster to the model raster
    raster_projected = raster.rio.reproject_match(model_raster)
    # Set the no data value
    if isinstance(raster, xr.DataArray):
        raster_projected = raster_projected.where(raster_projected != raster.rio.nodata, np.nan)
        raster_projected.rio.write_nodata(np.nan, inplace=True)
    else:
        for var in raster.data_vars:
            raster_projected[var] = raster_projected[var].where(raster_projected[var] != raster_projected[var].rio.nodata, np.nan)
            raster_projected[var].rio.write_nodata(np.nan, inplace=True)
    
    return raster_projected

def tile_bounding_box(xmin, ymin, xmax, ymax, tile_shape, resolution):
    width = int(tile_shape[0] * resolution)
    height = int(tile_shape[1] * resolution)

    # Calculate padding needed for x and y dimensions
    x_padding = width - ((xmax - xmin) % width)
    y_padding = height - ((ymax - ymin) % height)

    # Update xmax and ymax to include padding
    xmax += x_padding
    ymax += y_padding

    x_tiles = (xmax - xmin) // width
    y_tiles = (ymax - ymin) // height

    tiles = []
    for i in range(int(y_tiles)):
        for j in range(int(x_tiles)):
            tile_xmin = xmin + j * width
            tile_xmax = tile_xmin + width
            tile_ymin = ymin + i * height
            tile_ymax = tile_ymin + height
            tiles.append((tile_xmin, tile_ymin, tile_xmax, tile_ymax))

    return tiles
