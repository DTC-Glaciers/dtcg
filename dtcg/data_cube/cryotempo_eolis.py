"""Functionality for retrieving and resampling CryoTEMPO-EOLIS derived elevation change data."""
import logging
import os
import pyproj
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from affine import Affine
from pandas import Timedelta
from pyproj import Proj
from salem.gis import Grid
from specklia import Specklia

logger = logging.getLogger(__name__)

SPECKLIA_DATASET_NAME_EOLIS_ELEVATION_CHANGE = 'CryoTEMPO-EOLIS Processed Elevation Change Maps'
EOLIS_STATIC_KEYS = [
    'Conventions', 'DOI', 'Metadata_Conventions', 'baseline', 'cdm_data_type',
    'comment', 'contact', 'creator_email', 'creator_url', 'institution',
    'keywords', 'keywords_vocabulary', 'platform', 'processing_level',
    'project', 'references', 'region', 'source', 'summary', 'title', 'version'
]
EOLIS_PRODUCT_KEYS = [
    'science_pds_path', 'Earth_Explorer_Header', 'science_pds_download_link',
    'date_created', 'date_modified', 'geospatial_projection', 'geospatial_resolution',
    'geospatial_resolution_units', 'geospatial_x_max', 'geospatial_x_min',
    'geospatial_x_units', 'geospatial_y_max', 'geospatial_y_min',
    'geospatial_y_units', 'time_coverage_duration', 'time_coverage_end',
    'time_coverage_start', 'version'
]


def convert_gridded_dataframe_to_array(
        gridded_df: pd.DataFrame, value_column_names: list[str], x_coordinate_column: str, y_coordinate_column: str,
        spatial_resolution: float, xy_projection: Proj, y_affine_negative: bool = True,
        t_coordinate_column: str = None) -> tuple[dict[str, np.ndarray], Grid, np.ndarray | None]:
    """
    Resolve arrays from sparse gridded data stored in a dataframe.

    For each column name specified, an array (either 2d, or 3d if a time coordinate is supplied) is created using the
    x and y extent in the gridded dataframe and the provided resolution, and is populated with the sparse gridded data
    from the dataframe.   

    Parameters
    ----------
    gridded_df : pd.DataFrame
        DataFrame containing sparse gridded data, including coordinate columns and value columns.
    value_column_names : list[str]
        List of column names whose values are to be gridded.
    x_coordinate_column : str
        Name of the column containing x coordinates.
    y_coordinate_column : str
        Name of the column containing y coordinates.
    spatial_resolution : float
        Resolution to use when constructing the regular grid.
    xy_projection : Proj
        PyProj projection object representing the spatial reference system.
    y_affine_negative : bool, optional
        Whether to invert the y-axis when computing affine transformation, by default True.
    t_coordinate_column : str, optional
        Optional name of time coordinate column to create a 3D array, by default None.

    Returns
    -------
    tuple[dict[str, np.ndarray], Grid, np.ndarray or None]
        - Dictionary of gridded arrays keyed by column name
        - Salem Grid object describing the spatial extent
        - Optional array of sorted time coordinates (if time is used)
    """
    # Validate input
    for col in [x_coordinate_column, y_coordinate_column] + value_column_names:
        if col not in gridded_df.columns:
            raise ValueError(f"Missing required column '{col}' in gridded_df")

    unique_x_points = np.sort(gridded_df[x_coordinate_column].unique())
    unique_y_points = np.sort(gridded_df[y_coordinate_column].unique())

    # prepare coordinate axes
    desired_x_axis = np.arange(unique_x_points[0], unique_x_points[-1] + spatial_resolution, spatial_resolution)
    desired_y_axis = np.arange(unique_y_points[0], unique_y_points[-1] + spatial_resolution, spatial_resolution)        

    # we need to offset both these axes by one in order to use np.searchsorted() correctly below.
    y_idx = np.searchsorted(desired_y_axis, gridded_df[y_coordinate_column])
    x_idx = np.searchsorted(desired_x_axis, gridded_df[x_coordinate_column])

    # time axis handling
    if t_coordinate_column is not None:
        desired_t_axis = np.sort(gridded_df[t_coordinate_column].unique())
        t_idx = np.searchsorted(desired_t_axis, gridded_df[t_coordinate_column])
    else:
        desired_t_axis = None

    # Affine transformation
    if y_affine_negative:
        desired_y_axis = desired_y_axis[::-1]
        transform = Affine.translation(
            desired_x_axis.min() - spatial_resolution / 2,
            desired_y_axis.max() + spatial_resolution / 2) * Affine.scale(
                spatial_resolution, -spatial_resolution)
    else:
        transform = Affine.translation(
            desired_x_axis.min() - spatial_resolution / 2,
            desired_y_axis.min() - spatial_resolution / 2) * Affine.scale(
                spatial_resolution, spatial_resolution)

    # create the output raster, initially filling it with NaN
    output_arrays = {}
    for column_name in value_column_names:
        if desired_t_axis is not None:
            grid_arr = np.full((len(desired_t_axis), len(desired_y_axis), len(desired_x_axis)), np.nan)
            grid_arr[t_idx, y_idx, x_idx] = gridded_df[column_name]
        else:
            grid_arr = np.full((len(desired_y_axis), len(desired_x_axis)), np.nan)
            grid_arr[y_idx, x_idx] = gridded_df[column_name]
        if y_affine_negative:
            grid_arr = np.flip(grid_arr, axis=-2)
        output_arrays[column_name] = grid_arr
    
    grid = Grid(
        proj=xy_projection,
        nxny=(len(desired_x_axis), len(desired_y_axis)),
        dxdy=(transform.a, transform.e),
        x0y0=(transform.c, transform.f),
        pixel_ref='center')

    return output_arrays, grid, desired_t_axis


def prepare_eolis_metadata(specklia_source_data: list[dict], preliminary_dataset: bool = True) -> dict:
    """
    Extract metadata from Specklia dataset source information.

    Parameters
    ----------
    specklia_source_data : list[dict]
        List of source information dictionaries returned by Specklia.
    preliminary_dataset : bool, optional
        Whether to return all metadata from the first product without parsing, by default True.

    Returns
    -------
    dict
        Dictionary of static metadata and product-specific attributes, grouped by filename.
    """
    first_source = specklia_source_data[0].get('source_information', {})
    
    if preliminary_dataset:
        return first_source

    metadata = {k: first_source.get(k) for k in EOLIS_STATIC_KEYS if k in first_source}
    product_attributes = {}
    for item in specklia_source_data:
        source_info = item.get('source_information', {})
        science_path = source_info.get('science_pds_path', '')
        file_name = os.path.basename(science_path) if science_path else 'unknown_file'
        product_attributes[file_name] = {
            ('original_' + k if 'geospatial' in k else k): source_info[k]
            for k in EOLIS_PRODUCT_KEYS if k in source_info
        }
    metadata['product_attributes'] = product_attributes
    return metadata


def create_query_polygon(oggm_ds: xr.Dataset) -> shapely.Polygon:
    """
    Create a WGS84-aligned polygon bounding box from the spatial extent of an OGGM dataset.

    Parameters
    ----------
    oggm_ds : xr.Dataset
        OGGM dataset a geospatial CRS to extract extent from.

    Returns
    -------
    shapely.Polygon
        Polygon in EPSG:4326 (lat/lon) describing the outer bounds of the OGGM dataset.
    """
    # Create query polygon for Specklia, matching the extent of the oggm data cube and then reprojected to WGS84
    # should consider buffering this a bit to avoid edge features after resampling
    if not oggm_ds.rio.crs:
        oggm_ds.rio.write_crs(oggm_ds.pyproj_srs, inplace=True)

    # Reproject the dataset to EPSG:4326
    ds_ll = oggm_ds.rio.reproject("EPSG:4326")

    # build query polygon from bounds
    bounds = ds_ll.rio.bounds()

    return shapely.box(*bounds)


def retrieve_data_from_specklia(
        query_polygon: shapely.Polygon, specklia_data_set_name: str, specklia_api_key: str,
        min_timestamp: int = None, max_timestamp: int = None) -> tuple[gpd.GeoDataFrame, list[dict], pd.Series]:
    """
    Query and retrieve data from the Specklia API for a given spatial and temporal extent.

    Parameters
    ----------
    query_polygon : shapely.Polygon
        WGS84 polygon representing the spatial area to query.
    specklia_data_set_name : str
        Name of the dataset to query from Specklia.
    specklia_api_key : str
        API key used to authenticate with the Specklia service.
    min_timestamp : int, optional
        Optional minimum timestamp to filter results, by default None. If not set then the minimum timestamp
        of the dataset in specklia will be used.
    max_timestamp : int, optional
        Optional maximum timestamp to filter results, by default None. If not set then the maximum timestamp
        of the dataset in specklia will be used.

    Returns
    -------
    tuple[gpd.GeoDataFrame, list[dict], pd.Series]
        - GeoDataFrame of geospatial data queried from Specklia
        - List of metadata dictionaries for all data in the retrieved geodataframe
        - Dataset-level metadata for the Specklia dataset queried
    """
    client = Specklia(specklia_api_key)
    available_datasets = client.list_datasets()

    matching_dataset = available_datasets[available_datasets['dataset_name'] == specklia_data_set_name]
    if matching_dataset.empty:
        raise ValueError(f"No dataset named '{specklia_data_set_name}' found in Specklia.")
    specklia_dataset_info = matching_dataset.iloc[0]

    if min_timestamp is None:
        min_timestamp = specklia_dataset_info['min_timestamp'] - Timedelta(seconds=1)
    
    if max_timestamp is None:
        max_timestamp = specklia_dataset_info['max_timestamp'] + Timedelta(seconds=1)

    # query the data from Specklia
    query_start_time = perf_counter()

    gdf, source_info = client.query_dataset(
        dataset_id=specklia_dataset_info['dataset_id'],
        epsg4326_polygon=query_polygon,
        min_datetime=min_timestamp,
        max_datetime=max_timestamp)

    logger.debug(f'Specklia query took {perf_counter()-query_start_time:.2f} seconds')

    return gdf, source_info, specklia_dataset_info


def retrieve_prepare_eolis_gridded_data(oggm_ds: xr.Dataset, grid: Grid) -> xr.Dataset:
    """
    Retrieve EOLIS gridded elevation change data and resample it to the given OGGM grid.

    Parameters
    ----------
    oggm_ds : xr.Dataset
        OGGM xarray dataset to which EOLIS data will be added.
    grid : Grid
        Grid object representing the OGGM grid projection and extent.

    Returns
    -------
    xr.Dataset
        Modified OGGM dataset including resampled EOLIS elevation change and uncertainty arrays.
    """
    # build query polygon from OGGM data cube
    specklia_query_polygon = create_query_polygon(oggm_ds)

    # query EOLIS dataset from specklia over full time period
    eolis_gridded_data, eolis_gridded_sources, eolis_gridded_dataset = retrieve_data_from_specklia(
        specklia_query_polygon, SPECKLIA_DATASET_NAME_EOLIS_ELEVATION_CHANGE, os.getenv("SPECKLIA_API_KEY"))
    
    # convert spare dataframe to data cube
    eolis_arrays, eolis_grid, time_coordinates = convert_gridded_dataframe_to_array(
        eolis_gridded_data,
        ['elevation_change', 'standard_error'],
        'x',
        'y',
        np.nanmin(np.abs(np.diff(eolis_gridded_data.x.unique()))),
        eolis_gridded_sources[0]['source_information']['xy_cols_proj4'],
        y_affine_negative=True,
        t_coordinate_column='timestamp')

    resampling_start_time = perf_counter()

    # add EOLIS elevation and uncertainty to OGGM data cube
    eolis_metadata = prepare_eolis_metadata(eolis_gridded_sources)
    for col in ['elevation_change', 'standard_error']:
        data_name = f'eolis_gridded_{col}'
        eolis_resampled_grid = grid.map_gridded_data(eolis_arrays[col], eolis_grid, interp='linear')
        column_info = [d for d in eolis_gridded_dataset['columns'] if d['name'] == col][0]
        eolis_resampled_grids_xarr = xr.DataArray(
            eolis_resampled_grid,
            coords={'t': time_coordinates, 'y': oggm_ds.y, 'x': oggm_ds.x},
            dims=('t', 'y', 'x'),
            name=data_name,
            attrs={'units': column_info['unit'], 'long_name': f'EOLIS {col.replace("_", " ").title()}',
                   'description': column_info['description']} | eolis_metadata
        )

        oggm_ds[data_name] = eolis_resampled_grids_xarr

    logger.debug(f'Resampling of EOLIS and creation of dataset took {perf_counter()-resampling_start_time:.2f} seconds')

    return oggm_ds
