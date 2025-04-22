import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray
import xarray as xr
from pyproj import Proj
from salem import Grid
from shapely.geometry import box

import dtcg.data_cube.cryotempo_eolis as cryotempo_eolis_utils

XY_PROJ = Proj(3413)


@pytest.fixture
def dataframe_2d():
    return pd.DataFrame({
        'x': [0, 1, 0, 1],
        'y': [0, 0, 1, 1],
        'elevation_change': [1, 2, 3, 4],
        'standard_error': [0.1, 0.2, 0.3, 0.4]
    })


@pytest.fixture
def dataframe_3d():
    return pd.DataFrame({
        'x': [0, 1, 0, 1, 0, 1],
        'y': [0, 0, 1, 1, 0, 1],
        'timestamp': [1, 1, 1, 1, 2, 2],
        'elevation_change': [1, 2, 3, 4, 5, 6],
        'standard_error': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    })


@pytest.fixture
def oggm_dataset(test_inputs_path):
    ds = rioxarray.open_rasterio(os.path.join(test_inputs_path, "oggm_shop.tif"))
    print(ds)
    return ds


@patch("dtcg.data_cube.cryotempo_eolis.Specklia")
def test_retrieve_data_from_specklia(mock_specklia_class):
    mock_client = MagicMock()
    mock_client.list_datasets.return_value = pd.DataFrame([{
        'dataset_name': 'CryoTEMPO-EOLIS Processed Elevation Change Maps',
        'dataset_id': 'abc123',
        'min_timestamp': pd.Timestamp(0),
        'max_timestamp': pd.Timestamp(100),
    }])
    mock_client.query_dataset.return_value = (
        gpd.GeoDataFrame({'x': [0], 'y': [0]}),
        [{'source_information': {'science_pds_path': 'mock.nc'}}]
    )
    mock_specklia_class.return_value = mock_client

    gdf, meta, info = cryotempo_eolis_utils.retrieve_data_from_specklia(
        query_polygon=box(0, 0, 1, 1),
        specklia_data_set_name='CryoTEMPO-EOLIS Processed Elevation Change Maps',
        specklia_api_key='dummy-key'
    )

    assert not gdf.empty
    assert isinstance(meta, list)
    assert 'dataset_id' in info


@pytest.mark.parametrize("is_3d", [False, True])
def test_convert_gridded_dataframe_to_array(is_3d, dataframe_2d, dataframe_3d):
    df = dataframe_3d if is_3d else dataframe_2d
    t_col = 'timestamp' if is_3d else None

    output, grid, t_axis = cryotempo_eolis_utils.convert_gridded_dataframe_to_array(
        gridded_df=df,
        value_column_names=['elevation_change', 'standard_error'],
        x_coordinate_column='x',
        y_coordinate_column='y',
        spatial_resolution=1.0,
        xy_projection=XY_PROJ,
        t_coordinate_column=t_col
    )

    assert isinstance(output, dict)
    assert 'elevation_change' in output
    assert 'standard_error' in output
    assert isinstance(grid, Grid)
    if is_3d:
        assert output['elevation_change'].ndim == 3
        assert t_axis is not None
    else:
        assert output['elevation_change'].ndim == 2
        assert t_axis is None


@pytest.mark.parametrize("preliminary", [True, False])
def test_prepare_eolis_metadata(preliminary):
    source = [{
        'source_information': {
            'science_pds_path': 'mock/file.nc',
            'version': '1.0',
            'cdm_data_type': 'Grid',
            'region': 'Global',
            'geospatial_x_max': 10,
            'geospatial_x_min': 0
        }
    }]
    result = cryotempo_eolis_utils.prepare_eolis_metadata(source, preliminary_dataset=preliminary)
    assert isinstance(result, dict)
    if not preliminary:
        assert 'product_attributes' in result


def test_create_query_polygon(oggm_dataset):
    polygon = cryotempo_eolis_utils.create_query_polygon(oggm_dataset)
    assert polygon.bounds == (-17.634867503383262,
                              64.59053260897588,
                              -17.524957449239906,
                              64.63902233874501)


@pytest.mark.skip
@patch("dtcg.data_cube.cryotempo_eolis.retrieve_data_from_specklia")
def test_retrieve_prepare_eolis_gridded_data(mock_retrieve, oggm_dataset):
    mock_retrieve.return_value = (
        pd.DataFrame({
            'x': [-23469.592, -23269.592, -23069.592, -22869.592],
            'y': [7168181., 7167981., 7167781., 7167581.],
            'timestamp': [1, 1, 2, 2],
            'elevation_change': [1.0, 2.0, 4.0, 8.0],
            'standard_error': [0.1, 0.2, 0.4, 0.5]
        }),
        [{'source_information': {'xy_cols_proj4': XY_PROJ}}],
        {'columns': [{'name': 'elevation_change', 'unit': 'm', 'description': 'Elevation change'},
                     {'name': 'standard_error', 'unit': 'm', 'description': 'Error'}]}
    )
    print(oggm_dataset.shape)
    grid = Grid(
        proj=Proj(oggm_dataset.pyproj_srs),
        nxny=(len(oggm_dataset.x), len(oggm_dataset.y)),
        dxdy=(oggm_dataset.x[1] - oggm_dataset.x[0],
              oggm_dataset.y[1] - oggm_dataset.y[0]),
        x0y0=(oggm_dataset.x[0], oggm_dataset.y[0]),
        pixel_ref='center')

    result = cryotempo_eolis_utils.retrieve_prepare_eolis_gridded_data(oggm_dataset, grid)
    assert isinstance(result, xr.Dataset)
    assert 'eolis_gridded_elevation_change' in result
    assert 'eolis_gridded_standard_error' in result
