from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Proj
from salem import Grid
from shapely.geometry import box

import dtcg.datacube.cryotempo_eolis as cryotempo_eolis_utils


@pytest.fixture
def dataframe_2d():
    return pd.DataFrame(
        {
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "elevation_change": [1, 2, 3, 4],
            "standard_error": [0.1, 0.2, 0.3, 0.4],
        }
    )


@pytest.fixture
def dataframe_3d():
    return pd.DataFrame(
        {
            "x": [0, 1, 0, 1, 0, 1],
            "y": [0, 0, 1, 1, 0, 1],
            "timestamp": [1, 1, 1, 1, 2, 2],
            "elevation_change": [1, 2, 3, 4, 5, 6],
            "standard_error": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


@pytest.fixture
def oggm_dataset(sample_data_path):
    oggm_path = Path(sample_data_path / "oggm_data.nc")
    ds = xr.open_dataset(oggm_path)
    yield ds


class TestDataCubeCryoTempoEolis:
    """Test processing of CryoTEMPO-EOLIS data.

    Attributes
    ----------
    XY_PROJ : pyproj.proj.Proj
        Map projection.
    """

    XY_PROJ = Proj(3057)

    @patch("dtcg.datacube.cryotempo_eolis.Specklia")
    def test_retrieve_data_from_specklia(self, mock_specklia_class):
        mock_client = MagicMock()
        mock_client.list_datasets.return_value = pd.DataFrame(
            [
                {
                    "dataset_name": "CryoTEMPO-EOLIS Processed Elevation Change Maps",
                    "dataset_id": "abc123",
                    "min_timestamp": pd.Timestamp(0),
                    "max_timestamp": pd.Timestamp(100),
                }
            ]
        )
        mock_client.query_dataset.return_value = (
            gpd.GeoDataFrame({"x": [0], "y": [0]}),
            [{"source_information": {"science_pds_path": "mock.nc"}}],
        )
        mock_specklia_class.return_value = mock_client

        gdf, meta, info = cryotempo_eolis_utils.retrieve_data_from_specklia(
            query_polygon=box(0, 0, 1, 1),
            specklia_data_set_name="CryoTEMPO-EOLIS Processed Elevation Change Maps",
            specklia_api_key="dummy-key",
        )

        assert not gdf.empty
        assert isinstance(meta, list)
        assert "dataset_id" in info

    @pytest.mark.parametrize("is_3d", [False, True])
    def test_convert_gridded_dataframe_to_array(
        self, is_3d, dataframe_2d, dataframe_3d
    ):
        df = dataframe_3d if is_3d else dataframe_2d
        t_col = "timestamp" if is_3d else None

        output, grid, t_axis = cryotempo_eolis_utils.convert_gridded_dataframe_to_array(
            gridded_df=df,
            value_column_names=["elevation_change", "standard_error"],
            x_coordinate_column="x",
            y_coordinate_column="y",
            spatial_resolution=1.0,
            xy_projection=self.XY_PROJ,
            t_coordinate_column=t_col,
        )

        assert isinstance(output, dict)
        assert "elevation_change" in output
        assert "standard_error" in output
        assert isinstance(grid, Grid)
        if is_3d:
            assert output["elevation_change"].ndim == 3
            assert t_axis is not None
        else:
            assert output["elevation_change"].ndim == 2
            assert t_axis is None

    @pytest.mark.parametrize("preliminary", [True, False])
    def test_prepare_eolis_metadata(self, preliminary):
        source = [
            {
                "source_information": {
                    "science_pds_path": "mock/file.nc",
                    "version": "1.0",
                    "cdm_data_type": "Grid",
                    "region": "Global",
                    "geospatial_x_max": 10,
                    "geospatial_x_min": 0,
                }
            }
        ]
        result = cryotempo_eolis_utils.prepare_eolis_metadata(
            source, preliminary_dataset=preliminary
        )
        assert isinstance(result, dict)
        if not preliminary:
            assert "product_attributes" in result

    def test_create_query_polygon(self, oggm_dataset):
        polygon = cryotempo_eolis_utils.create_query_polygon(oggm_dataset)
        assert polygon.bounds == (
            -17.634867503383262,
            64.59053260897588,
            -17.524957449239906,
            64.63902233874501,
        )

    @patch("dtcg.datacube.cryotempo_eolis.retrieve_data_from_specklia")
    def test_retrieve_prepare_eolis_gridded_data(self, mock_retrieve, oggm_dataset):
        xs, ys = np.meshgrid(
            np.array(np.arange(566000, 614000, 2000)),
            np.array(np.arange(388000, 460000, 2000)),
        )
        n_coords = len(xs.flatten())
        np.random.seed(21)
        mock_retrieve.return_value = (
            pd.DataFrame(
                {
                    "x": xs.flatten(),
                    "y": ys.flatten(),
                    "timestamp": np.ones(n_coords),
                    "elevation_change": np.random.rand(n_coords),
                    "standard_error": np.random.rand(n_coords),
                }
            ),
            [{"source_information": {"xy_cols_proj4": self.XY_PROJ}}],
            {
                "columns": [
                    {
                        "name": "elevation_change",
                        "unit": "m",
                        "description": "Elevation change",
                    },
                    {"name": "standard_error", "unit": "m", "description": "Error"},
                ]
            },
        )

        oggm_dataset.rio.write_crs(oggm_dataset.pyproj_srs, inplace=True)

        grid = Grid(
            proj=Proj(oggm_dataset.pyproj_srs),
            nxny=(len(oggm_dataset.x), len(oggm_dataset.y)),
            dxdy=(
                oggm_dataset.x[1] - oggm_dataset.x[0],
                oggm_dataset.y[1] - oggm_dataset.y[0],
            ),
            x0y0=(oggm_dataset.x[0], oggm_dataset.y[0]),
            pixel_ref="center",
        )

        result = cryotempo_eolis_utils.retrieve_prepare_eolis_gridded_data(
            oggm_dataset, grid
        )

        assert isinstance(result, xr.Dataset)

        expected_dims = {"x", "y", "t"}
        # Assert that all expected dimensions exist
        assert expected_dims.issubset(
            result.dims
        ), f"Missing dimensions: {expected_dims - set(result.dims)}"

        # check eolis data was added
        assert "eolis_gridded_elevation_change" in result
        assert "eolis_gridded_standard_error" in result
        assert (
            np.count_nonzero(np.isfinite(result["eolis_gridded_elevation_change"]))
            == 480
        )
        np.testing.assert_almost_equal(
            np.nanmean(result["eolis_gridded_elevation_change"]), 0.3771391
        )
