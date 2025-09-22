"""Copyright 2025 DTCG Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

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
            "elevation_change_sigma": [0.1, 0.2, 0.3, 0.4],
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
            "elevation_change_sigma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
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

    def get_datacube_cryotempo_eolis(self):
        return cryotempo_eolis_utils.DatacubeCryotempoEolis()

    @pytest.fixture(
        name="DatacubeCryotempoEolis", autouse=False, scope="function"
    )
    def fixture_datacube_cryotempo_eolis(self):
        return self.get_datacube_cryotempo_eolis()

    @patch("dtcg.datacube.cryotempo_eolis.Specklia")
    def test_retrieve_data_from_specklia(
        self, mock_specklia_class, DatacubeCryotempoEolis
    ):
        specklia_ds_name = "CryoTEMPO-EOLIS Processed Elevation Change Maps"
        mock_client = MagicMock()
        mock_client.list_datasets.return_value = pd.DataFrame(
            [
                {
                    "dataset_name": specklia_ds_name,
                    "dataset_id": "abc123",
                    "min_timestamp": pd.Timestamp(0),
                    "max_timestamp": pd.Timestamp(100),
                }
            ]
        )
        mock_client.query_dataset.return_value = (
            gpd.GeoDataFrame({"x": [0], "y": [0]},
                             geometry=gpd.points_from_xy([0], [0]),
                             crs=4326),
            [{"source_information": {"science_pds_path": "mock.nc"}}],
        )
        mock_specklia_class.return_value = mock_client

        gdf, meta, info = DatacubeCryotempoEolis.retrieve_data_from_specklia(
            query_polygon=box(0, 0, 1, 1),
            specklia_data_set_name=specklia_ds_name,
            specklia_api_key="dummy-key",
        )

        assert not gdf.empty
        assert isinstance(meta, list)
        assert "dataset_id" in info

    @pytest.mark.parametrize("is_3d", [False, True])
    def test_convert_gridded_dataframe_to_array(
        self, is_3d, dataframe_2d, dataframe_3d, DatacubeCryotempoEolis
    ):
        df = dataframe_3d if is_3d else dataframe_2d
        t_col = "timestamp" if is_3d else None

        output, grid, t_axis = (
            DatacubeCryotempoEolis.convert_gridded_dataframe_to_array(
                gridded_df=df,
                value_column_names=["elevation_change",
                                    "elevation_change_sigma"],
                x_coordinate_column="x",
                y_coordinate_column="y",
                spatial_resolution=1.0,
                xy_projection=self.XY_PROJ,
                t_coordinate_column=t_col,
            )
        )

        assert isinstance(output, dict)
        assert "elevation_change" in output
        assert "elevation_change_sigma" in output
        assert isinstance(grid, Grid)
        if is_3d:
            assert output["elevation_change"].ndim == 3
            assert t_axis is not None
        else:
            assert output["elevation_change"].ndim == 2
            assert t_axis is None

    @pytest.mark.parametrize("preliminary", [True, False])
    def test_prepare_eolis_metadata(self, preliminary, DatacubeCryotempoEolis):
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
        result = DatacubeCryotempoEolis.prepare_eolis_metadata(
            source, preliminary_dataset=preliminary
        )
        assert isinstance(result, dict)
        if not preliminary:
            assert "product_attributes" in result

    def test_create_query_polygon(self, oggm_dataset, DatacubeCryotempoEolis):
        polygon = DatacubeCryotempoEolis.create_query_polygon(oggm_dataset)
        assert polygon.bounds == (
            -17.634867503383262,
            64.59053260897588,
            -17.524957449239906,
            64.63902233874501,
        )

    @patch(
        "dtcg.datacube.cryotempo_eolis.DatacubeCryotempoEolis.retrieve_data_from_specklia"
    )
    def test_retrieve_prepare_eolis_gridded_data(
        self, mock_retrieve, oggm_dataset, DatacubeCryotempoEolis
    ):
        xs, ys = np.meshgrid(
            np.array(np.arange(566000, 570000, 500)),
            np.array(np.arange(458000, 464000, 500)),
        )
        n_coords = len(xs.flatten())
        np.random.seed(21)
        mock_retrieve.return_value = (
            gpd.GeoDataFrame(
                {
                    "x": xs.flatten(),
                    "y": ys.flatten(),
                    "timestamp": np.ones(n_coords),
                    "elevation_change": np.random.rand(n_coords),
                    "elevation_change_sigma": np.random.rand(n_coords),
                },
                geometry=gpd.points_from_xy(xs.flatten(), ys.flatten()),
                crs=self.XY_PROJ.crs
            ),
            [{"source_information":
              {"xy_cols_proj4": self.XY_PROJ,
               "elevation_change": {"long_name": "Elevation Change",
                                    "source": 'dummy'},
               "elevation_change_sigma": {"long_name": "Error",
                                          "source": 'dummy'}}}],
            {
                "columns": [
                    {
                        "name": "elevation_change",
                        "unit": "m",
                        "description": "Elevation change",
                    },
                    {
                        "name": "elevation_change_sigma",
                        "unit": "m",
                        "description": "Error"
                    },
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

        result = DatacubeCryotempoEolis.retrieve_prepare_eolis_gridded_data(
            oggm_dataset, grid
        )

        assert isinstance(result, xr.Dataset)

        expected_dims = {"x", "y", "t"}
        # Assert that all expected dimensions exist
        assert expected_dims.issubset(
            result.dims
        ), f"Missing dimensions: {expected_dims - set(result.dims)}"

        # check eolis data was added
        expected_vars = {
            "eolis_gridded_elevation_change": ("t", "y", "x"),
            "eolis_gridded_elevation_change_sigma": ("t", "y", "x"),
            "eolis_elevation_change_timeseries": ("t",),
            "eolis_elevation_change_sigma_timeseries": ("t",)
        }
        for var_name, var_dims in expected_vars.items():
            assert var_name in result
            assert var_dims == result[var_name].dims
        assert (
            np.count_nonzero(
                np.isfinite(result["eolis_gridded_elevation_change"])) == 198
        )
        np.testing.assert_almost_equal(
            np.nanmean(result["eolis_gridded_elevation_change"]), 0.49841077
        )
        assert result.eolis_gridded_elevation_change.attrs == {
            "units": "m", "long_name": "Elevation Change", "source": "dummy"}
        assert result.eolis_elevation_change_timeseries.attrs == {
            "ancillary_variables": "eolis_elevation_change_sigma_timeseries",
            "units": "m", "long_name": "Elevation Change",
            "source": "dummy Values represent glacier-wide mean elevation change,"
            " computed as the average of all valid grid cells within the glacier mask."}

    def test_gaussian_filter_fill(self, DatacubeCryotempoEolis):
        arr = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, np.nan, np.nan],
            [7.0, np.nan, 9.0]
        ])

        result = DatacubeCryotempoEolis.gaussian_filter_fill(
            arr, sigma=1)

        # Original finite values should be preserved
        np.testing.assert_array_equal(result[np.isfinite(arr)],
                                      arr[np.isfinite(arr)])
        assert np.all(np.isfinite(result[np.isnan(arr)]))

        # If input is all-NaN, output should stay all-NaN
        all_nan = np.full((3, 3), np.nan)
        all_nan_result = DatacubeCryotempoEolis.gaussian_filter_fill(
            all_nan, sigma=1)
        assert np.all(np.isnan(all_nan_result))

    def test_generate_1d_timeseries_basic(self, DatacubeCryotempoEolis):
        # Build fake gridded data with two timestamps
        df = pd.DataFrame({
            "x": [0, 1, 0, 1],
            "y": [0, 0, 1, 1],
            "timestamp": [1, 1, 2, 2],
            "elevation_change": [10.0, 12.0, 20.0, 22.0],
            "elevation_change_sigma": [1.0, 1.0, 2.0, 2.0],
        })
        # Turn into GeoDataFrame so we can pass to groupby
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

        means, errors = DatacubeCryotempoEolis.generate_1d_timeseries(
            gdf, "elevation_change", "elevation_change_sigma", length_scale=1e6
        )

        # There should be 2 timestamps
        assert len(means) == 2
        assert len(errors) == 2
        # Means should be the average per timestamp
        assert np.allclose(means, [11.0, 21.0], atol=1e-6)
        # Errors should be positive
        assert all(e > 0 for e in errors)

    def test_create_vector_glacier_mask(
            self, oggm_dataset, DatacubeCryotempoEolis):
        # Ensure dataset has CRS
        oggm_dataset.rio.write_crs(oggm_dataset.pyproj_srs, inplace=True)

        # Call function
        mask_gdf = DatacubeCryotempoEolis.create_vector_glacier_mask(
            oggm_dataset, target_crs="EPSG:4326"
        )

        # Result should be a GeoDataFrame with one geometry
        assert isinstance(mask_gdf, gpd.GeoDataFrame)
        assert not mask_gdf.empty
        assert mask_gdf.crs.to_string() == "EPSG:4326"
        assert mask_gdf.geometry.iloc[0].is_valid
