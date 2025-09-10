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

from datetime import date

import pytest
import xarray as xr

import dtcg.datacube.desp_era5 as desp_era5

pytest_plugins = "oggm.tests.conftest"  # for hef_gdir


def check_desp_api_key():
    """Check if user has access to DESP."""

    try:
        with xr.open_dataset(
            f"{desp_era5.DESP_SERVER}era5/reanalysis-era5-single-levels-monthly-means-v0.zarr",
            storage_options={"client_kwargs": {"trust_env": True}},
            chunks={},
            engine="zarr",
        ) as ds:
            return True
    except:
        return False


_has_desp_access = check_desp_api_key()


class Test_DatacubeDespERA5:

    basenames_patch = {
        "BASENAMES": {
            "ERA5_DESP": "public/test-dataset-v0.zarr",
            "ERA5_DESP_hourly": "public/test-dataset-v0.zarr",
        }
    }
    has_desp_access = _has_desp_access

    def test_get_desp_datastream(self, conftest_boilerplate, monkeypatch):
        test_cube = desp_era5.DatacubeDespEra5()
        conftest_boilerplate.patch_variable(
            monkeypatch, desp_era5, self.basenames_patch
        )

        for d in desp_era5.BASENAMES.keys():
            assert isinstance(test_cube.get_desp_datastream(d), xr.Dataset)

        with pytest.raises(ValueError):
            test_cube.get_desp_datastream("ERA5")
        with pytest.raises(ValueError):
            test_cube.get_desp_datastream("Wrong_Set")

    @pytest.mark.skipif(
        not has_desp_access,
        reason="No access to DESP. Check your .netrc file has a valid API key.",
    )
    @pytest.mark.parametrize("arg_dataset", ["ERA5_DESP"])
    @pytest.mark.parametrize(
        "arg_y0", [pytest.param(None, marks=pytest.mark.slow), 2023]
    )
    @pytest.mark.parametrize(
        "arg_y1", [pytest.param(None, marks=pytest.mark.slow), 2024]
    )
    def test_process_desp_era5_data(self, arg_dataset, arg_y0, arg_y1, hef_gdir):

        gdir = hef_gdir
        test_cube = desp_era5.DatacubeDespEra5()

        test_cube.process_desp_era5_data(
            gdir=gdir, dataset=arg_dataset, y0=arg_y0, y1=arg_y1
        )
        with xr.open_dataset(gdir.get_filepath("climate_historical")) as ds:
            ds = ds.load()

        if not arg_y0:
            assert ds.yr_0 == 1940
        else:
            assert ds.yr_0 == arg_y0
        if not arg_y1:
            # This may fail each new year's day
            assert ds.yr_1 == int(date.today().year)
        else:
            assert ds.yr_1 == arg_y1
