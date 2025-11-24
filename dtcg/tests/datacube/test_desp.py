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

import dtcg.datacube.desp as desp

pytest_plugins = "oggm.tests.conftest"  # for hef_gdir


def check_desp_api_key():
    """Check if user has access to DESP."""

    try:
        with xr.open_dataset(
            f"{desp.DESP_SERVER}era5/reanalysis-era5-single-levels-monthly-means-v0.zarr",
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

    @pytest.mark.skipif(
        not has_desp_access,
        reason="No access to DESP. Check your .netrc file has a valid API key.",
    )
    def test_get_desp_datastream(self, conftest_boilerplate, monkeypatch):
        conftest_boilerplate.patch_variable(monkeypatch, desp, self.basenames_patch)

        for d in desp.BASENAMES.keys():
            assert isinstance(desp.get_desp_datastream(d), xr.Dataset)

        with pytest.raises(ValueError):
            desp.get_desp_datastream("ERA5")
        with pytest.raises(ValueError):
            desp.get_desp_datastream("Wrong_Set")

    @pytest.mark.skipif(
        not has_desp_access,
        reason="No access to DESP. Check your .netrc file has a valid API key.",
    )
    @pytest.mark.parametrize("arg_frequency", ["monthly", "daily"])
    def test_process_desp_era5_data(self, arg_frequency, hef_gdir):

        gdir = hef_gdir

        arg_y0, arg_y1 = (2023, 2024)
        desp.process_desp_era5_data(
            gdir=gdir, frequency=arg_frequency, y0=arg_y0, y1=arg_y1
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

    @pytest.mark.skipif(
        not has_desp_access,
        reason="No access to DESP. Check your .netrc file has a valid API key.",
    )
    @pytest.mark.parametrize(
        "arg_years", [(None, 1941), (2023, None), (2023, 2024), (None, None)]
    )
    def test_process_desp_era5_data_years(self, arg_years, hef_gdir):
        gdir = hef_gdir

        arg_y0, arg_y1 = arg_years
        desp.process_desp_era5_data(
            gdir=gdir, frequency="monthly", y0=arg_y0, y1=arg_y1
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
