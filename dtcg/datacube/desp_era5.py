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

=====

Functionality for retrieving and resampling ERA5 data via DESP.
"""

from __future__ import annotations

import logging
import os

import pyproj
import xarray as xr
from aiohttp.client_exceptions import ClientResponseError
from oggm import cfg
from oggm.exceptions import InvalidParamsError

os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
DESP_EDH_KEY = os.environ.get("DESP_EDH_KEY")
logger = logging.getLogger(__name__)


if DESP_EDH_KEY:
    DESP_SERVER = f"https://edh:{DESP_EDH_KEY}data.earthdatahub.destine.eu/"
else:
    DESP_SERVER = "https://data.earthdatahub.destine.eu/"

BASENAMES = {
    "ERA5_DESP": "era5/reanalysis-era5-single-levels-monthly-means-v0.zarr",
    "ERA5_DESP_hourly": "era5/reanalysis-era5-single-levels-v0.zarr",
}


class DatacubeDespEra5:
    def get_desp_datastream(self, dataset: str = "ERA5_DESP") -> xr.Dataset:
        """Stream a DESP dataset.

        Parameters
        ----------
        dataset : str, default "ERA5_DESP"
            Name of dataset, either "ERA5_DESP" or "ERA5_DESP_hourly"

        Returns
        -------
        xr.Dataset
            Streamed DESP data.
        """

        if dataset not in BASENAMES.keys():
            raise InvalidParamsError(
                "DESP dataset {} not " "in {}".format(dataset, BASENAMES.keys())
            )

        dataset_name = f"{DESP_SERVER}{BASENAMES[dataset]}"

        try:
            dataset = xr.open_dataset(
                dataset_name,
                storage_options={"client_kwargs": {"trust_env": True}},
                chunks={},
                engine="zarr",
            )
        except ClientResponseError as e:
            if e.status == 401 or "401" in e.message:
                raise SystemExit(
                    "Access to DESP requires an API key. Check your .netrc file."
                )
            else:
                raise SystemExit(e.message)

        return dataset

    def process_desp_era5_data(
        self,
        gdir,
        settings_filesuffix: str = "",
        dataset: str = "ERA5_DESP",
        y0: int = None,
        y1: int = None,
        output_filesuffix: str = None,
    ):
        """Processes and writes the ERA5 baseline climate data for this glacier.

        Extracts the nearest timeseries and writes everything to a NetCDF file.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory to process
        settings_filesuffix: str
            You can use a different set of settings by providing a filesuffix. This
            is useful for sensitivity experiments. Code-wise the settings_filesuffix
            is set in the @entity-task decorater.
        dataset : str, default ERA5_DESP
            Dataset name.
        y0 : int
            the starting year of the timeseries to write. The default is to take
            the entire time period available in the file, but with this kwarg
            you can shorten it (to save space or to crop bad data).
        y1 : int
            the starting year of the timeseries to write. The default is to take
            the entire time period available in the file, but with this kwarg
            you can shorten it (to save space or to crop bad data).
        output_filesuffix : str
            add a suffix to the output file (useful to avoid overwriting
            previous experiments).
        """

        longitude = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
        latitude = gdir.cenlat

        with self.get_desp_datastream(dataset) as ds:
            # DESTINE recommends spatial selection first
            ds = ds.sel(longitude=longitude, latitude=latitude, method="nearest")
            yrs = ds["valid_time.year"]

            y0 = yrs[0].astype("int").values if y0 is None else y0
            y1 = yrs[-1].astype("int").values if y1 is None else y1
            ds = ds.sel(valid_time=slice(f"{y0}-01-01", f"{y1}-12-01"))
            # oggm.shop.ecmwf._check_ds_validity(ds)

            temperature = ds.t2m.astype("float32") - 273.15
            precipitation = (
                ds.tp.astype("float32") * 1000 * ds["valid_time.daysinmonth"]
            )

            temperature = temperature.compute().data
            precipitation = precipitation.compute().data

            time = ds.valid_time.compute().data

            ref_lon = ds.longitude.astype("float32").compute()
            ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon
            ref_lat = ds.latitude.astype("float32").compute()

        with self.get_desp_datastream("ERA5_DESP_hourly") as ds:
            # height is not included in monthly data
            ds = ds.sel(longitude=longitude, latitude=latitude, method="nearest")
            yrs = ds["valid_time.year"]

            # don't recalculate years in case of mismatch
            ds = ds.sel(valid_time=slice(f"{y0}-01-01", f"{y1}-12-01"))
            hgt = ds.z.astype("float32") / cfg.G
            hgt = hgt.compute().data

        temp_std = None
        # OK, ready to write
        gdir.write_monthly_climate_file(
            time,
            precipitation,
            temperature,
            hgt,
            ref_lon,
            ref_lat,
            filesuffix=output_filesuffix,
            temp_std=temp_std,
            source=dataset,
        )
