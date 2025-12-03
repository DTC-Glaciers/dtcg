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

Requires an Earth Data Hub personal access token in a .netrc file:

```
machine data.earthdatahub.destine.eu
  password <your personal access token>
```

For more information on authentication please read the `EDH
documentation`_.
.. _EDH documentation: https://earthdatahub.destine.eu/getting-started
"""

from __future__ import annotations

import logging

import xarray as xr
from aiohttp.client_exceptions import ClientResponseError
from oggm import cfg
from oggm.exceptions import InvalidParamsError

logger = logging.getLogger(__name__)

DESP_SERVER = "https://data.earthdatahub.destine.eu/"

BASENAMES = {
    "ERA5_DESP": "era5/reanalysis-era5-single-levels-monthly-means-v0.zarr",
    "ERA5_DESP_hourly": "era5/reanalysis-era5-single-levels-v0.zarr",
}


def get_desp_datastream(dataset: str = "ERA5_DESP") -> xr.Dataset:
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
    gdir,
    settings_filesuffix: str = "",
    frequency: str = "monthly",
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
    settings_filesuffix: str, optional
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    frequency : str, default "monthly"
        'monthly' (default) to use monthly DESP dataset, or "daily" to
        use the hourly DESP dataset resampled to daily aggregates.
    y0 : int, default None
        The starting year of the desired timeseries. The default is to
        take the entire time period available in the file, but with
        this argument you can shorten it to save space or to crop bad
        data.
    y1 : int, default None
        The end year of the desired timeseries. The default is to
        take the entire time period available in the file, but with
        this argument you can shorten it to save space or to crop bad
        data.
    output_filesuffix : str
        Add a suffix to the output file (useful to avoid overwriting
        previous experiments).
    """

    longitude = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    latitude = gdir.cenlat

    if frequency == "monthly":
        with get_desp_datastream("ERA5_DESP") as ds:
            # DESTINE recommends spatial selection first
            ds = ds.sel(longitude=longitude, latitude=latitude, method="nearest")
            years = ds["valid_time.year"]

            y0 = years[0].astype("int").values if y0 is None else y0
            y1 = years[-1].astype("int").values if y1 is None else y1
            ds = ds.sel(valid_time=slice(f"{y0}-01-01", f"{y1}-12-01"))
            # oggm.shop.ecmwf._check_ds_validity(ds)

            temperature = ds.t2m.astype("float32") - 273.15
            precipitation = (
                ds.tp.astype("float32") * 1000 * ds["valid_time.daysinmonth"]
            )
            time = ds.valid_time.compute().data

            ref_lon = ds.longitude.astype("float32").compute()
            ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon
            ref_lat = ds.latitude.astype("float32").compute()

        with get_desp_datastream("ERA5_DESP_hourly") as ds:
            # height is not included in monthly data
            ds = ds.sel(longitude=longitude, latitude=latitude, method="nearest")
            years = ds["valid_time.year"]

            # don't recalculate years in case of mismatch
            ds = ds.sel(valid_time=slice(f"{y0}-01-01", f"{y1}-12-01"))
            height = ds.z.isel(valid_time=0),astype("float32") / cfg.G

    elif frequency == "daily":
        # use the hourly dataset, resample to daily
        with get_desp_datastream("ERA5_DESP_hourly") as ds_hr:
            ds_hr = ds_hr.sel(longitude=longitude, latitude=latitude, method="nearest")
            years = ds_hr["valid_time.year"]

            y0 = years[0].astype("int").values if y0 is None else y0
            y1 = years[-1].astype("int").values if y1 is None else y1
            ds_hr = ds_hr.sel(valid_time=slice(f"{y0}-01-01", f"{y1}-12-31"))

            # hourly fields
            temp_hour = ds_hr.t2m.astype("float32") - 273.15
            # assume meters per time-step
            # convert precipitation to mm (meters -> mm)
            tp_hour = ds_hr.tp.astype("float32") * 1000

            # resample to daily: temperature mean, precipitation sum
            temperature = temp_hour.resample(valid_time="1D").mean()
            precipitation = tp_hour.resample(valid_time="1D").sum()
            time = precipitation.valid_time.compute().data
            ref_lon = ds_hr.longitude.astype("float32").compute()
            ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon
            ref_lat = ds_hr.latitude.astype("float32").compute()

            height = ds_hr.z.isel(valid_time=0).astype("float32") / cfg.G

    else:
        raise InvalidParamsError("frequency must be 'monthly' or 'daily'")

    temperature = temperature.compute().data
    precipitation = precipitation.compute().data
    
    height = height.compute().data

    # OK, ready to write
    gdir.write_climate_file(
        time,
        precipitation,
        temperature,
        height,
        ref_lon,
        ref_lat,
        filesuffix=output_filesuffix,
        temp_std=None,
        source=f"DESP_{frequency}",
        daily=frequency == "daily",
    )
