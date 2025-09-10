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
from oggm import cfg
from oggm.exceptions import InvalidParamsError

os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
logger = logging.getLogger(__name__)


ECMWF_SERVER = "https://data.earthdatahub.destine.eu/"

BASENAMES = {
    "ERA5": "era5/reanalysis-era5-single-levels-monthly-means-v0.zarr",
    "TEST": "public/test-dataset-v0.zarr",
}


def get_ecmwf_file(dataset="ERA5", var=None):
    """Returns a path to the desired ECMWF baseline climate file.

    If the file is not present, download it.

    Parameters
    ----------
    dataset : str
        'ERA5', 'ERA5L', 'CERA', 'ERA5L-HMA', 'ERA5dr'
    var : str
        'inv' for invariant
        'tmp' for temperature
        'pre' for precipitation

    Returns
    -------
    str
        path to the file
    """

    # Be sure input makes sense

    if dataset not in BASENAMES.keys():
        raise InvalidParamsError(
            "ECMWF dataset {} not " "in {}".format(dataset, BASENAMES.keys())
        )

    dataset_name = f"{ECMWF_SERVER}{BASENAMES[dataset]}"
    dataset = xr.open_dataset(
        dataset_name,
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )

    return dataset


def process_ecmwf_data(
    gdir,
    settings_filesuffix="",
    dataset="ERA5",
    y0=None,
    y1=None,
    output_filesuffix=None,
):
    """Processes and writes the ECMWF baseline climate data for this glacier.

    Extracts the nearest timeseries and writes everything to a NetCDF file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    dataset : str, default ERA5
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

    with get_ecmwf_file(dataset) as ds:
        # DESTINE recommends spatial selection first
        ds = ds.sel(longitude=longitude, latitude=latitude, method="nearest")
        yrs = ds["valid_time.year"]

        y0 = yrs[0] if y0 is None else y0
        y1 = yrs[-1] if y1 is None else y1
        ds = ds.sel(valid_time=slice(f"{y0}-01-01", f"{y1}-12-01"))
        # oggm._check_ds_validity(ds)

        temperature = ds.t2m.astype("float32") - 273.15
        precipitation = ds.tp.astype("float32") * 1000 * ds["valid_time.daysinmonth"]
        hgt = ds.z.astype("float32") / cfg.G
        temperature = temperature.compute().data
        precipitation = precipitation.compute().data
        hgt = hgt.compute().data
        time = ds.valid_time.compute().data

        ref_lon = ds.longitude.astype("float32").compute()
        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon
        ref_lat = ds.latitude.astype("float32").compute()
        temp_std = None

    # OK, ready to write
    gdir.write_climate_file(
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
