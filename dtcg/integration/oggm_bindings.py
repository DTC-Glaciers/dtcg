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

Bindings between DTCG and OGGM.

Executes an OGGM-API request.
"""

import csv
from pathlib import Path

import geopandas as gpd
import shapely.geometry as shpg
from oggm import DEFAULT_BASE_URL, cfg, graphics, tasks, utils, workflow

# DEFAULT_BASE_URL = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup"

DEFAULT_BASE_URL = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs"


def get_rgi_id(region_name: str) -> tuple:
    """Get RGI ID from a region/subregion name.

    This can be replaced with OGGM backend if available.

    Parameters
    ----------
    region_name : str
        name of region called by user.

    Returns
    -------
    RGI version and region ID

    Raises
    ------
    KeyError
        If the region name is not available.
    TypeError
        If region name is not a string.
    """

    # region_db = {"rofental": ("61", "11")}

    try:
        rgi_ids = get_matching_region_ids(region_name=region_name)
    except KeyError as e:
        raise KeyError(f"{region_name} region not found.")
    except (TypeError, AttributeError) as e:
        raise TypeError(f"{region_name} is not a string.")

    return rgi_ids


def get_matching_region_ids(region_name: str) -> set:

    if not region_name:
        raise KeyError(f"{region_name} region not found.")

    region_db = get_rgi_metadata()
    matching_region = set()
    for row in region_db:
        if region_name.lower() in row["Full_name"].lower():
            id = (row["O1"].zfill(2), row["O2"].zfill(2))
            matching_region.add(id)
    if not matching_region:
        raise KeyError(f"{region_name} region not found.")

    return matching_region


def init_oggm(region_name: str, **oggm_params) -> None:
    """Initialise OGGM run parameters."""
    cfg.initialize(logging_level="WORKFLOW")
    cfg.PARAMS["border"] = 80
    WORKING_DIR = utils.gettempdir(f"OGGM_{region_name}")
    utils.mkdir(WORKING_DIR, reset=True)
    cfg.PATHS["working_dir"] = WORKING_DIR
    valid_keys = cfg.PARAMS.keys()
    for key, value in oggm_params.items():
        if key in valid_keys:
            cfg.PARAMS[key] = value
    # if use_multiprocessing:
    #     cfg.PARAMS["use_multiprocessing"] = True


def get_rgi_metadata(path: str = "") -> list:
    """Get RGI metadata."""

    if not path:  # fallback to sample-data
        path = utils.get_demo_file("rgi_subregions_V6.csv")

    # Using csv is faster/lighter than pandas for <1K rows
    metadata = []
    with open(path, "r") as file:
        csv_data = csv.DictReader(file, delimiter=",")
        for row in csv_data:
            metadata.append(row)
    return metadata


def get_rgi_file(region_name: str) -> list:
    """Get RGI shapefile from a region name.

    Parameters
    ----------
    region_name : str
        Name of region.

    Returns
    -------
        List of RGI shapefiles.
    """

    rgi_ids = get_rgi_id(region_name=region_name)
    # rgi_version, rgi_region = rgi_ids
    rgi_files = []
    for ids in rgi_ids:
        path = utils.get_rgi_region_file(ids[0], version="61")
        # path = utils.get_rgi_region_file(rgi_region, version=rgi_version)
        rgi_files.append(gpd.read_file(path))

    return rgi_files


def get_rgi_from_shapefile(path: str) -> gpd.GeoDataFrame:
    """Get shapefile from user path."""
    shapefile = gpd.read_file(path)

    return shapefile


def get_rgi_basin_file(subregion_name: str) -> gpd.GeoDataFrame:
    """Placeholder for getting shapefiles from DTCG database."""
    path = utils.get_demo_file(f"{subregion_name.lower()}_hydrosheds.shp")
    basin_file = gpd.read_file(path)

    return basin_file


def get_glaciers_in_subregion(region, subregion):
    """Get only the glaciers in a given subregion.

    Parameters
    ----------
    region : gpd.GeoDataFrame
        Contains all glaciers in a region.
    subregion : gpd.GeoDataFrame
        Subregion shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        Only glaciers inside the given subregion.
    """
    subregion_mask = [
        subregion.geometry.contains(shpg.Point(x, y))[0]
        for (x, y) in zip(region.CenLon, region.CenLat)
    ]
    region = region.loc[subregion_mask]
    region = region.sort_values("Area", ascending=False)

    return region


def get_glacier_directories(glaciers: gpd.GeoDataFrame):
    """

    Returns
    -------
    gdirs : List of :py:class:`oggm.GlacierDirectory` objects for the
        initialised glacier directories.
    """

    gdirs = workflow.init_glacier_directories(
        glaciers, from_prepro_level=4, prepro_base_url=DEFAULT_BASE_URL
    )

    return gdirs


def plot_glacier_domain(gdirs):
    graphics.plot_domain(gdirs, figsize=(6, 5))


def get_user_subregion(region_name: str, shapefile_path: str = None, **oggm_params):
    """Get user-selected subregion.

    This should be called via gateway.
    """

    init_oggm(region_name=region_name, **oggm_params)

    region_file = get_rgi_file(region_name=region_name)[0]
    if shapefile_path:  # if user uploads/selects a shapefile
        user_shapefile = get_rgi_from_shapefile(subregion_name=region_name)
        region_file = get_glaciers_in_subregion(
            region=region_file, subregion=user_shapefile
        )

    return region_file
