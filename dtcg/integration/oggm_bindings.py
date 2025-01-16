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

    region_db = {"rofental": ("61", "11")}

    try:
        rgi_ids = region_db[region_name.lower()]
    except KeyError as e:
        raise KeyError(f"{region_name} region not available.")
    except (TypeError, AttributeError) as e:
        raise TypeError(f"{region_name} is not a string.")

    return rgi_ids


def init_oggm(region_name: str) -> None:
    """Initialise OGGM run parameters."""
    cfg.initialize(logging_level="WORKFLOW")
    cfg.PARAMS["border"] = 80
    WORKING_DIR = utils.gettempdir(f"OGGM_{region_name}")
    utils.mkdir(WORKING_DIR, reset=True)
    cfg.PATHS["working_dir"] = WORKING_DIR


def get_rgi_file(region_name):
    """Get RGI shapefile from a region name.

    Parameters
    ----------
    region_name : str
        Name of region.

    Returns
    -------
    gpd.GeoDataFrame
        RGI shapefile.
    """

    rgi_version, rgi_region = get_rgi_id(region_name=region_name)
    path = utils.get_rgi_region_file(rgi_region, version=rgi_version)
    rgi_file = gpd.read_file(path)

    return rgi_file


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

    for key in region.keys():
        print(key)
    return region


def get_glacier_directories(glaciers: gpd.GeoDataFrame):
    """

    Returns
    -------
    gdirs : List of :py:class:`oggm.GlacierDirectory` objects for the
        initialised glacier directories.
    """

    gdirs = workflow.init_glacier_regions(glaciers, from_prepro_level=4)

    return gdirs


def get_glacier_details(gdirs) -> list:
    details = []
    for glacier in gdirs:
        details.append(f"Region: {glacier.region}", f"Subregion: {glacier.subregion}")
    return details


def plot_glacier_domain(gdirs):
    graphics.plot_domain(gdirs, figsize=(6, 5))


def get_user_subregion(request=None):
    """Get user-selected subregion.

    This should be called via gateway.
    """
    if request is not None:
        region_name = request.region_name
    else:  # placeholder until API format is defined.
        region_name = "Rofental"

    init_oggm(region_name=region_name)

    region_file = get_rgi_file(region_name=region_name)
    basin_file = get_rgi_basin_file(subregion_name=region_name)
    basin_glaciers = get_glaciers_in_subregion(region=region_file, subregion=basin_file)
    gdirs = get_glacier_directories(glaciers=basin_glaciers)
    details = get_glacier_details(gdirs=gdirs)
    for i in details:
        print(f"{i[0]} {i[1]}")

    return gdirs
