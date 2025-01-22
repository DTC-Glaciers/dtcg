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

import geopandas as gpd
import shapely.geometry as shpg
from oggm import DEFAULT_BASE_URL, cfg, graphics, utils, workflow

# TODO: Link to DTCG instead.
DEFAULT_BASE_URL = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs"


def get_rgi_region_codes(region_name: str) -> tuple:
    """Get RGI region and subregion codes from a subregion name.

    This can be replaced with OGGM backend if available.

    Parameters
    ----------
    region_name : str
        Name of subregion called by user.

    Returns
    -------
    tuple[str]
        RGI region (O1) and subregion (O2) codes.

    Raises
    ------
    KeyError
        If the region name is not available.
    TypeError
        If region name is not a string.
    """

    try:
        rgi_codes = get_matching_region_codes(region_name=region_name)
    except KeyError as e:
        raise KeyError(f"{region_name} region not found.")
    except (TypeError, AttributeError) as e:
        raise TypeError(f"{region_name} is not a string.")

    return rgi_codes


def get_matching_region_codes(region_name: str) -> set:
    """Get region and subregion codes matching a given name.

    TODO: Fuzzy-finding

    Parameters
    ----------
    region_name : str
        The full name of a region.

    Returns
    -------
    set[str]
        All pairs of region and subregion codes matching the given
        region's name.

    Raises
    ------
    KeyError
        If the region name is not found.
    """

    if not region_name:
        raise KeyError(f"{region_name} region not found.")

    region_db = get_rgi_metadata()
    matching_region = set()
    for row in region_db:
        if region_name.lower() in row["Full_name"].lower():
            region_codes = (row["O1"].zfill(2), row["O2"].zfill(2))
            matching_region.add(region_codes)
    if not matching_region:
        raise KeyError(f"{region_name} region not found.")

    return matching_region


def set_oggm_kwargs(oggm_params: dict) -> None:
    """Set OGGM configuration parameters.

    .. note:: This may eventually be moved to ``api.internal._parse_oggm``.

    Parameters
    ----------
    oggm_params : dict
        Key/value pairs corresponding to valid OGGM configuration
        parameters.
    """

    valid_keys = cfg.PARAMS.keys()
    for key, value in oggm_params.items():
        if key in valid_keys:
            cfg.PARAMS[key] = value


def init_oggm(dirname: str, **oggm_params) -> None:
    """Initialise OGGM run parameters.

    TODO: Add kwargs for cfg.PATH.

    Parameters
    ----------
    dirname : str
        Name of temporary directory

    """

    cfg.initialize(logging_level="WORKFLOW")
    cfg.PARAMS["border"] = 80
    WORKING_DIR = utils.gettempdir(dirname)  # already handles empty strings
    utils.mkdir(WORKING_DIR, reset=True)  # TODO: this should be an API parameter
    cfg.PATHS["working_dir"] = WORKING_DIR

    set_oggm_kwargs(oggm_params=oggm_params)


def get_rgi_metadata(path: str = "") -> list:
    """Get RGI metadata.

    Parameters
    ----------
    path : str
        Path to database with subregion names and O1/O2 codes.
    """

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
        Name of region or subregion.

    Returns
    -------
        List of RGI shapefiles.
    """

    rgi_region_codes = get_rgi_region_codes(region_name=region_name)
    rgi_files = []
    for codes in rgi_region_codes:
        path = utils.get_rgi_region_file(codes[0])
        rgi_files.append(gpd.read_file(path))

    return rgi_files


def get_shapefile(path: str) -> gpd.GeoDataFrame:
    """Get shapefile from user path.

    Parameters
    ----------
    path : str
        Path to shapefile.
    """

    shapefile = gpd.read_file(path)
    return shapefile


def get_shapefile_from_web(shapefile_name: str) -> gpd.GeoDataFrame:
    """Placeholder for getting shapefiles from an online repository.

    Parameters
    ----------
    shapefile_name : str
        Name of file in ``oggm-sample-data`` repository.
    """

    path = utils.get_demo_file(shapefile_name)
    shapefile = gpd.read_file(path)

    return shapefile


def get_glaciers_in_subregion(region, subregion) -> gpd.GeoDataFrame:
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


def get_glacier_directories(glaciers: list):
    """

    Returns
    -------
    gdirs : list
        List of :py:class:`oggm.GlacierDirectory` objects for the
        initialised glacier directories.
    """

    gdirs = workflow.init_glacier_directories(
        glaciers,
        from_prepro_level=4,
        prepro_border=80,
        prepro_base_url=DEFAULT_BASE_URL,
    )

    return gdirs


def plot_glacier_domain(gdirs):
    graphics.plot_domain(gdirs, figsize=(6, 5))


def get_user_subregion(region_name: str, shapefile_path: str = None, **oggm_params):
    """Get user-selected subregion.

    This should be called via gateway.
    """

    init_oggm(dirname=f"OGGM_{region_name}", **oggm_params)

    region_file = get_rgi_file(region_name=region_name)[0]
    if shapefile_path:  # if user uploads/selects a shapefile
        user_shapefile = get_shapefile(path=shapefile_path)
        region_file = get_glaciers_in_subregion(
            region=region_file, subregion=user_shapefile
        )

    return region_file
