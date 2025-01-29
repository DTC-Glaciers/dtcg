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
import pandas as pd
import shapely.geometry as shpg
import xarray as xr
from oggm import DEFAULT_BASE_URL, cfg, graphics, tasks, utils, workflow

# TODO: Link to DTCG instead.
DEFAULT_BASE_URL = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs"
SHAPEFILE_PATH = (
    "https://cluster.klima.uni-bremen.de/~dtcg/test_files/case_study_regions/austria"
)


def get_rgi_region_codes(region_name: str = "", subregion_name: str = "") -> tuple:
    """Get RGI region and subregion codes from a subregion name.

    This can be replaced with OGGM backend if available.

    Parameters
    ----------
    subregion_name : str
        Name of subregion called by user.

    Returns
    -------
    tuple[str]
        RGI region (O1) and subregion (O2) codes.

    Raises
    ------
    KeyError
        If the subregion name is not available.
    TypeError
        If subregion name is not a string.
    """

    try:
        rgi_codes = get_matching_region_codes(
            region_name=region_name, subregion_name=subregion_name
        )
    except KeyError as e:
        raise KeyError(
            f"No regions or subregion matching {', '.join(region_name, subregion_name)}."
        )
    except (TypeError, AttributeError) as e:
        raise TypeError(f"{subregion_name} is not a string.")

    return rgi_codes


def get_matching_region_codes(region_name: str = "", subregion_name: str = "") -> set:
    """Get region and subregion codes matching a given subregion name.

    Subregion names take priority over region names.

    TODO: Fuzzy-finding

    Parameters
    ----------
    region_name : str
        The full name of a region.
    subregion_name : str
        The full name of a subregion.

    Returns
    -------
    set[str]
        All pairs of region and subregion codes matching the given
        subregion or region's name.

    Raises
    ------
    KeyError
        If the region or subregion name is not found.
    ValueError
        If no valid region or subregion name is supplied.
    """

    matching_region = set()
    if subregion_name:
        region_db = get_rgi_metadata("rgi_subregions_V6.csv", from_web=True)
        for row in region_db:
            if subregion_name.lower() in row["Full_name"].lower():
                region_codes = (row["O1"], row["O2"])
                # zfill should be applied only when using utils.parse_rgi_meta
                # region_codes = (row["O1"].zfill(2), row["O2"].zfill(2))
                matching_region.add(region_codes)
    elif region_name:
        region_db = get_rgi_metadata("rgi_regions.csv", from_web=True)
        for row in region_db:
            if region_name.lower() in row["Full_name"].lower():
                region_codes = (row["RGI_Code"].zfill(2),)
                # zfill should be applied only when using utils.parse_rgi_meta
                # region_codes = (row["O1"].zfill(2), row["O2"].zfill(2))
                matching_region.add(region_codes)
    else:
        raise ValueError("No valid region or subregion name supplied.")

    if not matching_region:
        if region_name and subregion_name:
            msg = f"{region_name}, {subregion_name}"
        else:
            msg = f"{region_name}{subregion_name}"  # since one is empty
        raise KeyError(f"No region found for {msg}.")

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

    cfg.initialize(logging_level="CRITICAL")
    cfg.PARAMS["border"] = 80
    WORKING_DIR = utils.gettempdir(dirname)  # already handles empty strings
    utils.mkdir(WORKING_DIR, reset=True)  # TODO: this should be an API parameter
    cfg.PATHS["working_dir"] = WORKING_DIR

    set_oggm_kwargs(oggm_params=oggm_params)


def get_rgi_metadata(
    path: str = "rgi_subregions_V6.csv", from_web: bool = False
) -> list:
    """Get RGI metadata.

    TODO: Replace with call to ``oggm.utils.parse_rgi_meta``.

    Parameters
    ----------
    path : str
        Path to database with subregion names and O1/O2 codes.
    from_web : bool
        Use data from oggm-sample-data.
    """

    if from_web:  # fallback to sample-data
        path = utils.get_demo_file(path)

    # Using csv is faster/lighter than pandas for <1K rows
    metadata = []
    with open(path, "r") as file:
        csv_data = csv.DictReader(file, delimiter=",")
        for row in csv_data:
            metadata.append(row)
    return metadata


def get_rgi_files_from_subregion(
    region_name: str = "", subregion_name: str = ""
) -> list:
    """Get RGI shapefile from a subregion name.

    Parameters
    ----------
    subregion_name : str
        Name of subregion.

    Returns
    -------
        List of RGI shapefiles.

    Raises
    ------
    KeyError
        If no glaciers are found for the given subregion.
    """

    rgi_region_codes = get_rgi_region_codes(
        region_name=region_name, subregion_name=subregion_name
    )
    rgi_files = []
    for codes in rgi_region_codes:
        path = utils.get_rgi_region_file(region=codes[0])
        candidate = gpd.read_file(path)
        if subregion_name:
            rgi_files.append(candidate[candidate["O2Region"] == codes[1]])
        else:
            rgi_files.append(candidate)
    try:
        rgi_files = rgi_files[0]
    except KeyError as e:
        raise KeyError(f"No glaciers found for {subregion_name}.")
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


def set_subregion_constraints(region, subregion, subregion_name: str = ""):
    if region.crs != subregion.crs:
        subregion = subregion.to_crs(region.crs)

    if subregion_name:
        subregion = subregion[subregion["id"] == subregion_name]
    return subregion


def get_glaciers_in_subregion(
    region, subregion, subregion_name: str = ""
) -> gpd.GeoDataFrame:
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

    # frame indices are fixed, so index by location instead
    subregion_mask = [
        subregion.geometry.contains(shpg.Point(x, y)).iloc[0]
        for (x, y) in zip(region.CenLon, region.CenLat)
    ]

    region = region.loc[subregion_mask]
    region = region.sort_values("Area", ascending=False)

    return region


def get_polygon_difference(frame: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Get the difference of overlapping polygons.

    .. note:: GPD's overlay() only acts on GeoDataFrames, but this
    function manipulates GeoSeries.

    Parameters
    ----------
    frame : gpd.GeoDataFrame
        Contains overlapping polygons.

    Returns
    -------
    gpd.GeoSeries
        Polygon geometries adjusted so no areas overlap.
    """

    diffs = frame.iloc[1:].geometry.difference(frame.iloc[:-1].geometry, align=False)
    # use GeoSeries instead of pd.Series for consistent typing & CRS
    new_geometry = pd.concat(
        [gpd.GeoSeries(frame.geometry.iloc[0], crs=diffs.crs), diffs]
    )

    return new_geometry


def set_polygon_overlap(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Set overlapping boundaries.

    If two polygons overlap, the overlapping area is removed from the
    second polygon.
    Parameters
    ----------
    frame : gpd.GeoDataFrame
        Contains overlapping polygons.

    Returns
    -------
    gpd.GeoDataFrame
        Dataframe with the polygon geometries adjusted so no areas
        overlap.
    """

    merge = frame.copy()
    new_geometry = get_polygon_difference(frame=frame)
    merge.geometry = new_geometry

    return merge


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


def get_gdir_by_name(data: list, name: str):
    rgi_ids = get_glacier_by_name(data=data, name=name)
    gdirs = workflow.init_glacier_directories(
        rgi_ids,
        from_prepro_level=5,
        prepro_border=160,
        prepro_base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup",
    )
    return gdirs[0]


def get_glacier_by_name(data, name: str) -> gpd.GeoDataFrame:
    glacier = data[data["Name"] == name]
    if glacier.empty:  # fallback in case of RGI ID
        glacier = get_glacier_by_rgi_id(data=data, rgi_id=name)
    return glacier


def get_glacier_by_rgi_id(data, rgi_id: str) -> gpd.GeoDataFrame:
    glacier = data[data["RGIId"] == rgi_id]
    if glacier.empty:
        raise KeyError(f"{rgi_id} not found.")
    return glacier


def get_named_glaciers(glaciers) -> gpd.GeoDataFrame:
    return glaciers.dropna(subset="Name")


def get_hydro_climatology(gdir, nyears: int = 20) -> xr.Dataset:
    tasks.run_with_hydro(
        gdir,
        run_task=tasks.run_constant_climate,
        nyears=nyears,
        y0=2014,
        halfsize=5,
        init_model_filesuffix="_spinup_historical",
        store_monthly_hydro=True,
        output_filesuffix="_ct",
    )
    with xr.open_dataset(
        gdir.get_filepath("model_diagnostics", filesuffix="_ct")
    ) as ds:
        ds = ds.isel(time=slice(0, -1)).load()
    return ds


def get_annual_runoff(ds: xr.Dataset):
    sel_vars = [v for v in ds.variables if "month_2d" not in ds[v].dims]
    df_annual = ds[sel_vars].to_dataframe()
    runoff_vars = [
        "melt_off_glacier",
        "melt_on_glacier",
        "liq_prcp_off_glacier",
        "liq_prcp_on_glacier",
    ]
    df_runoff = df_annual[runoff_vars] * 1e-9
    return df_runoff


def get_min_max_runoff_years(annual_runoff: pd.Series) -> tuple:
    runoff = annual_runoff.sum(axis=1)
    runoff_year_min = int(runoff.idxmin())
    runoff_year_max = int(runoff.idxmax())
    return runoff_year_min, runoff_year_max


def get_monthly_runoff(ds: xr.Dataset):
    monthly_runoff = (
        ds["melt_off_glacier_monthly"]
        + ds["melt_on_glacier_monthly"]
        + ds["liq_prcp_off_glacier_monthly"]
        + ds["liq_prcp_on_glacier_monthly"]
    )
    monthly_runoff *= 1e-9
    return monthly_runoff


def get_climatology(data, name: str = ""):
    glacier = get_gdir_by_name(data=data, name=name)
    climatology = get_hydro_climatology(gdir=glacier)
    return climatology

def get_aggregate_runoff(data:gpd.GeoDataFrame)->dict:
    annual_runoff = []
    monthly_runoff = []
    for rgi_id in data["RGIId"]:
        climatology = get_climatology(
            data=data, name = rgi_id
        )
        annual_runoff.append(get_annual_runoff(ds=climatology))
        monthly_runoff.append(get_monthly_runoff(ds=climatology))
    total_annual_runoff = sum(annual_runoff)
    total_monthly_runoff = sum(monthly_runoff)
    min_year, max_year =get_min_max_runoff_years(annual_runoff=total_annual_runoff)
    runoff_data = {
        "annual_runoff": total_annual_runoff,
        "monthly_runoff": total_monthly_runoff,
        "runoff_year_min": min_year,
        "runoff_year_max": max_year,
    }
    return runoff_data

def get_runoff(data: xr.Dataset) -> dict:
    annual_runoff = get_annual_runoff(ds=data)
    min_year, max_year = get_min_max_runoff_years(annual_runoff=annual_runoff)
    monthly_runoff = get_monthly_runoff(ds=data)
    runoff_data = {
        "annual_runoff": annual_runoff,
        "monthly_runoff": monthly_runoff,
        "runoff_year_min": min_year,
        "runoff_year_max": max_year,
    }
    return runoff_data


def get_user_subregion(
    region_name: str = "",
    subregion_name: str = "",
    shapefile_path: str = "",
    **oggm_params,
):
    """Get user-selected subregion.

    This should be called via gateway.
    """

    if subregion_name:
        temp_name = f"OGGM_{subregion_name}"
    elif region_name:
        temp_name = f"OGGM_{region_name}"
    else:
        raise ValueError("No region or subregion name supplied.")

    init_oggm(dirname=temp_name, **oggm_params)

    if shapefile_path:  # if user uploads/selects a shapefile
        shapefile_path = f"{SHAPEFILE_PATH}/{shapefile_path}"
        rgi_file = get_rgi_files_from_subregion(region_name=region_name)
        user_shapefile = get_shapefile(path=shapefile_path)
        user_shapefile = set_subregion_constraints(
            region=rgi_file, subregion=user_shapefile, subregion_name=subregion_name
        )
        rgi_file = get_glaciers_in_subregion(
            region=rgi_file, subregion=user_shapefile, subregion_name=subregion_name
        )
    else:
        rgi_file = get_rgi_files_from_subregion(
            region_name=region_name, subregion_name=subregion_name
        )

    return {"glacier_data": rgi_file, "shapefile": user_shapefile}
