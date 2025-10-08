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
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as shpg
import xarray as xr
from oggm import cfg, tasks, utils, workflow
from oggm.shop import its_live, w5e5

import dtcg.datacube.cryotempo_eolis as cryotempo_eolis
import dtcg.integration.calibration
from dtcg.datacube.geozarr import GeoZarrHandler

# TODO: Link to DTCG instead.
# DEFAULT_BASE_URL = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs"
# SHAPEFILE_PATH = (
#     "https://cluster.klima.uni-bremen.de/~dtcg/test_files/case_study_regions/austria"
# )


class BindingsOggmModel:
    """Bindings for interacting with OGGM and web repositories.

    Attributes
    ----------
    DEFAULT_BASE_URL : str
        Base URL for OGGM data.
    SHAPEFILE_PATH : str
        Base URL for shapefile data.
    WORKING_DIR : str, default None
        Temporary working directory.
    """

    def __init__(
        self,
        base_url: str = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs",
        working_dir: str = "",
        oggm_params: dict = None,
    ):
        super().__init__()
        self.DEFAULT_BASE_URL = base_url
        self.WORKING_DIR = utils.gettempdir(working_dir)
        if oggm_params is None:
            self.oggm_params = {}
        else:
            self.oggm_params = oggm_params

    def set_oggm_params(self, **new_params: dict) -> None:
        self.oggm_params.update(new_params)

    def set_oggm_kwargs(self) -> None:
        """Set OGGM configuration parameters.

        .. note:: This may eventually be moved to ``api.internal._parse_oggm``.

        Parameters
        ----------
        oggm_params : dict
            Key/value pairs corresponding to valid OGGM configuration
            parameters.
        """

        valid_keys = cfg.PARAMS.keys()
        for key, value in self.oggm_params.items():
            if key in valid_keys:
                cfg.PARAMS[key] = value

    def init_oggm(self, dirname: str, **kwargs) -> None:
        """Initialise OGGM run parameters.

        TODO: Add kwargs for cfg.PATH.

        Parameters
        ----------
        dirname : str
            Name of temporary directory

        """

        cfg.initialize(logging_level="CRITICAL")
        cfg.PARAMS["border"] = kwargs.get("border", 80)
        self.WORKING_DIR = utils.gettempdir(dirname)  # already handles empty strings
        utils.mkdir(
            self.WORKING_DIR, reset=True
        )  # TODO: this should be an API parameter
        cfg.PATHS["working_dir"] = self.WORKING_DIR

        self.set_oggm_params(**kwargs)
        self.set_oggm_kwargs()

    def get_matching_region_codes(
        self, region_name: str = "", subregion_name: str = ""
    ) -> set:
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
            region_db = self.get_rgi_metadata("rgi_subregions_V6.csv", from_web=True)
            for row in region_db:
                if subregion_name.lower() in row["Full_name"].lower():
                    region_codes = (row["O1"], row["O2"])
                    # zfill should be applied only when using utils.parse_rgi_meta
                    # region_codes = (row["O1"].zfill(2), row["O2"].zfill(2))
                    matching_region.add(region_codes)
        elif region_name:
            region_db = self.get_rgi_metadata("rgi_regions.csv", from_web=True)
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

    def get_rgi_region_codes(
        self, region_name: str = "", subregion_name: str = ""
    ) -> set:
        """Get RGI region and subregion codes from a subregion name.

        This can be replaced with OGGM backend if available.

        Parameters
        ----------
        region_name : str, optional
            Name of RGI region.
        subregion_name : str, optional
            Name of RGI subregion.

        Returns
        -------
        set[str]
            RGI region (O1) and subregion (O2) codes.

        Raises
        ------
        KeyError
            If the subregion name is not available.
        TypeError
            If subregion name is not a string.
        """

        try:
            rgi_codes = self.get_matching_region_codes(
                region_name=region_name, subregion_name=subregion_name
            )
        except KeyError as e:
            region_names = ", ".join(filter(None, (region_name, subregion_name)))
            raise KeyError(f"No regions or subregion matching {region_names}.")
        except (TypeError, ValueError, AttributeError) as e:
            region_names = ", ".join(filter(None, (region_name, subregion_name)))
            raise TypeError(f"{region_names} is not a string.")

        return rgi_codes

    def get_rgi_metadata(
        self, path: str = "rgi_subregions_V6.csv", from_web: bool = False
    ) -> list:
        """Get RGI metadata.

        TODO: Replace with call to ``oggm.utils.parse_rgi_meta``.

        Parameters
        ----------
        path : str, default "rgi_subregions_V6.csv".
            Path to database with subregion names and O1/O2 codes.
        from_web : bool, default False
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
        self, region_name: str = "", subregion_name: str = ""
    ) -> list:
        """Get RGI shapefile from a subregion name.

        Parameters
        ----------
        region_name : str, optional
            Name of RGI region.
        subregion_name : str, optional
            Name of RGI subregion.

        Returns
        -------
            List of RGI shapefiles.

        Raises
        ------
        KeyError
            If no glaciers are found for the given subregion.
        """

        rgi_region_codes = self.get_rgi_region_codes(
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

    def get_glacier_directories(
        self,
        rgi_ids: list,
        base_url: str = "",
        prepro_level: int = 4,
        prepro_border: int = 80,
        **kwargs,
    ):
        """Get OGGM glacier directories.

        Parameters
        ----------
        rgi_ids : list
            RGI IDs of glaciers.
        base_url : str, default empty string
            URL to OGGM data. If empty, uses the ``DEFAULT_BASE_URL``
            attribute.
        prepro_level : int, default 4
            File preprocessing level.
        prepro_border : int, default 80
            Grid border buffer around the glacier.
        **kwargs
            Extra arguments to ``workflow.init_glacier_directories``.

        Returns
        -------
        list
            :py:class:`oggm.GlacierDirectory` objects for the
            initialised glacier directories.
        """

        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        gdirs = workflow.init_glacier_directories(
            rgi_ids,
            prepro_base_url=base_url,
            from_prepro_level=prepro_level,
            prepro_border=prepro_border,
            **kwargs,
        )

        return gdirs

    def set_flowlines(self, gdir) -> None:
        """Compute glacier flowlines if missing from glacier directory."""
        if not os.path.exists(gdir.get_filepath("inversion_flowlines")):
            if not os.path.exists(gdir.get_filepath("elevation_band_flowline")):
                tasks.elevation_band_flowline(gdir=gdir, preserve_totals=True)
            tasks.fixed_dx_elevation_band_flowline(gdir, preserve_totals=True)


class BindingsOggmWrangler(BindingsOggmModel):
    """Wrangles input data for OGGM workflows."""

    def __init__(
        self,
        base_url: str = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs",
        working_dir: str = "",
        oggm_params: dict = None,
        shapefile_path: str = "https://cluster.klima.uni-bremen.de/~dtcg/test_files/case_study_regions/austria",
    ):
        super().__init__(
            base_url=base_url, working_dir=working_dir, oggm_params=oggm_params
        )
        self.SHAPEFILE_PATH = shapefile_path

    def get_shapefile(self, path: str) -> gpd.GeoDataFrame:
        """Get shapefile from user path.

        Parameters
        ----------
        path : str
            Path to shapefile.
        """

        shapefile = gpd.read_file(path)
        return shapefile

    def get_shapefile_from_web(self, shapefile_name: str) -> gpd.GeoDataFrame:
        """Placeholder for getting shapefiles from an online repository.

        Parameters
        ----------
        shapefile_name : str
            Name of file in ``oggm-sample-data`` repository.
        """

        path = utils.get_demo_file(shapefile_name)
        shapefile = gpd.read_file(path)

        return shapefile

    def set_subregion_constraints(
        self,
        region: gpd.GeoDataFrame,
        subregion: gpd.GeoDataFrame,
        subregion_id: str = "",
    ) -> gpd.GeoDataFrame:
        """Reproject subregional data to match regional data, and
        optionally mask by subregion name.

        Parameters
        ----------
        region : gpd.GeoDataFrame
            Shapefile data for region.
        subregion : gpd.GeoDataFrame
            Shapefile data for subregion.
        subregion_id : str, optional
            Subregion id.

        Returns
        -------
        gpd.GeoDataFrame
            Reprojected subregion data, optionally masked to a specific
            subregion name.
        """
        if region.crs != subregion.crs:
            subregion = subregion.to_crs(region.crs)

        if subregion_id:
            subregion = subregion[subregion["id"] == subregion_id]
        return subregion

    def get_glaciers_in_subregion(
        self, region: gpd.GeoDataFrame, subregion: gpd.GeoDataFrame
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

    def get_polygon_difference(self, frame: gpd.GeoDataFrame) -> pd.DataFrame:
        """Get the difference of overlapping polygons.

        .. note:: GPD's overlay() only acts on GeoDataFrames, but this
        function manipulates GeoSeries.

        Parameters
        ----------
        frame : gpd.GeoDataFrame
            Overlapping polygons.

        Returns
        -------
        gpd.GeoSeries
            Polygon geometries adjusted so no areas overlap.
        """

        diffs = frame.iloc[1:].geometry.difference(
            frame.iloc[:-1].geometry, align=False
        )
        # use GeoSeries instead of pd.Series for consistent typing & CRS
        new_geometry = pd.concat(
            [gpd.GeoSeries(frame.geometry.iloc[0], crs=diffs.crs), diffs]
        )

        return new_geometry

    def set_polygon_overlap(self, frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
            Polygon geometries adjusted so no areas overlap.
        """
        merge = frame.copy()
        new_geometry = self.get_polygon_difference(frame=frame)
        merge.geometry = new_geometry

        return merge

    def get_gdir_by_name(self, data: gpd.GeoDataFrame, name: str, **kwargs):
        """Get glacier directory from a glacier's name or RGI ID.

        Parameters
        ----------
        data : gpd.GeoDataFrame
            Glacier data.
        name : str
            Name or RGI ID of glacier.
        **kwargs
            Additional arguments passed to ``workflow.init_glacier_directories``.
        """
        glacier = self.get_glacier_by_name(data=data, name=name)
        gdirs = self.get_glacier_directories(glaciers=[glacier], **kwargs)

        return gdirs[0]

    def get_glacier_by_name(
        self, data: gpd.GeoDataFrame, name: str
    ) -> gpd.GeoDataFrame:
        """Get glacier data by full name or RGI ID.

        Parameters
        ----------
        data : gpd.GeoDataFrame
            Glacier data.
        name : str
            Full name or RGI ID of glacier.

        Returns
        -------
        gpd.GeoDataFrame
            Glacier with matching name or RGI ID.
        """
        glacier = data[data["Name"] == name]
        if glacier.empty:  # fallback in case of RGI ID
            glacier = self.get_glacier_by_rgi_id(data=data, rgi_id=name)
        return glacier

    def get_glacier_by_rgi_id(self, data, rgi_id: str) -> gpd.GeoDataFrame:
        """Get glacier data by RGI ID.

        Parameters
        ----------
        data : gpd.GeoDataFrame
            Glacier data.
        rgi_id : str
            RGI ID of glacier.

        Returns
        -------
        gpd.GeoDataFrame
            Glacier with matching RGI ID.
        """
        glacier = data[data["RGIId"] == rgi_id]
        if glacier.empty:
            raise KeyError(f"{rgi_id} not found.")
        return glacier

    def get_named_glaciers(self, glaciers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Get all glaciers with a full name.

        Parameters
        ----------
        glaciers : gpd.GeoDataFrame
            Glacier data.

        Returns
        -------
        gpd.GeoDataFrame
            All named glaciers.
        """
        return glaciers.dropna(subset="Name")

    def get_user_subregion(
        self,
        region_name: str = "",
        subregion_name: str = "",
        shapefile_path: str = "",
        **oggm_params,
    ) -> dict:
        """Get user-selected subregion in a catchment area.

        This should be called via gateway.

        Parameters
        ----------
        region_name : str, optional
            Name of RGI region.
        subregion_name : str, optional
            Name of RGI subregion.
        shapefile_path : str, optional
            Path to shapefile with catchment area outlines.
        **oggm_params
            Additional OGGM configuration parameters.
        """
        if subregion_name:
            temp_name = f"OGGM_{subregion_name}"
        elif region_name:
            temp_name = f"OGGM_{region_name}"
        else:
            raise ValueError("No region or subregion name supplied.")

        self.init_oggm(dirname=temp_name, **oggm_params)

        if shapefile_path:  # if user uploads/selects a shapefile
            if isinstance(shapefile_path, str):
                shapefile_path = f"{self.SHAPEFILE_PATH}/{shapefile_path}"
            user_shapefile = self.get_shapefile(path=shapefile_path)
            rgi_file = self.get_rgi_files_from_subregion(region_name=region_name)
            user_shapefile = self.set_subregion_constraints(
                region=rgi_file, subregion=user_shapefile, subregion_id=subregion_name
            )
            rgi_file = self.get_glaciers_in_subregion(
                region=rgi_file, subregion=user_shapefile
            )
        else:
            rgi_file = self.get_rgi_files_from_subregion(
                region_name=region_name, subregion_name=subregion_name
            )
            user_shapefile = None

        return {"glacier_data": rgi_file, "shapefile": user_shapefile}

    def get_shapefile_metadata(self, shapefile: gpd.GeoDataFrame) -> dict:
        """Get metadata from a catchment shapefile."""

        metadata = {}
        for key in ["id", "name"]:
            if key in shapefile.keys():
                metadata[key] = shapefile[key].dropna().values.tolist()
        metadata["glacier_names"] = [""]

        return metadata


class BindingsHydro(BindingsOggmWrangler):
    """Bindings for running model with hydrological spinup.

    Attributes
    ----------
    hydro_location : str
        Output directory name.
    """

    def __init__(
        self,
        base_url: str = "https://cluster.klima.uni-bremen.de/~oggm/demo_gdirs",
        working_dir: str = "",
        oggm_params: dict = None,
        hydro_location: str = "_spinup_historical_hydro",
    ):
        super().__init__(
            base_url=base_url, working_dir=working_dir, oggm_params=oggm_params
        )
        self.hydro_location = hydro_location

    def get_compiled_run_output(
        self,
        gdir,
        keys: list = None,
        input_filesuffix: str = "",
    ) -> xr.Dataset:
        if not input_filesuffix:
            input_filesuffix = self.hydro_location
        ds = utils.compile_run_output(gdir, input_filesuffix=input_filesuffix)
        if keys:
            ds = ds[keys]
        return ds

    def get_specific_mass_balance(self, ds, rgi_id: str):
        volume = ds.loc[{"rgi_id": rgi_id}].volume.values
        area = ds.loc[{"rgi_id": rgi_id}].area.values
        smb = (volume[1:] - volume[:-1]) / area[1:] * cfg.PARAMS["ice_density"] / 1000
        return smb
        # volume = ds.volume_m3.values
        # area = ds.area_m2.values
        # smb = (volume[1:] - volume[:-1]) / area[1:] * cfg.PARAMS['ice_density'] / 1000
        # return smb

    def get_hydro_climatology(self, gdir, nyears: int = 20) -> xr.Dataset:
        workflow.execute_entity_task(
            tasks.run_with_hydro,
            gdir,  # Select the one glacier but actually should be a list
            run_task=tasks.run_from_climate_data,
            init_model_filesuffix="_spinup_historical",
            init_model_yr=1979,
            ref_area_yr=2000,  # Important ! Fixed gauge runoff with ref year 2000
            ys=1979,
            ye=2020,
            output_filesuffix=self.hydro_location,
            store_monthly_hydro=True,
        )[0]
        with xr.open_dataset(
            gdir.get_filepath("model_diagnostics", filesuffix=self.hydro_location)
        ) as ds:
            ds = ds.isel(time=slice(0, -1)).load()
        return ds

    def get_annual_runoff(self, ds: xr.Dataset):
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

    def get_min_max_runoff_years(
        self, annual_runoff: pd.Series, nyears: int = 20
    ) -> tuple:
        """Get the years with minimum and maximum runoff.

        Parameters
        ----------
        annual_runoff : pd.Series
            Time series of annual runoff.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        tuple[int]
            The years with minimum and maximum runoff.
        """
        if len(annual_runoff) > nyears:
            annual_runoff = annual_runoff.iloc[-nyears:]
        runoff = annual_runoff.sum(axis=1)
        runoff_year_min = int(runoff.idxmin())
        runoff_year_max = int(runoff.idxmax())

        return runoff_year_min, runoff_year_max

    def get_monthly_runoff(self, ds: xr.Dataset, nyears: int = 20) -> xr.DataArray:
        """Get the monthly glacier runoff.

        Parameters
        ----------
        ds : xr.Dataset
            Glacier climatology.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        xr.Dataset
            Monthly runoff.
        """
        monthly_runoff = (
            ds["melt_off_glacier_monthly"]
            + ds["melt_on_glacier_monthly"]
            + ds["liq_prcp_off_glacier_monthly"]
            + ds["liq_prcp_on_glacier_monthly"]
        )
        if len(monthly_runoff) > nyears:
            monthly_runoff = monthly_runoff.isel(time=slice(nyears, None))
        monthly_runoff *= 1e-9

        return monthly_runoff

    def get_climatology(self, data: gpd.GeoDataFrame, name: str = "") -> tuple:
        """Get climatological data for a glacier.

        Parameters
        ----------
        data : gpd.GeoDataFrame
            Contains data for one or more glaciers.
        name : str, optional
            Name of glacier.

        Returns
        -------
        tuple[xr.Dataset,np.ndarray]
            Climatological data and specific mass balance for a glacier.
        """
        glacier = self.get_gdir_by_name(
            data=data,
            name=name,
            from_prepro_level=5,
            prepro_border=160,
            prepro_base_url="https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup",
        )
        climatology = self.get_hydro_climatology(gdir=glacier)

        try:
            mass_balance = self.get_compiled_run_output(
                gdir=glacier,
                # keys=["volume", "area"],
                input_filesuffix=self.hydro_location,
            )
            mass_balance = self.get_specific_mass_balance(
                ds=mass_balance, rgi_id=glacier.rgi_id
            )
            ref_data = glacier.get_ref_mb_data()

        except RuntimeError:
            ref_data = None
            mass_balance = None
        return climatology, ref_data, mass_balance

    def get_aggregate_runoff(self, data: gpd.GeoDataFrame) -> dict:

        annual_runoff = []
        monthly_runoff = []
        mass_balance = []
        observations = []
        for rgi_id in data["RGIId"]:
            climatology, ref_data, specific_mb = self.get_climatology(
                data=data, name=rgi_id
            )
            mass_balance.append(specific_mb)
            observations.append(ref_data)
            annual_runoff.append(self.get_annual_runoff(ds=climatology))
            monthly_runoff.append(self.get_monthly_runoff(ds=climatology))
        total_annual_runoff = sum(annual_runoff)
        total_monthly_runoff = sum(monthly_runoff)
        min_year, max_year = self.get_min_max_runoff_years(
            annual_runoff=total_annual_runoff
        )
        runoff_data = {
            "annual_runoff": total_annual_runoff,
            "monthly_runoff": total_monthly_runoff,
            "runoff_year_min": min_year,
            "runoff_year_max": max_year,
            "mass_balance": mass_balance,
            "wgms": observations,
        }
        return runoff_data

    def get_runoff(self, data: xr.Dataset, name: str) -> dict:

        climatology, observations, mass_balance = self.get_climatology(
            data=data, name=name
        )
        annual_runoff = self.get_annual_runoff(ds=climatology)
        monthly_runoff = self.get_monthly_runoff(ds=climatology)
        min_year, max_year = self.get_min_max_runoff_years(annual_runoff=annual_runoff)

        runoff_data = {
            "annual_runoff": annual_runoff,
            "monthly_runoff": monthly_runoff,
            "runoff_year_min": min_year,
            "runoff_year_max": max_year,
            "mass_balance": mass_balance,
            "wgms": observations,
        }
        return runoff_data


class BindingsCryotempo(BindingsOggmWrangler):
    """Bindings for interacting with CryoTEMPO-EOLIS data.

    Attributes
    ----------
    hydro_location : str
        Output directory name.
    """

    def __init__(
        self,
        base_url: str = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup",
        working_dir: str = "",
        oggm_params: dict = {
            "border": 160,
            "store_model_geometry": True,
        },
        hydro_location: str = "_spinup_historical_hydro",
    ):
        super().__init__(
            base_url=base_url, working_dir=working_dir, oggm_params=oggm_params
        )
        self.datacube_manager = cryotempo_eolis.DatacubeCryotempoEolis()
        self.calibrator = dtcg.integration.calibration.CalibratorCryotempo()
        self.hydro_location = hydro_location

    def init_oggm(self, dirname="", **kwargs):
        return super().init_oggm(dirname, **kwargs)

    def get_glacier_directories(
        self,
        rgi_ids: list,
        base_url: str = "",
        prepro_level: int = 4,
        prepro_border: int = 80,
        **kwargs,
    ):
        return super().get_glacier_directories(
            rgi_ids, base_url, prepro_level, prepro_border, **kwargs
        )

    def get_glacier_data(self, gdirs: list) -> None:
        """Add velocity data, monthly and daily W5E5 data."""
        workflow.execute_entity_task(tasks.glacier_masks, gdirs)
        workflow.execute_entity_task(its_live.itslive_velocity_to_gdir, gdirs)

        # monthly data needed to prevent silent failures
        workflow.execute_entity_task(gdirs=gdirs, task=w5e5.process_w5e5_data)
        workflow.execute_entity_task(
            gdirs=gdirs, task=w5e5.process_w5e5_data, daily=True
        )

    def get_eolis_data(self, gdir):
        """Get gridded data enhanced with CryoTEMPO-EOLIS data."""
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as datacube:
            datacube = datacube.load()

        self.datacube_manager.retrieve_prepare_eolis_gridded_data(
            oggm_ds=datacube, grid=gdir.grid
        )

        geozarr_handler = GeoZarrHandler(datacube)

        return gdir, geozarr_handler

    def get_aggregate_runoff(self, gdir) -> dict:
        """Get the computed runoff from OGGM."""

        annual_runoff = []
        monthly_runoff = []
        mass_balance = []
        observations = []
        climatology, ref_data, specific_mb = self.get_climatology(
            name=gdir.rgi_id, gdir=gdir
        )
        mass_balance.append(specific_mb)
        observations.append(ref_data)
        annual_runoff.append(self.get_annual_runoff(ds=climatology))
        monthly_runoff.append(self.get_monthly_runoff(ds=climatology))

        total_annual_runoff = sum(annual_runoff)
        total_monthly_runoff = sum(monthly_runoff)
        min_year, max_year = self.get_min_max_runoff_years(
            annual_runoff=total_annual_runoff
        )
        runoff_data = {
            "annual_runoff": total_annual_runoff,
            "monthly_runoff": total_monthly_runoff,
            "runoff_year_min": min_year,
            "runoff_year_max": max_year,
            "mass_balance": mass_balance,
            "wgms": observations,
        }
        return runoff_data

    def get_monthly_runoff(self, ds: xr.Dataset, nyears: int = 20) -> xr.DataArray:
        """Get the monthly glacier runoff.

        Parameters
        ----------
        ds : xr.Dataset
            Glacier climatology.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        xr.Dataset
            Monthly runoff.
        """
        monthly_runoff = (
            ds["melt_off_glacier_monthly"]
            + ds["melt_on_glacier_monthly"]
            + ds["liq_prcp_off_glacier_monthly"]
            + ds["liq_prcp_on_glacier_monthly"]
        )
        if len(monthly_runoff) > nyears:
            monthly_runoff = monthly_runoff.isel(time=slice(nyears, None))
        monthly_runoff *= 1e-9

        return monthly_runoff

    def get_climatology(self, gdir, name: str = "") -> tuple:
        """Get climatological data for a glacier.

        Parameters
        ----------
        data : gpd.GeoDataFrame
            Contains data for one or more glaciers.
        name : str, optional
            Name of glacier.

        Returns
        -------
        tuple[xr.Dataset,np.ndarray]
            Climatological data and specific mass balance for a glacier.
        """

        climatology = self.get_hydro_climatology(gdir=gdir)

        try:
            mass_balance = self.get_compiled_run_output(
                gdir=gdir,
                # keys=["volume", "area"],
                input_filesuffix=self.hydro_location,
            )
            mass_balance = self.get_specific_mass_balance(
                ds=mass_balance, rgi_id=gdir.rgi_id
            )
            ref_data = gdir.get_ref_mb_data()

        except RuntimeError:
            ref_data = None
            mass_balance = None
        return climatology, ref_data, mass_balance

    def get_annual_runoff(self, ds: xr.Dataset):
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

    def get_min_max_runoff_years(
        self, annual_runoff: pd.Series, nyears: int = 20
    ) -> tuple:
        """Get the years with minimum and maximum runoff.

        Parameters
        ----------
        annual_runoff : pd.Series
            Time series of annual runoff.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        tuple[int]
            The years with minimum and maximum runoff.
        """
        if len(annual_runoff) > nyears:
            annual_runoff = annual_runoff.iloc[-nyears:]
        runoff = annual_runoff.sum(axis=1)
        runoff_year_min = int(runoff.idxmin())
        runoff_year_max = int(runoff.idxmax())

        return runoff_year_min, runoff_year_max

    def get_hydro_climatology(self, gdir, nyears: int = 20) -> xr.Dataset:
        workflow.execute_entity_task(
            tasks.run_with_hydro,
            gdir,  # Select the one glacier but actually should be a list
            run_task=tasks.run_from_climate_data,
            init_model_filesuffix="_spinup_historical",
            init_model_yr=1979,
            ref_area_yr=2000,  # Important ! Fixed gauge runoff with ref year 2000
            ys=1979,
            ye=2020,
            output_filesuffix=self.hydro_location,
            store_monthly_hydro=True,
        )[0]
        with xr.open_dataset(
            gdir.get_filepath("model_diagnostics", filesuffix=self.hydro_location)
        ) as ds:
            ds = ds.isel(time=slice(0, -1)).load()
        return ds

    def get_compiled_run_output(
        self,
        gdir,
        keys: list = None,
        input_filesuffix: str = "",
    ) -> xr.Dataset:
        if not input_filesuffix:
            input_filesuffix = self.hydro_location
        ds = utils.compile_run_output(gdir, input_filesuffix=input_filesuffix)
        if keys:
            ds = ds[keys]
        return ds

    def get_specific_mass_balance(self, ds, rgi_id: str):
        volume = ds.loc[{"rgi_id": rgi_id}].volume.values
        area = ds.loc[{"rgi_id": rgi_id}].area.values
        smb = (volume[1:] - volume[:-1]) / area[1:] * cfg.PARAMS["ice_density"] / 1000
        return smb

    def get_cached_data(self, rgi_id: str, cache="../../ext/data/l2_precompute/"):

        if isinstance(cache, str):
            cache = Path(cache)
        cache_path = cache / rgi_id
        with open(cache_path / "gdir.json", mode="r", encoding="utf-8") as file:
            raw = file.read()
            gdir = dict(json.loads(raw))

        smb = np.load(cache_path / "smb.npz")
        with xr.open_dataarray(cache_path / "runoff.nc") as file:
            runoff = file.load()
        runoff_data = {
            "monthly_runoff": runoff,
            "runoff_year_min": gdir["runoff_data"]["runoff_year_min"],
            "runoff_year_max": gdir["runoff_data"]["runoff_year_max"],
        }

        return gdir, smb, runoff_data

    def get_cached_data(self, rgi_id: str, cache="../../ext/data/l2_precompute/"):

        if isinstance(cache, str):
            cache = Path(cache)
        cache_path = cache / rgi_id
        gdir = self.get_cached_gdir_data(cache_path=cache_path)

        smb = self.get_cached_smb_data(cache_path=cache_path)
        runoff = self.get_cached_runoff_data(cache_path=cache_path)
        runoff["runoff_year_min"] = gdir["runoff_data"]["runoff_year_min"]
        runoff["runoff_year_max"] = gdir["runoff_data"]["runoff_year_max"]

        return gdir, smb, runoff

    def get_cached_l1_data(self, rgi_id: str, cache="../../ext/data/l1_precompute/"):

        if isinstance(cache, str):
            cache = Path(cache)
        cache_path = cache / rgi_id
        gdir = self.get_cached_gdir_data(cache_path=cache_path)

        return gdir

    def get_cached_gdir_data(self, cache_path: Path) -> dict:

        with open(cache_path / "gdir.json", mode="r", encoding="utf-8") as file:
            raw = file.read()
            gdir = dict(json.loads(raw))
        return gdir

    def get_cached_smb_data(self, cache_path: Path) -> np.ndarray:
        smb = np.load(cache_path / "smb.npz")

        return smb

    def get_cached_runoff_data(self, cache_path: Path) -> dict:
        runoff = xr.open_dataarray(cache_path / "runoff.nc")
        runoff_data = {"monthly_runoff": runoff}

        return runoff_data

    def get_cached_metadata(
        self, index="glacier_index", cache="../../ext/data/l2_precompute/"
    ):
        if isinstance(cache, str):
            cache = Path(cache)
        cache_path = cache / index
        with open(
            cache_path / f"glacier_index.json", mode="r", encoding="utf-8"
        ) as file:
            raw = file.read()
            metadata = dict(json.loads(raw))

        return metadata
