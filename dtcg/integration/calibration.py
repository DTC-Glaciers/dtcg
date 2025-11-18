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

Calibrate OGGM models.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.tz import UTC
from oggm import GlacierDirectory, cfg, utils
from oggm.core import massbalance
from tqdm import tqdm


class Calibrator:
    """Bindings for calibrating OGGM models.

    Attributes
    ----------
    """

    def __init__(self, model_matrix: dict = None):
        super().__init__()
        if not model_matrix:
            self.model_matrix = {}
        else:
            self.model_matrix = model_matrix

    def set_model_matrix(self, name: str, model, geo_period: str, **kwargs):
        """Set model parameters for calibration.

        Parameters
        ----------
        name : str
            Name of configuration parameters. Ideally this should match
            the settings filesuffix.
        model : oggm.core.massbalance.MassBalanceModel
            OGGM mass balance model class.
        geo_period : str
            Reference mass balance period in the format
            YYYY-MM-DD_YYYY-MM-DD.
        kwargs
            Additional arguments passed to the mass balance model
            class.
        """

        matrix = {
            name: {
                "model": model,
                "geo_period": geo_period,
            }
        }
        if kwargs:
            matrix[name].update(kwargs)
        self.model_matrix.update(matrix)

    def get_nearest(self, items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def get_calibrated_models(
        self,
        gdir: GlacierDirectory,
        model_class,
        ref_mb: pd.DataFrame,
        geodetic_period: str = "",
        years: list = None,
        model_calib: dict = None,
        model_flowlines: dict = None,
        smb: dict = None,
        daily: bool = False,
        calibration_filesuffix: str = "",
        calibration_parameters: dict = None,
        **kwargs,
    ) -> tuple:
        """Get calibrated models.

        Note this uses all three calibration parameters, with ``prcp_fac``
        as the first parameter.

        Parameters
        ----------
        gdir : GlacierDirectory
            Glacier directory.
        model_class : oggm.MassBalanceModel
            Any mass balance model that subclasses MonthlyTIModel.
        ref_mb : pd.DataFrame
            Reference mass balance.
        geodetic_period : str, default empty string
            The reference calibration period in the format: "Y-M-D_Y-M-D"
        years : list, default None
            Years for which to calculate the specific mass balance. Ensure
            these are float years when using ``MonthlyTI``.
        model_calib : dict
            Store calibrated models derived from ``mb_model_class``
        model_flowlines : dict
            Store calibrated ``MultipleFlowlineMassBalanceModel``.
        smb : dict
            Store specific mass balance.
        daily : bool, default False
            Process daily specific mass balance.
        calibration_filesuffix : str, default empty string
            Calibration filesuffix.
        calibration_parameters : dict, default None
            Extra arguments passed to ``mb_calibration_from_scalar_mb``.
        kwargs
            Extra arguments passed to the mass balance model used for
            calibration.

        Returns
        -------
        tuple
            Calibrated model instances for each calibration parameter,
            calibrated flowline models for each parameter,
            and specific mass balance for each calibrated flowline model.
        """
        model_name = model_class.__name__
        if not geodetic_period:
            geodetic_period = cfg.PARAMS["geodetic_mb_period"]

        if model_calib is None:
            model_calib = {}
        if model_flowlines is None:
            model_flowlines = {}
        if smb is None:
            smb = {}

        if not calibration_parameters:
            calibration_parameters = {
                "calibrate_param1": "melt_f",
                "calibrate_param2": "prcp_fac",
                "calibrate_param3": "temp_bias",
            }

        if not calibration_filesuffix:
            calibration_filesuffix = f"{model_name}_{geodetic_period}"
        model_key = calibration_filesuffix.removesuffix("Model")

        # This follows mb_calibration_from_geodetic_mb
        prcp_fac = massbalance.decide_winter_precip_factor(gdir)
        mi, ma = cfg.PARAMS["prcp_fac_min"], cfg.PARAMS["prcp_fac_max"]
        prcp_fac_min = utils.clip_scalar(prcp_fac * 0.8, mi, ma)
        prcp_fac_max = utils.clip_scalar(prcp_fac * 1.2, mi, ma)

        if "SfcType_Cryosat_2015" in calibration_filesuffix:
            calibration_parameters.update(
                {
                    "calibrate_param1": "prcp_fac",
                    "calibrate_param2": "temp_bias",
                    "calibrate_param3": "melt_f",
                    "melt_f": 4.728225163624522,
                }
            )

        massbalance.mb_calibration_from_scalar_mb(
            gdir,
            ref_mb=ref_mb,
            ref_mb_period=geodetic_period,
            **calibration_parameters,
            prcp_fac=prcp_fac,
            prcp_fac_min=prcp_fac_min,
            prcp_fac_max=prcp_fac_max,
            mb_model_class=model_class,
            overwrite_gdir=True,
            use_2d_mb=False,
            settings_filesuffix=calibration_filesuffix,
            observations_filesuffix=calibration_filesuffix,
            **kwargs,
        )

        if not kwargs:
            model_calib[model_key] = model_class(
                gdir,
                settings_filesuffix=calibration_filesuffix,
            )
            model_flowlines[model_key] = massbalance.MultipleFlowlineMassBalance(
                gdir,
                mb_model_class=model_class,
                use_inversion_flowlines=True,
                settings_filesuffix=calibration_filesuffix,
            )
        else:
            model_calib[model_key] = model_class(
                gdir,
                settings_filesuffix=calibration_filesuffix,
                **kwargs,
            )
            model_flowlines[model_key] = massbalance.MultipleFlowlineMassBalance(
                gdir,
                mb_model_class=model_class,
                use_inversion_flowlines=True,
                settings_filesuffix=calibration_filesuffix,
                **kwargs,
            )
        fls = gdir.read_pickle("inversion_flowlines")
        if not years:
            years = utils.float_years_timeseries(
                y0=2000, y1=2019, include_last_year=True, monthly=False
            )
        if not daily:
            smb[model_key] = model_flowlines[model_key].get_specific_mb(
                fls=fls, year=years
            )
        else:
            smb[model_key] = model_flowlines[model_key].get_specific_mb(
                fls=fls, year=years, time_resolution="daily"
            )

        return model_calib, model_flowlines, smb

    def get_geodetic_mb(self, gdir: GlacierDirectory) -> pd.DataFrame:
        """Get geodetic mass balances for a glacier.

        Returns
        -------
        pd.DataFrame
            Geodetic mass balances for a specific glacier for different
            reference periods.
        """
        pd_geodetic = utils.get_geodetic_mb_dataframe()

        return pd_geodetic.loc[gdir.rgi_id]

    def get_calibration_mb(self, ref_mb: pd.DataFrame, geo_period: str) -> float:
        """Get calibration mass balance for a specific reference period.

        Parameters
        ----------
        ref_mb : pd.DataFrame
            Reference mass balances for a glacier.
        geo_period : str
            Reference calibration period. This should be a value in
            ref_mb["period"].

        Returns
        -------
        float
            Mass balance used for calibration.
        """
        geodetic_mb = ref_mb.loc[ref_mb.period == geo_period].dmdtda * 1000
        return geodetic_mb

    def calibrate(
        self, gdir: GlacierDirectory, model_matrix: dict, ref_mb: float
    ) -> tuple:
        """Calibrate an OGGM glacier model.

        Parameters
        ----------
        gdir : GlacierDirectory
            Glacier directory.
        model_matrix : dict
            Model parameters for different run configurations.
        ref_mb : float
            Reference mass balance.

        Returns
        -------
        tuple
            Calibrated mass balance model instances, flowlines, and
            specific mass balances for each run configuration in the
            model matrix.
        """
        # Store results
        mb_model_calib = {}
        mb_model_flowlines = {}
        smb = {}

        for matrix_name, model_params in tqdm(model_matrix.items()):
            mb_model_class = model_params["model"]
            geo_period = model_params["geo_period"]
            calibration_filesuffix = f"{matrix_name}_{geo_period}"
            cfg.PARAMS["geodetic_mb_period"] = geo_period

            mb_geodetic = self.get_calibration_mb(ref_mb=ref_mb, geo_period=geo_period)

            mb_model_calib, mb_model_flowlines, smb = self.get_calibrated_models(
                gdir=gdir,
                model_class=mb_model_class,
                ref_mb=mb_geodetic,
                geodetic_period=geo_period,
                model_calib=mb_model_calib,
                model_flowlines=mb_model_flowlines,
                smb=smb,
                daily=model_params.get("daily", False),
                calibration_filesuffix=calibration_filesuffix,
            )

        return mb_model_calib, mb_model_flowlines, smb


class CalibratorCryotempo(Calibrator):

    def __init__(self, model_matrix: dict = None):
        super().__init__(model_matrix=model_matrix)

    def set_model_matrix(
        self,
        name: str = "SfcType_Cryosat",
        model=massbalance.SfcTypeTIModel,
        geo_period: str = "2011-01-01_2020-01-01",
        daily: bool = True,
        source: str = "CryoTEMPO-EOLIS",
        **kwargs,
    ):
        """Set model parameters for calibration.

        Parameters
        ----------
        name : str
            Name of configuration parameters. Ideally this should match
            the settings filesuffix.
        model : oggm.core.massbalance.MassBalanceModel
            OGGM mass balance model class.
        geo_period : str
            Reference mass balance period in the format
            YYYY-MM-DD_YYYY-MM-DD.
        kwargs
            Additional arguments passed to the mass balance model
            class.
        """

        return super().set_model_matrix(
            name,
            model,
            geo_period,
            daily=daily,
            source=source,
            **kwargs,
        )

    def get_eolis_dates(self, ds: xr.Dataset) -> np.ndarray:
        """Get time index from CryoTEMPO-EOLIS dataset.

        Parameters
        ----------
        ds : xr.Dataset
            CryoTEMPO-EOLIS dataset.

        Returns
        -------
        np.ndarray
            Time index adjusted to UTC.
        """
        return np.array([datetime.fromtimestamp(t, tz=UTC) for t in ds.t.values])

    def get_eolis_mean_dh(self, ds: xr.Dataset) -> np.ndarray:
        """Get

        Parameters
        ----------
        ds : xr.Dataset
            CryoTEMPO-EOLIS dataset.

        Returns
        -------
        np.ndarray
            Time series of spatial mean of elevation change.
        """
        mean_time_series = [
            np.nanmean(elevation_change_map.where(ds.glacier_mask == 1))
            for elevation_change_map in ds.eolis_gridded_elevation_change
        ]
        return np.array(mean_time_series)

    def get_temporal_bounds(self, dates: list, year_start: int, year_end: int) -> tuple:
        """Get start and end dates of geodetic and observational periods.

        This matches the nearest available dates for irregular time
        indices.

        Parameters
        ----------
        dates : list
            Time index. This may be irregularly spaced.
        year_start : int
            Start year of a desired period.
        year_end : int
            End year of a desired period.

        Returns
        -------
        tuple[datetime]
            Start and end dates of geodetic reference period, and the
            nearest available start and end dates for observations.
        """
        year_start = datetime(year_start, 1, 1, tzinfo=UTC)
        year_end = datetime(year_end, 1, 1, tzinfo=UTC)

        # EOLIS data is in 30-day periods, so get closest available date
        data_start = self.get_nearest(dates, year_start)
        data_end = self.get_nearest(dates, year_end)

        return year_start, year_end, data_start, data_end

    def get_dmdtda(
        self, dataset, dates: list, year_start: datetime, year_end: datetime
    ) -> float:
        """
        Get dmdtdA from a CryoTEMPO-EOLIS dataset.

        Parameters
        ----------
        dataset : xr.DataArray
            CryoTEMPO-EOLIS data.
        dates : list
            Datetimes for each timestep in the data period.
        year_start : datetime
            Start of reference period.
        year_end : datetime
            End of reference period.

        Returns
        -------
        float
            dmdtdA for a glacier.
        """
        calib_frame = pd.DataFrame(
            {
                "dh": dataset.eolis_elevation_change_timeseries,
                "dh_sigma": dataset.eolis_elevation_change_sigma_timeseries,
            },
            index=dates,
        )

        dt = (year_end - year_start).total_seconds() / cfg.SEC_IN_YEAR

        # dmdtda in kg m-2 yr-1, area not needed as we already have a mean dh
        # (dh = dV / A)
        bulk_density = 850  # not cfg.PARAMS["ice_density"]?
        dh = calib_frame["dh"].loc[year_end] - calib_frame["dh"].loc[year_start]
        # Convert to meters water-equivalent per year to have the same unit
        # as Hugonnet
        dmdtda = (dh * bulk_density / dt) / 1000

        return dmdtda

    def set_geodetic_mb_from_dataset(
        self,
        gdir: GlacierDirectory,
        dataset: xr.Dataset,
        year_start: int = 2011,
        year_end: int = 2020,
    ) -> pd.DataFrame:
        """Set the geodetic mass balance from enhanced gridded data.

        Parameters
        ----------
        gdir : GlacierDirectory
            Glacier directory.
        dataset : xr.Dataset
            CryoTEMPO-EOLIS dataset.
        year_start : int, default 2011
            Start of reference period.
        year_end : int, default 2020
            End of reference period.

        Returns
        -------
        pd.DataFrame
            Geodetic mass balances for various glaciers.
        """

        dates = self.get_eolis_dates(dataset)
        year_start, year_end, data_start, data_end = self.get_temporal_bounds(
            dates=dates, year_start=year_start, year_end=year_end
        )

        dmdtda = self.get_dmdtda(
            dataset=dataset, dates=dates, year_start=data_start, year_end=data_end
        )

        geodetic_mb_period = (
            f"{year_start.strftime('%Y-%m-%d')}_{year_end.strftime('%Y-%m-%d')}"
        )
        observations_period = (
            f"{data_start.strftime('%Y-%m-%d')}_{data_end.strftime('%Y-%m-%d')}"
        )
        geodetic_mb = {
            "rgiid": [gdir.rgi_id],
            "period": geodetic_mb_period,
            "observations_period": observations_period,
            "area": gdir.rgi_area_m2,
            "dmdtda": dmdtda,
            "source": "CryoTEMPO-EOLIS",
            "err_dmdtda": 0.0,
            "reg": 6,
            "is_cor": False,
        }

        return pd.DataFrame.from_records(geodetic_mb, index="rgiid")

    def get_geodetic_mb(
        self, gdir: GlacierDirectory, dataset: xr.Dataset = None
    ) -> pd.DataFrame:
        """Get geodetic mass balances for a glacier.

        Parameters
        ----------
        gdir : GlacierDirectory
            Glacier directory.
        dataset : xr.Dataset
            CryoTEMPO-EOLIS data.

        Returns
        -------
        pd.DataFrame
            Geodetic mass balances for a specific glacier RGI ID.
        """
        pd_geodetic = utils.get_geodetic_mb_dataframe()
        pd_geodetic["source"] = "Hugonnet"

        if dataset:
            period = [(2011, 2020), (2015, 2016)]
            for years in period:
                geodetic_mb = self.set_geodetic_mb_from_dataset(
                    gdir=gdir, dataset=dataset, year_start=years[0], year_end=years[1]
                )
                pd_geodetic = pd.concat([pd_geodetic, geodetic_mb])

        return pd_geodetic.loc[gdir.rgi_id]

    def get_calibration_mb(
        self, ref_mb: pd.DataFrame, geo_period: str, source: str
    ) -> float:
        """

        Parameters
        ----------
        ref_mb : pd.DataFrame
            Reference geodetic mass balances.
        geo_period : str
            Geodetic reference period in the format <YYYY-MM-DD_YYYY-MM-DD>.
        source : str
            Reference mass balance source, e.g. "Hugonnet".

        Returns
        -------

        float
            Geodetic mass balance for calibration period.

        """
        geodetic_mb = (
            ref_mb.loc[
                np.logical_and(ref_mb["source"] == source, ref_mb.period == geo_period)
            ].dmdtda.values[0]
            * 1000
        )
        return geodetic_mb

    def calibrate(
        self, gdir: GlacierDirectory, model_matrix: dict, ref_mb: pd.DataFrame, **kwargs
    ) -> tuple:
        """Calibrate and run OGGM models using a model matrix.

        Parameters
        ----------
        gdir : GlacierDirectory
            Glacier directory.
        model_matrix : dict
            Model parameters for various calibration runs.
        ref_mb : Geodetic mass balances.

        Returns
        -------
        tuple
            Calibrated mass balance models, flowlines, and specific
            mass balances for each set of configuration parameters in
            ``model_matrix``.
        """
        # Store results
        mb_model_calib = {}
        mb_model_flowlines = {}
        smb = {}

        for matrix_name, model_params in tqdm(model_matrix.items()):
            daily = model_params["daily"]
            mb_model_class = model_params["model"]
            geo_period = model_params["geo_period"]
            source = model_params["source"]
            calibration_filesuffix = f"{matrix_name}_{geo_period}"

            mb_geodetic = self.get_calibration_mb(
                ref_mb=ref_mb, geo_period=geo_period, source=source
            )

            if issubclass(mb_model_class, massbalance.SfcTypeTIModel) and daily:
                sfc_kwargs = {"climate_resolution": "daily"}
            else:
                sfc_kwargs = {}

            mb_model_calib, mb_model_flowlines, smb = self.get_calibrated_models(
                gdir=gdir,
                model_class=mb_model_class,
                ref_mb=mb_geodetic,
                geodetic_period=geo_period,
                model_calib=mb_model_calib,
                model_flowlines=mb_model_flowlines,
                smb=smb,
                daily=daily,
                calibration_filesuffix=calibration_filesuffix,
                **kwargs,
                **sfc_kwargs,
            )

        return mb_model_calib, mb_model_flowlines, smb

    def run_calibration(
        self, gdir: GlacierDirectory, datacube, model=massbalance.SfcTypeTIModel
    ):
        """Run calibration.

        Parameters
        ----------
        gdir : GlacierDirectory
            Glacier directory.
        datacube : GeoZarrHandler
            L1 or L2 datacube.
        model : massbalance.MassBalanceModel, default SfcTypeTIModel.
            OGGM mass balance model class, ideally DailyTIModel or similar.

        Returns
        -------
        tuple[dict, dict, dict]
            Calibrated mass balance models, flowlines, and specific
            mass balances for each set of configuration parameters in
            ``model_matrix``.
        """
        if not datacube:
            ref_mb = self.get_geodetic_mb(gdir=gdir)
        else:
            ref_mb = self.get_geodetic_mb(gdir=gdir, dataset=datacube)

        if isinstance(model, str):
            try:
                model = getattr(massbalance, model)
            except:
                model = massbalance.DailyTIModel

        if not datacube:
            source = "Hugonnet"
            geo_period = cfg.PARAMS["geodetic_mb_period"]
            sfc_model_kwargs = {}
        else:
            source = "Cryosat"
            geo_period = ("2011-01-01_2020-01-01",)
            sfc_model_kwargs = {
                "resolution": "day",
                "gradient_scheme": "annual",
                "check_data_exists": False,
            }

        self.set_model_matrix(
            name=f"{model.__name__}_{source}",
            model=model,
            geo_period=geo_period,
            daily=True,
            source=source,
            **sfc_model_kwargs,
        )

        mb_model_calib, mb_model_flowlines, smb = self.calibrate(
            model_matrix=self.model_matrix, gdir=gdir, ref_mb=ref_mb
        )
        return mb_model_calib, mb_model_flowlines, smb
