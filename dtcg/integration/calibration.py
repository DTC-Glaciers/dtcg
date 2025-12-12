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
import warnings
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from dateutil.tz import UTC
from oggm import GlacierDirectory, cfg, utils, workflow, tasks
from oggm.core import massbalance
from tqdm import tqdm

import SALib.sample.sobol as sampler


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

    def get_nearest_datetime(self, dates: np.ndarray, target_date: str):
        """

        Parameters
        ----------
        dates
        target_date: str
            "YYYY-MM-DD"

        Returns
        -------

        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="no explicit representation of timezones available for np.datetime64",
                category=UserWarning, )
            dates_ns = dates.astype("datetime64[ns]")
        target_ns = np.datetime64(
            datetime.strptime(target_date, "%Y-%m-%d"), "ns")
        diffs = np.abs(dates_ns - target_ns)
        idx = int(np.argmin(diffs))
        return dates[idx]

    def mcs_mb_calibration_workflow(
        self, gdir, settings_filesuffix, settings_parent_filesuffix,
        observations_filesuffix,
        mb_model_class, filename, input_filesuffix,
        ref_mb, ref_mb_period, ref_mb_err, ref_mb_unit,
        prcp_fac, melt_f, temp_bias,
        calibration_parameters=None,
    ):
        """
        This defines the calibration workflow executed for each ensemble member
        of the Monte Carlo simulation.

        Parameters
        ----------
        gdir
        settings_filesuffix
        settings_parent_filesuffix: str
            the settings where from where we use the bounds for the mb
            parameters prcp_fac, melt_f and temp_bias
        observations_filesuffix
        mb_model_class
        filename: str
            climatic filename
        input_filesuffix: str
            climatic filenamesuffix
        ref_mb
        ref_mb_period
        ref_mb_err
        ref_mb_unit
        prcp_fac
        melt_f
        temp_bias
        calibration_parameters: dict
            here you can define the order of calibration parameters, the default
            is {"calibrate_param1": "prcp_fac", "calibrate_param2": "melt_f",
            "calibrate_param3": "temp_bias",}.

        Returns
        -------

        """

        if calibration_parameters is None:
            calibration_parameters = {
                "calibrate_param1": "prcp_fac",
                "calibrate_param2": "melt_f",
                "calibrate_param3": "temp_bias",
            }

        # add observations data to observations file
        gdir.observations_filesuffix = observations_filesuffix
        gdir.observations['ref_mb'] = {'err': ref_mb_err,
                                       'period': ref_mb_period,
                                       'value': ref_mb,
                                       'unit': ref_mb_unit}

        # get bounds from parent
        bounds_to_define = ['prcp_fac_min', 'prcp_fac_max',
                            'melt_f_min', 'melt_f_max',
                            'temp_bias_min', 'temp_bias_max',
                            ]
        gdir.settings_filesuffix = settings_parent_filesuffix
        default_bounds = {}
        for bound in bounds_to_define:
            default_bounds[bound] = gdir.settings[bound]

        # create the new settings file with the correct parent_filesuffix
        utils.ModelSettings(gdir, filesuffix=settings_filesuffix,
                            parent_filesuffix=settings_parent_filesuffix)
        gdir.settings_filesuffix = settings_filesuffix
        gdir.settings['use_winter_prcp_fac'] = False
        # set parent bounds
        for bound in bounds_to_define:
            gdir.settings[bound] = default_bounds[bound]

        # the settings here mimic mb_calibration_from_hugonnet_mb
        wrkflw_return = workflow.execute_entity_task(
            massbalance.mb_calibration_from_scalar_mb,
            gdir,
            settings_filesuffix=settings_filesuffix,
            observations_filesuffix=observations_filesuffix,
            overwrite_gdir=True,
            **calibration_parameters,
            prcp_fac=prcp_fac,
            melt_f=melt_f,
            temp_bias=temp_bias,
            mb_model_class=mb_model_class,
            filename=filename,
            input_filesuffix=input_filesuffix,
            return_mb_model=True,
            )
        # the finally calibrated mass balance model is returned because of
        # return_mb_model=True
        mb_model = wrkflw_return[0][1]

        return mb_model

    def get_mcs_parameter_sample(
            self, gdir, control_filesuffix, nr_samples=2**4,
            ref_mb_err_sample_interval=1,
            prcp_fac_sample_interval=0.5,
            temp_bias_sample_interval=0.5,
            melt_f_sample_interval=0.25,):
        """
        This function defines the distribution of the observation and model
        parameters and creates a sample for the Monte Carlo simulation.

        Parameters
        ----------
        gdir
        control_filesuffix: str
            The settings filesuffix of the control run, which used only the most
            likely values of the default calibration.
        nr_samples: int
            This needs to be a multiple of 2 (e.g. 2**4). The total number of
            generated samples equals nr_samples * (4 + 2). 4 is the number of
            variables we use for sampling (ref_mb, prcp_fac, melt_f, temp_bias).
        ref_mb_err_sample_interval: float
            A factor which is used for multiplication of the ref_mb_err. Is is
            assumed that ref_mb has a normal distribution.
        prcp_fac_sample_interval: float
            This value defines the width of the bounds by adding and
            subtracting it from the prcp_fac of the control run. We use an
            uniform distribution during sampling.
        temp_bias_sample_interval
            See docstring for prcp_fac_sample_interval, but for temp_bias.
        melt_f_sample_interval
            See docstring for prcp_fac_sample_interval, but for melt_f.

        Returns
        -------

        """

        # to have a better overview I provide the parameter distribution in a
        # more readable way first
        parameter_distribution = {}

        # add geodetic mb observation
        gdir.observations_filesuffix = control_filesuffix
        ref_mb = gdir.observations['ref_mb']
        parameter_distribution['ref_mb'] = {
            'bounds': [ref_mb['value'],
                       ref_mb['err'] * ref_mb_err_sample_interval],
            'dists': 'norm'
        }

        # get default maximum values for mb parameters
        gdir.settings_filesuffix = ''
        prcp_fac_min = gdir.settings['prcp_fac_min']
        prcp_fac_max = gdir.settings['prcp_fac_max']
        melt_f_min = gdir.settings['melt_f_min']
        melt_f_max = gdir.settings['melt_f_max']
        temp_bias_min = gdir.settings['temp_bias_min']
        temp_bias_max = gdir.settings['temp_bias_max']

        # add mb parameter distributions
        gdir.settings_filesuffix = control_filesuffix

        # prcp_fac
        prcp_fac_control = gdir.settings['prcp_fac']
        prcp_fac_bounds = [
            max(prcp_fac_control - prcp_fac_sample_interval, prcp_fac_min),
            min(prcp_fac_control + prcp_fac_sample_interval, prcp_fac_max)]
        gdir.settings['prcp_fac_min'] = prcp_fac_bounds[0]
        gdir.settings['prcp_fac_max'] = prcp_fac_bounds[1]
        parameter_distribution['prcp_fac'] = {
            'bounds': prcp_fac_bounds,
            'dists': 'unif'
        }

        # melt_f
        melt_f_control = gdir.settings['melt_f']
        melt_f_bounds = [
            max(melt_f_control - melt_f_sample_interval, melt_f_min),
            min(melt_f_control + melt_f_sample_interval, melt_f_max)]
        gdir.settings['melt_f_min'] = melt_f_bounds[0]
        gdir.settings['melt_f_max'] = melt_f_bounds[1]
        parameter_distribution['melt_f'] = {
            'bounds': melt_f_bounds,
            'dists': 'unif'
        }

        # temp_bias
        temp_bias_control = gdir.settings['temp_bias']
        temp_bias_bounds = [
            max(temp_bias_control - temp_bias_sample_interval, temp_bias_min),
            min(temp_bias_control + temp_bias_sample_interval, temp_bias_max)]
        gdir.settings['temp_bias_min'] = temp_bias_bounds[0]
        gdir.settings['temp_bias_max'] = temp_bias_bounds[1]
        parameter_distribution['temp_bias'] = {
            'bounds': temp_bias_bounds,
            'dists': 'unif'
        }

        # now convert to format which is expected by SALib
        num_vars = 0
        problem = {
            'names': [],
            'bounds': [],
            'dists': [],
        }
        for key in parameter_distribution:
            num_vars += 1
            problem['names'].append(key)
            problem['bounds'].append(parameter_distribution[key]['bounds'])
            problem['dists'].append(parameter_distribution[key]['dists'])
        problem['num_vars'] = num_vars

        # create a sample of parameters
        param_values = sampler.sample(problem, nr_samples,
                                      calc_second_order=False)

        return param_values, problem['names']

    def calculate_and_add_total_runoff(self, ds):
        """
        Calculate total annual and monthly runoff from runoff components.

        Parameters
        ----------
        ds

        Returns
        -------

        """
        ds['runoff_monthly'] = (ds['melt_off_glacier_monthly'] +
                                ds['melt_on_glacier_monthly'] +
                                ds['liq_prcp_off_glacier_monthly'] +
                                ds['liq_prcp_on_glacier_monthly'])
        ds.runoff_monthly.attrs = {
            'unit': 'kg month-1',
            'description': (
                'Monthly glacier runoff from sum of monthly melt and liquid '
                'precipitation on and off the glacier using a fixed-gauge with '
                'a glacier minimum reference area from year 2000'
            )
        }

        ds['runoff'] = (ds['melt_off_glacier'] + ds['melt_on_glacier'] +
                        ds['liq_prcp_off_glacier'] + ds['liq_prcp_on_glacier'])
        ds.runoff.attrs = {
            'unit': 'kg yr-1',
            'description': (
                'Annual glacier runoff: sum of annual melt and liquid '
                'precipitation on and off the glacier using a fixed-gauge with '
                'a glacier minimum reference area from year 2000'
            )
        }

        return ds

    def calculate_and_add_specific_mb(self, ds, unit):
        """
        Calculate specific mass-balance in mm w.e. from a volume change time
        series.

        Parameters
        ----------
        ds

        Returns
        -------

        """
        ds['specific_mb'] = (
                ds.volume.diff(dim='time', label='lower').reindex(time=ds.time) /
                ds.area * cfg.PARAMS['ice_density'])

        return self.add_specific_mb_attrs(ds, unit)

    def add_specific_mb_attrs(self, ds, unit):
        ds.specific_mb.attrs = {
            'unit': unit,
            'description': ('Specific mass-balance')
        }

        return ds

    def calibrate_mb_and_create_datacubes(
        self,
        gdir: GlacierDirectory,
        mb_model_class,
        ref_mb: float,
        ref_mb_err: float,
        ref_mb_unit: str = 'kg m-2 yr-1',
        ref_mb_period: str = "",
        first_guess_settings: str = "",
        datacubes_requested: str | list = 'monthly',
        calibration_filesuffix: str = "",
        mcs_sampling_settings: dict = None,
        calibration_parameters_control: dict = None,
        calibration_parameters_mcs: dict = None,
        quantiles: dict = None,
        climate_input_filesuffix: str = None,
        show_log: bool = False,
        **kwargs,
    ) -> dict:
        """Calibrate model with provided reference mass balance and generate
        resulting model datacube. This also includes a uncertainty propagation
        through a Monte Carlo simulation.

        Parameters
        ----------
        gdir
        mb_model_class
        ref_mb
        ref_mb_err
        ref_mb_unit
        ref_mb_period
        first_guess_settings: str
            the mb parameters which should be used as a first guess
        datacubes_requested: str or list
            What data the returned datacubes should contain. You either can
            select one or provide a list. Options are:
            'monthly': monthly volume, area, length and mass-balance
            'annual_hydro': annual volume, area, length and mass-balance, and annual
              and monthly runoff components
            'daily_smb': daily mass-balance
            Default: 'monthly'
        calibration_filesuffix: str
            the filesuffix used for all outputs of this datacube creation
        mcs_sampling_settings: dict
            Defines the settings of the Monte Carlo simulation parameter
            sampling. For default values look at signature of
            Calibrator.get_mcs_parameter_sample.
        calibration_parameters_control: dict
            The order of calibration parameters for the control run. Default is
            {"calibrate_param1": "melt_f", "calibrate_param2": "prcp_fac",
            "calibrate_param3": "temp_bias",}
        calibration_parameters_mcs: dict
            The order of calibration parameters for the Monte Carlo simulation
            runs. Default is {"calibrate_param1": "prcp_fac",
            "calibrate_param2": "melt_f", "calibrate_param3": "temp_bias",}
        quantiles: dict
            The quantiles used for the aggregation of the Monte Carlo simulation
            results, including labels for the final datacube. Default is
            {0.05: "5th Percentile", 0.50: "Median", 0.95: "95th Percentile",}
        climate_input_filesuffix: str
            The filesuffix of the climate input file.
        show_log: bool
            If some basic logs should be shown during execution. Default is
            False.
        kwargs: dict
            kwargs to pass to the mb_model_class instance

        Returns
        -------
        xr.Dataset
            The model datacube generated by using the provided data for
            calibration, including uncertainties.

        """

        if not calibration_filesuffix:
            model_name = mb_model_class.__name__
            calibration_filesuffix = f"{model_name}_{ref_mb_period}"

        if climate_input_filesuffix is None:
            if mb_model_class.__name__ in ['DailyTIModel']:
                climate_input_filesuffix = '_daily'
            else:
                climate_input_filesuffix = ''

        # we apply the kwargs to the mb_model class here, to be sure all
        # subsequent tasks use them correctly
        if kwargs:
            mb_model_class = partial(mb_model_class, **kwargs)

        if quantiles is None:
            quantiles = {0.05: "5th Percentile",
                         0.50: "Median",
                         0.95: "95th Percentile",}

        if calibration_parameters_control is None:
            calibration_parameters_control = {
                "calibrate_param1": "melt_f",
                "calibrate_param2": "prcp_fac",
                "calibrate_param3": "temp_bias",
            }

        # define what should be the default settings we use as first guess for
        # control calibration
        gdir.settings_filesuffix = first_guess_settings
        prcp_fac_fg = gdir.settings['prcp_fac']
        melt_f_fg = gdir.settings['melt_f']
        temp_bias_fg = gdir.settings['temp_bias']

        # conduct a control calibration, this is the approach without MCS
        control_filesuffix = f"_{calibration_filesuffix}_control"
        wrkflw_return = workflow.execute_entity_task(
            tasks.mb_calibration_from_scalar_mb,
            gdir,
            settings_filesuffix=control_filesuffix,
            observations_filesuffix=control_filesuffix,
            ref_mb=ref_mb,
            ref_mb_unit=ref_mb_unit,
            ref_mb_err=ref_mb_err,
            ref_mb_period=ref_mb_period,
            overwrite_gdir=True,
            **calibration_parameters_control,
            prcp_fac=prcp_fac_fg,
            melt_f=melt_f_fg,
            temp_bias=temp_bias_fg,
            mb_model_class=mb_model_class,
            filename='climate_historical',
            input_filesuffix=climate_input_filesuffix,
            return_mb_model=True,
        )
        # the finally calibrated mass balance model is returned because of
        # return_mb_model=True
        mb_model_control = wrkflw_return[0][1]

        # now sample parameters of control calibration
        if mcs_sampling_settings is None:
            mcs_sampling_settings = {}
        parameter_sample, parameter_names = self.get_mcs_parameter_sample(
            gdir=gdir, control_filesuffix=control_filesuffix,
            **mcs_sampling_settings)

        # now conduct MCS by execute calibration for each sample, which is
        # currently just a for loop with a try statement (if a combination of
        # parameters fails). Here we maybe can add multiprocessing in the future
        # by using OGGMs @entity_task together with workflow.execute_entity_task
        gdir.observations_filesuffix = control_filesuffix
        ref_mb_control = gdir.observations['ref_mb']

        if show_log:
            print(f"{calibration_filesuffix}:\n"
                  f"  Starting Monte Carlo Simulation for "
                  f"{parameter_sample.shape[0]} ensemble members.")

        # we save successful parameter combinations for later
        working_samples = []
        mb_model_samples = []
        for i, param_val in enumerate(parameter_sample):
            try:
                param_filesuffix = f"_{calibration_filesuffix}_{i}"
                mb_model_tmp = self.mcs_mb_calibration_workflow(
                    gdir,
                    settings_filesuffix=param_filesuffix,
                    settings_parent_filesuffix=control_filesuffix,
                    observations_filesuffix=param_filesuffix,
                    mb_model_class=mb_model_class,
                    filename='climate_historical',
                    input_filesuffix=climate_input_filesuffix,
                    ref_mb_period=ref_mb_control['period'],
                    ref_mb_err=ref_mb_control['err'],
                    ref_mb_unit=ref_mb_control['unit'],
                    ref_mb=param_val[parameter_names.index('ref_mb')],
                    prcp_fac=param_val[parameter_names.index('prcp_fac')],
                    melt_f=param_val[parameter_names.index('melt_f')],
                    temp_bias=param_val[parameter_names.index('temp_bias')],
                    calibration_parameters=calibration_parameters_mcs,
                )
                working_samples.append(param_filesuffix)
                mb_model_samples.append(mb_model_tmp)
            except RuntimeError:
                pass

        if show_log:
            print(f"  Finished Monte Carlo Simulation: {len(working_samples)} "
                  f"ensemble members for output aggregation available.\n"
                  f"  Start generating datacubes")

        datacube_dict = {}
        if isinstance(datacubes_requested, str):
            datacubes_requested = [datacubes_requested]

        for datacube_request in datacubes_requested:
            if show_log:
                print(f"    Start generation of {datacube_request} datacube")

            if datacube_request in ['annual_hydro', 'monthly']:
                # define kwargs which are individual for each run
                all_sample_runs = []
                for sample_filesuffix in working_samples + [control_filesuffix]:
                    all_sample_runs.append(
                        (gdir, dict(
                            settings_filesuffix=sample_filesuffix,
                            output_filesuffix=f"{sample_filesuffix}_{datacube_request}", ))
                    )
            if datacube_request == 'annual_hydro':
                datacube_vars = ['volume', 'area', 'length', 'off_area',
                                 'on_area', 'melt_off_glacier', 'melt_on_glacier',
                                 'liq_prcp_off_glacier', 'liq_prcp_on_glacier',
                                 'snowfall_off_glacier', 'snowfall_on_glacier',
                                 'melt_off_glacier_monthly',
                                 'melt_on_glacier_monthly',
                                 'liq_prcp_off_glacier_monthly',
                                 'liq_prcp_on_glacier_monthly',
                                 'snowfall_off_glacier_monthly',
                                 'snowfall_on_glacier_monthly',
                                 'runoff_monthly', 'runoff', 'specific_mb']

                # run dynamic model with different calibration options
                # kwargs which are the same for all runs, can be provided directly
                workflow.execute_entity_task(
                    tasks.run_with_hydro,
                    all_sample_runs,
                    run_task=tasks.run_from_climate_data,
                    fixed_geometry_spinup_yr=2000,
                    ref_area_yr=2000,
                    ys=2000,
                    store_monthly_hydro=True,
                    mb_model_class=mb_model_class,
                    climate_filename='climate_historical',
                    climate_input_filesuffix=climate_input_filesuffix,
                )
            elif datacube_request == 'monthly':
                datacube_vars = ['volume', 'area', 'length', 'specific_mb']

                # run dynamic model with different calibration options
                # kwargs which are the same for all runs, can be provided directly
                workflow.execute_entity_task(
                    tasks.run_from_climate_data,
                    all_sample_runs,
                    fixed_geometry_spinup_yr=2000,
                    ys=2000,
                    store_monthly_step=True,
                    mb_elev_feedback='monthly',
                    mb_model_class=mb_model_class,
                    climate_filename='climate_historical',
                    climate_input_filesuffix=climate_input_filesuffix,
                )

            # create a xarray datacube including all ensemble members
            if datacube_request in ['annual_hydro', 'monthly']:
                # compile run outputs and merge
                ds_all = []
                # open each ensemble member and add a new coordinate for later
                # merging
                for sample_filesuffix in working_samples + [control_filesuffix]:
                    ds_tmp = utils.compile_run_output(
                        gdir,
                        input_filesuffix=f"{sample_filesuffix}_{datacube_request}")

                    if datacube_request == 'annual_hydro':
                        ds_tmp = self.calculate_and_add_total_runoff(ds_tmp)
                        mb_unit = 'mm w.e. yr-1'
                    else:
                        mb_unit = 'mm w.e. month-1'
                    ds_tmp = self.calculate_and_add_specific_mb(ds_tmp,
                                                                unit=mb_unit)
                    ds_tmp = ds_tmp[datacube_vars]
                    if sample_filesuffix == control_filesuffix:
                        ds_control = ds_tmp.expand_dims(member=["Control"])
                    ds_tmp.coords['sample_name'] = sample_filesuffix
                    ds_tmp = ds_tmp.expand_dims('sample_name')
                    ds_all.append(ds_tmp)

                # merge all ensemble members into a single dataset
                ds_all = xr.combine_by_coords(ds_all, fill_value=np.nan,
                                              combine_attrs="override")
            elif datacube_request == 'daily_smb':
                # define daily float year timeseries
                y1 = int(mb_model_control.ye)
                mb_years = utils.float_years_timeseries(
                    y0=2000, y1=y1, include_last_year=True, monthly=False
                )

                # loop through all ensemble members and get specific mb
                smb_all = []
                fls = gdir.read_pickle('inversion_flowlines')
                for sample_filesuffix, mb_model in zip(
                    working_samples + [control_filesuffix],
                    mb_model_samples + [mb_model_control]
                ):
                    smb_tmp = mb_model.get_specific_mb(
                        fls=fls, year=mb_years, time_resolution="daily")
                    if sample_filesuffix == control_filesuffix:
                        smb_control = smb_tmp
                    smb_all.append(smb_tmp)

                # combine all ensemble members in a dataset for later quantile
                # computation
                ds_all = xr.Dataset(
                    data_vars=dict(specific_mb=(["sample_name", "time"],
                                                np.array(smb_all)),),
                    coords=dict(sample_name=working_samples + [control_filesuffix],
                                time=mb_years),)
                ds_all = self.add_specific_mb_attrs(ds_all,
                                                    unit='mm w.e. day-1')

                ds_control = xr.Dataset(
                    data_vars=dict(specific_mb=(["member", "time"],
                                                [smb_control])),
                    coords=dict(member=["Control"], time=mb_years),
                )
                ds_control = self.add_specific_mb_attrs(ds_control,
                                                        unit='mm w.e. day-1')

            else:
                raise NotImplementedError(f"{datacube_request}")

            # calculate quantiles and rename, this is the same for all datacubes
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="All-NaN slice encountered",
                                        category=RuntimeWarning)
                ds_all = ds_all.quantile(q=list(quantiles.keys()),
                                         dim='sample_name')
            ds_all = ds_all.rename({"quantile": "member"})
            # make sure to match ordering
            labels = [quantiles[float(v)] for v in ds_all.member.values]
            ds_all = ds_all.assign_coords(member=("member", labels))

            # add control run as an extra member
            ds_all = xr.concat([ds_control, ds_all], dim="member")
            ds_all.coords["member"].attrs = {
                "description": "Members include Monte Carlo ensemble percentiles "
                               f"(calculated out of {len(working_samples)} "
                               "ensemble members) and a Control run calibrated "
                               "using median input values."
            }

            datacube_dict[datacube_request] = ds_all

            if show_log:
                print(f"    Finished generation of {datacube_request} datacube")

        if show_log:
            print(f"  Finished generating datacubes\n")

        return datacube_dict

    def get_ref_mb(
            self,
            gdir: GlacierDirectory,
            l1_datacube: xr.Dataset,
            ref_mb_period: str,
            source: str = 'Hugonnet') -> tuple:
        """Get observed mass balance for a specific reference period.

        Parameters
        ----------
        gdir
        l1_datacube: xr.Dataset
            If the reference mass balance should be derived from an observation
            data, which is available in a L1 datacube.
        ref_mb_period: str
            Reference calibration period in the format <YYYY-MM-DD_YYYY-MM-DD>.
        source: str
            Define the observation source

        Returns
        -------
        tuple
            ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period: the reference mass
            balance to match with unit, associated uncertainty and valid period.
        """

        if source == 'Hugonnet':
            pd_geodetic = utils.get_geodetic_mb_dataframe()
            df_ref_mb = pd_geodetic.loc[gdir.rgi_id]
            df_ref_mb = df_ref_mb.loc[df_ref_mb.period == ref_mb_period]
            ref_mb = df_ref_mb.dmdtda.item() * 1000
            ref_mb_unit = "kg m-2 yr-1"
            ref_mb_err = df_ref_mb.err_dmdtda.item() * 1000
            return ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period
        else:
            raise NotImplementedError(f"Source {source} is not implemented.")

    def calibrate(
            self,
            gdir: GlacierDirectory,
            model_matrix: dict,
            l1_datacube: xr.Dataset = None,
            **kwargs
    ) -> dict:
        """Calibrate the mass balance model and create L2 datacubes  using a
        model matrix.

        Parameters
        ----------
        gdir: GlacierDirectory
            Glacier directory.
        model_matrix: dict
            Model parameters for various calibration runs.
        l1_datacube: xr.Dataset
            If the reference mass balance should be derived from an observation
            data, which is available in a L1 datacube.
        kwargs: dict
            kwargs passed on to Clibrator.calibrate_mb_and_create_datacubes

        Returns
        -------
        dict
            Containing the created L2 datacubes for each set of configuration
            parameters in ``model_matrix``
        """
        # Store results
        l2_datacubes = {}

        for matrix_name, model_params in tqdm(model_matrix.items()):
            mb_model_class = model_params["model"]
            ref_mb_period = model_params["ref_mb_period"]
            source = model_params["source"]
            calibration_filesuffix = f"{matrix_name}_{ref_mb_period}"

            ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period = self.get_ref_mb(
                gdir=gdir, l1_datacube=l1_datacube, ref_mb_period=ref_mb_period,
                source=source
            )

            l2_datacubes[matrix_name] = self.calibrate_mb_and_create_datacubes(
                gdir=gdir,
                mb_model_class=mb_model_class,
                ref_mb=ref_mb,
                ref_mb_err=ref_mb_err,
                ref_mb_unit=ref_mb_unit,
                ref_mb_period=ref_mb_period,
                calibration_filesuffix=calibration_filesuffix,
                ** kwargs,
            )

        return l2_datacubes


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

    def get_temporal_bounds(self, dates: np.ndarray,
                            date_start: str,
                            date_end: str
                            ) -> tuple:
        """Get start and end dates of geodetic and observational periods.

        This matches the nearest available dates for irregular time
        indices.

        Parameters
        ----------
        dates : list
            Time index. This may be irregularly spaced.
        date_start : str
            Start year of a desired period.
        date_end : str
            End year of a desired period.

        Returns
        -------
        tuple[datetime]
            Nearest available start and end dates for observations.
        """

        # EOLIS data is in 30-day periods, so get closest available date
        data_start = self.get_nearest_datetime(dates, date_start)
        data_end = self.get_nearest_datetime(dates, date_end)

        return data_start, data_end

    def get_geodetic_mb_from_dataset(
        self,
        dataset: xr.Dataset,
        date_start="2011-01-01",
        date_end="2020-01-01",
    ) -> tuple:
        """Get the geodetic mass balance in kg m-2 from enhanced gridded data.

        Parameters
        ----------
        dataset : xr.Dataset
            CryoTEMPO-EOLIS dataset.
        date_start : str, default "2011-01-01"
            Start of reference period.
        date_end : str, default "2020-01-01"
            End of reference period.

        Returns
        -------
        tuple
            ref_mb, ref_mb_unit, ref_mb_err: the reference mass balance to match
            with unit and associated uncertainty.
        """

        dates = self.get_eolis_dates(dataset)
        data_start, data_end = self.get_temporal_bounds(
            dates=dates, date_start=date_start, date_end=date_end
        )
        ref_mb_period = (f"{data_start.strftime('%Y-%m-%d')}_"
                         f"{data_end.strftime('%Y-%m-%d')}")

        calib_frame = pd.DataFrame(
            {
                "dh": dataset.eolis_elevation_change_timeseries,
                "dh_sigma": dataset.eolis_elevation_change_sigma_timeseries,
            },
            index=dates,
        )

        # ref_mb in kg m-2, area not needed as we already have a mean dh
        # (dh = dV / A)
        bulk_density = 850
        dh = calib_frame["dh"].loc[data_end] - calib_frame["dh"].loc[data_start]
        ref_mb = (dh * bulk_density)
        ref_mb_unit = "kg m-2"

        # for now we just use the sigma of the last year times 0.66
        ref_mb_err = calib_frame["dh_sigma"].loc[data_end] * bulk_density# * 0.66

        return ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period

    def get_ref_mb(
            self,
            gdir: GlacierDirectory,
            l1_datacube: xr.Dataset,
            ref_mb_period: str,
            source: str = 'Hugonnet'
    ) -> tuple:
        """Get observed mass balance for a specific reference period.

        Parameters
        ----------
        gdir
        l1_datacube: xr.Dataset
            If the reference mass balance should be derived from an observation
            data, which is available in a L1 datacube.
        ref_mb_period: str
            Reference calibration period in the format <YYYY-MM-DD_YYYY-MM-DD>.
        source: str
            Define the observation source

        Returns
        -------
        tuple
            ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period: the reference mass
            balance to match with unit, associated uncertainty and valid period.
        """
        if source == "CryoTEMPO-EOLIS":
            dates = ref_mb_period.split('_')
            date_start = dates[0]
            date_end = dates[1]

            ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period = self.get_geodetic_mb_from_dataset(
                dataset=l1_datacube,
                date_start=date_start, date_end=date_end,
            )

            return ref_mb, ref_mb_unit, ref_mb_err, ref_mb_period
        else:
            return super().get_ref_mb(
                gdir=gdir, l1_datacube=l1_datacube, ref_mb_period=ref_mb_period,
                source=source)
