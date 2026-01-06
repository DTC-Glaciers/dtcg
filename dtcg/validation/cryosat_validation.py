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

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns

from oggm import utils

from dtcg.validation.validation_metrics import (
    get_supported_metrics, bootstrap_metric_obs_normal_mdl_quantiles)
from dtcg.validation.validation_plotting import (
    autoscale_y_from_fill_between, add_line_with_unc)


def get_cryosat_data(l1_datacube=None, l2_datacube=None, baseline_date=2011.0):
    """

    Parameters
    ----------
    l1_datacube
    l2_datacube
    baseline_date: float
        baseline date where the cumulative elevation change is starting

    Returns
    -------

    """
    returns = []

    if l1_datacube is not None:
        if 'eolis_elevation_change_timeseries' not in l1_datacube:
            raise ValueError("No Cryosat2 data available.")
        cryosat_elev = l1_datacube.eolis_elevation_change_timeseries
        cryosat_elev_unc = l1_datacube.eolis_elevation_change_sigma_timeseries
        returns.append(cryosat_elev)
        returns.append(cryosat_elev_unc)

    if l2_datacube is not None:
        # get model data
        if 'monthly' not in l2_datacube:
            raise ValueError(
                "For validation with CryoSat2 data, we need a 'monthly' "
                f"datacube. Available are {list(l2_datacube.keys())}.")
        model_volume = l2_datacube['monthly'].volume
        model_elev = ((model_volume - model_volume.sel(time=baseline_date)) /
                      l2_datacube['monthly'].area)
        model_elev = sort_quantiles_keep_control_last(
            model_elev.isel(rgi_id=0).load())
        returns.append(model_elev)

    return returns


def sort_quantiles_keep_control_last(da: xr.DataArray, control_label="Control"
                                     ) -> xr.DataArray:
    # Identify control vs quantile members
    is_control = da["member"].astype(str) == control_label

    # If control not present, just treat everything as quantiles
    if bool(is_control.any()):
        da_control = da.sel(member=control_label)
        da_q = da.sel(member=da["member"].where(~is_control, drop=True))
    else:
        da_control = None
        da_q = da

    # Order quantile labels numerically (these are strings like "0.05")
    q_num = da_q["member"].astype(float)
    q_order = np.argsort(q_num.values)
    da_q = da_q.isel(member=q_order)
    q_labels_sorted = da_q["member"]  # keep the original string labels

    # Sort values along member *within each timestep*
    da_q_sorted_values = xr.apply_ufunc(
        np.sort,
        da_q,
        input_core_dims=[["member"]],
        output_core_dims=[["member"]],
        vectorize=True,
    ).assign_coords(member=q_labels_sorted)

    # Append control at the end (preserving its original values)
    if da_control is not None:
        da_out = xr.concat([da_q_sorted_values,
                            da_control.expand_dims(member=[control_label])],
                           dim="member")
    else:
        da_out = da_q_sorted_values

    return da_out


def validate_with_cryosat(l1_datacube, l2_datacube, return_bootstrap_args=False,
                          baseline_date=2011.0, **kwargs):
    validation_metrics = {}

    # CryoSat2 observations
    if 'eolis_elevation_change_timeseries' in l1_datacube:
        # get validation data
        cryosat_elev, cryosat_elev_unc, model_elev = get_cryosat_data(
            l1_datacube=l1_datacube, l2_datacube=l2_datacube,
            baseline_date=baseline_date)

        # get overlapping years
        # convert cryosat dates into floatyears, use always first of month
        cryosat_dates = cryosat_elev.t.values
        cryosat_floatyrs = utils.date_to_floatyear(
            y=cryosat_dates.astype('datetime64[Y]').astype(int) + 1970,
            m=(cryosat_dates.astype('datetime64[M]').astype(int) % 12) + 1,
            d=1  # (cryosat_dates.astype('datetime64[D]') -
                 #  cryosat_dates.astype('datetime64[M]')).astype(int) + 1
        )
        # common years
        years = np.intersect1d(model_elev.time, cryosat_floatyrs)
        # exclude nan values of model
        model_elev = model_elev.sel(time=years)
        years = years[~np.isnan(model_elev.sel(member='0.5').values)]
        # finally select data
        # obs
        cyrosat_elev_years = cryosat_dates[np.isin(cryosat_floatyrs, years)]
        cryosat_elev = cryosat_elev.sel(t=cyrosat_elev_years).values
        cryosat_elev_unc = cryosat_elev_unc.sel(t=cyrosat_elev_years).values
        # model
        model_elev_q_levels = [float(q) for q in model_elev.member
                               if q != 'Control']
        model_elev_q_levels_sorted = sorted(model_elev_q_levels)
        model_elev_q_levels_sorted_str = [str(q) for q in model_elev_q_levels_sorted]
        model_elev = model_elev.sel(time=years,
                                    member=model_elev_q_levels_sorted_str)

        # conduct the actual calculation of validation metrics
        supported_metrics = get_supported_metrics()

        results = bootstrap_metric_obs_normal_mdl_quantiles(
                     obs_median=cryosat_elev,
                     obs_unc=cryosat_elev_unc,
                     mdl_q_levels=model_elev_q_levels_sorted,
                     mdl_quantiles=model_elev.values,
                     metrics=[supported_metrics[metric]['fct_name']
                              for metric in supported_metrics],
                     **kwargs
                 )

        for i, metric in enumerate(supported_metrics):
            metric_key = metric
            if supported_metrics[metric]['add_unit']:
                metric_key += f" (m)"

            metric_fmt = supported_metrics[metric]['fmt']
            validation_metrics.update({
                metric_key: [
                    f"{results['point_estimate'][i]:{metric_fmt}} "
                    f"({results['ci'][i][0]:{metric_fmt}}, "
                    f"{results['ci'][i][-1]:{metric_fmt}})"]
            })

    if validation_metrics != {}:
        if return_bootstrap_args:
            bootstrap_args = {
                key: results[key]
                for key in ['ci_level', 'n', 'n_boot', 'block_length', 'seed']
            }
            return validation_metrics, bootstrap_args
        else:
            return validation_metrics
    else:
        return None


def plot_cryosat(l1_datacube, datatree, l2_name_list):
    if 'eolis_elevation_change_timeseries' not in l1_datacube:
        raise ValueError("For CryoSat2 plot we need the "
                         "'eolis_elevation_change_timeseries' variable in the "
                         "L1 datacube.")

    color_palette = sns.color_palette("colorblind")

    legend_handles = []
    legend_labels = []

    c_cryosat = color_palette[0]

    fig, ax = plt.subplots()

    # CryoSat2
    cryosat_elev, cryosat_elev_unc = get_cryosat_data(l1_datacube=l1_datacube)
    cryosat_dates = cryosat_elev.t.values
    cryosat_floatyrs = utils.date_to_floatyear(
        y=cryosat_dates.astype('datetime64[Y]').astype(int) + 1970,
        m=(cryosat_dates.astype('datetime64[M]').astype(int) % 12) + 1,
        d=(cryosat_dates.astype('datetime64[D]') -
           cryosat_dates.astype('datetime64[M]')).astype(int) + 1
    )
    add_line_with_unc(ax=ax, x=cryosat_floatyrs, y=cryosat_elev.values,
                      y_unc=cryosat_elev_unc.values, c=c_cryosat,
                      label='CryoSat2', legend_handles=legend_handles,
                      legend_labels=legend_labels, alpha=0.25)

    # L2 datacubes
    for l2_name, c in zip(l2_name_list, color_palette[1:]):
        l2_datacube = datatree[l2_name]

        model_elev = get_cryosat_data(l2_datacube=l2_datacube)[0]

        add_line_with_unc(ax=ax, x=model_elev.time,
                          y=model_elev.sel(member='0.5'),
                          y_unc=[
                              model_elev.sel(member='0.05'),
                              model_elev.sel(member='0.95'),
                          ],
                          c=c, label=f"{l2_name} median with",
                          label_unc=f"5th and 95th percentile",
                          legend_handles=legend_handles,
                          legend_labels=legend_labels,
                          alpha=0.25)

    ax.set_xlim([2011, None])
    # Recompute y-limits based on visible data only
    autoscale_y_from_fill_between(ax)

    ax.set_xlabel('Year')
    ax.set_ylabel('Elevation change (m)');
    ax.legend(legend_handles, legend_labels,
              handler_map={tuple: HandlerTuple()},
              loc='upper center',
              bbox_to_anchor=(0.5, -0.12),
              )
    ax.grid(True, alpha=0.3)
    ax.set_title("Cumulative elevation change")

    return fig
