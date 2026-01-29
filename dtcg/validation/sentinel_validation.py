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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.legend_handler import HandlerTuple
from oggm import utils

from dtcg.validation.validation_metrics import (
    bootstrap_metric_obs_normal_mdl_quantiles,
    get_supported_metrics,
)
from dtcg.validation.validation_plotting import add_line_with_unc


def get_sentinel_data(l1_datacube=None, l2_datacube=None):

    returns = []

    if l1_datacube is not None:
        if "sfc_type_snowline" not in l1_datacube:
            raise ValueError("No Cryosat2 data available.")
        sentinel_snowline = l1_datacube.sfc_type_snowline
        returns.append(sentinel_snowline)

    if l2_datacube is not None:
        # get model data
        if "daily_smb" not in l2_datacube:
            raise ValueError(
                "For validation with Sentinel2 data, we need a 'daily_smb' "
                f"datacube. Available are {list(l2_datacube.keys())}."
            )
        if "snowline" not in l2_datacube["daily_smb"]:
            raise ValueError(
                "For validation with Sentinel2 data, we need a modelled "
                "snowline in the 'daily_smb' datacube. This is only available "
                "when using a mass balance model which includes snow tracking."
            )
        model_snowline = l2_datacube["daily_smb"].snowline
        returns.append(model_snowline)

    return returns


def set_infs(var):
    # handle special cases of fully snowcovered and fully snowfree,
    # those values are defined as the highest or lowest elevation of observation
    # or modelling domain
    inf_values = eval(var.attrs["inf_values"], {"np": np, "inf": np.inf})
    if np.inf in inf_values:
        var = xr.where(~np.isposinf(var), var, inf_values[np.inf])
    if -np.inf in inf_values:
        var = xr.where(~np.isneginf(var), var, inf_values[-np.inf])
    return var


def validate_with_sentinel(
    l1_datacube,
    l2_datacube,
    validation_period=None,
    return_bootstrap_args=False,
    **kwargs,
):
    validation_metrics = {}
    used_period = None

    if validation_period is not None:
        raise NotImplementedError()

    # Sentinel2 observations
    if "sfc_type_snowline" in l1_datacube:
        # get validation data
        sentinel_snowline, model_snowline = get_sentinel_data(
            l1_datacube=l1_datacube, l2_datacube=l2_datacube
        )

        # get overlapping years
        # convert model dates into datetime64
        model_floatyears_all = model_snowline.time
        yrs, mnths, dys = utils.floatyear_to_date(model_floatyears_all, return_day=True)
        model_dates = np.array(
            [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(yrs, mnths, dys)],
            dtype="datetime64[D]",
        )

        def get_model_floatyears(dates):
            return model_floatyears_all[np.isin(model_dates, dates)]

        # common dates
        common_dates = np.intersect1d(model_dates, sentinel_snowline.t_sfc_type)
        # exclude nan values of model
        model_snowline = model_snowline.sel(time=get_model_floatyears(common_dates))
        common_dates = common_dates[~np.isnan(model_snowline.sel(member="0.5").values)]
        # exclude nan values of obseravation
        sentinel_snowline = sentinel_snowline.sel(t_sfc_type=common_dates)
        common_dates = common_dates[
            ~np.isnan(sentinel_snowline.sel(snowcover_frac=0.5).values)
        ]
        # finally select data
        # obs
        sentinel_snowline = sentinel_snowline.sel(t_sfc_type=common_dates)
        # model
        model_snowline_q_levels = [
            float(q) for q in model_snowline.member if q != "Control"
        ]
        model_snowline_q_levels_sorted = sorted(model_snowline_q_levels)
        model_snowline_q_levels_sorted_str = [
            str(q) for q in model_snowline_q_levels_sorted
        ]
        model_snowline = model_snowline.sel(
            time=get_model_floatyears(common_dates),
            member=model_snowline_q_levels_sorted_str,
        )

        # save actual used period for label
        used_yrs, used_months, used_days = utils.floatyear_to_date(
            [
                get_model_floatyears(common_dates)[0],
                get_model_floatyears(common_dates)[-1],
            ],
            return_day=True,
        )
        used_period = (
            f"{used_yrs[0]}-{used_months[0]:02d}-{used_days[0]:02d}_"
            f"{used_yrs[1]}-{used_months[1]:02d}-{used_days[1]:02d}"
        )

        # conduct the actual calculation of validation metrics
        supported_metrics = get_supported_metrics()

        results = bootstrap_metric_obs_normal_mdl_quantiles(
            obs_median=set_infs(sentinel_snowline).values.T,
            obs_unc=sentinel_snowline.snowcover_frac.values,
            mdl_q_levels=model_snowline_q_levels_sorted,
            mdl_quantiles=set_infs(model_snowline).values.T,
            interpret_obs_as_quantiles=True,
            metrics=[
                supported_metrics[metric]["fct_name"] for metric in supported_metrics
            ],
            **kwargs,
        )

        for i, metric in enumerate(supported_metrics):
            metric_key = metric
            if supported_metrics[metric]["add_unit"]:
                metric_key += f" (m)"

            metric_fmt = supported_metrics[metric]["fmt"]
            validation_metrics.update(
                {
                    metric_key: [
                        f"{results['point_estimate'][i]:{metric_fmt}} "
                        f"({results['ci'][i][0]:{metric_fmt}}, "
                        f"{results['ci'][i][-1]:{metric_fmt}})"
                    ]
                }
            )

    if validation_metrics != {}:
        if return_bootstrap_args:
            bootstrap_args = {
                key: results[key]
                for key in ["ci_level", "n", "n_boot", "block_length", "seed"]
            }
            return validation_metrics, used_period, bootstrap_args
        else:
            return validation_metrics, used_period
    else:
        return None


def plot_sentinel(l1_datacube, datatree, l2_name_list, x_lim=None, dh_special=20):
    if "sfc_type_snowline" not in l1_datacube:
        raise ValueError(
            "For Sentinel2 plot we need the 'sfc_type_snowline' "
            "variable in the L1 datacube."
        )

    # define a x_range for demonstration
    if x_lim is None:
        x_lim = [np.array("2015-01-01", dtype="datetime64[D]"), None]
    else:
        y, m, d = utils.floatyear_to_date(x_lim, return_day=True)
        x_lim = np.array(
            [f"{y[0]}-{m[0]:02d}-{d[0]:02d}", f"{y[1]}-{m[1]:02d}-{d[1]:02d}"],
            dtype="datetime64[D]",
        )

    # open all data
    sentinel_snowline = get_sentinel_data(l1_datacube=l1_datacube)[0]
    model_snowlines = {}
    for l2_name in l2_name_list:
        l2_datacube = datatree[l2_name]
        try:
            model_snowlines[l2_name] = get_sentinel_data(l2_datacube=l2_datacube)[0]
        except ValueError:
            # if no snowline data is available
            continue

    if model_snowlines == {}:
        raise ValueError("No L2 datacubes including a snowline available!")

    # set one common value for special cases fully snow cover and fully snow free
    inf_values = eval(sentinel_snowline.attrs["inf_values"], {"np": np, "inf": np.inf})
    for l2_name in model_snowlines:
        model_inf_values = eval(
            model_snowlines[l2_name].attrs["inf_values"], {"np": np, "inf": np.inf}
        )

        if np.inf in model_inf_values:
            if model_inf_values[np.inf] > inf_values[np.inf]:
                inf_values[np.inf] = model_inf_values[np.inf]

        if -np.inf in model_inf_values:
            if model_inf_values[-np.inf] < inf_values[-np.inf]:
                inf_values[-np.inf] = model_inf_values[-np.inf]

    # plotting
    color_palette = sns.color_palette("colorblind")

    legend_handles = []
    legend_labels = []

    c_sentinel = "black"

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    def set_infs(var):
        var = np.where(np.isposinf(var), inf_values[np.inf], var)
        return np.where(np.isneginf(var), inf_values[-np.inf], var)

    # define observations with errorbars
    snowline_obs = sentinel_snowline.sel(snowcover_frac=0.5).values
    snowline_obs_lower = sentinel_snowline.sel(snowcover_frac=0.25).values
    snowline_obs_upper = sentinel_snowline.sel(snowcover_frac=0.75).values
    snowline_err = np.array(
        tuple(
            zip(
                set_infs(snowline_obs) - set_infs(snowline_obs_lower),
                set_infs(snowline_obs_upper) - set_infs(snowline_obs),
            )
        )
    ).T
    # convert datetime64 to floatyear
    snowline_dates = sentinel_snowline.t_sfc_type.values.astype("datetime64[D]")
    # plot observations
    line = ax.errorbar(
        snowline_dates,
        set_infs(snowline_obs),
        yerr=snowline_err,
        fmt=".",
        ecolor=c_sentinel,
        c=c_sentinel,
        capsize=2,
        lw=0.5,
        zorder=10,
    )
    legend_handles.append(line)
    legend_labels.append(
        "Sentinel2, snowcover area fraction of 50% (dot) and 25% and 75% " "(errorbar)"
    )

    # L2 datacubes
    for l2_name, c in zip(model_snowlines, color_palette[1:]):
        model_snowline = model_snowlines[l2_name]

        yrs, mnths, dys = utils.floatyear_to_date(model_snowline.time, return_day=True)
        model_dates = np.array(
            [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(yrs, mnths, dys)],
            dtype="datetime64[D]",
        )

        add_line_with_unc(
            ax=ax,
            x=model_dates,
            y=set_infs(model_snowline.sel(member="0.5")),
            y_unc=[
                set_infs(model_snowline.sel(member="0.05")),
                set_infs(model_snowline.sel(member="0.95")),
            ],
            c=c,
            label=f"{l2_name} median with",
            label_unc=f"5th and 95th percentile",
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            alpha=0.25,
        )

    ax.set_xlim(x_lim)
    # Recompute y-limits based on visible data only
    ax.set_ylim([inf_values[-np.inf] - dh_special, inf_values[np.inf] + dh_special])

    # add special areas fully snow covered and fully snow free
    y_lims = ax.get_ylim()
    line = ax.axhspan(
        y_lims[0],
        inf_values[-np.inf] + dh_special,
        color="lightblue",
        alpha=0.7,
        zorder=0,
    )
    legend_handles.insert(0, line)
    legend_labels.insert(0, "Completely snow covered")
    line = ax.axhspan(
        inf_values[np.inf] - dh_special, y_lims[1], color="red", alpha=0.4, zorder=0
    )
    legend_handles.insert(0, line)
    legend_labels.insert(0, "Completely snow free")

    ax.set_xlabel("Year")
    ax.set_ylabel("Altitude (m)")
    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple()},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(True, alpha=0.3)
    ax.set_title("Snowline Altitude")

    return fig
