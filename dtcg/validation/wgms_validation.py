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
from matplotlib.legend_handler import HandlerTuple

from dtcg.validation.validation_metrics import (
    bootstrap_metric_obs_normal_mdl_quantiles,
    get_supported_metrics,
)
from dtcg.validation.validation_plotting import (
    add_line_with_unc,
    autoscale_y_from_fill_between,
)


def get_annual_data_wgms(l1_datacube=None, l2_datacube=None):

    returns = []

    if l1_datacube is not None:
        if "wgms_mb" not in l1_datacube:
            raise ValueError("No annual WGMS data available.")
        wgms_mb = l1_datacube.wgms_mb
        wgms_mb_unc = l1_datacube.wgms_mb_unc
        returns.append(wgms_mb)
        returns.append(wgms_mb_unc)

    if l2_datacube is not None:
        # get model data
        if "annual_hydro" not in l2_datacube:
            raise ValueError(
                "For validation with annual wgms data, we need a 'annual_hydro' "
                f"datacube. Available are {list(l2_datacube.keys())}."
            )
        model_mb = l2_datacube["annual_hydro"].specific_mb.isel(rgi_id=0)
        returns.append(model_mb)

    return returns


def validate_with_wgms(
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

    # Annual WGMS observations
    if "wgms_mb" in l1_datacube:
        # get validation data
        wgms_mb, wgms_mb_unc, model_mb = get_annual_data_wgms(
            l1_datacube=l1_datacube, l2_datacube=l2_datacube
        )

        # get overlapping years
        # common years
        years = np.intersect1d(wgms_mb.t_wgms, model_mb.time)
        # exclude nan values of model
        model_mb = model_mb.sel(time=years)
        years = years[~np.isnan(model_mb.sel(member="0.5").values)]
        # finally select data
        # obs
        wgms_mb = wgms_mb.sel(t_wgms=years).values
        wgms_mb_unc = wgms_mb_unc.sel(t_wgms=years).values
        # model
        model_mb_q_levels = [float(q) for q in model_mb.member if q != "Control"]
        model_mb_q_levels_sorted = sorted(model_mb_q_levels)
        model_mb_q_levels_sorted_str = [str(q) for q in model_mb_q_levels_sorted]
        model_mb = model_mb.sel(time=years, member=model_mb_q_levels_sorted_str)

        # save actual used period for label
        used_period = f"{years[0]}-{years[-1]}"

        # conduct the actual calculation of validation metrics
        supported_metrics = get_supported_metrics()

        results = bootstrap_metric_obs_normal_mdl_quantiles(
            obs_median=wgms_mb,
            obs_unc=wgms_mb_unc,
            mdl_q_levels=model_mb_q_levels_sorted,
            mdl_quantiles=model_mb.values.T,
            metrics=[
                supported_metrics[metric]["fct_name"] for metric in supported_metrics
            ],
            **kwargs,
        )

        for i, metric in enumerate(supported_metrics):
            metric_key = metric
            if supported_metrics[metric]["add_unit"]:
                metric_key += f" ({model_mb.units})"

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


def plot_wgms_annual(l1_datacube, datatree, l2_name_list):
    if "wgms_mb" not in l1_datacube:
        raise ValueError(
            "For annual WGMS plot we need the 'wgms_mb' variable " "in the L1 datacube."
        )

    color_palette = sns.color_palette("colorblind")

    legend_handles = []
    legend_labels = []

    c_wgms = color_palette[0]

    fig, ax = plt.subplots()

    # wgms
    wgms_mb, wgms_mb_unc = get_annual_data_wgms(l1_datacube=l1_datacube)
    add_line_with_unc(
        ax=ax,
        x=wgms_mb.t_wgms,
        y=wgms_mb.values,
        y_unc=wgms_mb_unc.values,
        c=c_wgms,
        label="WGMS",
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        alpha=0.25,
    )

    # L2 datacubes
    for l2_name, c in zip(l2_name_list, color_palette[1:]):
        l2_datacube = datatree[l2_name]

        model_mb = get_annual_data_wgms(l2_datacube=l2_datacube)[0]

        add_line_with_unc(
            ax=ax,
            x=model_mb.time,
            y=model_mb.sel(member="0.5"),
            y_unc=[
                model_mb.sel(member="0.05"),
                model_mb.sel(member="0.95"),
            ],
            c=c,
            label=f"{l2_name} median with",
            label_unc=f"5th and 95th percentile",
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            alpha=0.25,
        )

    ax.set_xlim([1999.5, None])
    # Recompute y-limits based on visible data only
    autoscale_y_from_fill_between(ax)

    ax.set_xlabel("Year")
    ax.set_ylabel("Specific MB (mm w.e. yr-1)")
    ax.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple()},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.grid(True, alpha=0.3)
    ax.set_title("Annual Specific Mass-Balance")

    return fig
