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

Plotting utilities for the frontend.
"""

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FormatStrFormatter


def get_title(title: str, suffix: str = "") -> str:
    """Get a concatenated title."""
    if suffix:
        title = f"{title} for {suffix}"
    return title


def get_colour_palette(name: str) -> tuple:
    """Get a preset colourmap."""

    palettes = {
        "brown_blue_pastel": ("#e0beb3", "#b3d5e0"),
        "brown_blue_vivid": ("#f6beac", "#ace4fc"),
    }

    if name.lower() not in palettes.keys():
        raise KeyError("{name} not found. Try:{'\n'.join(palettes.keys())}")
    return palettes[name]


def set_time_constraint(dataset, nyears: int = 20):
    """Set a dataset's time period to a specific number of years."""
    if isinstance(dataset, (xr.DataArray, xr.Dataset)):
        if len(dataset) > nyears:
            dataset = dataset.isel(time=slice(-nyears, None))
    elif isinstance(dataset, (pd.DataFrame, pd.Series, list, tuple)):
        if len(dataset) > nyears:
            dataset = dataset.iloc[-nyears:]
    elif isinstance(dataset, np.ndarray):
        if len(dataset) > nyears:
            dataset = dataset[-nyears:]
    else:
        raise TypeError(f"{type(dataset)} not supported.")

    return dataset


def plot_annual_runoff(runoff: xr.Dataset, name: str = "", nyears=20, ax=None):
    """Plot the annual runoff of a glacier or basin.

    Parameters
    ----------
    runoff : xr.Dataset
        Annual runoff data.
    name : str
        Glacier or basin name.
    nyears : int
        Number of years to plot from present.
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = plt.gcf()
    # fig, ax = plt.subplots(figsize=(10, 3.5), sharex=True)
    runoff = set_time_constraint(dataset=runoff, nyears=nyears).sum(axis=1)
    ax = runoff.plot(ax=ax, color="k", lw=0.8)
    ax.set_ylabel("Runoff (Mt)")
    ax.set_xlabel("Years")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))

    title = get_title(title="Total Annual Runoff", suffix=name)
    ax.set_title(title)
    ax.margins(x=0)

    return fig, ax


def plot_monthly_runoff(
    runoff: xr.DataArray,
    runoff_year_min: int,
    runoff_year_max: int,
    nyears: int = 20,
    name: str = "",
    ax=None,
):
    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = plt.gcf()
    runoff = set_time_constraint(dataset=runoff, nyears=nyears)
    cmap = plt.cm.get_cmap("jet_r")(np.linspace(0, 1, 2))
    ax.set_prop_cycle("color", list(cmap))
    runoff.sel(time=[runoff_year_max, runoff_year_min]).plot(
        hue="time",
        label=[f"Maximum: {runoff_year_max}", f"Minimum: {runoff_year_min}"],
        lw=0.8,
        ax=ax,
    )
    mean_runoff = runoff.mean(dim="time")
    mean_runoff.plot(
        label=[f"{nyears}-year mean"], color="black", ls="--", lw=0.8, ax=ax
    )

    title = get_title(title="Mean Monthly Runoff Cycle", suffix=name)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    # ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    # ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b" ))
    ax.legend()
    ax.set_ylabel("Runoff (Mt)")
    ax.margins(x=0)

    fig = plt.gcf()
    return fig, ax


def plot_runoff_partitioning(
    runoff: xr.DataArray, name: str = "", nyears: int = 20, ax=None
):
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = plt.gcf()
    runoff = set_time_constraint(dataset=runoff, nyears=nyears)

    runoff.plot.area(ax=ax, color=sns.color_palette("Paired"))
    ax.set_xlabel("Years")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_ylabel("Runoff (Mt)")
    title = get_title(title="Annual Runoff", suffix=name)
    ax.set_title(title)
    ax.margins(x=0)
    ax.legend(loc="lower left", framealpha=1)

    return fig, ax


def plot_basin_selection(basin_shapefile, glacier_data, subregion_name: str, ax=None):

    colours = get_colour_palette("brown_blue_vivid")

    mask = basin_shapefile.id == subregion_name
    region_name = basin_shapefile[mask]["name"].values[0]
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = plt.gcf()
    basin_shapefile[mask].plot(
        facecolor=colours[0], alpha=0.4, edgecolor="k", lw=1.0, figsize=(10, 10), ax=ax
    )
    glacier_data.plot(
        ax=ax,
        label="RGI outlines",
        color=colours[1],
        zorder=6,
        edgecolor="black",
        lw=0.3,
        alpha=0.8,
    )
    glacier_data.plot(ax=ax)
    ax.set_title(f"{region_name} Basin")
    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")
    ax.grid(alpha=0.3, ls="--")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f E"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f N"))

    fig = plt.gcf()
    fig.suptitle(f"Dashboard for {region_name}", fontsize=24)
    return fig, ax


def plot_glacier_highlight(glacier_data, name: str, ax):
    glacier_data[glacier_data["Name"] == name].plot(
        ax=ax,
        label=name,
        zorder=7,
        color="#beacf6",
        edgecolor="black",
        lw=0.8,
        alpha=0.9,
    )
    fig = plt.gcf()
    return fig, ax


def plot_mass_balance(
    mass_balance, observations, ax=None, nyears: int = 20, name: str = ""
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    observations = set_time_constraint(observations, nyears=nyears)
    mass_balance = set_time_constraint(mass_balance, nyears=nyears)
    ax.plot(observations.index, mass_balance, label="OGGM", color="r", lw=0.8)
    ax.plot(
        observations.ANNUAL_BALANCE / 1000, label="WGMS", color="k", lw=0.8, ls="--"
    )
    ax.legend()
    title = get_title("Specific Mass Balance", name)
    ax.set_xlabel("Years")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_ylabel("Specific Mass Balance (m w.e.)")
    title = get_title(title="Mass Balance", suffix=name)
    ax.set_title(title)
    ax.margins(x=0)
    ax.legend(loc="lower left", framealpha=1)

    fig = plt.gcf()
    return fig, ax


def plot_runoff_dashboard(
    response,
    subregion_name,
    basin_shapefile,
    annual_runoff,
    monthly_runoff,
    min_runoff_year,
    max_runoff_year,
    observations=None,
    mass_balance=None,
    name: str = "",
):

    if observations is not None and mass_balance is not None:
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        plot_mass_balance(
            mass_balance=mass_balance,
            observations=observations,
            ax=axes[2][0],
            name=name,
        )

    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    plot_annual_runoff(runoff=annual_runoff, name=name, ax=axes[0][0])
    plot_runoff_partitioning(runoff=annual_runoff, name=name, ax=axes[1][0])
    plot_monthly_runoff(
        monthly_runoff, min_runoff_year, max_runoff_year, name=name, ax=axes[1][1]
    )
    plot_basin_selection(
        basin_shapefile=basin_shapefile,
        glacier_data=response,
        subregion_name=subregion_name,
        ax=axes[0][1],
    )
    if name:
        fig.suptitle(f"Dashboard for {name}", fontsize=24)
        plot_glacier_highlight(glacier_data=response, name=name, ax=axes[0][1])

    return fig, axes
