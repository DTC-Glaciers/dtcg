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

import sys
from datetime import datetime

import bokeh.models
import bokeh.plotting
import geoviews as gv
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter
from dateutil.tz import UTC

hv.extension("bokeh")


class BokehFigureFormat:
    """Regulates plot annotations and formatting for Bokeh plots.

    This also works for Holoviews when using Bokeh as a backend.
    """

    def __init__(self):
        super().__init__()
        self.defaults = self.get_default_opts()
        self.tooltips = []
        self.hover_tool = bokeh.models.HoverTool()

    def check_holoviews(self):
        if "holoviews" not in sys.modules:
            raise SystemError("Holoviews is not installed.")

    def initialise_formatting(
        self, figure: bokeh.plotting.figure
    ) -> bokeh.plotting.figure:
        """Initialise default formatting for Bokeh figures.

        Called separately from other plotting functions in this module
        to avoid overwriting on-the-fly formatting changes.

        Parameters
        ----------
        figure : bokeh.plotting.figure
            An unformatted Bokeh plot.
        """
        default_options = self.get_default_opts()
        figure = figure.opts(default_options)

        return figure

    def set_formatting_kwargs(self, figure: bokeh.plotting.figure, **kwargs):
        """Set kwargs for generic formatting.

        Applies annotations, titles, labels.

        Parameters
        ----------
        figure : bokeh.plotting.figure
            An unformatted Bokeh plot.
        """
        figure = figure.opts(**kwargs)

        return figure

    def get_title(
        self, title: str, suffix: str = "", timestamp: str = "", location: str = ""
    ) -> str:
        """Get a title including optional parameters.

        Parameters
        ----------
        title : str
            Title of figure.
        suffix : str
            Comment appended to returned title. Default empty string.
        timestamp : str, datetime.datetime, optional
            Timestamp. Default empty string.
        location : str
            Location. Default empty string.

        Returns
        -------
        str
            A title optionally including location, timestamp, and a suffix.
        """
        if location:
            title = f"{title} at {location}"
        if timestamp:
            if not isinstance(timestamp, str):
                timestamp = timestamp.strftime("%d %B %Y")
            title = f"{title},\n{timestamp}"
        if suffix:
            title = f"{title},\n{suffix}"

        return title

    def get_default_opts(self) -> dict:
        """Default options for Bokeh figures.

        Returns
        -------
        dict
            Default options for Bokeh figures. Not necessarily
            compatible with Holoviews or Geoviews objects.
        """
        default_options = {
            "aspect": 2,
            "active_tools": ["pan", "wheel_zoom"],
            # "fontsize": {"title": 18},
            "fontscale": 1.2,
            # "yformatter": NumeralTickFormatter(format="0.00 N"),
            "bgcolor": "white",
            "backend_opts": {"title.align": "center", "toolbar.autohide": True},
            "scalebar_opts": {"padding": 5},
            # "apply_hard_bounds":True,
            "show_frame": False,
            "margin": 0,
            "border": 0,
        }

        return default_options

    def get_color_palette(self, name: str) -> tuple:
        """Get a preset palette.

        Parameters
        ----------
        name : str
            Name of palette.

        Returns
        -------
        tuple[str]
            Palette of hex colours.
        """
        palettes = {
            "brown_blue_pastel": ("#e0beb3", "#b3d5e0", "#beacf6"),
            "brown_blue_vivid": ("#f6beac", "#ace4fc", "#beacf6"),
            "hillshade_glacier": ("#f6beac", "#ffffff", "#33b5cb"),
            "lines_jet_r": ("#ffffff", "#d62728", "#1f77b4"),
        }
        if name.lower() not in palettes.keys():
            try:
                palettes[name] = list(hv.Cycle.default_cycles[name])
            except:
                raise KeyError("{name} not found. Try:{'\n'.join(palettes.keys())}")

        return palettes[name]

    def get_hover_tool(self, mode="mouse") -> bokeh.models.HoverTool:
        """Get the hover tool attribute.

        Returns
        -------
        bokeh.models.HoverTool
            Hover tool with tooltips and formatter.
        """
        return bokeh.models.HoverTool(
            tooltips=self.tooltips, formatters=self.get_tooltip_format(), mode=mode
        )

    def set_hover_tool(self, mode: str = "mouse"):
        """Set the hover tool attribute.

        Parameters
        ----------
        mode : str
            Hover mode, e.g. "vline". Default "mouse".
        """
        self.hover_tool = bokeh.models.HoverTool(
            tooltips=self.tooltips, formatters=self.get_tooltip_format(), mode=mode
        )

    def set_tooltips(self, tooltips: list, mode: str = "mouse"):
        """Set the tooltips attribute and update the hover tool.

        Parameters
        ----------
        tooltips : list
            List of Bokeh figure tooltips. The tooltips should only
            contain printf formatters.
        mode : Hover mode for hover tool, e.g. "vline". Default "mouse".
        """
        self.tooltips = tooltips
        self.set_hover_tool(mode=mode)

    def get_tooltip_format(self) -> dict:
        """Get a tooltip formatter from the tooltips attribute.

        This ensures all tooltips with a format string (in curly
        braces) are formatted as fstrings. Skips tooltips without a
        format string.

        Returns
        -------
        dict
            Tooltip field names with an associated "printf" value.
        """
        # formatters = {}
        # for tooltip in self.tooltips:
        #     label = tooltip[1].split("{")
        #     if len(label) > 1:
        #         formatters[tooltip[1]] = "printf"

        # one-liner
        formatters = {label[1].split("{")[0]: "printf" for label in self.tooltips}
        return formatters

    def set_defaults(self, updated_options: dict):
        """Set and overwrite default options for Bokeh figures.

        Parameters
        ----------
        updated_options : dict
            New key/value pairs which will overwrite the default options.
        """
        self.defaults.update(updated_options)

    def get_help_button(
        self, text: list, position: str = "right", add_colors=None
    ) -> bokeh.models.HelpButton:
        """Get an interactive help button from a list of fields

        Parameters
        ----------
        text : list[str]
            Help text.
        position : str
            Tooltip position.
        add_colors : hv.Cycle, default None
            Add coloured boxes next to text. Used for legend entries.

        Returns
        -------
        bokeh.models.HelpButton
            An interactive help button that displays text on hover and
            click.
        """
        if add_colors:
            text = [
                f"<div style='color:{i};font-size:2em;display:inline'>&#9632;</div> {j}"
                for i, j in zip(add_colors.values[: len(text)], text)
            ]
        help_text = "<br />".join([i for i in text])

        help_text = bokeh.models.dom.HTML(help_text)
        tooltip = bokeh.models.Tooltip(content=help_text, position=position)
        help_button = bokeh.models.HelpButton(tooltip=tooltip)

        return help_button


class BokehMap(BokehFigureFormat):
    """Plot data as a map.

    Attributes
    ----------
    tooltips : list
        A list of tooltips corresponding to a polygon's metadata.
    hover_tool : bokeh.models.HoverTool
        Hover tool for Bokeh figures.
    palette : tuple[str]
        Colour palette.
    defaults : dict
        Default options for all Bokeh figures.
    """

    def __init__(self):
        super().__init__()

        self.tooltips = [
            ("Name", "@Name"),
            ("RGI ID", "@RGIId"),
            ("GLIMS ID", "@GLIMSId"),
            ("Area", "@Area{%0.2f km²}"),
            ("Max Elevation", "@Zmax{%.2f m}"),
            ("Min Elevation", "@Zmin{%.2f m}"),
            ("Latitude", "@CenLat{%.2f °N}"),
            ("Longitude", "@CenLon{%.2f °E}"),
        ]
        self.set_hover_tool()
        self.hover_tool = self.get_hover_tool()
        self.palette = self.get_color_palette("hillshade_glacier")
        self.set_defaults({"xlabel": "Longitude (°E)", "ylabel": "Latitude (°N)"})

    def plot_shapefile(self, shapefile, **kwargs) -> gv.Polygons:
        """Plot a shapefile.

        Parameters
        ----------
        shapefile : geopandas.GeoDataFrame
            Must contain polygon geometry.

        **kwargs
            Extra arguments for plotting Polygons. View
            ``gv.help(gv.Polygons)`` for all styling options.
        """
        plot = gv.Polygons(shapefile).opts(**kwargs)

        return plot

    def plot_subregion(
        self,
        shapefile,
        glacier_data,
        subregion_name: str,
    ) -> hv.Overlay:
        """Plot a subregion with all its glaciers.

        Parameters
        ----------
        shapefile : geopandas.GeoDataFrame
            Subregion shapefile.
        glacier_data: geopandas.GeoDataFrame
            Glacier outlines.
        subregion_name : str
            Name of subregion.

        Returns
        -------
        hv.Overlay
            Interactive figure of all glaciers in a subregion.
        """

        mask = shapefile.id == subregion_name
        region_name = shapefile[mask]["name"].values[0]
        title = self.get_title(title=f"{region_name} Basin")
        yformat = NumeralTickFormatter(format="0.00a")

        overlay = (
            (
                self.plot_shapefile(
                    shapefile[mask],
                    fill_color=self.palette[0],
                    line_color="black",
                    line_width=1.0,
                    fill_alpha=0.4,
                    color_index=None,
                    scalebar=True,  # otherwise won't appear in overlay
                    # tools=["hover"],
                    # yformatter=NumeralTickFormatter(format="0.00 N"),
                )
                * gv.tile_sources.EsriWorldHillshadeDark()
                * self.plot_shapefile(
                    glacier_data,
                    fill_color=self.palette[1],
                    line_color="black",
                    line_width=0.3,
                    fill_alpha=0.9,
                    color_index=None,
                    tools=[self.hover_tool],
                )
            )
            .opts(**self.defaults)
            .opts(
                yformatter=yformat,
                title=title,
                tools=[self.hover_tool],
            )
        )

        return overlay

    def plot_glacier_highlight(
        self, glacier_data, name: str, ax: hv.Overlay
    ) -> hv.Overlay:
        """Highlight a glacier.

        This overlays a highlighted colour and does not modify the
        figure contained in ax.

        Parameters
        ----------
        glacier_data : geopandas.GeoDataFrame
        name : str
            Name of glacier.
        ax : hv.Overlay
            Figure containing multiple glaciers.

        Returns
        -------
        hv.Overlay
            All glaciers with a single highlighted glacier.
        """
        overlay = ax * self.plot_shapefile(
            glacier_data[glacier_data["Name"] == name],
            fill_color=self.palette[2],
            line_color="black",
            line_width=0.8,
            fill_alpha=0.4,
            color_index=None,
        ).opts(
            **self.defaults,
            # scalebar=True,
            tools=[self.hover_tool],
        )

        return overlay

    def plot_subregion_with_glacier(
        self, shapefile, glacier_data, subregion_name: str, glacier_name: str = ""
    ) -> hv.Overlay:
        """Plot a subregion, its glaciers, and optional highlight.

        Parameters
        ----------
        shapefile: geopandas.GeoDataFrame
            Shapefile of all subregions.
        glacier_data: geopandas.GeoDataFrame
            Glacier data.
        subregion_name : str
            Name of subregion.
        glacier_name : str, optional
            Name of glacier. If empty, no glacier is highlighted.
            Defaults to empty string.

        Returns
        -------
        hv.Overlay
            Map of subregion with glacier outlines, and optionally a
            highlighted glacier.
        """
        overlay = self.plot_subregion(
            shapefile=shapefile,
            glacier_data=glacier_data,
            subregion_name=subregion_name,
        )
        if glacier_name:
            fig_basin_highlight = self.plot_glacier_highlight(
                glacier_data=glacier_data, name=glacier_name, ax=overlay
            )
            overlay = overlay * fig_basin_highlight

        return overlay


class BokehGraph(BokehFigureFormat):
    """Wrangle data and plot with Bokeh.

    Attributes
    ----------

    defaults : dict
        Default options for all Bokeh figures.
    tooltips : list
        A list of tooltips corresponding to a polygon's metadata.
    hover_tool : bokeh.models.HoverTool
        Hover tool for Bokeh figures.
    palette : tuple
        Colour palette.
    """

    def __init__(self):
        super().__init__()

        self.set_defaults(
            {
                "ylabel": "Runoff (Mt)",
                "yformatter": PrintfTickFormatter(format="%.2f"),
                "padding": (0.1, (0, 0.1)),
                "ylim": (0, None),
                "autorange": "y",
            }
        )
        self.tooltips = [
            ("Runoff", "$y{%.2f Mt}"),
        ]
        self.set_hover_tool(mode="vline")
        self.palette = self.get_color_palette("lines_jet_r")

    def set_time_constraint(self, dataset, nyears: int = 20):
        """Set a dataset's time period to a specific number of years.

        Parameters
        ----------
        dataset : Any
            Data with a time index in years.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        Any
            Data for the given time period up to the most recent
            available year.
        """
        if isinstance(dataset, (xr.DataArray, xr.Dataset)):
            if len(dataset) > nyears:
                dataset = dataset.isel(time=slice(-nyears, None))
        elif isinstance(dataset, (pd.DataFrame, pd.Series)):
            if len(dataset) > nyears:
                dataset = dataset.iloc[-nyears:]
        elif isinstance(dataset, (np.ndarray, list, tuple)):
            if len(dataset) > nyears:
                dataset = dataset[-nyears:]
        else:
            raise TypeError(f"{type(dataset)} not supported.")

        return dataset

    def get_time_index(self, data: xr.DataArray, end_date: int):
        index = [datetime.date(end_date, i, 1) for i in data.index.values]
        return index

    def plot_annual_runoff(
        self, runoff: xr.Dataset, name: str = "", nyears=20
    ) -> hv.Curve:
        """Plot the annual runoff of a glacier or basin.

        Parameters
        ----------
        runoff : xr.Dataset
            Annual runoff data.
        name : str, optional
            Glacier or basin name.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        hv.Curve
            Time series of annual runoff for a given number of years.
        """
        runoff = self.set_time_constraint(dataset=runoff, nyears=nyears).sum(axis=1)
        title = self.get_title(title="Total Annual Runoff", suffix=name)
        curve = (
            hv.Curve(runoff, group="runoff")
            .opts(**self.defaults)
            .opts(
                title=title,
                xlabel="Year",
                xformatter=f"%d",
                line_color="black",
                line_width=0.8,
                tools=[self.hover_tool],
            )
        )

        return curve

    def plot_monthly_runoff(
        self,
        runoff: xr.DataArray,
        runoff_year_min: int,
        runoff_year_max: int,
        name: str = "",
        nyears: int = 20,
    ) -> hv.Overlay:
        """Plot climatology of monthly runoff cycles.

        Parameters
        ----------
        runoff : xr.DataArray
            Runoff data.
        runoff_year_min : int
            Year of minimum runoff.
        runoff_year_max: int
            Year of maximum runoff.
        name : str, optional
            Glacier or basin name.
        nyears : int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        hv.Overlay
            Time series of minimum, maximum, and climatological mean
            monthly runoff cycles.
        """
        runoff = self.set_time_constraint(dataset=runoff, nyears=nyears)
        latest_year = int(runoff.time.values[-1])
        mean_runoff = runoff.mean(dim="time")
        index = [datetime.date(latest_year, i, 1) for i in mean_runoff.month_2d.values]

        title = self.get_title(
            title=f"Mean Monthly Runoff Cycles ({int(runoff.time.values[0])} - {latest_year})",
            # timestamp=f"{int(runoff.time.values[0])} - {latest_year}",
        )

        overlay = (
            (
                hv.Curve(
                    (index, mean_runoff), group="runoff", label=f"{nyears}-year mean"
                ).opts(
                    color="black",
                    line_dash="dashed",
                    tools=[self.hover_tool],
                )
                * hv.Curve(
                    (index, runoff.sel(time=runoff_year_max)),
                    label=f"Maximum: {runoff_year_max}",
                    group="runoff",
                ).opts(
                    color=self.palette[1],
                )
                * hv.Curve(
                    (index, runoff.sel(time=runoff_year_min)),
                    label=f"Minimum: {runoff_year_min}",
                    group="runoff",
                ).opts(color=self.palette[2])
            )
            .opts(**self.defaults)
            .opts(
                shared_axes=False,
                title=title,
                ylabel="Runoff (Mt)",
                xlabel="Month",
                xformatter=bokeh.models.DatetimeTickFormatter(months="%b"),
                legend_position="top_right",
                autorange="y",
                tools=["xwheel_zoom", "xpan", self.hover_tool],
                # tools=[self.hover_tool],
                fixed_bounds=True,
            )
        )
        return overlay

    def plot_runoff_stack(
        self, runoff: xr.DataArray, name: str = "", nyears: int = 20
    ) -> hv.Overlay:
        """Plot a stack of annual runoff split by source.

        Parameters
        ----------
        runoff : xr.DataArray
            Annual runoff data.
        name: str, optional
            Name of glacier. Defaults to empty string.
        nyears: int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        hv.Overlay
            Stack plot of annual runoff split by source.
        """
        title = self.get_title(title="Annual Runoff", location=name)
        palette = hv.Cycle(self.get_color_palette("Paired"))
        runoff = self.set_time_constraint(dataset=runoff, nyears=nyears)

        legend_parser = {
            "melt_off_glacier": (
                "melt off-glacier",
                "Snow melt on areas that are now glacier-free",
            ),
            "melt_on_glacier": (
                "melt on-glacier",
                "Ice and seasonal snow melt on the glacier",
            ),
            "liq_prcp_off_glacier": (
                "precip off-glacier",
                "Liquid precipitation off-glacier",
            ),
            "liq_prcp_on_glacier": (
                "precip on-glacier",
                "Liquid precipitation on glacier",
            ),
        }
        help_button = self.get_help_button(
            text=[i[1] for i in legend_parser.values()],
            position="left",
            add_colors=palette,
        )

        overlay = hv.Overlay(
            [
                hv.Area(runoff[key], label=legend_parser[key][0], group="runoff").opts(
                    color=palette
                )
                for key in runoff.keys()
            ]
        )
        overlay = (hv.Area.stack(overlay).opts(**self.defaults)).opts(
            title=title,
            xlabel="Year",
            xformatter=f"%d",
            legend_position="bottom_left",
            tools=["xwheel_zoom", "xpan", self.hover_tool],
            active_tools=["xwheel_zoom"],
            fixed_bounds=True,
            # apply_hard_bounds=True,
            # hooks = [self.plot_limits],
            legend_opts={
                "elements": [help_button],
                "orientation": "vertical",
                "css_variables": {"font-size": "1em", "display": "inline"},
            },
        )

        return overlay

    def plot_limits(self, plot, element):
        plot.handles["x_range"].min_interval = 0
        plot.handles["x_range"].max_interval = 20
        plot.handles["y_range"].min_interval = 0
        plot.handles["y_range"].max_interval = 40

    def plot_mass_balance(
        self, mass_balance, observations, nyears: int = 20, name: str = ""
    ) -> hv.Overlay:
        """Plot specific mass balance and WGMS observations.

        Parameters
        ----------
        mass_balance: xr.DataSet
            Specific mass balance data.
        observations : xr.DataSet
            WGMS observations.
        name: str, optional
            Name of glacier. Defaults to empty string.
        nyears: int, default 20
            Time period in years. The end date is always the most
            recent available year.

        Returns
        -------
        hv.Overlay
            Time series of specific mass balance and WGMS observations.
        """

        observations = self.set_time_constraint(observations, nyears=nyears)
        mass_balance = self.set_time_constraint(mass_balance, nyears=nyears)

        try:
            if isinstance(observations, list):
                observations = next(item for item in observations if item is not None)
            if isinstance(mass_balance, list):
                mass_balance = next(item for item in mass_balance if item is not None)
        except StopIteration:
            return None
        index = [datetime.date(i, 1, 1) for i in observations.index.values]
        title = self.get_title(title="Mass Balance")

        defaults = self.defaults
        defaults.update({"ylim": (None, None)})

        overlay = (
            (
                hv.Curve((index, mass_balance), label="OGGM").opts(
                    line_color=self.palette[1],
                    line_width=0.8,
                )
                * hv.Curve(
                    (index, observations.ANNUAL_BALANCE / 1000), label="WGMS"
                ).opts(
                    line_color="black",
                    line_width=0.8,
                    line_dash="dashed",
                    tools=[self.hover_tool],
                )
            )
            .opts(**defaults)
            .opts(
                shared_axes=False,
                title=title,
                ylabel="Specific Mass Balance (m w.e.)",
                xlabel="Year",
                xformatter=bokeh.models.DatetimeTickFormatter(months="%Y"),
                legend_position="bottom_left",
                tools=[self.hover_tool],
                legend_opts={"orientation": "vertical"},
            )
        )

        return overlay


class BokehCryotempo(BokehFigureFormat):
    """Wrangle data and plot with Bokeh.

    Attributes
    ----------

    defaults : dict
        Default options for all Bokeh figures.
    tooltips : list
        A list of tooltips corresponding to a polygon's metadata.
    hover_tool : bokeh.models.HoverTool
        Hover tool for Bokeh figures.
    palette : tuple
        Colour palette.
    """

    def __init__(self):
        super().__init__()

        self.set_defaults(
            {
                "ylabel": "Runoff (Mt)",
                "yformatter": PrintfTickFormatter(format="%.2f"),
                "padding": (0.1, (0, 0.1)),
                "ylim": (0, None),
                "autorange": "y",
            }
        )
        self.tooltips = [
            ("Runoff", "$y{%.2f Mt}"),
        ]
        self.set_hover_tool(mode="vline")
        self.palette = self.get_color_palette("lines_jet_r")

    def get_date_mask(self, dataframe: pd.DataFrame, start_date: str, end_date: str):
        date_mask = (
            datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
            <= dataframe.index
        ) & (
            dataframe.index
            <= datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC)
        )
        return date_mask

    def get_mean_by_doy(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return (
            dataframe.groupby([dataframe.index.day_of_year])
            .mean()
            .rename_axis(index=["doy"])
        )

    def add_curve_to_figures(
        self,
        figures: list,
        data: dict,
        key: str,
        label: str = "",
        line_width=0.8,
        **kwargs,
    ) -> list:
        if not label:
            label = self.get_label_from_key(key)
        curve = hv.Curve(data[key], label=label).opts(line_width=line_width, **kwargs)
        figures.append(curve)
        return figures

    def get_label_from_key(self, key: str) -> str:

        key_split = key.split("_")
        model_name = key_split[0]
        if "Sfc" in model_name:
            model_name = f"{model_name.removesuffix('Sfc')}, Tracking"
        label = f"{model_name}, {key_split[1]}"

        if len(key_split) > 4:
            if key_split[2] != "month":
                label = f"{label} ({key_split[2]})"
        else:
            years = [i.split("-")[0] for i in key_split[-2:]]
            geo_period = "-".join(years)
            label = f"{label} ({geo_period})"
        return label

    def get_eolis_dates(self, ds):
        return np.array([datetime.fromtimestamp(t, tz=UTC) for t in ds.t.values])

    def get_eolis_mean_dh(self, ds):
        mean_time_series = [
            np.nanmean(elevation_change_map.where(ds.glacier_mask == 1))
            for elevation_change_map in ds.eolis_gridded_elevation_change
        ]
        return np.array(mean_time_series)

    def plot_mb_comparison(
        self,
        smb: dict,
        years=None,
        glacier_name: str = "",
        geodetic_period: str = "2000-01-01_2020-01-01",
        ref_year: int = 2015,
        datacube=None,
        gdir=None,
        resample: bool = False,
    ):
        """Plot daily SMB for a specific year and geodetic mean.

        Parameters
        ----------
        smb : dict
            Specific mass balance data for various OGGM models.
        years : list, default None
            OGGM output years.
        glacier_name : str, default empty string
            Name of glacier.
        geodetic_period : str, default "2000-01-01_2020-01-01"
            Period over which to take the mean daily specific mass balance.
        ref_year : int, default 2015
            Reference year.
        datacube : xr.DataArray, default None
            CryoTEMPO-EOLIS observations for elevation change.
        gdir : GlacierDirectory, default None
            Glacier of interest.
        resample : bool, default False
            If True, resample observations to begin on the first day of the
            month.
        """
        self.check_holoviews()

        plot_data = {}
        figures = []

        if years is None:
            years = np.arange(1979, 2020)
        geodetic_period = geodetic_period.split("_")
        start_year = geodetic_period[0][:4]
        end_year = geodetic_period[1][:4]

        plot_dates_day = pd.date_range(
            f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="1D", tz=UTC
        )

        if datacube:
            if not gdir:
                raise ValueError("Provide a glacier directory.")
            cryotempo_dates = self.get_eolis_dates(datacube)
            cryotempo_dh = self.get_eolis_mean_dh(datacube)

            df = pd.DataFrame(cryotempo_dh, columns=["smb"], index=cryotempo_dates)
            if resample:
                df = df.resample("1MS").mean()

            date_mask = self.get_date_mask(
                df, f"{ref_year}-01-01", f"{ref_year+1}-01-01"
            )
            df = df[date_mask]
            if not df.empty:
                df["smb"] = (
                    1000 * (df["smb"] - df["smb"].iloc[0]) * 850 / gdir.rgi_area_km2
                )

                df_daily_mean = self.get_mean_by_doy(df)
                df_daily_mean.index = pd.to_datetime(df_daily_mean.index, format="%j")
                plot_data["CryoTEMPO-EOLIS Observations"] = df_daily_mean["smb"] / 30

                label = f"CryoTEMPO-EOLIS Observations ({ref_year})"
                curve = hv.Curve(
                    plot_data["CryoTEMPO-EOLIS Observations"], label=label
                ).opts(line_width=1.0, color="grey", line_dash="dotted")
                figures.append(curve)

        for k, v in smb.items():
            if "Daily" in k:
                label = self.get_label_from_key(k)

                df = pd.DataFrame(v, columns=["smb"], index=plot_dates_day)

                geodetic_mask = self.get_date_mask(df, *geodetic_period)

                df_daily_mean = self.get_mean_by_doy(df[geodetic_mask])
                df_daily_mean.index = pd.to_datetime(df_daily_mean.index, format="%j")
                plot_data[k] = df_daily_mean["smb"]

                label = f"{start_year}-{end_year} Mean"
                figures = self.add_curve_to_figures(
                    data=plot_data, key=k, figures=figures, line_color="k", label=label
                )

                date_mask = self.get_date_mask(
                    df, f"{ref_year}-01-01", f"{ref_year+1}-01-01"
                )

                df_daily_mean = self.get_mean_by_doy(df[date_mask])
                df_daily_mean.index = pd.to_datetime(df_daily_mean.index, format="%j")
                plot_data[k] = df_daily_mean["smb"]
                label = f"{ref_year}"
                figures = self.add_curve_to_figures(
                    data=plot_data,
                    key=k,
                    figures=figures,
                    line_color="#d62728",
                    label=label,
                )

        default_opts = self.get_default_opts()
        if glacier_name:
            glacier_name = f"{glacier_name}, "
        overlay = (
            hv.Overlay(figures)
            .opts(**default_opts)
            .opts(
                aspect=4,
                ylabel="Daily SMB (mm w.e.)",
                title=f"Daily Specific Mass Balance\n {glacier_name}{start_year}-{end_year}",
                xlabel="Month",
                # xformatter=f"%j",
                xformatter=bokeh.models.DatetimeTickFormatter(months="%B"),
                tools=["xwheel_zoom", "xpan"],
                active_tools=["xwheel_zoom"],
                legend_position="top",
                legend_opts={
                    "orientation": "vertical",
                    "css_variables": {"font-size": "1em", "display": "inline"},
                },
            )
        )
        layout = (
            hv.Layout([overlay])
            .cols(1)
            .opts(sizing_mode="stretch_width", shared_axes=False)
        )
        return layout


class HoloviewsDashboard(BokehFigureFormat):
    """Holoviews dashboard for runoff data.

    Attributes
    ----------
    plot_map : BokehMap
    plot_graph : BokehGraph
    title : str
    figures : list
    dashboard : hv.Layout
    """

    def __init__(self):
        super().__init__()

        self.plot_map = BokehMap()
        self.plot_graph = BokehGraph()
        self.title = "Dashboard"
        self.figures = []
        self.dashboard = hv.Layout()

    def set_dashboard_title(self, name: str = "", subregion_name: str = "") -> str:
        if name and subregion_name:
            location = f"{name} ({subregion_name})"
        elif subregion_name:
            location = subregion_name
        elif name:
            location = name
        else:
            location = ""
        self.title = self.get_title(title=location)

    def set_layout(self, figures: list) -> hv.Layout:
        """Compose Layout from a sequence of overlays or layouts.

        Dynamically adds a sequence of overlays to a layout.

        Parameters
        ----------
        figures : list[hv.Overlay|hv.Layout]
            A sequence of figures.
        """
        # columns = len(figures)
        if isinstance(figures, list):
            layout = figures[0]
            if len(figures) > 1:
                layout = figures
            layout = hv.Layout(layout).cols(2)
        else:
            layout = hv.Layout([figures])

        layout = layout.opts(sizing_mode="stretch_both")

        return layout

    def set_dashboard(self, figures: list):
        """Set dashboard from a sequence of figures.

        Parameters
        ----------
        figures : list[hv.Overlay|hv.Layout]
            A sequence of figures.
        """
        self.dashboard = self.set_layout(figures=figures).opts(
            shared_axes=False,
            title=self.title,
            fontsize={"title": 18},
            sizing_mode="scale_both",
            merge_tools=False,
        )

    def plot_runoff_dashboard(
        self,
        data,
        subregion_name,
        glacier_name: str = "",
    ) -> hv.Layout:
        """Plot a dashboard showing runoff data.

        Parameters
        ----------
        data : dict
            Contains glacier data, shapefile, and optionally runoff
            data and observations.
        subregion_name : str
            Name of subregion.
        glacier_name : str, optional
            Name of glacier in subregion. Default empty string.

        Returns
        -------
        hv.Layout
            Dashboard showing a map of the subregion and runoff data.
        """
        fig_basin_selection = self.plot_map.plot_subregion_with_glacier(
            shapefile=data["shapefile"],
            glacier_data=data["glacier_data"],
            subregion_name=subregion_name,
            glacier_name=glacier_name,
        )
        self.figures = fig_basin_selection

        if "runoff_data" in data.keys():
            runoff_data = data["runoff_data"]
            if runoff_data is not None:
                # fig_annual_runoff = self.plot_graph.plot_annual_runoff(
                #     runoff=runoff_data["annual_runoff"], name=glacier_name
                # )
                fig_runoff_stack = self.plot_graph.plot_runoff_stack(
                    runoff=runoff_data["annual_runoff"], name=glacier_name
                )
                fig_monthly_runoff = self.plot_graph.plot_monthly_runoff(
                    runoff_data["monthly_runoff"],
                    runoff_data["runoff_year_min"],
                    runoff_data["runoff_year_max"],
                    name=glacier_name,
                )

                self.figures = [
                    fig_basin_selection,
                    fig_runoff_stack,
                    fig_monthly_runoff,
                ]
            if "wgms" in runoff_data.keys():
                observations = runoff_data["wgms"]
                mass_balance = runoff_data["mass_balance"]
                # TODO: Bug where different types get passed to Curve
                if observations is not None and mass_balance is not None:
                    fig_mass_balance = self.plot_graph.plot_mass_balance(
                        mass_balance=mass_balance,
                        observations=observations,
                        name=glacier_name,
                    )
                    self.figures.append(fig_mass_balance)

        if glacier_name:
            self.set_dashboard_title(name=glacier_name)
        self.set_dashboard(figures=self.figures)
        return self.dashboard
