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

import logging

import bokeh.models
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

from datetime import datetime

import numpy as np
from dateutil.tz import UTC

import dtcg.interface.plotting as dtcg_plotting

pytest_plugins = "oggm.tests.conftest"


class TestBokehFigureFormat:

    def get_bokeh_figure_format(self):
        return dtcg_plotting.BokehFigureFormat()

    @pytest.fixture(name="BokehFigureFormat", autouse=False, scope="function")
    def fixture_bokeh_figure_format(self):
        return self.get_bokeh_figure_format()

    def test_init(self):

        test_figure = dtcg_plotting.BokehFigureFormat()
        for attribute_name in ["defaults", "tooltips", "hover_tool"]:
            assert hasattr(test_figure, attribute_name)
        assert isinstance(test_figure.defaults, dict)
        assert isinstance(test_figure.tooltips, list)
        assert isinstance(test_figure.hover_tool, bokeh.models.HoverTool)

    def test_get_default_opts(self, BokehFigureFormat):
        test_figure = BokehFigureFormat
        test_opts = test_figure.get_default_opts()
        assert isinstance(test_opts, dict)
        assert test_opts
        # These are either deprecated, or clash with Overlay/Layout
        invalid_opts = ["scalebar_opts", "css_variables", "apply_hard_bounds"]
        for option in invalid_opts:
            assert option not in test_opts.keys()

    def test_get_all_palettes(self, BokehFigureFormat):
        test_figure = BokehFigureFormat
        palettes = test_figure.get_all_palettes()
        assert isinstance(palettes, dict)
        assert all([isinstance(i, tuple) for i in palettes.values()])

        # clearer than one-liner or itertools
        for palette in palettes.values():
            assert all(isinstance(hex_colour, str) for hex_colour in palette)
            assert all(hex_colour[0] == "#" for hex_colour in palette)
            assert all(len(hex_colour) == 7 for hex_colour in palette)

    @pytest.mark.parametrize(
        "arg_name",
        [
            "brown_blue_pastel",
            "brown_blue_vivid",
            "hillshade_glacier",
            "lines_jet_r",
            "Set1",  # Holoviews
        ],
    )
    def test_get_color_palette(self, BokehFigureFormat, arg_name):
        test_figure = BokehFigureFormat
        palette = test_figure.get_color_palette(name=arg_name)
        assert isinstance(palette, (tuple, list))
        assert all(isinstance(i, str) for i in palette)

    def test_get_color_palette_missing(self, BokehFigureFormat):
        test_figure = BokehFigureFormat
        test_name = "Missing Palette"
        test_palettes = test_figure.get_all_palettes()
        palette_names = "', '".join(test_palettes.keys())
        msg = f"{test_name} not found. Try: {palette_names}"
        with pytest.raises(KeyError, match=msg):
            test_figure.get_color_palette(name=test_name)

    @pytest.mark.parametrize("arg_mode", ["mouse", "vline"])
    def test_get_hover_tool(self, BokehFigureFormat, arg_mode):
        test_figure = BokehFigureFormat
        test_tool = test_figure.get_hover_tool(mode=arg_mode)
        assert isinstance(test_tool, bokeh.models.HoverTool)
        assert test_tool.mode == arg_mode

    def test_set_hover_tool(self, BokehFigureFormat):
        test_figure = BokehFigureFormat
        # default mode is mouse
        test_figure.set_hover_tool(mode="vline")
        assert isinstance(test_figure.hover_tool, bokeh.models.HoverTool)
        assert test_figure.hover_tool.mode == "vline"

    @pytest.mark.parametrize("arg_location", ["", "test location"])
    @pytest.mark.parametrize(
        "arg_timestamp,expected_timestamp",
        [
            ("", None),
            ("2000-01-01", "2000-01-01"),
            (datetime(2000, 1, 1, tzinfo=UTC), "1 January 2000"),
        ],
    )
    @pytest.mark.parametrize("arg_suffix", ["", "some suffix"])
    def test_get_title(
        self,
        BokehFigureFormat,
        arg_location,
        arg_timestamp,
        arg_suffix,
        expected_timestamp,
    ):
        test_figure = BokehFigureFormat
        original_title = "Mass Balance"
        test_title = test_figure.get_title(
            title=original_title,
            suffix=arg_suffix,
            timestamp=arg_timestamp,
            location=arg_location,
        )
        assert isinstance(test_title, str)
        if arg_location:
            assert f"{arg_location}" in test_title
        else:
            assert f"{original_title} at " not in test_title
        if arg_suffix:
            assert f"{arg_suffix}" in test_title

        if arg_timestamp:
            assert f"0{expected_timestamp}" not in test_title
            assert expected_timestamp in test_title


class TestBokehCryotempo:

    def get_bokeh_figure(self):
        return dtcg_plotting.BokehCryotempo()

    @pytest.fixture(name="BokehFigure", autouse=False, scope="function")
    def fixture_bokeh_figure_format(self):
        return self.get_bokeh_figure()

    @pytest.fixture(name="cte_frame", autouse=False, scope="function")
    def fixture_cte_frame(self):
        date_range = pd.date_range("2000-01-01", "2020-12-31", freq="1D", tz=UTC)
        values = np.sin(np.arange(len(date_range)))
        frame = pd.DataFrame(values, columns=["smb"], index=date_range)
        yield frame.copy(deep=True)

    def test_init(self):

        test_figure = dtcg_plotting.BokehCryotempo()
        for attribute_name in ["defaults", "tooltips", "hover_tool", "palette"]:
            assert hasattr(test_figure, attribute_name)

        test_defaults = {
            "ylabel": "Runoff (Mt)",
            "padding": (0.1, (0, 0.1)),
            "ylim": (0, None),
            "autorange": "y",
        }

        compare_defaults = test_figure.defaults
        for k, v in test_defaults.items():
            assert k in compare_defaults.keys()
            assert compare_defaults[k] == v

        # Can't compare directly as IDs are different
        test_yformatter = bokeh.models.PrintfTickFormatter(format="%.2f")
        assert compare_defaults["yformatter"].format == test_yformatter.format

        assert test_figure.tooltips == [("Runoff", "$snap_y{%.2f Mt}")]

        assert test_figure.hover_tool.mode == "vline"
        assert isinstance(test_figure.palette, tuple)

    def test_get_date_mask(self, BokehFigure, cte_frame):
        test_figure = BokehFigure
        test_frame = cte_frame

        start_date = "2003-04-01"
        end_date = "2010-12-11"

        mask = test_figure.get_date_mask(
            dataframe=test_frame, start_date=start_date, end_date=end_date
        )
        assert isinstance(mask, np.ndarray)
        all(isinstance(i, (np.bool, bool)) for i in mask)
        assert len(mask) == len(test_frame)

        filter_frame = test_frame[mask]
        assert filter_frame.index[0] >= datetime.strptime(
            start_date, "%Y-%m-%d"
        ).replace(tzinfo=UTC)
        assert filter_frame.index[-1] <= datetime.strptime(
            end_date, "%Y-%m-%d"
        ).replace(tzinfo=UTC)

    def test_get_mean_by_doy(self, BokehFigure, cte_frame):
        test_figure = BokehFigure
        test_frame = cte_frame

        test_mean = test_figure.get_mean_by_doy(dataframe=test_frame)
        assert isinstance(test_mean, pd.DataFrame)
        assert test_mean.index.name == "doy"
        assert len(test_mean.index)
        np.testing.assert_array_equal(test_mean.index.values, np.arange(1, 367))
        np.testing.assert_array_less(test_mean, 1.0)  # cte_frame is sine function
