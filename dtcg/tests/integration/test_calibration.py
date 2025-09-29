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
from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil.tz import UTC
from oggm import utils
from oggm.core import massbalance

import dtcg.integration.calibration as calibration

logger = logging.getLogger(__name__)
pytest_plugins = "oggm.tests.conftest"


class TestCalibrator:
    """Tests OGGM bindings for API queries."""

    # Fixtures
    def get_calibrator(self):
        return calibration.Calibrator()

    @pytest.fixture(name="Calibrator", autouse=False, scope="function")
    def fixture_calibrator(self):
        return self.get_calibrator()

    def test_init_calibrator(self):
        test_calibrator = calibration.Calibrator()
        assert hasattr(test_calibrator, "model_matrix")
        assert isinstance(test_calibrator.model_matrix, dict)

        test_calibrator = calibration.Calibrator(model_matrix={"test_matrix": 0})

        assert hasattr(test_calibrator, "model_matrix")
        assert isinstance(test_calibrator.model_matrix, dict)
        assert test_calibrator.model_matrix == {"test_matrix": 0}

    def test_set_model_matrix(self, Calibrator):

        test_calibrator = Calibrator
        model_matrix = {
            "model": massbalance.DailyTIModel,
            "geo_period": "2000-01-01_2020-01-01",
            "test_kwarg": 100,
        }
        for suffix in range(0, 2):
            matrix_name = f"DailyTI_0{suffix}"
            test_calibrator.set_model_matrix(name=matrix_name, **model_matrix)

        compare_matrix = test_calibrator.model_matrix
        for suffix in range(0, 2):  # ensure keys aren't overwritten
            matrix_name = f"DailyTI_0{suffix}"
            assert matrix_name in compare_matrix.keys()
            assert isinstance(compare_matrix[matrix_name], dict)
            matrix_attrs = compare_matrix[matrix_name]
            assert matrix_attrs["geo_period"] == "2000-01-01_2020-01-01"
            assert "test_kwarg" in matrix_attrs.keys()

    # @pytest.mark.parametrize("arg_period", ["", None])
    # def test_set_model_matrix_no_cfg(self, arg_period, Calibrator):
    #     test_calibrator = Calibrator
    #     msg = "Either set the `geodetic_mb_period` parameter in cfg.PARAMS, or pass it explicitly"
    #     with pytest.raises(InvalidParamsError, match=msg) as excinfo:
    #         test_calibrator.set_model_matrix(
    #             name="FailMatrix",
    #             model=massbalance.DailyTIModel,
    #             geo_period=arg_period,
    #         )

    # def test_get_calibrated_models(self, Calibrator, hef_gdir):
    #     test_calibrator = Calibrator
    #     gdir = hef_gdir

    #     workflow.execute_entity_task(gdirs=[gdir], task=w5e5.process_w5e5_data)
    #     workflow.execute_entity_task(
    #         gdirs=[gdir], task=w5e5.process_w5e5_data, daily=True
    #     )
    #     test_ref_mb = pd.DataFrame()

    #     # Default settings
    #     test_calib, test_flowlines, smb = test_calibrator.get_calibrated_models(
    #         gdir=gdir,
    #         model_class=massbalance.MonthlyTIModel,
    #         ref_mb=test_ref_mb,
    #         geodetic_period="2000-01-01_2020-01-01",
    #     )

    #     for data in [test_calib, test_flowlines, smb]:
    #         assert isinstance(data, dict)
    #     print(test_calib.keys())
    #     assert False

    @pytest.mark.parametrize("arg_rgiid", ["RGI60-11.00897"])
    def test_get_geodetic_mb(self, Calibrator, hef_gdir, arg_rgiid):
        test_calibrator = Calibrator
        gdir = hef_gdir
        # instead of inititialising data for multiple glaciers
        gdir.rgi_id = arg_rgiid
        test_mb = utils.get_geodetic_mb_dataframe()

        if gdir.rgi_id in test_mb.index:
            test_mb = test_mb.loc[gdir.rgi_id]
        else:
            # Construct explicitly as DTCG adds other columns
            test_mb = pd.DataFrame(
                columns=["period", "area", "dmdtda", "err_dmdtda", "reg", "is_cor"]
            ).rename_axis(index="rgiid")

        compare_mb = test_calibrator.get_geodetic_mb(gdir)
        assert isinstance(compare_mb, pd.DataFrame)
        pd.testing.assert_frame_equal(compare_mb, test_mb)

    def test_get_geodetic_mb_missing(self, Calibrator, hef_gdir):
        test_calibrator = Calibrator
        gdir = hef_gdir
        # in case hef_gdir changes - RGI00 may later be used for non-RGI
        # glaciers
        gdir.rgi_id = "RGIXX-00.00.0000"
        # msg = f"No reference mb available for {gdir.rgi_id}."
        with pytest.raises(KeyError, match=f"{gdir.rgi_id}") as excinfo:
            test_calibrator.get_geodetic_mb(gdir)
            print(excinfo)


class TestCalibratorCryotempoEolis(TestCalibrator):

    # Fixtures
    def get_calibrator_cryotempo(self):
        return calibration.CalibratorCryotempo()

    @pytest.fixture(name="Calibrator", autouse=False, scope="function")
    def fixture_calibrator(self):
        return self.get_calibrator_cryotempo()

    def test_init_calibrator(self):
        test_calibrator = calibration.CalibratorCryotempo()
        assert hasattr(test_calibrator, "model_matrix")
        assert isinstance(test_calibrator.model_matrix, dict)

        test_calibrator = calibration.CalibratorCryotempo(
            model_matrix={"test_matrix": 0}
        )

        assert hasattr(test_calibrator, "model_matrix")
        assert isinstance(test_calibrator.model_matrix, dict)
        assert test_calibrator.model_matrix == {"test_matrix": 0}

    def test_set_model_matrix(self, Calibrator):

        # test_period = cfg.PARAMS["geodetic_mb_period"]
        test_calibrator = Calibrator
        model_matrix = {
            "model": massbalance.DailyTIModel,
            "geodetic_mb_period": "2000-01-01_2020-01-01",
            "test_kwarg": 100,
        }
        for suffix in range(0, 2):
            matrix_name = f"DailyTI_0{suffix}"
            test_calibrator.set_model_matrix(name=matrix_name, **model_matrix)

        compare_matrix = test_calibrator.model_matrix
        for suffix in range(0, 2):  # ensure keys aren't overwritten
            matrix_name = f"DailyTI_0{suffix}"
            assert matrix_name in compare_matrix.keys()
            assert isinstance(compare_matrix[matrix_name], dict)
            matrix_attrs = compare_matrix[matrix_name]
            assert matrix_attrs["geodetic_mb_period"] == "2000-01-01_2020-01-01"
            assert "test_kwarg" in matrix_attrs.keys()

            for key in ["source", "daily", "extra_kwargs", "test_kwarg"]:
                assert key in matrix_attrs.keys()

    # @pytest.mark.parametrize("arg_period", ["", None])
    # def test_set_model_matrix_no_cfg(self, arg_period, Calibrator):
    #     return super().test_set_model_matrix_no_cfg(
    #         arg_period=arg_period, Calibrator=Calibrator
    #     )

    @pytest.mark.xfail(
        reason="Issue with OGGM where 'auto_skip_task' not present in cfg.PARAMS"
    )
    @pytest.mark.parametrize("arg_rgiid", ["RGI60-11.00897"])
    def test_get_geodetic_mb(self, Calibrator, hef_gdir, arg_rgiid):
        test_calibrator = Calibrator
        gdir = hef_gdir
        # instead of inititialising data for multiple glaciers
        gdir.rgi_id = arg_rgiid
        test_mb = utils.get_geodetic_mb_dataframe()

        if gdir.rgi_id in test_mb.index:
            test_mb = test_mb.loc[gdir.rgi_id]
        else:
            # Construct explicitly as DTCG adds other columns
            test_mb = pd.DataFrame(
                columns=["period", "area", "dmdtda", "err_dmdtda", "reg", "is_cor"]
            ).rename_axis(index="rgiid")
        test_mb["source"] = "Hugonnet"

        compare_mb = test_calibrator.get_geodetic_mb(gdir)
        assert isinstance(compare_mb, pd.DataFrame)
        pd.testing.assert_frame_equal(compare_mb, test_mb)

    @pytest.mark.xfail(
        reason="Issue with OGGM where 'auto_skip_task' not present in cfg.PARAMS"
    )
    def test_get_geodetic_mb_missing(self, Calibrator, hef_gdir):
        test_calibrator = Calibrator
        gdir = hef_gdir
        return super().test_get_geodetic_mb_missing(
            Calibrator=test_calibrator, hef_gdir=gdir
        )

    @pytest.mark.parametrize(
        "arg_years,expected",
        [
            ((2000, 2004), datetime(2003, 11, 11, tzinfo=UTC)),
            ((2001, 2003), datetime(2003, 1, 15, tzinfo=UTC)),
        ],
    )
    def test_get_temporal_bounds(self, Calibrator, arg_years, expected):
        test_calibrator = Calibrator
        # test_dates = [datetime(i)] for i in
        base_time = datetime(2000, 1, 1, tzinfo=UTC)
        test_dates = [base_time + timedelta(days=x * 30) for x in range(48)]
        assert len(test_dates) == 48

        test_year_start = arg_years[0]
        test_year_end = arg_years[1]

        year_start, year_end, data_start, data_end = (
            test_calibrator.get_temporal_bounds(
                dates=test_dates, year_start=test_year_start, year_end=test_year_end
            )
        )

        assert year_start == datetime(test_year_start, 1, 1, tzinfo=UTC)
        assert year_end == datetime(test_year_end, 1, 1, tzinfo=UTC)
        test_data_start = base_time + timedelta(days=(arg_years[0] - 2000) * 12 * 30)
        assert data_start == test_data_start
        assert data_end == expected
