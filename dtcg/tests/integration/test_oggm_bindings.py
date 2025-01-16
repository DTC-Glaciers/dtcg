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

import pytest

logger = logging.getLogger(__name__)
import geopandas as gpd
import numpy as np
from oggm import cfg, utils

import dtcg.integration.oggm_bindings as integration_ob


class TestOGGMBindings:

    @pytest.mark.parametrize("arg_name", ["rofental", "Rofental"])
    def test_get_rgi_id(self, arg_name):
        ids = integration_ob.get_rgi_id(region_name=arg_name)
        assert isinstance(ids, tuple)
        assert all(isinstance(i, str) for i in ids)

    @pytest.mark.parametrize("arg_name", ["Illegal name", "", None])
    def test_get_rgi_id_missing_key(self, arg_name):
        with pytest.raises((KeyError, TypeError)) as excinfo:
            integration_ob.get_rgi_id(region_name=arg_name)
        assert str(arg_name) in str(excinfo.value)

    def test_get_rgi_file(self, class_case_dir):

        cfg.initialize()
        cfg.PATHS["working_dir"] = class_case_dir

        compare_file = integration_ob.get_rgi_file(region_name="rofental")

        assert isinstance(compare_file, gpd.GeoDataFrame)
        assert not compare_file.empty
        assert not compare_file["RGIId"].empty
        assert (compare_file["O1Region"] == "11").all()

    def test_get_rgi_basin_file(self, class_case_dir):
        cfg.initialize()
        cfg.PATHS["working_dir"] = class_case_dir

        compare_file = integration_ob.get_rgi_basin_file(subregion_name="rofental")

        assert isinstance(compare_file, gpd.GeoDataFrame)
        assert not compare_file.empty
        assert not compare_file["geometry"].empty
        assert (compare_file["HYBAS_ID"] == 2120509820).all()

    def test_get_glaciers_in_subregion(self, class_case_dir):
        cfg.initialize()
        cfg.PATHS["working_dir"] = class_case_dir

        region = gpd.read_file(utils.get_rgi_region_file("11", version="61"))
        subregion = gpd.read_file(utils.get_demo_file("rofental_hydrosheds.shp"))
        compare_file = integration_ob.get_glaciers_in_subregion(
            region=region, subregion=subregion
        )

        assert isinstance(compare_file, gpd.GeoDataFrame)
        assert not compare_file.empty
        assert (compare_file["O1Region"] == "11").all()
        assert compare_file.size <= region.size
        # np.in1d is deprecated
        assert np.isin(compare_file.CenLon.values, region.CenLon.values).all()
