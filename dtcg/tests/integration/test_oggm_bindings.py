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

import itertools
import logging

import geopandas as gpd
import numpy as np
import pytest
from oggm import GlacierDirectory, cfg, utils

import dtcg.integration.oggm_bindings as integration_ob

logger = logging.getLogger(__name__)
pytest_plugins = "oggm.tests.conftest"


class TestOGGMBindings:
    """Tests OGGM bindings for API queries."""

    # Fixtures
    def get_oggm_model(self):
        return integration_ob.BindingsOggmModel()

    @pytest.fixture(name="OggmModel", autouse=False, scope="function")
    def fixture_oggm_model(self):
        return self.get_oggm_model()

    def get_oggm_wrangler(self):
        return integration_ob.BindingsOggmWrangler()

    @pytest.fixture(name="OggmWrangler", autouse=False, scope="function")
    def fixture_oggm_wrangler(self):
        return self.get_oggm_wrangler()

    def _get_sample_region_file(
        self, rgi_region: str = "11", rgi_version: str = "62", reset: bool = False
    ):
        """Get a sample region file."""

        sample_gdirs = gpd.read_file(
            utils.get_rgi_region_file(rgi_region, version=rgi_version, reset=reset)
        )

        return sample_gdirs

    def test_get_sample_region_file(self):
        rgi_file = self._get_sample_region_file(
            rgi_region="11", rgi_version="62", reset=False
        )

        assert isinstance(rgi_file, gpd.GeoDataFrame)
        assert (rgi_file["O1Region"] == "11").all()

    @pytest.fixture(name="sample_region_file", scope="function", autouse=False)
    def fixture_get_sample_region_file(self):
        return self._get_sample_region_file()

    def test_get_rgi_metadata(self, OggmModel):
        metadata = OggmModel.get_rgi_metadata(from_web=True)
        assert isinstance(metadata, list)
        for row in metadata:
            assert isinstance(row, dict)

    @pytest.mark.parametrize("arg_name", ["alps", "Alps"])
    def test_get_rgi_region_codes(self, OggmModel, arg_name):
        region_codes = OggmModel.get_rgi_region_codes(subregion_name=arg_name)
        assert isinstance(region_codes, set)
        assert all(isinstance(i, tuple) for i in region_codes)
        assert all(
            isinstance(i, str) for i in itertools.chain.from_iterable(region_codes)
        )

    @pytest.mark.parametrize("arg_name", ["Illegal name"])
    def test_get_rgi_region_codes_missing(self, OggmModel, arg_name):
        msg = f"No regions or subregion matching {arg_name}"

        with pytest.raises((KeyError, TypeError, AttributeError), match=msg) as excinfo:
            OggmModel.get_rgi_region_codes(subregion_name=arg_name)
        assert str(arg_name) in str(excinfo.value)

    @pytest.mark.parametrize("arg_name", ["", None, 127831])
    def test_get_rgi_region_codes_incorrect_type(self, OggmModel, arg_name):
        # msg = f"{arg_name} is not a string."
        with pytest.raises((TypeError)) as excinfo:
            OggmModel.get_rgi_region_codes(subregion_name=arg_name)
        # assert str(arg_name) in str(excinfo.value)

    @pytest.mark.parametrize("arg_name", ["alps", "Alps"])
    def test_get_matching_region_codes(self, OggmModel, arg_name):
        compare_region = OggmModel.get_matching_region_codes(subregion_name=arg_name)

        assert isinstance(compare_region, set)
        for element in itertools.chain.from_iterable(compare_region):
            assert isinstance(element, str)
            # assert len(element) == 2
        for codes in compare_region:
            assert codes[0] == "11"
            assert codes[1] == "1"

    @pytest.mark.parametrize("arg_name", ["Illegal name"])
    def test_get_matching_region_codes_missing(self, OggmModel, arg_name):
        msg = f"No region found for {arg_name}"

        with pytest.raises((KeyError, AttributeError), match=msg) as excinfo:
            OggmModel.get_matching_region_codes(subregion_name=arg_name)
        assert str(arg_name) in str(excinfo.value)

    @pytest.mark.parametrize("arg_name", ["", None])
    def test_get_matching_region_codes_incorrect_type(self, OggmModel, arg_name):
        msg = "No valid region or subregion name supplied."
        with pytest.raises((ValueError, TypeError), match=msg) as excinfo:
            OggmModel.get_matching_region_codes(subregion_name=arg_name)
        # assert str(arg_name) in str(excinfo.value)

    def test_get_rgi_files_from_subregion(self, OggmModel, class_case_dir):

        cfg.initialize()
        cfg.PATHS["working_dir"] = class_case_dir

        compare_files = OggmModel.get_rgi_files_from_subregion(subregion_name="Alps")

        # for frame in compare_files:
        #     assert isinstance(frame, gpd.GeoDataFrame)
        assert isinstance(compare_files, gpd.GeoDataFrame)
        assert not compare_files.empty
        assert not compare_files["RGIId"].empty
        assert (compare_files["O1Region"] == "11").all()
        assert (compare_files["O2Region"] == "1").all()

    def test_get_shapefile_from_web(self, OggmWrangler, class_case_dir):
        cfg.initialize()
        cfg.PATHS["working_dir"] = class_case_dir

        compare_file = OggmWrangler.get_shapefile_from_web(
            shapefile_name="rofental_hydrosheds.shp"
        )

        assert isinstance(compare_file, gpd.GeoDataFrame)
        assert not compare_file.empty
        assert not compare_file["geometry"].empty
        assert (compare_file["HYBAS_ID"] == 2120509820).all()

    def test_get_glaciers_in_subregion(
        self, OggmWrangler, class_case_dir, sample_region_file
    ):
        cfg.initialize()
        cfg.PATHS["working_dir"] = class_case_dir
        region = sample_region_file

        subregion = gpd.read_file(utils.get_demo_file("rofental_hydrosheds.shp"))
        compare_file = OggmWrangler.get_glaciers_in_subregion(
            region=region, subregion=subregion
        )

        assert isinstance(compare_file, gpd.GeoDataFrame)
        assert not compare_file.empty
        assert (compare_file["O1Region"] == "11").all()
        assert compare_file.size <= region.size
        # np.in1d is deprecated
        assert np.isin(compare_file.CenLon.values, region.CenLon.values).all()

    # def test_get_glacier_directories(self, class_case_dir, patch_data_urls):
    #     cfg.initialize()
    #     cfg.PATHS["working_dir"] = class_case_dir

    #     region = gpd.read_file(utils.get_rgi_region_file("11", version="61"))
    #     subregion = gpd.read_file(utils.get_demo_file("rofental_hydrosheds.shp"))
    #     test_glaciers = integration_ob.get_glaciers_in_subregion(
    #         region=region, subregion=subregion
    #     )

    #     compare_gdirs = integration_ob.get_glacier_directories(test_glaciers)
    #     assert isinstance(compare_gdirs, gpd.GeoDataFrame)
    #     assert not compare_gdirs.empty
    #     assert (compare_gdirs["O1Region"] == "11").all()

    def test_get_outline_details(self, outline_shapefile, OggmWrangler):
        test_shapefile = outline_shapefile
        binder = OggmWrangler
        details = binder.get_outline_details(polygon=test_shapefile.iloc[0])

        keys = [
            "Name",
            "RGI ID",
            "GLIMS ID",
            "Area",
            "Max Elevation",
            "Min Elevation",
            "Latitude",
            "Longitude",
            "Outline Date",
        ]
        assert isinstance(details, dict)
        for key in keys:
            assert key in details.keys()
            assert isinstance(details[key], dict)
            assert isinstance(details[key]["value"], str)
            assert isinstance(details[key]["unit"], str)

        assert details["Name"]["value"] == "Hintereisferner"
