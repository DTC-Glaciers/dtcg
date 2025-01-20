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

import itertools

logger = logging.getLogger(__name__)
import geopandas as gpd
import numpy as np
from oggm import cfg, utils

import dtcg.interface.gateway as interface_gateway

pytest_plugins = "oggm.tests.conftest"


class TestOGGMRequestAPIConstructor:
    attributes = ["query", "region_name", "shapefile_path"]

    def test_init_RequestAPIConstructor(self):
        test_constructor = interface_gateway.RequestAPIConstructor(query="test_action")
        assert isinstance(test_constructor, interface_gateway.RequestAPIConstructor)
        for attribute_name in self.attributes:
            assert attribute_name in test_constructor.__dict__
        assert isinstance(test_constructor.query, str)
        assert test_constructor.query == "test_action"


class TestOGGMGateway:

    attributes = ["query", "region_name", "shapefile_path"]

    def test_set_user_query(self):
        test_query = interface_gateway._set_user_query(query="test_action")

        assert isinstance(test_query, interface_gateway.RequestAPIConstructor)
        for attribute_name in self.attributes:
            assert attribute_name in test_query.__dict__
        assert test_query.query == "test_action"

    def test_set_user_query_kwargs(self):
        kwargs = {"query": "test_action", "region_name": "Alps"}
        test_query = interface_gateway._set_user_query(**kwargs)

        for attribute_name in self.attributes:
            assert attribute_name in test_query.__dict__
        assert test_query.query == "test_action"
        assert test_query.region_name == "Alps"
        assert test_query.shapefile_path is None
