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

import dtcg.interface.gateway as interface_gateway

pytest_plugins = "oggm.tests.conftest"


class TestOGGMRequestAPIConstructor:
    attributes = ["query", "subregion_name", "shapefile_path"]

    def test_init_RequestAPIConstructor(self):
        test_constructor = interface_gateway.RequestAPICtor(action="test_action")
        assert isinstance(test_constructor, interface_gateway.RequestAPICtor)
        for attribute_name in self.attributes:
            assert attribute_name in test_constructor.__dict__
        assert isinstance(test_constructor.query, dict)
        assert test_constructor.action == "test_action"


class TestOGGMGateway:

    attributes = ["query", "subregion_name", "shapefile_path"]

    def get_gateway_handler(self):
        return interface_gateway.GatewayHandler

    @pytest.fixture(name="GatewayHandler", autouse=False, scope="function")
    def fixture_gateway_handler(self):
        return self.get_gateway_handler()

    @pytest.mark.xfail(reason="TODO: monkeypatch action")
    def test_set_user_query(self, GatewayHandler):
        test_query = GatewayHandler(query={"action": "test_action"})
        assert test_query.action == "test_action"
        test_query._set_user_query(action="compare_action")
        assert test_query.action == "compare_action"

        assert isinstance(test_query, interface_gateway.RequestAPIConstructor)
        for attribute_name in self.attributes:
            assert attribute_name in test_query.__dict__
        assert isinstance(test_query.query, dict)
        assert test_query.action == "test_action"

    @pytest.mark.xfail(reason="TODO: monkeypatch action")
    def test_set_user_query_kwargs(self, GatewayHandler):
        kwargs = {"action": "compare_action", "subregion_name": "Alps"}
        test_query = GatewayHandler(kwargs)
        test_query._set_user_query(**kwargs)

        for attribute_name in self.attributes:
            assert attribute_name in test_query.__dict__
        assert isinstance(test_query.query, dict)
        assert test_query.action == "compare_action"
        assert test_query.subregion_name == "Alps"
        assert test_query.shapefile_path is None
