"""Copyright 2026 DTCG Contributors

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

import pytest
from pathlib import Path

import dtcg.api.external.call as dtcg_call
from dtcg import DEFAULT_L1_DATACUBE_URL, DEFAULT_L2_DATACUBE_URL


class Test_StreamDatacube:
    """
    Docstring for Test_StreamDatacube

    Attributes
    ----------
    """

    def get_datacube_streamer(self):
        return dtcg_call.StreamDatacube()

    @pytest.fixture(name="DataCubeStreamer", autouse=False, scope="function")
    def fixture_datacube_streamer(self):
        return self.get_datacube_streamer()

    def test_init(self):
        streamer = dtcg_call.StreamDatacube()
        assert isinstance(streamer, dtcg_call.StreamDatacube)
        assert hasattr(streamer, "server")
        # assert hasattr(streamer, "binder")

        assert isinstance(streamer.server, str)
        assert streamer.server == DEFAULT_L1_DATACUBE_URL

        streamer = dtcg_call.StreamDatacube(server="sometestserver")
        assert streamer.server == "sometestserver"

    @pytest.mark.parametrize("arg_layer", ["L1", None])
    def test_stream_datacube(self, DataCubeStreamer, arg_layer):
        streamer = DataCubeStreamer

        datacube = streamer.stream_datacube(glacier="RGI60-06.00372", layer=arg_layer)

        assert datacube

    @pytest.mark.parametrize("arg_suffix", [".zip", ".zarr"])
    def test_get_zip_path(self, DataCubeStreamer, tmp_path, arg_suffix):
        streamer = DataCubeStreamer
        assert isinstance(tmp_path, Path)

        test_rgi_id = "RGI60-06.00372"
        zip_path = streamer.get_zip_path(zip_path=tmp_path, rgi_id=test_rgi_id)
        assert isinstance(zip_path, Path)
        assert test_rgi_id in str(zip_path)
        assert zip_path

    @pytest.mark.parametrize(
        "arg_server", [DEFAULT_L1_DATACUBE_URL, DEFAULT_L2_DATACUBE_URL]
    )
    def test_get_url(self, arg_server):
        streamer = dtcg_call.StreamDatacube(server=arg_server)
        test_rgi_id = "RGI60-06.00372"
        compare_url = streamer.get_url(rgi_id=test_rgi_id)

        assert isinstance(compare_url, str)
        assert compare_url == f"{arg_server}{test_rgi_id}.zarr/"
