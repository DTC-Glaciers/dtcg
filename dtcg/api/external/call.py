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

Create and read DTCG requests.
"""

import dtcg.integration.oggm_bindings as oggm_bindings
import xarray as xr


class StreamDatacube:

    def __init__(self):
        self.binder = oggm_bindings.BindingsCryotempo()
        self.binder.init_oggm()
        self.server = "https://cluster.klima.uni-bremen.de/~dtcg/test_zarr/"

    def stream_datacube(
        self, glacier: str, layer: str = "", region_name: str = "Iceland"
    ) -> xr.DataTree:
        """Stream datacube from server.

        Parameters
        ----------
        glacier : str
            Name or RGI-ID of glacier.
        layer : str, optional
            Datacube layer. Default will load all available layers.
        region : str, default "Iceland"
            RGI region name.

        Returns
        -------
        xr.DataTree
            Datacube or datacube layer.
        """
        if glacier[:3] != "RGI":

            glacier_data = self.binder.get_rgi_files_from_subregion(
                region_name=region_name, subregion_name=""
            )
            glacier = self.binder.get_glacier_by_name(
                glacier_data, glacier
            ).RGIId.values[0]
        stream_url = self.get_url(rgi_id=glacier)

        return xr.open_datatree(
            stream_url,
            group=layer,
            chunks={},
            engine="zarr",
            consolidated=True,
            decode_cf=True,
        )

    def get_url(self, rgi_id: str) -> str:
        """Get URL for a Zarr datacube.

        Parameters
        ----------
        rgi_id : str
            Glacier RGI ID.

        Returns
        -------
        str
            Server URL of Zarr datacube.
        """

        url = f"{self.server}{rgi_id}.zarr/"
        return url
