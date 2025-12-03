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

from pathlib import Path

import xarray as xr
import zarr
from zarr.errors import GroupNotFoundError

import dtcg.integration.oggm_bindings as oggm_bindings


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

    def get_zip_path(self, zip_path: Path, rgi_id: str):
        if not zip_path.suffix != ".zip":  # zarr is a directory
            Path(zip_path).mkdir(exist_ok=True)
            zip_path = zip_path.with_suffix(".zip")
        zip_path = Path(f"{zip_path}/{rgi_id}.zarr.zip")
        return zip_path

    def zip_datacube(self, rgi_id: str, zip_path: Path = ""):
        """Download and zip a datacube.

        Parameters
        ----------
        stream_url : str
            URL to a zarr folder.
        rgi_id : str, optional
            RGI-ID of glacier.
        zip_path : Path, optional
            Output path for zip file.

        Returns
        -------
        Path
            Path to output zipfile.
        """

        if rgi_id:
            stream_url = self.get_url(rgi_id=rgi_id)
        try:
            zip_path = self.get_zip_path(zip_path=zip_path, rgi_id=rgi_id)

            store = zarr.storage.ZipStore(zip_path, mode="w")
            with xr.open_datatree(
                stream_url,
                group=None,
                chunks={},
                engine="zarr",
                consolidated=True,
                decode_cf=True,
            ) as stream:
                stream.compute().to_zarr(
                    store=store,
                    mode="w",
                    consolidated=True,
                    zarr_format=2,
                    # encoding=stream.encoding,
                )

        except GroupNotFoundError as e:
            print(f"Error when zipping: {e}")

        return zip_path

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
