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

Functionality for exporting a GeoZarr file.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

from dtcg.datacube.update_metadata import MetadataMapper


class ZarrStorage(Enum):
    local_store = "local_store"
    memory_store = "memory_store"


class GeoZarrWriter(MetadataMapper):
    def __init__(
        self: GeoZarrWriter,
        ds: xr.Dataset,
        storage_type: ZarrStorage = ZarrStorage.local_store,
        storage_directory: Optional[str] = None,
        target_chunk_mb: float = 5.0,
        compressor: Optional[Blosc] = None,
        overwrite: bool = True,
        metadata_mapping_file_path: str = None
    ):
        """
        Initialise a GeoZarrWriter object.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        storage_type : ZarrStorage, optional
            Enum to specify storage backend, either ZarrStorage.local_store
            or ZarrStorage.memory_store. Default is local store.
        storage_directory : str or None, optional
            Required if using local_store. Path to write the Zarr data.
        target_chunk_mb : float, optional
            Approximate chunk size in megabytes for efficient storage.
            Default is 5 MB.
        compressor : Blosc or None, optional
            Compressor to apply on arrays. If None, defaults to Blosc with zstd.
        overwrite : bool, optional
            Whether to overwrite existing Zarr contents in the target location.
            Default is True.
        metadata_mapping_file_path: Optional[str] = None
            Path to the YAML file containing variable metadata mappings.
            If None, defaults to 'metadata_mapping.yaml' in the current
            directory.
        """
        super().__init__(metadata_mapping_file_path=metadata_mapping_file_path)
        self.ds = ds
        self.storage_type = storage_type
        self.storage_directory = storage_directory
        self.target_chunk_mb = target_chunk_mb
        self.compressor = compressor or Blosc(
            cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.overwrite = overwrite
        self.encoding = {}

        self._set_store()
        self._validate_dataset()
        self._define_encodings()

    def _set_store(self: GeoZarrWriter) -> None:
        """
        Set the Zarr storage backend based on the selected storage type.

        Raises
        ------
        TypeError
            If `storage_directory` is not provided when using `local_store`.
        ValueError
            If an invalid `storage_type` is provided.
        """
        if self.storage_type == ZarrStorage.memory_store:
            self.store = zarr.storage.MemoryStore()
        elif self.storage_type == ZarrStorage.local_store:
            if self.storage_directory is None:
                raise TypeError("Enter a valid storage location")
            self.store = self.storage_directory
        else:
            raise ValueError(
                "Invalid storage_type. Must be ZarrStorage.local_store or "
                "ZarrStorage.memory_store.")

    def _validate_dataset(self: GeoZarrWriter) -> None:
        """
        Validate the input dataset to ensure it includes required dimensions
        and associated coordinate variables.

        Raises
        ------
        ValueError
        - If 'x' or 'y' dimensions are missing.
        - If any dimension does not have an associated coordinate variable.
        """
        required_dims = {'x', 'y'}
        if not required_dims.issubset(self.ds.dims):
            raise ValueError(
                f"Dataset must have at least dimensions {required_dims}")
        for dim in self.ds.dims:
            if dim not in self.ds.coords:
                raise ValueError(
                    f"Coordinate variable for dimension '{dim}' is missing in "
                    "the dataset.")

    def _calculate_chunk_sizes(self: GeoZarrWriter, var: xr.DataArray) -> None:
        """
        Calculate chunk sizes for a given variable to match the target chunk
        size in megabytes.

        Parameters
        ----------
        var : xr.DataArray
            Data array whose dtype and dimensions are used to compute chunk
            sizes.

        Returns
        -------
        dict
            A dictionary of chunk sizes for dimensions 'x', 'y', and optionally
            't'.
        """
        target_bytes = self.target_chunk_mb * 1024 * 1024
        x_size = self.ds.sizes['x']
        y_size = self.ds.sizes['y']
        total_elements_target = target_bytes // var.dtype.itemsize
        side_length = int(np.sqrt(total_elements_target))
        chunk_x = min(x_size, side_length)
        chunk_y = min(y_size, side_length)
        chunk_sizes = {'x': chunk_x, 'y': chunk_y}
        if 't' in self.ds.dims:
            chunk_sizes['t'] = 1
        return chunk_sizes

    def _define_encodings(self: GeoZarrWriter) -> None:
        """
        Define encoding settings for each data variable in the dataset,
        including chunking and compression.

        Notes
        -----
        Chunk sizes are computed using `_calculate_chunk_sizes`, and the
        compressor is set according to the class-level setting.
        """
        for var in self.ds.data_vars:
            chunk_sizes = self._calculate_chunk_sizes(self.ds[var])
            chunks = tuple(chunk_sizes.get(dim) for dim in self.ds[var].dims)
            self.encoding[var] = {
                'chunks': chunks,
                'compressor': self.compressor
            }

    def write(self: GeoZarrWriter, zarr_format: int = 2) -> None:
        """
        Write the dataset to GeoZarr format.

        Parameters
        ----------
        zarr_format : int, optional
            Zarr format version to use (2 or 3). Default is 2.

        Notes
        -----
        Metadata is first updated using the `update_metadata` method. Each data
        variable is tagged with the `grid_mapping` attribute for spatial
        referencing.
        """
        ds_metadata_updated = self.update_metadata(self.ds)
        for var in self.ds.data_vars:
            self.ds[var].attrs["grid_mapping"] = "spatial_ref"
        ds_metadata_updated.to_zarr(
            self.store,
            mode='w' if self.overwrite else 'a',
            consolidated=True,
            zarr_format=zarr_format,
            encoding=self.encoding)
