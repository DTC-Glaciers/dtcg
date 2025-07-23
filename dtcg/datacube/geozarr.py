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
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

from dtcg.datacube.update_metadata import MetadataMapper


class GeoZarrHandler(MetadataMapper):
    def __init__(
        self: GeoZarrHandler,
        ds: xr.Dataset,
        target_chunk_mb: float = 5.0,
        compressor: Optional[Blosc] = None,
        metadata_mapping_file_path: str = None,
        zarr_format: int = 2
    ):
        """Initialise a GeoZarrWriter object.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        target_chunk_mb : float, default 5.0
            Approximate chunk size in megabytes for efficient storage.
        compressor : Blosc, default None
            Compressor to apply on arrays. If None, the compression will
            be Blosc with zstd.
        metadata_mapping_file_path: str, default None
            Path to the YAML file containing variable metadata mappings.
            If None, defaults to 'metadata_mapping.yaml' in the current
            directory.
        zarr_format : int, default 2
            Zarr format version to use (2 or 3).
        """
        super().__init__(metadata_mapping_file_path=metadata_mapping_file_path)
        self.ds = ds
        self.target_chunk_mb = target_chunk_mb
        self.compressor = compressor or Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
        )
        self.zarr_format = zarr_format
        self.encoding = {}
        self.memory_store = zarr.storage.MemoryStore()

        self._validate_dataset()
        self._define_encodings()
        self._update_metdata()
        self._write(self.memory_store)

    def _validate_dataset(self: GeoZarrHandler) -> None:
        """Validate the input dataset to ensure it includes required dimensions
        and associated coordinate variables.

        Raises
        ------
        ValueError
            - If 'x' or 'y' dimensions are missing.
            - If any dimension does not have an associated coordinate
              variable.
        """
        required_dims = {"x", "y"}
        if not required_dims.issubset(self.ds.dims):
            raise ValueError(f"Dataset must have at least dimensions {required_dims}")
        for dim in self.ds.dims:
            if dim not in self.ds.coords:
                raise ValueError(
                    f"Coordinate variable for dimension '{dim}' is missing in "
                    "the dataset."
                )

    def _calculate_chunk_sizes(
        self: GeoZarrHandler, var: xr.DataArray
    ) -> dict[str, int]:
        """Calculate chunk sizes for a given variable to match the
        target chunk size in megabytes.

        Parameters
        ----------
        var : xr.DataArray
            Data array whose dtype and dimensions are used to compute
            chunk sizes.

        Returns
        -------
        dict[str, int]
            A dictionary of chunk sizes for dimensions 'x', 'y', and
            optionally 't'.
        """
        target_bytes = self.target_chunk_mb * 1024 * 1024
        x_size = var.sizes["x"]
        y_size = var.sizes["y"]
        t_size = var.sizes.get("t", 1)  # Defaults to 1 if no 't' dimension

        # Calculate the number of elements allowed per chunk
        # After accounting for a full 't' slice
        elements_per_t_slice = target_bytes // (var.dtype.itemsize * t_size)

        # Determine side length based on remaining budget
        side_length = int(np.sqrt(elements_per_t_slice))

        chunk_x = min(x_size, side_length)
        chunk_y = min(y_size, side_length)

        chunk_sizes = {"x": chunk_x, "y": chunk_y}

        if "t" in var.dims:
            # Use the full length of 't' - this allows more efficient loading,
            # assuming the user is always interested in the full time series
            chunk_sizes["t"] = t_size

        return chunk_sizes

    def _define_encodings(self: GeoZarrHandler) -> None:
        """Define encoding settings for each data variable in the
        dataset, including chunking and compression.

        Notes
        -----
        Chunk sizes are computed using `_calculate_chunk_sizes`, and the
        compressor is set according to the class-level setting.
        """
        for var in self.ds.data_vars:
            chunk_sizes = self._calculate_chunk_sizes(self.ds[var])
            chunks = tuple(chunk_sizes.get(dim) for dim in self.ds[var].dims)
            self.encoding[var] = {"chunks": chunks, "compressor": self.compressor}

    def _update_metdata(self) -> None:
        """
        Metadata is first updated using the ``update_metadata`` method.
        Each data variable is tagged with the ``grid_mapping`` attribute
        for spatial referencing.
        """
        self.ds = self.update_metadata(self.ds)
        for var in self.ds.data_vars:
            self.ds[var].attrs["grid_mapping"] = "spatial_ref"

    def _write(
            self: GeoZarrHandler, store: str,
            overwrite: bool = True,
            ) -> None:
        """Write the dataset to GeoZarr format.

        Parameters
        ----------
        storage_directory : str
            Required if using ``local_store``. Path to write the Zarr
            data.
        overwrite : bool, default True
            Whether to overwrite existing Zarr contents in the target
            location.
        """
        self.ds.to_zarr(
            store,
            mode="w" if overwrite else "a",
            consolidated=True,
            zarr_format=self.zarr_format,
            encoding=self.encoding,
        )

    @property
    def dataset(self):
        """Load the written Zarr file as an xarray Dataset."""
        return xr.open_zarr(self.memory_store)

    def export(self: GeoZarrHandler, storage_directory: str,
               overwrite: bool = True) -> None:
        """Write the dataset to GeoZarr format.

        Parameters
        ----------
        storage_directory : str
            Path to write the Zarr data.
        overwrite : bool, default True
            Whether to overwrite existing Zarr contents in the target
            location.
        """
        dir_path = Path(storage_directory).parent
        if not dir_path.exists():
            raise FileNotFoundError(
                "Base directory of 'storage_directory' does not exist: "
                + dir_path
            )
        self._write(storage_directory, overwrite=overwrite)
