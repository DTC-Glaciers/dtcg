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

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from numcodecs import Blosc

from dtcg.datacube.update_metadata import MetadataMapper


class GeoZarrHandler(MetadataMapper):
    def __init__(
        self: GeoZarrHandler,
        ds: xr.Dataset,
        ds_name: str = "L1",
        target_chunk_mb: float = 5.0,
        compressor: Optional[Blosc] = None,
        metadata_mapping_file_path: str = None,
        zarr_format: int = 2,
    ):
        """Initialise a GeoZarrHandler object.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        ds_name : str, default 'L1'
            Name of datacube.
        target_chunk_mb : float, default 5.0
            Approximate chunk size in megabytes for efficient storage.
        compressor : Blosc, default None
            Compressor to apply on arrays. If None, the compression will
            be Blosc with zstd.
        metadata_mapping_file_path : str, default None
            Path to the YAML file containing variable metadata mappings.
            If None, defaults to 'metadata_mapping.yaml' in the current
            directory.
        zarr_format : int, default 2
            Zarr format version to use (2 or 3).
        """
        super().__init__(metadata_mapping_file_path=metadata_mapping_file_path)
        self.ds_name = ds_name
        self.target_chunk_mb = target_chunk_mb
        self.compressor = compressor or Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
        )
        self.zarr_format = zarr_format

        ds = self._validate_dataset(ds)
        ds = self._update_metadata(ds, ds_name)
        self.encoding = {}
        self._define_encodings(ds, ds_name)

        # convert dataset to datatree
        self.data_tree = xr.DataTree.from_dict({ds_name: ds})

    def _validate_dataset(self: GeoZarrHandler, ds: xr.Dataset) -> xr.Dataset:
        """Validate the input dataset to ensure it includes required
        dimensions and associated coordinate variables.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.

        Raises
        ------
        ValueError
            - If 'x' or 'y' dimensions are missing.
            - If any dimension does not have an associated coordinate
              variable.
        """
        accepted_dims = {"x", "y", "t"}
        if not set(ds.dims).issubset(accepted_dims):
            raise ValueError(
                "Incorrect dataset dimensions."
                f" Accepted data dimensions are: {accepted_dims}"
            )
        for dim in ds.dims:
            if dim not in ds.coords:
                raise ValueError(
                    f"Coordinate variable for dimension '{dim}' is missing in "
                    "the dataset."
                )
        return ds

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
        t_size = var.sizes.get("t", 1)  # Defaults to 1 if no 't' dimension
        chunk_sizes = {}

        if "x" in var.dims and "y" in var.dims:
            x_size = var.sizes["x"]
            y_size = var.sizes["y"]
            # Calculate the number of elements allowed per chunk
            # After accounting for a full 't' slice
            elements_per_t_slice = target_bytes // (var.dtype.itemsize * t_size)

            # Determine side length based on remaining budget
            side_length = int(np.sqrt(elements_per_t_slice))

            chunk_x = min(x_size, side_length)
            chunk_y = min(y_size, side_length)

            chunk_sizes["x"] = chunk_x
            chunk_sizes["y"] = chunk_y

        if "t" in var.dims:
            # Use the full length of 't' - this allows more efficient loading,
            # assuming the user is always interested in the full time series
            chunk_sizes["t"] = t_size

        return chunk_sizes

    def _define_encodings(self: GeoZarrHandler, ds: xr.Dataset, ds_name: str) -> None:
        """Define encoding settings for each data variable in the
        dataset, including chunking and compression.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        ds_name : str
            Dataset name to be used for this node of the tree.

        Notes
        -----
        Chunk sizes are computed using `_calculate_chunk_sizes`, and the
        compressor is set according to the class-level setting.
        """
        if ds_name not in self.encoding:
            self.encoding[f"/{ds_name}"] = {}

        for var in ds.data_vars:
            chunk_sizes = self._calculate_chunk_sizes(ds[var])
            chunks = tuple(chunk_sizes.get(dim) for dim in ds[var].dims)
            self.encoding[f"/{ds_name}"][var] = {
                "chunks": chunks,
                "compressor": self.compressor,
            }

    def _update_metadata(self, ds: xr.Dataset, ds_name: str) -> xr.Dataset:
        """Update metadata to Climate and Forecast convention.

        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with dimensions ('x', 'y') or ('t', 'x', 'y').
            Must include coordinate variables.
        ds_name : str
            Layer name for this node of the tree.

        Metadata is first updated using the ``update_metadata`` method.
        Each data variable is tagged with the ``grid_mapping`` attribute
        for spatial referencing.
        """
        ds = self.update_metadata(ds, ds_name)
        for var in ds.data_vars:
            var_dims = ds[var].dims
            if "x" in var_dims or "y" in var_dims:
                ds[var].attrs["grid_mapping"] = "spatial_ref"
        return ds

    def export(
        self: GeoZarrHandler, storage_directory: str, overwrite: bool = True
    ) -> None:
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
                f"Base directory of 'storage_directory' does not exist: {dir_path}"
            )
        self.data_tree.to_zarr(
            storage_directory,
            mode="w" if overwrite else "a",
            consolidated=True,
            zarr_format=self.zarr_format,
            encoding=self.encoding,
        )

    def add_layer(
        self: GeoZarrHandler, ds: xr.Dataset, ds_name: str, overwrite: bool = False
    ) -> None:
        """Add a new dataset as a child group of the DataTree at the root.

        Parameters
        ----------
        ds : xarray.Dataset
            New dataset layer to be added to the existing data tree.
        ds_name : str
            Layer name to be used for this node of the tree.
        overwrite : bool
            If True, allow a layer of the same name to be overwritten.
        """
        if ds_name in self.data_tree.children and not overwrite:
            raise ValueError(f"Group '{ds_name}' already exists.")

        # prepare new dataset
        ds = self._validate_dataset(ds)
        ds = self._update_metadata(ds, ds_name)

        # append additional encodings to the encodings class attribute
        self._define_encodings(ds, ds_name)

        # validate dataset attributes
        for var in ds.data_vars:
            attrs = ds[var].attrs.copy()
            attrs.pop("grid_mapping", None)
            self.METADATA_SCHEMA.validate(attrs)

        self.data_tree[ds_name] = xr.DataTree(dataset=ds)

    def get_layer(self: GeoZarrHandler, ds_name: str) -> xr.Dataset:
        """Get a dataset from a DataTree.

        Parameters
        ----------
        ds_name : str
            Layer name.

        Returns
        -------
        xr.Dataset
            Dataset layer in tree.

        Raises
        ------
        KeyError
            If the layer name is not present in the data tree.
        AttributeError
            If the layer does not contain a dataset.


        """
        try:
            layer = self.data_tree[ds_name].ds
        except KeyError:
            raise KeyError(f"{ds_name} layer not found in the data tree.")

        return layer
