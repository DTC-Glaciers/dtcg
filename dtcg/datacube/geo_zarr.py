from enum import Enum
from typing import Optional

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc


class ZarrStorage(Enum):
    local_store = "local_store"
    memory_store = "memory_store"


class GeoZarrWriter:
    def __init__(
        self,
        ds: xr.Dataset,
        storage_type: ZarrStorage = ZarrStorage.local_store,
        storage_directory: Optional[str] = None,
        target_chunk_mb: float = 5.0,
        compressor: Optional[object] = None,
        overwrite: bool = True
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
        compressor : Object or None, optional
            Compressor to apply on arrays. If None, defaults to Blosc with zstd.
        overwrite : bool, optional
            Whether to overwrite existing Zarr contents in the target location.
            Default is True.
        """
        self.ds = ds
        self.storage_type = storage_type
        self.storage_directory = storage_directory
        self.target_chunk_mb = target_chunk_mb
        self.compressor = compressor or Blosc(
            cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.overwrite = overwrite
        self.chunk_sizes = {}
        self.encoding = {}

        self._set_store()
        self._validate_dataset()
        self._add_cf_metadata()
        self._calculate_chunk_sizes()
        self._define_encodings()

    def _set_store(self):
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

    def _validate_dataset(self):
        required_dims = {'x', 'y'}
        if not required_dims.issubset(self.ds.dims):
            raise ValueError(
                f"Dataset must have at least dimensions {required_dims}")
        for dim in self.ds.dims:
            if dim not in self.ds.coords:
                raise ValueError(
                    f"Coordinate variable for dimension '{dim}' is missing in "
                    "the dataset.")

    def _add_cf_metadata(self):
        if 'standard_name' not in self.ds['x'].attrs:
            self.ds['x'].attrs.update({
                'standard_name': 'projection_x_coordinate',
                'units': 'meters'
            })
        if 'standard_name' not in self.ds['y'].attrs:
            self.ds['y'].attrs.update({
                'standard_name': 'projection_y_coordinate',
                'units': 'meters'
            })
        if 't' in self.ds.dims and 'standard_name' not in self.ds['t'].attrs:
            self.ds['t'].attrs.update({
                'standard_name': 'time'
            })

    def _calculate_chunk_sizes(self):
        bytes_per_element = 4
        target_bytes = self.target_chunk_mb * 1024 * 1024
        x_size = self.ds.sizes['x']
        y_size = self.ds.sizes['y']
        total_elements_target = target_bytes // bytes_per_element
        side_length = int(np.sqrt(total_elements_target))
        chunk_x = min(x_size, side_length)
        chunk_y = min(y_size, side_length)
        self.chunk_sizes = {'x': chunk_x, 'y': chunk_y}
        if 't' in self.ds.dims:
            self.chunk_sizes['t'] = 1

    def _define_encodings(self):
        for var in self.ds.data_vars:
            dims = self.ds[var].dims
            chunks = tuple(
                self.chunk_sizes.get(dim, self.ds.dims[dim]) for dim in dims)
            self.encoding[var] = {
                'chunks': chunks,
                'compressor': self.compressor
            }

    def write(self):
        mode = 'w' if self.overwrite else 'a'
        root = zarr.open_group(store=self.store, mode=mode, zarr_version=2)
        root.attrs['_ARRAY_DIMENSIONS'] = list(self.ds.dims)
        root.attrs['MAP_PROJECTION'] = self.ds.rio.crs.to_wkt()

        for var_name, da in self.ds.data_vars.items():
            chunks = tuple(self.chunk_sizes.get(dim, self.ds.dims[dim])
                           for dim in da.dims)
            z = root.create_dataset(
                name=var_name,
                shape=da.shape,
                dtype=da.dtype,
                chunks=chunks,
                compressors=self.compressor,
                overwrite=self.overwrite
            )
            z.attrs.update(da.attrs)
            z.attrs['_ARRAY_DIMENSIONS'] = list(da.dims)
            z[...] = da.values

        for coord_name in ['x', 'y', 't']:
            if coord_name in self.ds:
                coord_da = self.ds[coord_name]
                z = root.create_dataset(
                    name=coord_name,
                    shape=coord_da.shape,
                    dtype=coord_da.dtype,
                    chunks=(coord_da.shape[0],),
                    compressors=self.compressor,
                    overwrite=self.overwrite
                )
                z.attrs.update(coord_da.attrs)
                z.attrs['_ARRAY_DIMENSIONS'] = [coord_name]
                z[...] = coord_da.values

        zarr.consolidate_metadata(self.store)
