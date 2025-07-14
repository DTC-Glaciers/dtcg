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

Tests for GeoZarr datacubes.
"""

import os
import numpy as np
import pytest
import xarray as xr
import zarr
import yaml

from dtcg.datacube.geo_zarr import GeoZarrWriter, ZarrStorage


class TestGeoZarrWriter:
    """Tests for GeoZarrWriter functionality and metadata compliance."""

    @pytest.fixture(name="test_dataset", scope="function")
    def fixture_test_dataset(self, tmp_path):
        """Generate a test xarray dataset and metadata mapping file."""
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(("t", "y", "x"), np.random.rand(100, 100, 100)),
                precipitation=(("y", "x"), np.random.rand(100, 100)),
            ),
            coords=dict(
                x=("x", np.linspace(0, 1000, 100)),
                y=("y", np.linspace(0, 2000, 100)),
                t=("t", np.linspace(1e5, 1e6, 100)),
            ),
            attrs={
                "pyproj_srs": "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 "
                "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
            },
        )

        metadata_mapping = {
            "temperature": {
                "standard_name": "bond",
                "long_name": "james bond",
                "units": "007",
                "institution": "mi6",
                "source": "wibble",
                "comment": "wobble",
                "references": "moneypenny",
            }
        }
        metadata_path = os.path.join(tmp_path, "meta.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(metadata_mapping, f)

        return ds, metadata_path

    @pytest.mark.filterwarnings("ignore:Metadata mapping is missing")
    @pytest.mark.parametrize(
        "storage_type", [ZarrStorage.memory_store, ZarrStorage.local_store]
    )
    def test_geozarr_attributes(self, tmp_path, test_dataset, storage_type):
        """Test that spatial_ref, metadata, and dimensions are correct."""
        ds, metadata_path = test_dataset

        store_dir = None
        if storage_type == ZarrStorage.local_store:
            store_dir = tmp_path / "geozarr_store"

        writer = GeoZarrWriter(
            ds=ds,
            storage_type=storage_type,
            storage_directory=store_dir,
            overwrite=True,
            metadata_mapping_file_path=metadata_path,
        )

        writer.write()

        root = zarr.open_group(store=writer.store, mode="r")

        assert "spatial_ref" in root
        assert "crs_wkt" in root["spatial_ref"].attrs

        for param in ["temperature", "precipitation"]:
            assert param in root
            assert "_ARRAY_DIMENSIONS" in root[param].attrs
            assert root[param].attrs["grid_mapping"] == "spatial_ref"
            if param == "temperature":
                assert root[param].attrs["standard_name"] == "bond"
                assert root[param].attrs["references"] == "moneypenny"

        for coord in ["x", "y", "t"]:
            assert coord in root
            assert root[coord].attrs["_ARRAY_DIMENSIONS"] == [coord]

    def test_missing_required_dims_raises(self, test_dataset):
        """Test that missing required dimensions raises ValueError."""
        ds, _ = test_dataset
        ds = ds.drop_dims("x")

        with pytest.raises(ValueError, match="Dataset must have at least dimensions"):
            GeoZarrWriter(ds=ds, storage_type=ZarrStorage.memory_store)

    def test_zarr_storage(self):
        """Test ZarrStorage has appropriate attributes."""
        for storage_type in ["memory_store", "local_store"]:
            assert storage_type in ZarrStorage
            assert hasattr(ZarrStorage, storage_type)
        storage_type = "wrong_store"
        assert storage_type not in ZarrStorage

    @pytest.mark.filterwarnings("ignore:Metadata mapping is missing")
    def test_correct_chunking(self, test_dataset):
        """Test that chunking is computed as expected."""
        ds, _ = test_dataset
        writer = GeoZarrWriter(
            ds=ds,
            storage_type=ZarrStorage.memory_store,
            overwrite=True,
            target_chunk_mb=1,
        )

        writer.write()

        root = zarr.open_group(store=writer.store, mode="r")
        temp_chunks = root["temperature"].chunks
        precip_chunks = root["precipitation"].chunks

        assert isinstance(temp_chunks, tuple)
        assert temp_chunks == (100, 36, 36)
        assert isinstance(precip_chunks, tuple)
        assert precip_chunks == (100, 100)

    def test_missing_dimension(self, test_dataset):
        ds, _ = test_dataset
        assert "t" in ds.dims
        ds = ds.drop_vars("t")
        assert "t" not in ds.coords
        with pytest.raises(
            ValueError, match="Coordinate variable for dimension 't' is missing"
        ):
            GeoZarrWriter(ds=ds, storage_type=ZarrStorage.memory_store)

    def test_non_enum_storage_type(self, test_dataset):
        # ensure correct error handling when incorrect storage type is supplied
        ds, _ = test_dataset
        with pytest.raises(NotImplementedError, match="Invalid storage_type."):
            GeoZarrWriter(ds=ds, storage_type="blah")
