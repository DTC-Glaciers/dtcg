import numpy as np
import pytest
import rioxarray
import xarray as xr
import zarr

from dtcg.data_cube.geo_zarr import GeoZarrWriter, ZarrStorage


def create_test_dataset():
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(("t", "y", "x"), np.random.rand(2, 4, 5)),
            precipitation=(("y", "x"), np.random.rand(4, 5)),
        ),
        coords=dict(
            x=("x", np.linspace(0, 1000, 5)),
            y=("y", np.linspace(0, 2000, 4)),
            t=("t", np.array(["2020-01-01", "2020-01-02"],
               dtype="datetime64[ns]")),
        ),
    )
    ds.rio.write_crs("EPSG:3413", inplace=True)
    return ds


def test_memory_store_write():
    ds = create_test_dataset()
    writer = GeoZarrWriter(
        ds=ds,
        storage_type=ZarrStorage.memory_store,
        overwrite=True
    )
    writer.write()
    root = zarr.open_group(store=writer.store, mode='r')

    assert 'temperature' in root
    assert 'precipitation' in root
    assert 'x' in root
    assert 'y' in root
    assert 't' in root
    assert '_ARRAY_DIMENSIONS' in root.attrs
    assert 'MAP_PROJECTION' in root.attrs


def test_local_store_write(tmp_path):
    ds = create_test_dataset()
    store_dir = tmp_path / "geozarr_store"
    writer = GeoZarrWriter(
        ds=ds,
        storage_type=ZarrStorage.local_store,
        storage_directory=str(store_dir),
        overwrite=True
    )
    writer.write()
    root = zarr.open_group(str(store_dir), mode='r')

    assert 'temperature' in root
    assert 'precipitation' in root
    assert 'x' in root
    assert 'y' in root
    assert 't' in root
    assert '_ARRAY_DIMENSIONS' in root.attrs
    assert 'MAP_PROJECTION' in root.attrs


def test_missing_required_dims_raises():
    ds = create_test_dataset()
    ds = ds.drop_dims('x')

    with pytest.raises(ValueError):
        GeoZarrWriter(
            ds=ds,
            storage_type=ZarrStorage.memory_store
        )


def test_correct_chunking():
    ds = create_test_dataset()
    writer = GeoZarrWriter(
        ds=ds,
        storage_type=ZarrStorage.memory_store,
        overwrite=True
    )
    writer.write()
    root = zarr.open_group(store=writer.store, mode='r')
    temp_chunks = root['temperature'].chunks
    assert isinstance(temp_chunks, tuple)
    assert len(temp_chunks) == 3
