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

Functionality for ensuring metadata is CF complaint: https://cfconventions.org/.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import xarray as xr
import yaml
from schema import Optional, Schema


class MetadataMapper:
    """
    Class for applying CF-compliant metadata to xarray Datasets.

    Attributes
    ----------
    METADATA_SCHEMA : schema.Schema
        Validation schema for variable metadata.
    metadata_mappings : dict
        Dictionary of metadata mappings loaded from a YAML file.
    """

    def __init__(self: MetadataMapper,
                 metadata_mapping_file_path: os.PathLike = None):
        """
        Initialise MetadataMapper with a given or default mapping file.

        Parameters
        ----------
        metadata_mapping_file_path : os.PathLike, optional
            Path to the YAML file containing variable metadata mappings.
            If None, defaults to 'metadata_mapping.yaml' in the current
            directory.
        """
        if metadata_mapping_file_path is None:
            metadata_mapping_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'metadata_mapping.yaml')
        self.METADATA_SCHEMA = Schema({
            'standard_name': str,
            'long_name': str,
            'units': str,
            Optional('author'): str,
            'institution': str,
            'source': str,
            'comment': str,
            'references': str
        })
        self.load_metadata_mappings(metadata_mapping_file_path)

    def load_metadata_mappings(self: MetadataMapper,
                               metadata_mapping_file_path: str) -> None:
        """
        Load and validate metadata mappings from a YAML file.

        Parameters
        ----------
        metadata_mapping_file_path : str
            Path to the YAML file containing metadata mappings.

        Raises
        ------
        schema.SchemaError
            If any of the metadata entries fail schema validation.
        """
        with open(metadata_mapping_file_path) as f:
            config_dict = yaml.safe_load(f)

        for _, metadata in config_dict.items():
            self.METADATA_SCHEMA.validate(metadata)

        self.metadata_mappings = config_dict
    
    def _update_shared_metadata(self: MetadataMapper,
                                dataset: xr.Dataset) -> None:
        """
        Add shared metadata attributes to the dataset and ensure CRS is set.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset to which shared metadata and CRS should be applied.

        Notes
        -----
        If a CRS is not present, it is set from the dataset's `pyproj_srs`
        attribute. Shared metadata includes CF conventions, title, and
        summary.
        """
        # create a spatial_ref layer in the dataset
        if not dataset.rio.crs:
            dataset.rio.write_crs(dataset.pyproj_srs, inplace=True)

        # update metadata shared across all variables
        shared_metadata = {
            "Conventions": "CF-1.12",
            "title": "Datacube of Glacier-domain variables.",
            "summary": "Resampled Glacier-domain variables from multiple "
            "sources. Generated as part of the DTC-Glaciers project.",
            "comment": "The DTC-Glaciers project is developed under the "
            "European Space Agency's Digital Twin Earth initiative, as part of "
            "the Digital Twin Components (DTC) Early Development Actions.",
            "date_created": datetime.now().isoformat()
        }

        dataset.attrs.update(shared_metadata)

    def update_metadata(self: MetadataMapper,
                        dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply variable and shared metadata to an xarray Dataset.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to which metadata should be applied.

        Returns
        -------
        xarray.Dataset
            The input dataset with updated metadata.

        Warns
        -----
        UserWarning
            If any dataset variables are missing in the metadata mapping.

        Notes
        -----
        This function adds both per-variable and global metadata attributes.
        Missing variable mappings are reported as warnings, not errors.
        """
        # check there are mappings for all variables in the dataset
        difference = set(dataset.variables) - set(self.metadata_mappings.keys())
        if difference:
            warnings.warn(
                "Metadata mapping is missing for the following variables: "
                f"{sorted(difference)}. The metadata for these variables might "
                "not be compliant with Climate and Forecast conventions "
                "https://cfconventions.org/.")

        # simple function to apply metadata to all layers in an xarray dataset
        for data_name, metadata in self.metadata_mappings.items():
            dataset[data_name].attrs.update(metadata)

        self._update_shared_metadata(dataset)

        return dataset
