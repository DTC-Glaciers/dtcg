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
from schema import Schema


class MetadataMapper:
    def __init__(self: MetadataMapper,
                 metadata_mapping_file_path: os.PathLike = None):
        if metadata_mapping_file_path is None:
            metadata_mapping_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'metadata_mapping.yaml')
        self.METADATA_SCHEMA = Schema({
            'standard_name': str,
            'long_name': str,
            'units': str,
            'institution': str,
            'source': str,
            'comment': str,
            'references': str
        })
        self.load_metadata_mappings(metadata_mapping_file_path)

    def load_metadata_mappings(self: MetadataMapper,
                               metadata_mapping_file_path: str) -> None:
        with open(metadata_mapping_file_path) as f:
            config_dict = yaml.safe_load(f)

        for _, metadata in config_dict.items():
            self.METADATA_SCHEMA.validate(metadata)

        self.metadata_mappings = config_dict
    
    def _update_shared_metadata(self: MetadataMapper,
                                dataset: xr.Dataset) -> None:
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
