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

Functionality for retrieving WGMS data.
"""

import pandas as pd
import numpy as np
import xarray as xr

from oggm import utils


class DatacubeWGMS:
    """Functionality for adding WGMS data to a OGGM datacube.
    """
    def __init__(self):
        base_url = "https://cluster.klima.uni-bremen.de/~dtcg/test_files/wgms_data/"
        fp_glacier_ids = utils.file_downloader(
            base_url + "glacier_id_lut.csv"
        )
        fp_mb_dtcg = utils.file_downloader(
            base_url + "WGMS_MB-DTC-Glaciers.csv"
        )

        self.df_glacier_ids = pd.read_csv(fp_glacier_ids)[['NAME', 'WGMS_ID',
                                                           'RGI60_ID']]
        self.df_wgms_mbs = pd.read_csv(fp_mb_dtcg)[
            ['glacier_id', 'glacier_id.short_name', 'year', 'annual_balance',
             'annual_balance_unc']]

    def get_wgms_mb(self, rgi_id, default_uncertainty):
        df_row = self.df_glacier_ids['RGI60_ID'] == rgi_id
        wgms_id = self.df_glacier_ids[df_row]['WGMS_ID'].item()
        df_wgms_mbs_row = self.df_wgms_mbs['glacier_id'] == wgms_id
        df_mbs = self.df_wgms_mbs[df_wgms_mbs_row]
        df_mbs = df_mbs.rename(
            columns={'annual_balance': 'wgms_mb',
                     'annual_balance_unc': 'wgms_mb_unc'}).set_index('year')

        # convert from m w.e. to mm w.e.
        df_mbs['wgms_mb'] = df_mbs['wgms_mb'] * 1000

        # if now uncertainty is provided use fixed value
        df_mbs['wgms_mb_unc'] = np.where(
            np.isnan(df_mbs['wgms_mb_unc']),
            default_uncertainty, df_mbs['wgms_mb_unc']
        )

        return df_mbs

    def add_data_to_datacube(self, datacube, gdir):
        """Every datacube must support this method.

        It should be able to add data to the provided datacube and returns the
        final datacube.
        """
        # this try except statement is quick and dirty, could be more elegant in
        # the future
        try:
            default_uncertainty = 200  # mm w.e.
            df_wgms_mb = self.get_wgms_mb(gdir.rgi_id, default_uncertainty)

            wgms_mb_da = xr.DataArray(
                df_wgms_mb.wgms_mb.values,
                coords={"t_wgms": df_wgms_mb.index.values},
                dims=("t_wgms"),
                name="wgms_mb",
                attrs={
                    "units": "mm w.e.",
                    "source": "Specific mass balance observation as reported to"
                              "the World Glacier Monitoring Service"
                }
            )
            datacube["wgms_mb"] = wgms_mb_da
            datacube['t_wgms'].attrs = {
                'long_name': 'Year of WGMS observations'
            }

            wgms_mb_unc_da = xr.DataArray(
                df_wgms_mb.wgms_mb_unc.values,
                coords={"t_wgms": df_wgms_mb.index.values},
                dims=("t_wgms"),
                name="wgms_mb",
                attrs={
                    "units": "mm w.e.",
                    "source": "Specific mass balance observation uncertainty as "
                              "reported to the World Glacier Monitoring Service. "
                              f"If no value was reported it is set to "
                              f"{default_uncertainty} mm w.e.."
                }
            )
            datacube["wgms_mb_unc"] = wgms_mb_unc_da

            return datacube

        except:
            print(f"No WGMS data available for {gdir.rgi_id}.")
