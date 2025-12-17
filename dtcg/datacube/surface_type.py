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

Functionality for retrieving ENVEO surface type classification data.
"""

import warnings
import numpy as np
import pandas as pd

from oggm import utils
import salem


class DatacubeSurfaceType:
    def __init__(self):

        self.data_base_url = ('https://cluster.klima.uni-bremen.de/~dtcg/test_files/'
                              'case_study_regions/austria/sfc_type_obs/merged_data/')
        self.files_used = [
            '20150704_R022.tif', '20150803_R022.tif', '20150813_R022.tif',
            '20150826_R065.tif', '20151022_R022.tif', '20160522_R065.tif',
            '20160628_R022.tif', '20160718_R022.tif', '20160807_R022.tif',
            '20160827_R022.tif', '20160909_R065.tif', '20160926_R022.tif',
            '20161016_R022.tif', '20161029_R065.tif', '20170517_R065.tif',
            '20170527_R065.tif', '20170613_R022.tif', '20170626_R065.tif',
            '20170706_R065.tif', '20170716_R065.tif', '20170805_R065.tif',
            '20170807_R022.tif', '20170822_R022.tif', '20170825_R065.tif',
            '20170830_R065.tif', '20170921_R022.tif', '20180616_R065.tif',
            '20180623_R022.tif', '20180701_R065.tif', '20180713_R022.tif',
            '20180718_R022.tif', '20180726_R065.tif', '20180731_R065.tif',
            '20180817_R022.tif', '20180820_R065.tif', '20180827_R022.tif',
            '20180919_R065.tif', '20180921_R022.tif', '20180926_R022.tif',
            '20190502_R065.tif', '20190524_R022.tif', '20190601_R065.tif',
            '20190603_R022.tif', '20190613_R022.tif', '20190618_R022.tif',
            '20190628_R022.tif', '20190716_R065.tif', '20190723_R022.tif',
            '20190827_R022.tif', '20190830_R065.tif', '20190904_R065.tif',
            '20190916_R022.tif', '20190921_R022.tif', '20190929_R065.tif',
            '20200521_R065.tif', '20200602_R022.tif', '20200612_R022.tif',
            '20200627_R022.tif', '20200707_R022.tif', '20200712_R022.tif',
            '20200727_R022.tif', '20200801_R022.tif', '20200806_R022.tif',
            '20200809_R065.tif', '20200811_R022.tif', '20200819_R065.tif',
            '20200821_R022.tif', '20200903_R065.tif', '20200905_R022.tif',
            '20200908_R065.tif', '20200913_R065.tif', '20200915_R022.tif',
            '20210811_R022.tif', '20210814_R065.tif', '20210821_R022.tif',
            '20210903_R065.tif', '20210905_R022.tif', '20210910_R022.tif',
            '20210913_R065.tif', '20210918_R065.tif', '20210923_R065.tif',
            '20210925_R022.tif']

        # the above file handling can be improved in the future, e.g. by using:

        # import requests
        # from bs4 import BeautifulSoup
        #
        # response = requests.get(data_base_url)
        # soup = BeautifulSoup(response.text, "html.parser")
        #
        # files_used = []
        # sfc_type_dates = []
        # for link in soup.find_all("a", href=True):
        #     file = link['href']
        #     if file.lower().endswith('.tif'):
        #         date = file.split('_')[0]
        #         if date in sfc_type_dates:
        #             # their is already an observation for this date available
        #             raise ValueError(f'Surface type for {date} already added! '
        #                              'Check the provided input files, for each '
        #                              'date only one should be provided!')
        #         sfc_type_dates.append(date)
        #         files_used.append(file)

    def add_data_to_datacube(self, datacube, gdir):
        """Every datacube must support this method.

        It should be able to add data to the provided datacube and returns the
        final datacube.
        """
        # this try except statement is quick and dirty, could be more elegant in
        # the future
        try:
            return self.add_sfc_type_observation(ds=datacube, gdir=gdir)
        except RuntimeError:
            print(f"No surface type observation available for {gdir.rgi_id}")

    # This function is based on oggm.shop.millan22._filter_and_reproj
    def reproject_single_sfc_type_file(self, gdir, input_file):
        # Subset to avoid mega files
        dsb = salem.GeoTiff(input_file)

        x0, x1, y0, y1 = gdir.grid.extent_in_crs(dsb.grid.proj)
        with warnings.catch_warnings():
            # This can trigger an out of bounds warning
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message='.*out of bounds.*')
            dsb.set_subset(corners=((x0, y0), (x1, y1)),
                           crs=dsb.grid.proj,
                           margin=5)

        data_sfc_types = dsb.get_vardata(var_id=1)
        data_uncertainty = dsb.get_vardata(var_id=2)

        # Reproject now
        with warnings.catch_warnings():
            # This can trigger an out of bounds warning
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message='.*out of bounds.*')
            r_data_sfc_types = gdir.grid.map_gridded_data(data_sfc_types, dsb.grid,
                                                          interp='nearest')
            r_data_uncertainty = gdir.grid.map_gridded_data(data_uncertainty, dsb.grid,
                                                            interp='nearest')

        return r_data_sfc_types.data, r_data_uncertainty.data

    def add_sfc_type_observation(self, ds, gdir):

        # prepare structure for data
        sfc_type_data = np.zeros((len(self.files_used), *ds['glacier_mask'].shape))
        sfc_type_uncertainty = np.zeros((len(self.files_used), *ds['glacier_mask'].shape))

        # loop through all files and add one after the other
        for i, filename in enumerate(self.files_used):
            # download data
            input_file = utils.file_downloader(self.data_base_url + filename)

            r_data, r_uncertainty = self.reproject_single_sfc_type_file(
                gdir, input_file)
            sfc_type_data[i, :] = r_data
            sfc_type_uncertainty[i, :] = r_uncertainty

        # use nan for missing data
        missing_data_val = np.nan
        sfc_type_data = np.where(sfc_type_data == 255,
                                 missing_data_val, sfc_type_data)
        sfc_type_uncertainty = np.where(sfc_type_uncertainty == 255,
                                        missing_data_val, sfc_type_uncertainty)

        # add to gridded data with some attributes
        sfc_type_dates = [file.split('_')[0] for file in self.files_used]
        ds.coords['t_sfc_type'] = pd.to_datetime(sfc_type_dates, format="%Y%m%d")

        ds['t_sfc_type'].attrs = {
            'long_name': 'Timestamp of surface type observations'
        }
        ds['sfc_type_data'] = (('t_sfc_type', 'y', 'x'), sfc_type_data)
        ds['sfc_type_data'].attrs = {
            'long_name': 'Glacier facies classification',
            'data_source': 'ENVEO',
            'units': 'none',
            'code_description': 'Extract dict with ast.literal_eval(ds.code).',
            'code': str({
                0: 'unclassified',
                1: 'snow',
                2: 'firn / old snow / bright ice',
                3: 'clean ice',
                4: 'debris',
                5: 'cloud',
                'nan': 'no data',
            }),
        }

        ds['sfc_type_uncertainty'] = (('t_sfc_type', 'y', 'x'), sfc_type_uncertainty)
        ds['sfc_type_uncertainty'].attrs = {
            'long_name': 'Glacier facies classification uncertainty',
            'data_source': 'ENVEO',
            'units': 'none',
            'code_description': 'Extract dict with ast.literal_eval(ds.code).',
            'code': str({
                1: 'low uncertainty for illuminated pixel',
                2: 'medium uncertainty for illuminated pixel',
                3: 'high uncertainty for illuminated pixel',
                5: 'cloud',
                11: 'low uncertainty for shaded pixel',
                12: 'medium uncertainty for shaded pixel',
                13: 'high uncertainty for shaded pixel',
                'nan': 'no data',
            }),
        }

        ds = ds.sortby('t_sfc_type')

        return ds
