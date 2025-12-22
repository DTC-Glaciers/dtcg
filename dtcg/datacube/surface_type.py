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
import xarray as xr

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

    def add_data_to_datacube(self, datacube, gdir, bin_intervall=30,
                             snow_cover_thresholds=[0.25, 0.5, 0.75]):
        """Every datacube must support this method.

        It should be able to add data to the provided datacube and returns the
        final datacube.
        """
        # this try except statement is quick and dirty, could be more elegant in
        # the future
        try:
            return self.add_sfc_type_observation(
                ds=datacube, gdir=gdir, bin_intervall=bin_intervall,
                snow_cover_thresholds=snow_cover_thresholds)
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

    def add_sfc_type_observation(self, ds, gdir, bin_intervall=30,
                                 snow_cover_thresholds=[0.25, 0.5, 0.75]):

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

        ds = self.add_snowline_from_observation(
            gdir, ds, bin_intervall=bin_intervall,
            snow_cover_thresholds=snow_cover_thresholds)

        # convert time coordinate
        EPOCH = np.datetime64("1970-01-01T00:00:00")
        time = ds["t_sfc_type"]
        seconds = (
            (time - EPOCH)
            .astype("timedelta64[s]")
            .astype("int64")
        )

        seconds.attrs = {
            "long_name": time.attrs.get("long_name", ""),
            "standard_name": "time",
            "units": "seconds since 1970-01-01 00:00:00",
        }

        ds = ds.assign_coords(t_sfc_type=seconds)

        return ds

    def add_snowline_from_observation(self, gdir, ds, bin_intervall=30,
                                      snow_cover_thresholds=[0.25, 0.5, 0.75]):
        # currently we only support 3 thresholds, but can be adapted below
        assert len(snow_cover_thresholds) == 3

        ds_count = get_categories_per_elevation_band(
            ds=ds, bin_intervall=bin_intervall)
        ds_count = exclude_empty_elevation_bins(ds=ds_count)
        ds_count = exclude_dates_with_to_much_cloud_cover(ds=ds_count)

        snowline_obs = []
        snowline_obs_lower = []
        snowline_obs_upper = []
        for date in ds.t_sfc_type:
            if date.values not in ds_count.t_sfc_type.values:
                snowline_obs_lower.append(np.nan)  # 25% snow cover
                snowline_obs.append(np.nan)  # 50% snow cover
                snowline_obs_upper.append(np.nan)
                continue
            ds_use = ds_count.sel(t_sfc_type=date)

            tmp_idx, tmp_obs = get_snowline(
                ds_use, thresholds=snow_cover_thresholds)
            snowline_obs_lower.append(tmp_obs[0])  # 25% snow cover
            snowline_obs.append(tmp_obs[1])  # 50% snow cover
            snowline_obs_upper.append(tmp_obs[2])  # 75% snow cover

        ds.coords['snowcover_frac'] = snow_cover_thresholds
        ds['snowcover_frac'].attrs = {
            'long_name': 'Minimum snowcover fraction per elevation bin derived '
                         'from surface type observations'
        }
        ds['sfc_type_snowline'] = (
            ('snowcover_frac', 't_sfc_type'),
            np.array([snowline_obs_lower, snowline_obs, snowline_obs_upper]),)

        # derive values for inf (fully or no snowcover)
        elev_band_edges = get_elev_band_edges(ds, bin_intervall=bin_intervall)

        ds['sfc_type_snowline'].attrs = {
            'long_name': 'Snowline altitude derived from glacier facies '
                         'classification',
            'data_source': 'ENVEO',
            'units': 'm',
            'inf_values': str({
                np.inf: float(np.max(elev_band_edges)),
                -np.inf: float(np.min(elev_band_edges)),
            })
        }

        return ds


def get_elev_band_edges(ds, topo_data='topo_smoothed', bin_intervall=30):
    """Get elevation band heights for selected inbervall in m.

    Parameters
    ----------
    gdir
    ds
    topo_data
    bin_intervall

    Returns
    -------

    """
    glacier_topo_flat = ds[topo_data].data[ds.glacier_mask.astype(bool)]
    max_glacier_elevation = np.max(glacier_topo_flat)
    min_glacier_elevation = np.min(glacier_topo_flat)

    # + and - bin_intervall just to be sure everything is included
    max_band_height = np.ceil(max_glacier_elevation / bin_intervall
                              ) * bin_intervall + bin_intervall
    min_band_height = np.floor(min_glacier_elevation / bin_intervall
                               ) * bin_intervall - bin_intervall

    return np.arange(max_band_height, min_band_height - 1, -bin_intervall)


def get_categories_per_elevation_band(ds,
                                      topo_data='topo_smoothed',
                                      category_data='sfc_type_data',
                                      category_uncertainty='sfc_type_uncertainty',
                                      category_time_var='t_sfc_type',
                                      nodata_value=np.nan,
                                      bin_intervall=30,
                                      ):

    # open needed data for aggregation
    elevations = ds[topo_data].data[ds.glacier_mask.astype(bool)]
    categories = ds[category_data].data[:, ds.glacier_mask.astype(bool)]
    categories_attrs = ds[category_data].attrs
    uncertainties = ds[category_uncertainty].data[:, ds.glacier_mask.astype(bool)]
    uncertainties_attrs = ds[category_uncertainty].attrs
    time_dim = categories.shape[0]
    time_values = ds[category_time_var]
    # Mask out nodata
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(categories)
    else:
        valid_mask = ~np.isin(categories, nodata_value)

    # get elevation bands, need to be ascending for np.digitize
    elev_band_edges = get_elev_band_edges(ds, topo_data=topo_data,
                                          bin_intervall=bin_intervall)
    n_bins = elev_band_edges.size - 1

    # assign elevation band indexes, -1 to start from 0
    bin_ids = np.digitize(elevations, bins=elev_band_edges) - 1

    def count_values_per_elevation_bin(data):
        # Get unique valid categories (excluding nodata and -1)
        unique_values = np.unique(data)
        unique_values = np.sort(unique_values)
        if np.isnan(nodata_value):
            unique_values = unique_values[~np.isnan(unique_values)]
        else:
            unique_values = np.setdiff1d(unique_values, [nodata_value])
        # we assign values from 0 to len(unique_cats) to be able to use hist 2d later
        values_to_index = {value: i for i, value in enumerate(unique_values)}

        # Create result array: shape should be (time, elevation_bin, valid categories)
        result = np.zeros((time_dim, n_bins, len(unique_values)), dtype=int)

        # Efficient per-time counting
        for t in range(time_dim):
            valid = valid_mask[t]
            binned = bin_ids[valid]
            datavals = data[t][valid]

            # Map category values to indices
            value_idx = np.array([values_to_index[c] for c in datavals])
            hist2d = np.zeros((n_bins, len(unique_values)), dtype=int)

            # Use np.add.at for fast 2D histogram
            np.add.at(hist2d, (binned, value_idx), 1)
            result[t] = hist2d

        return result, unique_values

    category_counts, unique_cats = count_values_per_elevation_bin(categories)
    uncertainty_counts, unique_uncert = count_values_per_elevation_bin(uncertainties)

    # create dataset of result
    ds = xr.Dataset(
        data_vars={
            "category_counts": (
                ("t_sfc_type", "elevation_bin", "category"),
                category_counts,
                {"long_name": "Count of glacier facies classification per elevation band"},
            ),
            "uncertainty_counts": (
                ("t_sfc_type", "elevation_bin", "uncertainty_flag"),
                uncertainty_counts,
                {
                    "long_name": "Count of glacier facies classification  uncertainty per elevation band"},
            ),
        },
        coords={
            "t_sfc_type": time_values,
            "elevation_bin": (("elevation_bin"), range(n_bins),
                              {"long_name": "Index of elevation bin"}),
            "lower_elevation": (("elevation_bin"), elev_band_edges[1:],
                                {"long_name": "Lower boundary of elevation bin"}),
            "upper_elevation": (("elevation_bin"), elev_band_edges[:-1],
                                {"long_name": "Upper boundary of elevation bin"}),
            "category": (("category"), unique_cats, categories_attrs),
            "uncertainty_flag": (("uncertainty_flag"), unique_uncert, uncertainties_attrs),
        },
    )

    return ds


def exclude_empty_elevation_bins(ds):
    # get the total number of observations per elevation band
    number_grid_points_elev_bin = ds.category_counts.sum(dim='category')

    # check that this number is the same for each timestamp
    assert np.all(number_grid_points_elev_bin == number_grid_points_elev_bin.isel(t_sfc_type=0))

    # if the number of grid points is 0 their is no valid data
    return ds.isel(elevation_bin=np.where(number_grid_points_elev_bin.isel(t_sfc_type=0) != 0)[0])


def exclude_dates_with_to_much_cloud_cover(ds, cloud_cover=0.5):
    relative_cloud_cover = (ds.category_counts.sum(dim='elevation_bin').sel(category=5) /
                            ds.category_counts.sum(dim=['elevation_bin', 'category']))

    return ds.isel(t_sfc_type=np.where(relative_cloud_cover < 0.5)[0])


def get_snowline(ds, thresholds=[0.25, 0.5, 0.75]):
    # the provided ds should already be cleaned for scenes
    da_counts = ds.category_counts

    all_grid_points_elev_bin = da_counts.sum(dim='category')
    cloud_grid_points = da_counts.sel(category=5)

    # exclude completely cloud covered elevation bins
    da_cloudfree = da_counts.isel(
        elevation_bin=np.where(all_grid_points_elev_bin != cloud_grid_points)[0])

    # calculate relative contributions, excluding cloud grid points
    # we assume all surface types are equally covored by clouds
    da_rel_snow = da_cloudfree.sel(category=1) / da_cloudfree.sel(category=[1, 2, 3, 4]).sum(
        dim='category')

    # use different thresholds of the snow fraction for integrating uncertainty
    def get_lowest_elev_bin_exceeding_threshold(threshold):
        exceeding_threshold = da_rel_snow > threshold

        # check if lowest bin is snow covered
        if exceeding_threshold.isel(elevation_bin=-1):
            return da_counts.elevation_bin.values[-1] + 1, -np.inf
        # check if no bin exceeds threshold
        elif not np.any(exceeding_threshold):
            return -1, np.inf
        else:
            lowest_bin = da_rel_snow.sel(elevation_bin=exceeding_threshold).isel(elevation_bin=-1)
            return lowest_bin.elevation_bin.values.item(), lowest_bin.lower_elevation.values.item()

    lowest_elev_bin_idx = []
    lowest_elev_bin_values = []
    for threshold in thresholds:
        tmp_idx, tmp_value = get_lowest_elev_bin_exceeding_threshold(threshold)
        lowest_elev_bin_idx.append(tmp_idx)
        lowest_elev_bin_values.append(tmp_value)

    return lowest_elev_bin_idx, lowest_elev_bin_values

