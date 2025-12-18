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

"""

import pandas as pd

from dtcg.validation.wgms_validation import validate_with_wgms, plot_wgms_annual
from dtcg.validation.validation_metrics import get_supported_metrics_descriptions


class DatacubeValidator:
    def __init__(self, datatree):
        self.datatree = datatree

        # define here all available validation observations
        self.supported_obs_for_validation = {
            'WGMS': validate_with_wgms,
        }

        self.supported_obs_for_plotting = {
            'WGMS_annual': plot_wgms_annual,
        }

    def get_datacube_from_datatree(self, datacube_name):
        if datacube_name not in self.datatree.keys():
            raise ValueError(
                f"No datacube with name '{datacube_name}' available in provided "
                f"datatree. Options are {list(self.datatree.keys())}.")

        return self.datatree[datacube_name]

    def get_validation_for_layers(self, l2_name_list=None, l1_name="L1",
                                  return_bootstrap_args=False, **kwargs):
        l1_datacube = self.get_datacube_from_datatree(l1_name)

        if l2_name_list is None:
            l2_name_list = [L2_name for L2_name in self.datatree.keys()
                            if 'L2' in L2_name]

        validation_metrics = {}
        bootstrap_args = None

        for l2_name in l2_name_list:
            l2_datacube = self.get_datacube_from_datatree(l2_name)

            for obs, validation_fct in self.supported_obs_for_validation.items():
                validation_tmp = validation_fct(
                    l1_datacube=l1_datacube,
                    l2_datacube=l2_datacube,
                    return_bootstrap_args=return_bootstrap_args,
                    **kwargs
                )

                if validation_tmp is not None:
                    if return_bootstrap_args:
                        validation_tmp, bootstrap_args = validation_tmp
                    df_tmp = pd.DataFrame.from_dict(validation_tmp)
                    df_tmp.index = [l2_name]
                    df_tmp.index.name = obs

                    if obs not in validation_metrics:
                        validation_metrics[obs] = df_tmp
                    else:
                        validation_metrics[obs] = pd.concat(
                            [validation_metrics[obs], df_tmp])

        if return_bootstrap_args:
            return validation_metrics, bootstrap_args
        return validation_metrics

    def get_description_of_metrics(self):
        return get_supported_metrics_descriptions()

    def get_validation_plot_for_layers(self, obs_name, l2_name_list=None,
                                       l1_name="L1"):
        if obs_name not in self.supported_obs_for_plotting:
            raise ValueError(
                f"Currently not plotting for {obs_name} supported. Available "
                f"options are {list(self.supported_obs_for_plotting.keys())}."
            )
        l1_datacube = self.get_datacube_from_datatree(l1_name)

        if l2_name_list is None:
            l2_name_list = [L2_name for L2_name in self.datatree.keys()
                            if 'L2' in L2_name]

        return self.supported_obs_for_plotting[obs_name](
            l1_datacube=l1_datacube,
            datatree=self.datatree,
            l2_name_list=l2_name_list,
        )
