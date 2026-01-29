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

import logging

import pandas as pd

from dtcg.validation.annual_mb_validation import plot_annual_mb, validate_with_annual_mb
from dtcg.validation.cryosat_validation import plot_cryosat, validate_with_cryosat
from dtcg.validation.sentinel_validation import plot_sentinel, validate_with_sentinel
from dtcg.validation.validation_metrics import get_supported_metrics_descriptions
from dtcg.validation.wgms_validation import plot_wgms_annual, validate_with_wgms

# Module logger
log = logging.getLogger(__name__)


class DatacubeValidator:
    def __init__(self, datatree):
        self.datatree = datatree

        # define here all available validation observations
        self.supported_obs_for_validation = {
            "WGMS": {"fct": validate_with_wgms, "label": "L1 WGMS annual mass balance"},
            "CryoSat2": {
                "fct": validate_with_cryosat,
                "label": "L1 CryoSat2 elevation change",
            },
            "Sentinel2": {
                "fct": validate_with_sentinel,
                "label": "L1 Sentinel2 snowline",
            },
        }

        self.supported_user_obs = {
            "annual_mb": {
                "fct": validate_with_annual_mb,
                "label": "User provided annual mass balance",
            }
        }

        self.supported_obs_for_plotting = {
            "WGMS": plot_wgms_annual,
            "CryoSat2": plot_cryosat,
            "Sentinel2": plot_sentinel,
        }

        self.supported_user_obs_for_plotting = {
            "annual_mb": plot_annual_mb,
        }

    def get_datacube_from_datatree(self, datacube_name):
        if datacube_name not in self.datatree.keys():
            raise ValueError(
                f"No datacube with name '{datacube_name}' available in provided "
                f"datatree. Options are {list(self.datatree.keys())}."
            )

        return self.datatree[datacube_name]

    def get_validation_for_layers(
        self,
        l2_name_list=None,
        l1_name="L1",
        obs_list=None,
        user_observation=None,
        validation_period=None,
        return_bootstrap_args=False,
        ignore_missing_data=True,
        **kwargs,
    ):
        l1_datacube = self.get_datacube_from_datatree(l1_name)

        if l2_name_list is None:
            l2_name_list = [
                L2_name for L2_name in self.datatree.keys() if "L2" in L2_name
            ]

        validation_metrics = {}
        bootstrap_args = {}

        def add_result_to_dicts(validation_tmp, obs, label, name):
            if validation_tmp is not None:
                if return_bootstrap_args:
                    validation_tmp, used_period, bootstrap_args_tmp = validation_tmp
                    bootstrap_args[obs] = bootstrap_args_tmp
                else:
                    validation_tmp, used_period = validation_tmp
                df_tmp = pd.DataFrame.from_dict(validation_tmp)
                df_tmp.index = [name]
                df_tmp.index.name = f"{label}\n{used_period}"

                if obs not in validation_metrics:
                    validation_metrics[obs] = df_tmp
                else:
                    validation_metrics[obs] = pd.concat(
                        [validation_metrics[obs], df_tmp]
                    )

        for l2_name in l2_name_list:
            l2_datacube = self.get_datacube_from_datatree(l2_name)

            if user_observation is None:
                if obs_list is None:
                    obs_list = self.supported_obs_for_validation.keys()
                for obs in obs_list:
                    obs_args = self.supported_obs_for_validation[obs]
                    try:
                        validation_tmp = obs_args["fct"](
                            l1_datacube=l1_datacube,
                            l2_datacube=l2_datacube,
                            return_bootstrap_args=return_bootstrap_args,
                            validation_period=validation_period,
                            **kwargs,
                        )
                    except ValueError:
                        if ignore_missing_data:
                            log.warning(f"Not all data available for {obs}")
                            continue
                        else:
                            raise

                    add_result_to_dicts(
                        validation_tmp=validation_tmp,
                        obs=obs,
                        label=obs_args["label"],
                        name=l2_name,
                    )

            else:
                if user_observation["obs_type"] not in self.supported_user_obs:
                    raise NotImplementedError(
                        f"{user_observation['obs_type']} observation type not "
                        f"supported! Currently available are "
                        f"{list(self.supported_user_obs.keys())}."
                    )

                obs = user_observation["obs_type"]
                obs_dict = self.supported_user_obs[obs]
                validation_tmp = obs_dict["fct"](
                    observation=user_observation,
                    l2_datacube=l2_datacube,
                    return_bootstrap_args=return_bootstrap_args,
                    validation_period=validation_period,
                    **kwargs,
                )
                if "name" in user_observation:
                    label = user_observation["name"]
                else:
                    label = obs_dict["label"]
                add_result_to_dicts(
                    validation_tmp=validation_tmp,
                    obs=obs,
                    label=label,
                    name=l2_name,
                )

        if return_bootstrap_args:
            return validation_metrics, bootstrap_args
        return validation_metrics

    def get_description_of_metrics(self):
        return get_supported_metrics_descriptions()

    def get_validation_plot_for_layers(
        self,
        obs_name=None,
        user_observation=None,
        l2_name_list=None,
        l1_name="L1",
        **kwargs,
    ):

        if l2_name_list is None:
            l2_name_list = [
                L2_name
                for L2_name in self.datatree.keys()
                if "L2" in L2_name or "L3" in L2_name
            ]

        if user_observation is None:
            if obs_name not in self.supported_obs_for_plotting:
                raise ValueError(
                    f"Currently no plotting for {obs_name} supported. Available "
                    f"options are {list(self.supported_obs_for_plotting.keys())}."
                )

            l1_datacube = self.get_datacube_from_datatree(l1_name)

            return self.supported_obs_for_plotting[obs_name](
                l1_datacube=l1_datacube,
                datatree=self.datatree,
                l2_name_list=l2_name_list,
                **kwargs,
            )

        else:
            if user_observation["obs_type"] not in self.supported_user_obs_for_plotting:
                raise ValueError(
                    f"Currently no plotting for {user_observation['obs_type']} "
                    f"supported. Available options are "
                    f"{list(self.supported_user_obs_for_plotting.keys())}."
                )

            return self.supported_user_obs_for_plotting[user_observation["obs_type"]](
                observation=user_observation,
                datatree=self.datatree,
                l2_name_list=l2_name_list,
                **kwargs,
            )
