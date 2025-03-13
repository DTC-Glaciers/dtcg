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

Entry point for user DCTG API calls. All API requests and responses pass
through here.
"""

import dtcg.integration.oggm_bindings as oggm_bindings


class RequestAPICtor:
    """Mock API request. Placeholder for API query until the backend is
    set up.

    Attributes
    ----------
    query : str
        The type of query desired by the user.
    region_name : str
        The region selected by the user. Default None.
    subregion_name : str
        The subregion selected by the user. Default None.
    shapefile_path : str
        Path to a shapefile selected/uploaded by the user. Default None.
    glacier_name : str
        The glacier selected by the user. Default None.
    oggm_params : dict
        OGGM model parameters. Default None.
    """

    def __init__(
        self,
        action: str,
        region_name: str = None,
        subregion_name: str = None,
        glacier_name: str = None,
        shapefile_path: str = None,
        oggm_params: dict = None,
    ):
        super().__init__()

        self.action = action
        self.region_name = region_name
        self.subregion_name = subregion_name
        self.glacier_name = glacier_name
        self.shapefile_path = shapefile_path
        if oggm_params:
            self.oggm_params = oggm_params
        else:
            self.oggm_params = {}

        self.query = self.get_query()

    def get_query(self) -> dict:
        # return self.__dict__
        query = {}
        for attribute in self.__dict__.keys():
            if attribute[:2] != "__" and attribute != "query":
                value = getattr(self, attribute)
                if not callable(value):
                    query[attribute] = value
        return query


class GatewayHandler:
    def __init__(self, query: dict):
        super().__init__()

        self._set_user_query(query)
        self.response = self.get_response()

    def _set_user_query(self, query: dict, **kwargs):
        """Create a user query."""
        self.query = RequestAPICtor(**query, **kwargs)

    def get_response(self) -> dict:
        """Get response to API query."""

        # This should eventually link to ``dtcg.api.external``.
        response = self._get_query_handler(query=self.query)
        return response

    def _get_query_handler(self, query: RequestAPICtor) -> dict:
        """Redirect query to appropriate binding.

        This currently links directly to the binding. This should eventually
        be replaced by calling the binding via ``dtcg.api``.
        """

        # Currently we link directly to the bindings until the internal API is set up (dtcg.api)
        if query.action == "select_subregion":
            binder = oggm_bindings.BindingsHydro()
            data = binder.get_user_subregion(
                region_name=query.region_name,
                subregion_name=query.subregion_name,
                shapefile_path=query.shapefile_path,
                **query.oggm_params,
            )
            response = {"response_code": "200", "data": data}
            response["data"]["runoff_data"] = binder.get_aggregate_runoff(
                data=response["data"]["glacier_data"]
            )

        elif query.action == "select_glacier":
            binder = oggm_bindings.BindingsHydro()
            data = binder.get_user_subregion(
                region_name=query.region_name,
                subregion_name=query.subregion_name,
                shapefile_path=query.shapefile_path,
                **query.oggm_params,
            )
            response = {"response_code": "200", "data": data}
            if (
                query.glacier_name
                in response["data"]["glacier_data"]["Name"].dropna().values
            ):
                response["data"]["runoff_data"] = binder.get_runoff(
                    data=response["data"]["glacier_data"], name=query.glacier_name
                )
        else:
            response = {"response_code": "422"}
            raise NotImplementedError(f"{query.action} is not yet implemented.")
        return response


def main():
    pass


if __name__ == "__main__":
    main()
