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


# Mock API request - placeholder only until the full API is set up
class RequestAPIConstructor:
    """Placeholder for API object.

    Attributes
    ----------
    query : str
        The type of query desired by the user.
    region_name : str
        The region selected by the user. Default None.
    shapefile_path : str
        Path to a shapefile selected/uploaded by the user. Default None.
    """

    def __init__(
        self,
        query: str,
        region_name: str = None,
        shapefile_path: str = None,
        oggm_params: dict = None,
    ):
        self.query = query
        self.region_name = region_name
        self.shapefile_path = shapefile_path
        if oggm_params:
            self.oggm_params = oggm_params
        else:
            self.oggm_params = {}


def _set_user_query(query: str, **kwargs) -> RequestAPIConstructor:
    user_query = RequestAPIConstructor(query, **kwargs)
    return user_query


def get_query_response(query: RequestAPIConstructor) -> dict:

    # This should eventually link to ``dtcg.api.external``.
    response = _get_query_handler(query=query)
    return response


def _get_query_handler(query: RequestAPIConstructor) -> dict:
    """Redirect query to appropriate binding.

    Currently this links directly to the binding. This should eventually be replaced by calling the binding via ``dtcg.api``.
    """

    # Currently we link directly to the bindings until the internal API is set up (dtcg.api)
    if query.query == "select_region":
        data = oggm_bindings.get_user_subregion(
            region_name=query.region_name,
            shapefile_path=query.shapefile_path,
            **query.oggm_params,
        )
        response = {"response_code": "200", "data": data}
    else:
        response = {"response_code": "422"}
        raise NotImplementedError(f"{query.query} is not yet implemented.")
    return response


def main():
    pass


if __name__ == "__main__":
    main()
