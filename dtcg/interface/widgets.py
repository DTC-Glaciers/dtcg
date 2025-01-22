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

Widgets used by Jupyter-based interfaces.
"""

import json

import ipywidgets  # avoid collisions with ipywidgets.widgets
from IPython.display import display


class OutputWidget:
    """Parent class for output widgets."""

    def __init__(self, *args, **kwargs):
        super(OutputWidget, self).__init__(*args, **kwargs)
        self.layout = {
            "border": "1px solid black",
            "display": "flex",
            "flex_flow": "column nowrap",
            "justify_content": "space-between",
            "align_items": "stretch",
        }
        self.out = ipywidgets.widgets.Output(layout=self.layout)


class WidgetSelectSubRegion(OutputWidget):
    def __init__(self, *args, **kwargs):
        super(WidgetSelectSubRegion, self).__init__(*args, **kwargs)
        self.layout = {
            "border": "1px solid black",
            "width": "50%",
            "display": "flex",
            "flex_flow": "column nowrap",
            "justify_content": "space-between",
            "align_items": "stretch",
        }
        self.query = {"query": "select_subregion"}
        self.out = ipywidgets.widgets.Output(layout=self.layout)

        self.dropdown_subregion = self.get_dropdown_subregion()
        self.multiprocessing_checkbox = self.get_multiprocessing_checkbox()
        self.result = self.get_result_widget()

    # @classmethod
    def get_dropdown_subregion(self):
        self.dropdown_subregion = ipywidgets.widgets.Dropdown(
            options=["Alps", "Karakaoram", "Southern and Eastern Europe"],
            value="Alps",
            description="Subregion:",
            disabled=False,
        )
        return self.dropdown_subregion

    # @classmethod
    def get_multiprocessing_checkbox(self):
        self.checkbox_multiprocessing = ipywidgets.widgets.Checkbox(
            value=False,
            description="Enable multiprocessing:",
            disabled=False,
            indent=False,
        )
        return self.checkbox_multiprocessing

    # @classmethod
    def get_result_widget(self):
        self.result = ipywidgets.widgets.HTML(
            value="{}",
            description="Generated API Query:",
            style={"description_width": "initial"},
        )
        return self.result

    def get_result(self):
        return self.result

    def gen_query(self, _):
        query_params = {
            "query": "select_subregion",
            "subregion_name": self.dropdown_subregion.value,
            "oggm_params": {
                "use_multiprocessing": self.checkbox_multiprocessing.value,
                "rgi_version": "62",
            },
        }
        self.result.value = f"{query_params}"
        self.query = query_params

    # @classmethod
    def get_output(self):
        self.dropdown_subregion.observe(self.gen_query)
        self.checkbox_multiprocessing.observe(self.gen_query)
        # self.query = self.result.observe(self.gen_query)

        combi_widget = ipywidgets.widgets.VBox(
            [
                ipywidgets.widgets.HBox(
                    [self.dropdown_subregion, self.checkbox_multiprocessing]
                ),
                self.result,
            ]
        )
        self.out.append_display_data(combi_widget)
        display(self.out)
        return self.out

    def get_query(self):
        return self.query

    def get_query_json(self):
        if self.query:
            query = json.dumps(self.query)

        # else:
        #     query = json.loads("{}")
        return query
