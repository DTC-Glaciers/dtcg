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

Provides shared fixtures for tests.

Use these to replace duplicated code.

For generating objects within a test function's scope, call a fixture
directly:

    .. code-block:: python

        def test_foobar(self, conftest_mock_grid):
            grid_object = conftest_mock_grid
            grid_object.set_foo(foo=bar)
            ...
"""

pytest_plugins = "oggm.tests.conftest"
