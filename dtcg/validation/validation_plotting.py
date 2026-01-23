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

from matplotlib.collections import PolyCollection
import numpy as np


def add_line_with_unc(ax, x, y, y_unc, c, label, legend_handles, legend_labels,
                      label_unc="± uncertainty", alpha=0.25):
    line, = ax.plot(x, y, color=c, marker='.')

    if isinstance(y_unc, list):
        y_lower = y_unc[0]
        y_upper = y_unc[1]
    else:
        y_lower = y - y_unc
        y_upper = y + y_unc

    band = ax.fill_between(x, y_lower, y_upper,
                           color=c, alpha=alpha, )
    # save here the data for automatic scaling of y-axis later
    band._data_for_scaling = (x, y_lower, y_upper)

    # save entries for legend
    legend_handles.append((line, band))
    legend_labels.append(f"{label} {label_unc}")


def autoscale_y_from_fill_between(ax, margin=0.05):
    xmin, xmax = ax.get_xlim()
    ymins = []
    ymaxs = []

    for coll in ax.collections:
        if not isinstance(coll, PolyCollection):
            continue
        if not hasattr(coll, "_data_for_scaling"):
            continue

        x, ylo, yhi = coll._data_for_scaling
        x = x.astype(float)
        m = (x >= xmin) & (x <= xmax)
        if not np.any(m):
            continue

        ymins.append(np.nanmin(np.r_[ylo[m], yhi[m]]))
        ymaxs.append(np.nanmax(np.r_[ylo[m], yhi[m]]))

    if not ymins:
        return

    y_min = min(ymins)
    y_max = max(ymaxs)

    # padding similar to matplotlib autoscale
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0
    pad = margin * y_range
    ax.set_ylim(y_min - pad, y_max + pad)
