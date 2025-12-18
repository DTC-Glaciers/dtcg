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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, Literal

import numpy as np

# import metrics from oggm
from oggm.utils import mad, md, rmsd, corrcoef


def get_supported_metrics():
    # the keys define the column names of the final validation df output
    return {
        'MeanAbsD': {
            'fct_name': 'mad',
            'add_unit': True,
            'fmt': '.1f',
            'description': 'Mean Absolute Deviation'},
        'MeanD': {
            'fct_name': 'mean_bias',
            'add_unit': True,
            'fmt': '.1f',
            'description': 'Mean Deviation'},
        'MedianD': {
            'fct_name': 'median_bias',
            'add_unit': True,
            'fmt': '.1f',
            'description': 'Median Deviation'},
        'RMSD': {
            'fct_name': 'rmsd',
            'add_unit': True,
            'fmt': '.1f',
            'description': 'Root Mean Squared Deviation'},
        'CORRCOEF': {
            'fct_name': 'corrcoef',
            'add_unit': False,
            'fmt': '.2f',
            'description': 'Pearson correlation coefficient'},
    }


MetricName = Literal["median_bias", "mean_bias", "mad", "rmsd", "corrcoef"]


def get_supported_metrics_descriptions():
    supported_metrics = get_supported_metrics()
    metrics_descriptions = {}
    for key in supported_metrics:
        metrics_descriptions[key] = supported_metrics[key]['description']
    return metrics_descriptions


@dataclass(frozen=True)
class BootstrapMetricResult:
    metric: str
    point_estimate: float
    ci: Tuple[float, float]
    ci_level: float
    n: int
    n_boot: int
    block_length: int
    seed: Optional[int]


def _as_1d_float_array(x: Union[Iterable[float], np.ndarray], name: str
                       ) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(~np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/Inf.")
    return a


def _as_2d_float_array(x: Union[Iterable[Iterable[float]], np.ndarray],
                       name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D array (n_time, n_quantiles).")
    if a.size == 0:
        raise ValueError(f"{name} is empty.")
    if np.any(~np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/Inf.")
    return a


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (ppf) using Acklam's rational approximation.
    Accurate enough for typical CI levels used in uncertainty work.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1).")

    # Coefficients from Peter J. Acklam's approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den

    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return -(num / den)

    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return num / den


def _validate_quantiles(q_levels: np.ndarray, y2_q: np.ndarray,
                        tol: float = 0.0) -> None:
    if np.any(q_levels <= 0.0) or np.any(q_levels >= 1.0):
        raise ValueError("y2_q_levels must be strictly within (0,1).")
    if np.any(np.diff(q_levels) <= 0.0):
        raise ValueError("y2_q_levels must be strictly increasing.")

    # Quantiles must be non-decreasing per timestamp
    diffs = np.diff(y2_q, axis=1)
    if np.any(diffs < -tol):
        raise ValueError(
            "y2_quantiles must be non-decreasing across quantiles for each "
            "timestamp (row-wise). Fix upstream or increase tol slightly for "
            "tiny numerical inversions."
        )


def _inv_cdf_piecewise_linear(
    u: np.ndarray,
    q_levels: np.ndarray,
    q_values: np.ndarray,
    *,
    tail: str = "clamp",
) -> np.ndarray:
    """
    Inverse-CDF sampling via linear interpolation in quantile space.

    u: (n,) Uniform(0,1)
    q_levels: (m,) increasing in (0,1)
    q_values: (n,m) quantile values per sample
    tail: "clamp" (recommended) or "extrapolate"
    """
    n, m = q_values.shape
    if tail not in {"clamp", "extrapolate"}:
        raise ValueError('model_tail must be "clamp" or "extrapolate".')

    lo_q, hi_q = q_levels[0], q_levels[-1]
    seg = np.searchsorted(q_levels, u, side="right") - 1  # [-1, m-1]
    seg = np.clip(seg, 0, m - 2)

    if tail == "clamp":
        u_eff = np.clip(u, lo_q, hi_q)
    else:
        u_eff = u

    q0 = q_levels[seg]
    q1 = q_levels[seg + 1]
    y0 = q_values[np.arange(n), seg]
    y1 = q_values[np.arange(n), seg + 1]

    w = (u_eff - q0) / (q1 - q0)
    out = y0 + w * (y1 - y0)

    if tail == "clamp":
        out = np.where(u <= lo_q, q_values[:, 0], out)
        out = np.where(u >= hi_q, q_values[:, -1], out)
    else:
        below = u < lo_q
        above = u > hi_q
        if np.any(below):
            q0b, q1b = q_levels[0], q_levels[1]
            y0b = q_values[below, 0]
            y1b = q_values[below, 1]
            wb = (u[below] - q0b) / (q1b - q0b)
            out[below] = y0b + wb * (y1b - y0b)
        if np.any(above):
            q0a, q1a = q_levels[-2], q_levels[-1]
            y0a = q_values[above, -2]
            y1a = q_values[above, -1]
            wa = (u[above] - q0a) / (q1a - q0a)
            out[above] = y0a + wa * (y1a - y0a)

    return out


def _compute_metric(ref: np.ndarray, data: np.ndarray, metric: MetricName
                    ) -> float:
    if metric == "median_bias":
        return float(np.median(data - ref))
    elif metric == "mean_bias":
        return float(md(ref, data))
    elif metric == "mad":
        return float(mad(ref, data))
    elif metric == "rmsd":
        return float(rmsd(ref, data))
    elif metric == "corrcoef":
        return float(corrcoef(ref, data))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def bootstrap_metric_obs_normal_mdl_quantiles(
    obs_median: Union[Iterable[float], np.ndarray],
    obs_unc: Union[Iterable[float], np.ndarray],
    mdl_q_levels: Union[Iterable[float], np.ndarray],
    mdl_quantiles: Union[Iterable[Iterable[float]], np.ndarray],
    *,
    metric: MetricName = "median_bias",
    obs_bounds_level: float = 0.95,
    ci_level: float = 0.68,
    n_boot: int = 5000,
    block_length: Optional[int] = None,
    seed: Optional[int] = 0,
    model_tail: str = "clamp",
    allow_quantile_inversions_tol: float = 0.0,
    q50_tol: float = 1e-12,
) -> BootstrapMetricResult:
    """
    Compute a comparison metric between observed and modelled time series and
    estimate its uncertainty using a moving-block bootstrap that accounts for
    both temporal dependence and uncertainty in observations and model output.

    This function is intended for applications where observations have symmetric
    uncertainty and model output is represented by asymmetric predictive
    distributions (quantiles).

    Overview of the method
    ----------------------
    The procedure combines two sources of uncertainty:
      1) Temporal sampling uncertainty, addressed via a moving-block bootstrap
         that resamples contiguous multi-year blocks to preserve interannual
         persistence.
      2) Within-year uncertainty, addressed by sampling plausible realizations
         of observations and model output at each resampled time step.

    For each bootstrap replicate:
      - A sequence of time indices is generated using moving-block resampling.
      - Observation values are drawn from a Normal distribution centered on the
        observed median with standard deviation derived from the reported
        uncertainty.
      - Model values are drawn from the model’s predictive distribution using
        inverse-CDF sampling based on the provided quantiles.
      - Residuals between observations and model are computed and summarized
        using the selected metric.

    Repeating this process yields an empirical distribution of the metric, from
    which percentile-based confidence intervals are derived.

    Definitions of metrics
    ----------------------
    Let r_t = obs_t − mdl_t be the residual at time t.

    Supported metrics (argument `metric`):
      - "median_bias":
            median_t(r_t)
        Robust estimate of the typical signed difference.
      - "mean_bias":
            mean_t(r_t)
        Mean deviation (mean bias), sensitive to systematic offsets.
      - "mad":
            mean_t(|r_t|)
        Mean absolute deviation (or MAE), representing typical error magnitude.
      - "rmsd":
            sqrt(mean_t(r_t^2))
        Root-mean-square deviation, emphasizing larger deviations.
      - "corrcoef":
            Pearson correlation coefficient

    Uncertainty models
    ------------------
    Observations (obs):
      - Inputs: obs_median, obs_unc
      - obs_unc is interpreted as a symmetric half-width around obs_median
        corresponding to a central interval with coverage `obs_bounds_level`
        (e.g., 95%).
      - A Normal distribution is assumed:
            obs_t ~ Normal(obs_median_t, sigma_t),
        where sigma_t = obs_unc_t / z and
            z = Phi^{-1}((1 + obs_bounds_level) / 2).

    Model (mdl):
      - Inputs: mdl_q_levels and mdl_quantiles.
      - mdl_q_levels must include 0.5; the corresponding quantile is used as the
        model point estimate (p50).
      - Model sampling is performed via inverse-CDF sampling using piecewise
        linear interpolation in quantile space, allowing asymmetric uncertainty
        to be preserved.
      - Behavior outside the provided quantile range is controlled by
        model_tail`.

    Point estimate vs. uncertainty
    -------------------------------
    - The returned point estimate is computed using obs_median and the model p50
      (taken directly from mdl_quantiles at quantile level 0.5).
    - The confidence interval reflects uncertainty due to observation error,
      model uncertainty, temporal dependence, and finite sample size.

    Parameters
    ----------
    obs_median : array-like, shape (n,)
        Observed median time series.
    obs_unc : array-like, shape (n,)
        Symmetric uncertainty half-width for observations. Bounds are
        interpreted as  [obs_median − obs_unc, obs_median + obs_unc] at coverage
        `obs_bounds_level`.
    mdl_q_levels : array-like, shape (m,)
        Strictly increasing quantile levels in (0, 1). Must explicitly include
        0.5.
    mdl_quantiles : array-like, shape (n, m)
        Model quantile values corresponding to `mdl_q_levels` for each
        timestamp. Each row must be non-decreasing across quantiles.
    metric : str, default "median_bias"
        Metric used to summarize residuals. Options are: "median_bias",
        "mean_bias", "mad", "rmsd" and "corrcoef"
    obs_bounds_level : float, default 0.95
        Coverage level associated with obs_unc (e.g., 0.95 for a 95% interval).
    ci_level : float, default 0.68
        Confidence level of the bootstrap confidence interval.
    n_boot : int, default 5000
        Number of bootstrap replicates. Larger values reduce Monte Carlo noise.
    block_length : int or None, default None
        Block length for the moving-block bootstrap. If None, uses
        max(2, round(sqrt(n))).
    seed : int or None, default 0
        Random seed for reproducibility.
    model_tail : {"clamp", "extrapolate"}, default "clamp"
        Handling of sampling outside the provided quantile range:
          - "clamp": values are capped at the lowest/highest quantile
            (conservative).
          - "extrapolate": linear extrapolation beyond the outer quantiles.
    allow_quantile_inversions_tol : float, default 0.0
        Tolerance for small numerical inversions in mdl_quantiles rows. Values
        below this tolerance are ignored; larger inversions raise an error.
    q50_tol : float, default 1e-12
        Tolerance when checking that mdl_q_levels includes 0.5.

    Returns
    -------
    BootstrapMetricResult
        Dataclass containing:
          - metric: name of the metric
          - point_estimate: metric computed on obs_median and model p50
          - ci: (lower, upper) bootstrap confidence interval
          - ci_level, n, n_boot, block_length, seed

    Notes
    -----
    - For short annual records, confidence intervals may be wide; conclusions
      should focus on robustness rather than point estimates alone.
    - RMSD is particularly sensitive to tail behavior of the model distribution;
      providing additional intermediate quantiles (e.g., 10/25/75/90) improves
      stability.
    - If observational uncertainty contains large systematic (year-correlated)
      components, the resulting confidence intervals may be optimistic unless
      such effects are modeled explicitly.

    Raises
    ------
    ValueError
        If input shapes are inconsistent, quantile levels are invalid, 0.5 is
        missing from mdl_q_levels, or parameters are outside valid ranges.
    """

    obs_med = _as_1d_float_array(obs_median, "obs_median")
    obs_u = _as_1d_float_array(obs_unc, "obs_unc")

    if obs_med.size != obs_u.size:
        raise ValueError("obs_median and obs_unc must have the same length"
                         "(aligned timestamps).")
    if np.any(obs_u < 0.0):
        raise ValueError("obs_unc must be >= 0 at all timestamps.")

    n = obs_med.size

    q_levels = _as_1d_float_array(mdl_q_levels, "mdl_q_levels")
    mdl_q = _as_2d_float_array(mdl_quantiles, "mdl_quantiles")

    if mdl_q.shape != (n, q_levels.size):
        raise ValueError("mdl_quantiles must have shape "
                         "(n_time, len(mdl_q_levels)).")
    _validate_quantiles(q_levels, mdl_q, tol=allow_quantile_inversions_tol)

    idx50_candidates = np.where(np.isclose(q_levels, 0.5, atol=q50_tol,
                                           rtol=0.0))[0]
    if idx50_candidates.size == 0:
        raise ValueError("mdl_q_levels must include 0.5 so the model p50 can "
                         "be extracted.")
    idx50 = int(idx50_candidates[0])

    if not (0.0 < obs_bounds_level < 1.0):
        raise ValueError("obs_bounds_level must be between 0 and 1.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1.")
    if n_boot < 200:
        raise ValueError("n_boot is very small; use at least ~200 (prefer "
                         "2000+).")

    mdl_point = mdl_q[:, idx50]
    point_est = _compute_metric(ref=obs_med, data=mdl_point, metric=metric)

    z = _norm_ppf((1.0 + obs_bounds_level) / 2.0)
    obs_sigma = obs_u / z

    if block_length is None:
        block_length = int(max(2, round(np.sqrt(n))))
    if not (1 <= block_length <= n):
        raise ValueError("block_length must be in [1, n].")

    rng = np.random.default_rng(seed)
    starts = np.arange(n - block_length + 1)

    def _mbb_indices() -> np.ndarray:
        k = int(np.ceil(n / block_length))
        chosen_starts = rng.choice(starts, size=k, replace=True)
        idx = np.concatenate([np.arange(s, s + block_length)
                              for s in chosen_starts])
        return idx[:n]

    boot_stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = _mbb_indices()

        obs_draw = obs_med[idx] + rng.normal(0.0, obs_sigma[idx])
        u = rng.uniform(0.0, 1.0, size=idx.size)
        mdl_draw = _inv_cdf_piecewise_linear(u, q_levels, mdl_q[idx, :],
                                             tail=model_tail)

        boot_stats[b] = _compute_metric(ref=obs_draw, data=mdl_draw,
                                        metric=metric)

    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot_stats, alpha / 2.0))
    hi = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))

    return BootstrapMetricResult(
        metric=metric,
        point_estimate=float(point_est),
        ci=(lo, hi),
        ci_level=ci_level,
        n=n,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
    )
