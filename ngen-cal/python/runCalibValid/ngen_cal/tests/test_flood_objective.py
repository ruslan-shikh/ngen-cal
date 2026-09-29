"""The flood objectives score each listed flood's peak on the raw simulated series, and weight
KGE against the mean or the largest peak error.

Each case fails against a term scored on the observation-filtered series, a window refused at
the record's start, a flood left out when the simulation lacks an hour, or swapped weights.
"""
import numpy as np
import pandas as pd
import pytest

from ngen.cal import metric_functions as mf
from ngen.cal.search import _calc_metrics

HOURS = pd.date_range("2020-01-01 01:00", periods=2000, freq="h")


def hydrograph(at: int, height: float, name: str, base: float = 1.0) -> pd.Series:
    q = np.full(len(HOURS), base)
    for k in range(-12, 13):
        if 0 <= at + k < len(HOURS):
            q[at + k] = max(q[at + k], base + (height - base) * (1 - abs(k) / 12))
    return pd.Series(q, index=HOURS, name=name)


def flood_list(tmp_path, *floods) -> str:
    path = tmp_path / "floods.csv"
    pd.DataFrame({"peak_time": [HOURS[at].strftime("%Y-%m-%d %H:%M:%S") for at, _ in floods],
                  "obs_peak_cms": [q for _, q in floods]}).to_csv(path, index=False)
    return str(path)


def test_peak_error_of_a_late_half_size_flood(tmp_path):
    sim = hydrograph(503, 11.0, "sim_flow")
    floods = mf.read_floods(flood_list(tmp_path, (500, 21.0)))
    assert mf.flood_peak_errors(sim, floods).tolist() == pytest.approx([10 / 21])


def test_a_simulation_missing_an_hour_of_a_window_raises(tmp_path):
    sim = hydrograph(500, 21.0, "sim_flow").drop(HOURS[600])     # inside the ±168 h window
    floods = mf.read_floods(flood_list(tmp_path, (500, 21.0)))
    with pytest.raises(ValueError):
        mf.flood_peak_errors(sim, floods)


def test_a_window_at_the_start_is_cut_to_the_simulation(tmp_path):
    sim = hydrograph(20, 41.0, "sim_flow")
    floods = mf.read_floods(flood_list(tmp_path, (20, 21.0)))
    assert mf.flood_peak_errors(sim, floods).tolist() == pytest.approx([20 / 21])


def test_objectives_weight_kge_and_the_mean_or_largest_error():
    e = np.array([0.1, 0.5])
    assert mf.kge_flood_mean(0.8, e, 0.5) == pytest.approx(1 - (0.5 * 0.2 + 0.5 * 0.3))
    assert mf.kge_flood_max(0.8, e, 0.5) == pytest.approx(1 - (0.5 * 0.2 + 0.5 * 0.5))
    assert mf.kge_flood_max(0.8, e, 0.25) == pytest.approx(1 - (0.75 * 0.2 + 0.25 * 0.5))


def test_calc_metrics_scores_floods_on_the_raw_simulation(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    obs.iloc[503] = np.nan                   # the simulated peak's hour has no observation
    sim = hydrograph(503, 11.0, "sim_flow")
    params = {"floods": flood_list(tmp_path, (500, 21.0)), "flood_weight": 0.5}
    m = _calc_metrics(sim, obs, objective_params=params)
    assert m["FLOOD_MEAN_PEAK_ERR"] == pytest.approx(10 / 21)   # Q̂ = 11, at the unobserved hour
    assert m["KGE_FLOOD_MEAN"] == pytest.approx(1 - (0.5 * (1 - m["KGE"]) + 0.5 * 10 / 21))
    assert m["KGE_FLOOD_MAX"] == pytest.approx(m["KGE_FLOOD_MEAN"])   # one flood


def test_calc_metrics_without_floods_is_unchanged(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    sim = hydrograph(503, 11.0, "sim_flow")
    plain = _calc_metrics(sim, obs)
    params = {"floods": flood_list(tmp_path, (500, 21.0)), "flood_weight": 0.5}
    with_floods = _calc_metrics(sim, obs, objective_params=params)
    assert "KGE_FLOOD_MEAN" not in plain
    assert with_floods["KGE"] == plain["KGE"]


def test_floods_outside_the_evaluation_range_are_left_out(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    sim = hydrograph(503, 11.0, "sim_flow")
    params = {"floods": flood_list(tmp_path, (500, 21.0)), "flood_weight": 0.5}
    m = _calc_metrics(sim, obs, (HOURS[1000], HOURS[1999]), objective_params=params)
    assert np.isnan(m["KGE_FLOOD_MEAN"]) and np.isnan(m["FLOOD_MAX_PEAK_ERR"])


def test_calc_metrics_takes_the_engines_one_column_output_frame(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    sim = hydrograph(503, 11.0, "sim_flow").to_frame()     # the catchment and set outputs are frames
    params = {"floods": flood_list(tmp_path, (500, 21.0)), "flood_weight": 0.5}
    m = _calc_metrics(sim, obs, objective_params=params)
    assert m["FLOOD_MEAN_PEAK_ERR"] == pytest.approx(10 / 21)
