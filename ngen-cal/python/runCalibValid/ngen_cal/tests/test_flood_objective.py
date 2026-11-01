from unittest.mock import MagicMock

import numpy as np
import pydantic
import pytest

from ngen.cal import metric_functions as mf
from ngen.cal.model import EvaluationOptions
from ngen.cal.search import _calc_metrics, _evaluate
from ngen.cal.strategy import Objective

from .utils import HOURS, flood_list, hydrograph


def test_the_window_is_seven_days_either_side_cut_at_the_ends_of_the_run(tmp_path):
    week = 7 * 24
    sim = hydrograph((500, 11.0))
    sim.iloc[500 + week] = 30.0  # exactly seven days after the peak: counts
    sim.iloc[500 - week - 1] = 40.0  # an hour more than seven days before: ignored
    sim.iloc[0] = 50.0  # the first and last hours count for floods near the ends of the run
    sim.iloc[-1] = 60.0
    floods = mf.read_floods(flood_list(tmp_path, (500, 21.0), (20, 21.0), (1990, 21.0)))
    peaks = mf.flood_peaks(sim, floods)
    assert peaks["sim_peak_cms"].tolist() == [30.0, 50.0, 60.0]
    assert peaks["sim_peak_time"].tolist() == [HOURS[500 + week], HOURS[0], HOURS[-1]]


def test_flood_metrics(tmp_path):
    obs = hydrograph((500, 21.0), name="obs_flow")
    obs.iloc[503] = np.nan  # no observation at the hour of the simulated peak
    sim = hydrograph((503, 11.0)).to_frame()
    floods = flood_list(tmp_path, (500, 21.0))
    m = _calc_metrics(sim, obs, floods=floods, flood_weight=0.5)
    assert m["FLOOD_MEAN_PEAK_ERR"] == pytest.approx(10 / 21)
    assert m["FLOOD_MAX_PEAK_ERR"] == pytest.approx(10 / 21)
    assert m["FLOOD_MEAN_SHIFT_H"] == 3.0
    assert m["KGE_FLOOD"] == pytest.approx(1 - (0.5 * (1 - m["KGE"]) + 0.5 * 10 / 21))

    # without a flood list the metrics are as before
    plain = _calc_metrics(sim, obs)
    assert "KGE_FLOOD" not in plain
    assert plain["KGE"] == m["KGE"]

    # a NaN simulated flow within a flood's window gives NaN flood scores
    sim.iloc[503] = np.nan
    assert np.isnan(_calc_metrics(sim, obs, floods=floods, flood_weight=0.5)["KGE_FLOOD"])


def test_floods_outside_the_evaluation_range_give_nan(tmp_path, caplog):
    obs = hydrograph((500, 21.0), name="obs_flow")
    sim = hydrograph((503, 11.0)).to_frame()
    floods = flood_list(tmp_path, (500, 21.0))
    m = _calc_metrics(sim, obs, (HOURS[1000], HOURS[1999]), floods=floods, flood_weight=0.5)
    assert np.isnan(m["KGE_FLOOD"])
    assert "No flood in the flood list peaks in the evaluation range" in caplog.text


def test_the_kge_flood_objective_scores_one_minus_kge_flood(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no *.log here, so _evaluate skips its plots
    obs = hydrograph((500, 21.0), name="obs_flow")
    sim = hydrograph((503, 11.0)).to_frame()
    floods = flood_list(tmp_path, (500, 21.0))
    calibration = MagicMock(output=sim, observed=obs, evaluation_range=None, threshold=None,
                            floods=floods, flood_weight=0.5, objective=Objective.kge_flood,
                            target="min")
    score = _evaluate(0, calibration, MagicMock(algorithm="dds"))
    m = _calc_metrics(sim, obs, floods=floods, flood_weight=0.5)
    assert score == pytest.approx(1 - m["KGE_FLOOD"])


def test_flood_settings_are_checked_when_the_config_loads(tmp_path):
    path = flood_list(tmp_path, (500, 21.0))
    no_peaks = tmp_path / "no_peaks.csv"
    no_peaks.write_text("peak_time\n2020-01-21 21:00\n")
    for wrong in (
        {"objective": "kge_flood"},
        {"objective": "kge_flood", "flood_weight": 0.5},
        {"objective": "kge_flood", "floods": path},
        {"objective": "kge", "floods": path, "flood_weight": 7},
        {"objective": "kge", "floods": no_peaks, "flood_weight": 0.5},
    ):
        with pytest.raises(pydantic.ValidationError):
            EvaluationOptions(**wrong)

    # kge_flood needs a flood between evaluation_start and evaluation_stop
    later = {"evaluation_start": HOURS[1000], "evaluation_stop": HOURS[1999]}
    with pytest.raises(pydantic.ValidationError, match="no flood"):
        EvaluationOptions(objective="kge_flood", floods=path, flood_weight=0.5, **later)
    # other objectives only report the flood metrics, so they don't need one
    EvaluationOptions(objective="kge", floods=path, flood_weight=0.5, **later)
    EvaluationOptions(objective="kge_flood", floods=path, flood_weight=0.5)
