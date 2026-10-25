"""The storm stress test reruns the best parameters' validation run under each listed changed
forcing and reads the validation period's floods under every change beside the unchanged run.

Each case fails against a reading that drops the unchanged run, misnumbers the changes, reads a
flood outside the validation period, takes the peak's hour from the observation, fits the
response over the concentration changes, runs every forcing at once or in the worker's own
directory, or runs anything without the list.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ngen.cal import metric_functions as mf
from ngen.cal.stress_test import STRESS_RUNS_AT_ONCE, run_stress_test, stress_reading

HOURS = pd.date_range("2020-01-01 01:00", periods=2000, freq="h")
FIRST, SECOND = 500, 1500          # the two floods' observed peak hours, on HOURS
VALID = (HOURS[1000], HOURS[-1])   # the validation period holds the second flood only


def hydrograph(*peaks, name: str = "sim_flow", base: float = 1.0) -> pd.DataFrame:
    """An hourly hydrograph with a triangular peak of each (hour, height), as the engine's
    one-column frame."""
    q = np.full(len(HOURS), base)
    for at, height in peaks:
        for k in range(-12, 13):
            q[at + k] = max(q[at + k], base + (height - base) * (1 - abs(k) / 12))
    return pd.DataFrame({name: q}, index=HOURS)


def flood_list(tmp_path) -> Path:
    path = tmp_path / "floods.csv"
    pd.DataFrame({"peak_time": [HOURS[FIRST], HOURS[SECOND]],
                  "obs_peak_cms": [21.0, 11.0]}).to_csv(path, index=False)
    return path


def changes(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["depth", "concentration", "forcing"])


GRID = changes((1 / 1.5, 1.0, "/data/stress_01.nc"), (1.5, 1.0, "/data/stress_02.nc"),
               (1.0, 1.2, "/data/stress_03.nc"), (1.5, 1.2, "/data/stress_04.nc"))
# the simulated peak of the second flood under each row of GRID: Q̂ = 10 f², whatever γ adds
PEAKS = [10.0 * (1 / 1.5) ** 2, 10.0 * 1.5 ** 2, 13.0, 30.0]


def test_a_floods_simulated_peak_comes_with_its_own_hour(tmp_path):
    floods = mf.read_floods(str(flood_list(tmp_path)))
    peaks = mf.flood_peaks(hydrograph((FIRST + 3, 10.5), (SECOND - 2, 5.0))["sim_flow"], floods)
    assert peaks["sim_peak_cms"].tolist() == pytest.approx([10.5, 5.0])
    assert peaks["sim_peak_time"].tolist() == [HOURS[FIRST + 3], HOURS[SECOND - 2]]


def reading(tmp_path):
    floods = mf.read_floods(str(flood_list(tmp_path)))
    floods = floods[floods["peak_time"] >= VALID[0]]
    observed = hydrograph((SECOND, 11.0), name="obs_flow")
    runs = [hydrograph((SECOND + 2, q)) for q in PEAKS]
    return stress_reading(hydrograph((SECOND, 10.0)), runs, GRID, floods, observed, VALID)


def test_each_flood_is_read_under_the_unchanged_storm_and_every_change(tmp_path):
    table, kge, _ = reading(tmp_path)
    assert table["change"].tolist() == [0, 1, 2, 3, 4]
    assert table[["depth", "concentration"]].values.tolist() == [
        [1.0, 1.0], [1 / 1.5, 1.0], [1.5, 1.0], [1.0, 1.2], [1.5, 1.2]]
    assert (table["peak_time"] == HOURS[SECOND]).all()           # the first flood is not read
    assert table["sim_peak_cms"].tolist() == pytest.approx([10.0, *PEAKS])
    assert table["peak_error"].tolist() == pytest.approx([abs(q - 11.0) / 11.0 for q in [10.0, *PEAKS]])
    assert table["timing_h"].tolist() == [0.0, 2.0, 2.0, 2.0, 2.0]
    assert kge["change"].tolist() == [0, 1, 2, 3, 4]
    assert kge["KGE"].iloc[0] > kge["KGE"].iloc[4]               # the wettest storm's peak is farthest


def test_each_floods_summary_holds_its_worst_change_and_its_response_to_depth(tmp_path):
    _, _, summary = reading(tmp_path)
    assert len(summary) == 1
    flood = summary.iloc[0]
    assert flood["peak_error"] == pytest.approx(1 / 11)
    assert flood["largest_peak_error"] == pytest.approx(19 / 11)
    assert (flood["largest_at_depth"], flood["largest_at_concentration"]) == (1.5, 1.2)
    # over the unchanged run and the two depth-only changes ln Q̂ = ln 10 + 2 ln f; the
    # concentration changes would bend the fit
    assert flood["response"] == pytest.approx(2.0)


SHOW = (f'{sys.executable} -c "import json, pathlib; pathlib.Path(\'ran.txt\').write_text('
        'json.load(open(\'realization.json\'))[\'global\'][\'forcing\'][\'path\'])"')


def validation(tmp_path, listed=True):
    """A validation worker whose model writes the forcing it ran, and the calibration object of
    its basin, whose changed runs' hydrographs follow the forcing each ran under."""
    (tmp_path / "realization.json").write_text(json.dumps(
        {"global": {"forcing": {"path": "/data/forcings.nc", "provider": "NetCDF"}}}, indent=4))
    GRID.to_csv(tmp_path / "changes.csv", index=False)
    homes = []

    def output_in(home):
        homes.append(Path(home).name)
        forcing = (Path(home) / "ran.txt").read_text()
        return hydrograph((SECOND + 2, PEAKS[list(GRID["forcing"]).index(forcing)]))

    params = {"floods": str(flood_list(tmp_path)), "flood_weight": 0.5}
    if listed:
        params["stress_test"] = str(tmp_path / "changes.csv")
    agent = SimpleNamespace(job=SimpleNamespace(workdir=tmp_path, log_file=None), cmd=SHOW,
                            realization_file=Path("/elsewhere/realization.json"),
                            run_name="valid_best", workdir=tmp_path)
    calibration = SimpleNamespace(objective_params=params, valid_evaluation_range=VALID,
                                  output=hydrograph((SECOND, 10.0)), basinID="01234567",
                                  observed=hydrograph((SECOND, 11.0), name="obs_flow"),
                                  output_in=output_in)
    return agent, calibration, homes


def test_the_changed_forcings_run_a_few_at_a_time_and_the_reading_is_written(tmp_path):
    agent, calibration, homes = validation(tmp_path)
    run_stress_test(agent, calibration)
    # four forcings, three at a time: each batch reuses the changed runs' directories
    assert homes == ["storm_0", "storm_1", "storm_2", "storm_0"]
    assert STRESS_RUNS_AT_ONCE == 3
    assert not list(tmp_path.glob("storm_*"))                     # removed once read
    floods = pd.read_csv(tmp_path / "01234567_stress_floods_valid_best.csv")
    assert floods["sim_peak_cms"].tolist() == pytest.approx([10.0, *PEAKS])
    assert len(pd.read_csv(tmp_path / "01234567_stress_kge_valid_best.csv")) == 5
    assert len(pd.read_csv(tmp_path / "01234567_stress_summary_valid_best.csv")) == 1
    # the worker's own run is untouched
    assert json.loads((tmp_path / "realization.json").read_text())["global"]["forcing"][
        "path"] == "/data/forcings.nc"


def test_without_the_list_nothing_runs(tmp_path):
    agent, calibration, homes = validation(tmp_path, listed=False)
    run_stress_test(agent, calibration)
    assert homes == [] and not list(tmp_path.glob("*_stress_*"))


def test_a_validation_period_without_a_flood_runs_nothing(tmp_path):
    agent, calibration, homes = validation(tmp_path)
    calibration.valid_evaluation_range = (HOURS[600], HOURS[1400])   # between the two floods
    run_stress_test(agent, calibration)
    assert homes == [] and not list(tmp_path.glob("*_stress_*"))
