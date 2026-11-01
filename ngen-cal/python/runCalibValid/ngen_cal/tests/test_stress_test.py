import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydantic
import pytest

from ngen.cal import metric_functions as mf
from ngen.cal.model import EvaluationOptions
from ngen.cal.stress_test import run_stress_test, stress_reading

from .utils import HOURS, flood_list, hydrograph, worker

FIRST, SECOND = 500, 1500          # the two floods' observed peak hours, on HOURS
VALID = (HOURS[1000], HOURS[-1])   # the validation period holds the second flood only


def floods_csv(tmp_path) -> str:
    return flood_list(tmp_path, (FIRST, 21.0), (SECOND, 11.0))


GRID = pd.DataFrame([(1 / 1.5, 1.0, "/data/stress_01.nc"), (1.5, 1.0, "/data/stress_02.nc"),
                     (1.0, 1.2, "/data/stress_03.nc"), (1.5, 1.2, "/data/stress_04.nc")],
                    columns=["depth", "concentration", "forcing"])
# the second flood's simulated peak under each row of GRID
PEAKS = [10.0 * (1 / 1.5) ** 2, 10.0 * 1.5 ** 2, 13.0, 30.0]


def reading(tmp_path):
    floods = mf.read_floods(floods_csv(tmp_path))
    floods = floods[floods["peak_time"] >= VALID[0]]
    observed = hydrograph((SECOND, 11.0), name="obs_flow")
    runs = [hydrograph((SECOND + 2, q)).to_frame() for q in PEAKS]
    return stress_reading(hydrograph((SECOND, 10.0)).to_frame(), runs, GRID, floods, observed, VALID)


def test_each_flood_is_read_under_every_change(tmp_path):
    table, kge, _ = reading(tmp_path)
    assert table["change"].tolist() == [0, 1, 2, 3, 4]
    assert table[["depth", "concentration"]].values.tolist() == [
        [1.0, 1.0], [1 / 1.5, 1.0], [1.5, 1.0], [1.0, 1.2], [1.5, 1.2]]
    assert (table["peak_time"] == HOURS[SECOND]).all()           # the first flood is not read
    assert table["sim_peak_cms"].tolist() == pytest.approx([10.0, *PEAKS])
    assert table["peak_error"].tolist() == pytest.approx([abs(q - 11.0) / 11.0 for q in [10.0, *PEAKS]])
    assert table["timing_h"].tolist() == [0.0, 2.0, 2.0, 2.0, 2.0]
    assert table["sim_peak_time"].tolist() == [HOURS[SECOND]] + [HOURS[SECOND + 2]] * 4
    assert kge["change"].tolist() == [0, 1, 2, 3, 4]
    assert kge["KGE"].iloc[0] > kge["KGE"].iloc[4]               # the wettest storm's peak is farthest


def test_a_floods_summary_holds_its_worst_change_and_depth_response(tmp_path):
    _, _, summary = reading(tmp_path)
    assert len(summary) == 1
    flood = summary.iloc[0]
    assert flood["peak_error"] == pytest.approx(1 / 11)
    assert flood["largest_log_error"] == pytest.approx(np.log(30 / 11))
    assert (flood["largest_at_depth"], flood["largest_at_concentration"]) == (1.5, 1.2)
    # the fit takes the unchanged run and the two depth-only changes, where
    # ln(peak) = ln 10 + 2 ln(depth); the concentration changes would bend it
    assert flood["response"] == pytest.approx(2.0)


def validation(tmp_path, listed=True):
    """Return a validation worker whose model writes the forcing it ran, and a calibration
    object whose changed runs' hydrographs follow that forcing, with the homes it read."""
    GRID.to_csv(tmp_path / "changes.csv", index=False)
    homes = []

    def output_in(home):
        homes.append(Path(home).name)
        forcing = (Path(home) / "ran.txt").read_text()
        return hydrograph((SECOND + 2, PEAKS[list(GRID["forcing"]).index(forcing)])).to_frame()

    agent = worker(tmp_path, run_name="valid_best")
    calibration = SimpleNamespace(floods=floods_csv(tmp_path),
                                  stress_test=tmp_path / "changes.csv" if listed else None,
                                  valid_evaluation_range=VALID,
                                  basinID="01234567", output_in=output_in,
                                  observed=hydrograph((SECOND, 11.0), name="obs_flow"))
    return agent, calibration, homes


def test_the_stress_test_writes_its_reading(tmp_path):
    agent, calibration, homes = validation(tmp_path)
    run_stress_test(agent, calibration, hydrograph((SECOND, 10.0)).to_frame())
    assert len(homes) == len(GRID)
    assert not list(tmp_path.glob("storm_*"))                     # removed once read
    floods = pd.read_csv(tmp_path / "01234567_stress_floods_valid_best.csv")
    assert floods["sim_peak_cms"].tolist() == pytest.approx([10.0, *PEAKS])
    assert len(pd.read_csv(tmp_path / "01234567_stress_kge_valid_best.csv")) == 5
    assert len(pd.read_csv(tmp_path / "01234567_stress_summary_valid_best.csv")) == 1
    assert json.loads((tmp_path / "realization.json").read_text())["global"]["forcing"][
        "path"] == "/data/forcings.nc"


def test_nothing_runs_without_the_list_or_a_flood_in_the_validation_period(tmp_path):
    agent, calibration, homes = validation(tmp_path, listed=False)
    run_stress_test(agent, calibration, hydrograph((SECOND, 10.0)).to_frame())

    agent, calibration, homes_without_flood = validation(tmp_path)
    calibration.valid_evaluation_range = (HOURS[600], HOURS[1400])   # between the two floods
    run_stress_test(agent, calibration, hydrograph((SECOND, 10.0)).to_frame())

    assert homes == [] and homes_without_flood == []
    assert not list(tmp_path.glob("*_stress_*"))


def test_the_stress_test_list_is_checked_when_the_config_loads(tmp_path):
    (tmp_path / "dry.nc").touch()
    floods = {"floods": floods_csv(tmp_path), "flood_weight": 0.5}
    changes = tmp_path / "changes.csv"
    good = f"depth,concentration,forcing\n1.5,1.0,{tmp_path / 'dry.nc'}\n"
    changes.write_text(good)
    EvaluationOptions(objective="kge", stress_test=changes, **floods)
    for text, params, error in ((good, {}, "needs 'floods'"),
                                ("depth,forcing\n1.5,/data/dry.nc\n", floods, "no column concentration"),
                                ("depth,concentration,forcing\n1.5,1.0,/data/none.nc\n", floods, "does not exist")):
        changes.write_text(text)
        with pytest.raises(pydantic.ValidationError, match=error):
            EvaluationOptions(objective="kge", stress_test=changes, **params)
