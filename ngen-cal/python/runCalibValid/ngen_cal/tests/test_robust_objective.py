import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
import pydantic
import pytest

from ngen.cal import search
from ngen.cal.calibration_set import read_routed_flow
from ngen.cal.model import EvaluationOptions
from ngen.cal.ngen import NgenIndependent
from ngen.cal.search import _calc_metrics

from .utils import flood_list, hydrograph, worker


def flood_args(tmp_path, *floods) -> dict:
    """The flood settings for a flood list of (hour, observed peak) pairs, weighted 0.5."""
    return {"floods": flood_list(tmp_path, *floods), "flood_weight": 0.5}


def storm_params(tmp_path) -> dict:
    """Return the eval_params settings for kge_flood_robust, as read from the YAML."""
    (tmp_path / "dry.nc").touch()
    return {"floods": flood_list(tmp_path, (500, 21.0)), "flood_weight": 0.5,
            "storm_forcings": [str(tmp_path / "dry.nc")]}


def test_the_robust_score_takes_each_floods_worst_run(tmp_path):
    obs = hydrograph((500, 21.0), (1500, 4.0), name="obs_flow")
    p = flood_args(tmp_path, (500, 21.0), (1500, 4.0))
    unchanged = hydrograph((500, 21.0), (1500, 4.0)).to_frame()
    dry = hydrograph((500, 10.5), (1500, 4.0)).to_frame()       # halves the first flood only
    wet = hydrograph((500, 21.0), (1500, 16.0)).to_frame()      # quadruples the second only
    m = _calc_metrics(unchanged, obs, **p, storm_hydrographs=[dry, wet])
    # each flood's worst run: ln 2 for the first, ln 4 for the second
    worst = (np.log(2) + np.log(4)) / 2
    assert m["FLOOD_ROBUST_LOG_ERR"] == pytest.approx(worst)
    assert m["KGE_FLOOD_ROBUST"] == pytest.approx(1 - (0.5 * (1 - m["KGE"]) + 0.5 * worst))
    # without storm runs there are no robust scores
    assert "KGE_FLOOD_ROBUST" not in _calc_metrics(unchanged, obs, **p)


def test_storm_forcings_go_with_the_robust_objective_only(tmp_path):
    storms = storm_params(tmp_path)
    EvaluationOptions(objective="kge_flood_robust", **storms)
    no_storms = {k: v for k, v in storms.items() if k != "storm_forcings"}
    no_floods = {"storm_forcings": storms["storm_forcings"]}
    for objective, params in (("kge_flood_robust", no_storms), ("kge_flood_robust", no_floods),
                              ("kge_flood", storms)):
        with pytest.raises(pydantic.ValidationError):
            EvaluationOptions(objective=objective, **params)


def test_storm_forcings_and_the_stress_test_need_the_uniform_strategy(tmp_path):
    files = {}
    for key in ("realization", "catchments", "nexus", "crosswalk"):
        files[key] = tmp_path / key
        files[key].write_text("{}")
    storms = storm_params(tmp_path)
    (tmp_path / "changes.csv").write_text(f"depth,concentration,forcing\n1.5,1.0,{storms['storm_forcings'][0]}\n")
    stress = {"floods": storms["floods"], "flood_weight": 0.5, "stress_test": tmp_path / "changes.csv"}
    for eval_params in (EvaluationOptions(objective="kge_flood_robust", **storms),
                        EvaluationOptions(objective="kge", **stress)):
        with pytest.raises(pydantic.ValidationError, match="uniform strategy"):
            NgenIndependent(type="ngen", strategy="independent", params={}, eval_params=eval_params, **files)


def test_each_storm_forcing_runs_in_its_own_directory(tmp_path):
    search._execute(worker(tmp_path), 3)
    assert not list(tmp_path.glob("storm_*"))  # nothing extra without storm forcings

    storms = ["/data/dry.nc", "/data/wet.nc"]
    search._execute(worker(tmp_path), 3, storm_forcings=storms)
    assert (tmp_path / "ran.txt").read_text() == "/data/forcings.nc"
    assert (tmp_path / "storm_0" / "ran.txt").read_text() == "/data/dry.nc"
    assert (tmp_path / "storm_1" / "ran.txt").read_text() == "/data/wet.nc"
    assert json.loads((tmp_path / "realization.json").read_text())["global"]["forcing"][
        "path"] == "/data/forcings.nc"
    changed = json.loads((tmp_path / "storm_1" / "realization.json").read_text())
    assert changed["time"] == {"start_time": "2014-10-01 00:00:00"}

    (tmp_path / "storm_0" / "left_over.csv").write_text("old")
    search._execute(worker(tmp_path), 4, storm_forcings=storms)
    assert not (tmp_path / "storm_0" / "left_over.csv").exists()   # each iteration starts empty


def test_a_catchment_with_its_own_forcing_stops_the_storm_runs(tmp_path):
    meta = worker(tmp_path)
    realization = json.loads((tmp_path / "realization.json").read_text())
    realization["catchments"] = {"cat-1": {"forcing": {"path": "/data/cat-1.csv"}}}
    (tmp_path / "realization.json").write_text(json.dumps(realization))
    with pytest.raises(ValueError, match="own forcing"):
        search._execute(meta, 3, storm_forcings=["/data/dry.nc"])
    assert not (tmp_path / "ran.txt").exists()


def test_a_failed_storm_run_stops_the_iteration(tmp_path):
    fails_away_from_home = (f'{sys.executable} -c "import pathlib, sys; '
                            'sys.exit(pathlib.Path.cwd().name.startswith(\'storm_\'))"')
    with pytest.raises(subprocess.CalledProcessError):
        search._execute(worker(tmp_path, fails_away_from_home), 3, storm_forcings=["/d.nc"])
    search._execute(worker(tmp_path, fails_away_from_home), 3)     # the main run alone passes


def test_a_failed_main_run_stops_the_storm_runs_and_keeps_its_error(tmp_path):
    # the storm run marks its start, and its end two seconds later; the main run fails once
    # the storm run has started
    run = tmp_path / "run.py"
    run.write_text("import pathlib, sys, time\n"
                   "if pathlib.Path.cwd().name == 'storm_0':\n"
                   "    pathlib.Path('started').touch()\n"
                   "    time.sleep(2)\n"
                   "    pathlib.Path('ended').touch()\n"
                   "    sys.exit(3)\n"
                   "for _ in range(200):\n"
                   "    if pathlib.Path('storm_0', 'started').exists():\n"
                   "        break\n"
                   "    time.sleep(0.05)\n"
                   "sys.exit(2)\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        search._execute(worker(tmp_path, f"{sys.executable} {run}"), 3, storm_forcings=["/d.nc"])
    assert error.value.returncode == 2
    assert (tmp_path / "storm_0" / "started").exists()
    time.sleep(3)
    assert not (tmp_path / "storm_0" / "ended").exists()


def test_a_storm_run_without_output_names_its_log(tmp_path):
    def no_output(home):
        raise FileNotFoundError(home / "troute_output_201410010000.nc")
    (tmp_path / "storm_0").mkdir()
    calibration = SimpleNamespace(storm_forcings=["/data/dry.nc"], output_in=no_output)
    agent = SimpleNamespace(job=SimpleNamespace(workdir=tmp_path))
    with pytest.raises(RuntimeError, match="/data/dry.nc.*storm_0/run.log"):
        search._storm_outputs(calibration, agent)
    assert (tmp_path / "storm_0").exists()             # kept, with its log


def routed(directory: Path, flows) -> Path:
    """Write a routing output of two hourly steps and three features, and its nexus file."""
    out = directory / "troute_output_201410010000.nc"
    with netCDF4.Dataset(out, "w") as nc:
        nc.createDimension("feature_id", 3)
        nc.createDimension("time", 2)
        nc.createVariable("feature_id", "i8", ("feature_id",))[:] = [11, 22, 33]
        nc.createVariable("flow", "f8", ("feature_id", "time"))[:] = flows
    pd.DataFrame({"ts": [0, 1, 2], "time": ["2014-10-01 00:00:00", "2014-10-01 01:00:00",
                                              "2014-10-01 02:00:00"], "q": 0.0}).to_csv(
        directory / "nex-1_output.csv", header=False, index=False)
    return out


def test_routed_flow_is_read_from_the_given_directory(tmp_path):
    (tmp_path / "storm_0").mkdir()
    routed(tmp_path, [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    there = routed(tmp_path / "storm_0", [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]])
    flow = read_routed_flow(there, ["11", "33"])
    assert flow["sim_flow"].tolist() == [303.0, 404.0]
    assert list(flow.index) == [pd.Timestamp("2014-10-01 01:00"), pd.Timestamp("2014-10-01 02:00")]
