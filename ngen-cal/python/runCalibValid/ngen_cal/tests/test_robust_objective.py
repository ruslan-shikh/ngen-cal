"""The robust objectives run the model under changed forcings beside the unchanged one. They score
each listed flood by the log of its simulated peak over its observed one at the flood's worst run,
and the record by the lowest KGE among the runs.

Each case fails against a term that leaves a run out, takes the relative error in place of the
log ratio, averages the runs or takes one run as worst for every flood, scores a validation run
that has no changed runs, starts a changed run on the unchanged forcing or in the worker's own
directory, or lets a failed changed run pass.
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
import pytest

from ngen.cal import metric_functions as mf
from ngen.cal import search
from ngen.cal.calibration_set import read_routed_flow
from ngen.cal.search import _calc_metrics
from ngen.cal.strategy import Objective

HOURS = pd.date_range("2020-01-01 01:00", periods=2000, freq="h")


def hydrograph(at: int, height: float, name: str = "sim_flow", base: float = 1.0) -> pd.Series:
    q = np.full(len(HOURS), base)
    for k in range(-12, 13):
        q[at + k] = max(q[at + k], base + (height - base) * (1 - abs(k) / 12))
    return pd.Series(q, index=HOURS, name=name)


def params(tmp_path, *floods, storms=("dry.nc", "wet.nc")) -> dict:
    path = tmp_path / "floods.csv"
    pd.DataFrame({"peak_time": [HOURS[at].strftime("%Y-%m-%d %H:%M:%S") for at, _ in floods],
                  "obs_peak_cms": [q for _, q in floods]}).to_csv(path, index=False)
    return {"floods": str(path), "flood_weight": 0.5, "storm_forcings": list(storms)}


def test_peak_ratio_is_the_simulated_peak_over_the_observed(tmp_path):
    floods = mf.read_floods(params(tmp_path, (500, 21.0))["floods"])
    assert mf.flood_peak_ratios(hydrograph(503, 10.5), floods).tolist() == pytest.approx([0.5])
    assert mf.flood_peak_errors(hydrograph(503, 10.5), floods).tolist() == pytest.approx([0.5])


def test_the_robust_term_reads_a_halved_and_a_doubled_peak_alike(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    p = params(tmp_path, (500, 21.0))
    unchanged, dry, wet = hydrograph(500, 21.0), hydrograph(500, 10.5), hydrograph(500, 42.0)
    m = _calc_metrics(unchanged, obs, objective_params=p, storm_hydrographs=[dry, wet])
    # ratios 1, 1/2 and 2: either changed run is worst, at ln 2, where the relative error would
    # read the doubled peak as twice as far off as the halved one
    assert m["FLOOD_ROBUST_LOG_ERR"] == pytest.approx(np.log(2))
    assert m["KGE_FLOOD_ROBUST"] == pytest.approx(1 - (0.5 * (1 - m["KGE"]) + 0.5 * np.log(2)))
    assert m["FLOOD_MEAN_PEAK_ERR"] == pytest.approx(0.0)      # the unchanged run's, as before


def test_changed_runs_are_taken_as_the_engines_one_column_frames(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    p = params(tmp_path, (500, 21.0))
    m = _calc_metrics(hydrograph(500, 21.0).to_frame(), obs, objective_params=p,
                      storm_hydrographs=[hydrograph(500, 10.5).to_frame(),
                                         hydrograph(500, 42.0).to_frame()])
    assert m["FLOOD_ROBUST_LOG_ERR"] == pytest.approx(np.log(2))


def test_a_run_without_its_changed_runs_carries_no_robust_score(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    m = _calc_metrics(hydrograph(500, 21.0), obs, objective_params=params(tmp_path, (500, 21.0)))
    assert np.isnan(m["FLOOD_ROBUST_LOG_ERR"]) and np.isnan(m["KGE_FLOOD_ROBUST"])
    assert np.isnan(m["KGE_ROBUST"])
    assert m["KGE_FLOOD_MEAN"] == pytest.approx(1 - 0.5 * (1 - m["KGE"]))


def test_a_flood_objective_without_storm_forcings_is_unchanged(tmp_path):
    obs = hydrograph(500, 21.0, "obs_flow")
    p = params(tmp_path, (500, 21.0))
    del p["storm_forcings"]
    m = _calc_metrics(hydrograph(500, 21.0), obs, objective_params=p)
    assert not {"FLOOD_ROBUST_LOG_ERR", "KGE_FLOOD_ROBUST", "KGE_ROBUST"} & set(m)


def both(*parts: pd.Series) -> pd.Series:
    """One hydrograph holding the peak of each part."""
    return pd.concat(parts, axis=1).max(axis=1).rename(parts[0].name)


def test_the_robust_term_takes_each_floods_largest_log_error_among_the_runs(tmp_path):
    obs = both(hydrograph(500, 21.0, "obs_flow"), hydrograph(1500, 4.0, "obs_flow"))
    p = params(tmp_path, (500, 21.0), (1500, 4.0))
    unchanged = both(hydrograph(500, 21.0), hydrograph(1500, 4.0))
    dry = both(hydrograph(500, 10.5), hydrograph(1500, 4.0))       # halves the first flood only
    wet = both(hydrograph(500, 21.0), hydrograph(1500, 16.0))      # quadruples the second only
    m = _calc_metrics(unchanged, obs, objective_params=p, storm_hydrographs=[dry, wet])
    # the first flood is worst in the dry run and the second in the wet one: the mean of the
    # runs would read (ln 2 + ln 4) / 6, and the one run that is worst over both floods ln 4 / 2
    worst = (np.log(2) + np.log(4)) / 2
    assert m["FLOOD_ROBUST_LOG_ERR"] == pytest.approx(worst)
    assert m["KGE_FLOOD_ROBUST"] == pytest.approx(1 - (0.5 * (1 - m["KGE"]) + 0.5 * worst))


def test_the_robust_kge_is_the_lowest_kge_among_the_runs():
    obs = hydrograph(500, 21.0, "obs_flow")
    unchanged, dry, wet = hydrograph(500, 21.0), hydrograph(500, 10.5), hydrograph(500, 42.0)
    alone = [_calc_metrics(run, obs)["KGE"] for run in (unchanged, dry, wet)]
    assert alone[0] == pytest.approx(1.0) and min(alone) < 0.9
    # no flood list: the record's score needs only the changed runs
    m = _calc_metrics(unchanged, obs, objective_params={"storm_forcings": ["dry.nc", "wet.nc"]},
                      storm_hydrographs=[dry.to_frame(), wet.to_frame()])
    assert m["KGE_ROBUST"] == pytest.approx(min(alone))
    assert m["KGE"] == pytest.approx(1.0)                           # the unchanged run's, as before



@pytest.mark.parametrize("name", ["kge_flood_mean", "kge_flood_robust", "kge_robust"])
def test_each_objective_has_a_score_under_its_name(tmp_path, name):
    # the search reads an objective's score as metrics[objective.value.upper()]
    obs = hydrograph(500, 21.0, "obs_flow")
    m = _calc_metrics(hydrograph(500, 21.0), obs, objective_params=params(tmp_path, (500, 21.0)),
                      storm_hydrographs=[hydrograph(500, 10.5), hydrograph(500, 42.0)])
    assert np.isfinite(m[Objective(name).value.upper()])


SHOW = (f'{sys.executable} -c "import json, pathlib; pathlib.Path(\'ran.txt\').write_text('
        'json.load(open(\'realization.json\'))[\'global\'][\'forcing\'][\'path\'])"')


def worker(tmp_path, cmd=SHOW) -> SimpleNamespace:
    (tmp_path / "realization.json").write_text(json.dumps(
        {"global": {"forcing": {"path": "/data/forcings.nc", "provider": "NetCDF"}},
         "time": {"start_time": "2014-10-01 00:00:00"}}, indent=4))
    return SimpleNamespace(job=SimpleNamespace(workdir=tmp_path, log_file=None), cmd=cmd,
                           realization_file=Path("/somewhere/else/realization.json"),
                           run_name="calib")


def test_each_changed_forcing_runs_in_its_own_directory_beside_the_unchanged_run(tmp_path):
    meta = worker(tmp_path)
    search._execute(meta, 3, storm_forcings=["/data/dry.nc", "/data/wet.nc"])
    assert (tmp_path / "ran.txt").read_text() == "/data/forcings.nc"
    assert (tmp_path / "storm_0" / "ran.txt").read_text() == "/data/dry.nc"
    assert (tmp_path / "storm_1" / "ran.txt").read_text() == "/data/wet.nc"
    # the worker's own realization is left as the engine wrote it
    assert json.loads((tmp_path / "realization.json").read_text())["global"]["forcing"][
        "path"] == "/data/forcings.nc"
    changed = json.loads((tmp_path / "storm_1" / "realization.json").read_text())
    assert changed["time"] == {"start_time": "2014-10-01 00:00:00"}


def test_a_run_without_storm_forcings_starts_nothing_else(tmp_path):
    search._execute(worker(tmp_path), 3)
    assert (tmp_path / "ran.txt").read_text() == "/data/forcings.nc"
    assert not list(tmp_path.glob("storm_*"))


def test_a_changed_run_that_fails_stops_the_iteration(tmp_path):
    fails_away_from_home = (f'{sys.executable} -c "import pathlib, sys; '
                            'sys.exit(pathlib.Path.cwd().name.startswith(\'storm_\'))"')
    with pytest.raises(subprocess.CalledProcessError):
        search._execute(worker(tmp_path, fails_away_from_home), 3, storm_forcings=["/d.nc"])
    search._execute(worker(tmp_path, fails_away_from_home), 3)     # the unchanged run passes


def test_a_changed_runs_directory_starts_empty_each_iteration(tmp_path):
    meta = worker(tmp_path)
    search._execute(meta, 1, storm_forcings=["/data/dry.nc"])
    (tmp_path / "storm_0" / "left_over.csv").write_text("old")
    search._execute(meta, 2, storm_forcings=["/data/dry.nc"])
    assert not (tmp_path / "storm_0" / "left_over.csv").exists()



def test_a_changed_run_without_output_names_its_log(tmp_path):
    def no_output(home):
        raise FileNotFoundError(home / "troute_output_201410010000.nc")
    (tmp_path / "storm_0").mkdir()
    calibration = SimpleNamespace(objective_params={"storm_forcings": ["/data/dry.nc"]},
                                  output_in=no_output)
    agent = SimpleNamespace(job=SimpleNamespace(workdir=tmp_path))
    with pytest.raises(RuntimeError, match="/data/dry.nc.*storm_0/run.log"):
        search._storm_outputs(calibration, agent)
    assert (tmp_path / "storm_0").exists()             # kept, with its log

def routed(directory: Path, flows) -> Path:
    """A routing output of two hourly steps and three features, with the nexus file the engine
    takes its hours from."""
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


def test_routed_flow_is_summed_over_the_listed_features_in_the_directory_read(tmp_path):
    (tmp_path / "storm_0").mkdir()
    routed(tmp_path, [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    there = routed(tmp_path / "storm_0", [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]])
    flow = read_routed_flow(there, ["11", "33"])
    assert flow["sim_flow"].tolist() == [303.0, 404.0]
    assert list(flow.index) == [pd.Timestamp("2014-10-01 01:00"), pd.Timestamp("2014-10-01 02:00")]
