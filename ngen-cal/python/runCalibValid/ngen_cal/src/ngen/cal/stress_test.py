"""
This module reruns the best parameters under forcings with changed storms, after the validation
run, and reports how the validation period's flood peaks and KGE change.

'stress_test' names a CSV with one row per changed forcing and the columns depth (the factor the
storm depth is multiplied by), concentration (the exponent applied to the hourly rain) and
forcing (the forcing file).
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np
import pandas as pd

from .metric_functions import flood_peaks, read_floods
from .search import _calc_metrics, _finish_storms, _start_storms, _storm_outputs

if TYPE_CHECKING:
    from datetime import datetime
    from ngen.cal.agent import Agent
    from ngen.cal.calibratable import Evaluatable

# how many changed-forcing runs go at once; each uses as many cores as the validation run
STRESS_RUNS_AT_ONCE = 3


def _flood_summary(flood: pd.DataFrame) -> dict:
    """Summarize one flood across all runs.

    Gives the flood's peak error under the original forcing; its largest log error
    |ln(simulated / observed peak)| over all runs, which counts a peak half the observed and one
    twice it alike, and the change that caused it; and its response: the slope of
    ln(simulated peak) against ln(depth) over the runs that change depth only. A response of 1
    means the simulated peak grows in proportion to the storm depth.
    """
    original = flood[flood['change'] == 0].iloc[0]
    log_error = np.abs(np.log(flood['sim_peak_cms'] / flood['obs_peak_cms']))
    # idxmax fails when every run gave a NaN peak
    no_peak = pd.Series({'depth': np.nan, 'concentration': np.nan})
    worst = flood.loc[log_error.idxmax()] if log_error.notna().any() else no_peak
    depth_only = flood[flood['concentration'] == 1.0]
    response = float('nan')
    if depth_only['depth'].nunique() > 1 and (depth_only['sim_peak_cms'] > 0).all():
        slope, _ = np.polyfit(np.log(depth_only['depth']), np.log(depth_only['sim_peak_cms']), 1)
        response = float(slope)
    return {'peak_time': original['peak_time'], 'obs_peak_cms': original['obs_peak_cms'],
            'peak_error': original['peak_error'], 'largest_log_error': log_error.max(),
            'largest_at_depth': worst['depth'], 'largest_at_concentration': worst['concentration'],
            'response': response}


def stress_reading(
    unchanged: pd.DataFrame,
    runs: Sequence[pd.DataFrame],
    changes: pd.DataFrame,
    floods: pd.DataFrame,
    observed: pd.DataFrame,
    eval_range: Tuple['datetime', 'datetime'],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the floods and KGE across the original and changed forcings.

    Parameters
    ----------
    unchanged : Simulated flow of the validation run, under the original forcing
    runs : Simulated flow under each row of changes, in order
    changes : depth and concentration of each changed forcing
    floods : peak_time and obs_peak_cms of each flood to report
    observed : Observed flow
    eval_range : The period KGE is computed over

    Returns
    ----------
    Three tables. The first has one row per flood and run, with the simulated peak, its hour,
    the peak error and the timing error in hours. Run 0 is the original forcing and run k the
    k-th row of changes. The second has KGE for each run. The third has one row per flood
    (see _flood_summary).
    """
    settings = [(1.0, 1.0)] + list(zip(changes['depth'], changes['concentration']))
    hydrographs = [unchanged] + list(runs)

    flood_rows, kge_rows = [], []
    for change, ((depth, concentration), run) in enumerate(zip(settings, hydrographs)):
        peaks = flood_peaks(run['sim_flow'], floods)[['peak_time', 'obs_peak_cms', 'sim_peak_cms', 'sim_peak_time']]
        flood_rows.append(peaks.assign(
            change=change,
            depth=depth,
            concentration=concentration,
            peak_error=np.abs(peaks['sim_peak_cms'] / peaks['obs_peak_cms'] - 1.0),
            timing_h=(peaks['sim_peak_time'] - peaks['peak_time']) / pd.Timedelta(hours=1),
        ))
        kge = _calc_metrics(run, observed, eval_range)['KGE']
        kge_rows.append({'change': change, 'depth': depth, 'concentration': concentration, 'KGE': kge})

    columns = ['change', 'depth', 'concentration', 'peak_time', 'obs_peak_cms', 'sim_peak_cms',
               'sim_peak_time', 'peak_error', 'timing_h']
    table = pd.concat(flood_rows, ignore_index=True)[columns]
    by_flood = table.groupby('peak_time', sort=False)
    summary = pd.DataFrame([_flood_summary(flood) for _, flood in by_flood])
    return table, pd.DataFrame(kge_rows), summary


def run_stress_test(agent: 'Agent', calibration_object: 'Evaluatable',
                    unchanged: pd.DataFrame) -> None:
    """Rerun the best parameters under each changed forcing in 'stress_test'.

    Runs STRESS_RUNS_AT_ONCE at a time, then writes the three tables of stress_reading for the
    validation period's floods beside the validation metrics, as <basin>_stress_floods_<run>.csv,
    <basin>_stress_kge_<run>.csv and <basin>_stress_summary_<run>.csv. Does nothing if
    'stress_test' is not set.

    Parameters
    ----------
    agent : Agent object of the validation run
    calibration_object : Evaluatable object
    unchanged : Simulated flow of the validation run
    """
    if calibration_object.stress_test is None:
        return
    eval_range = calibration_object.valid_evaluation_range
    floods = read_floods(calibration_object.floods)
    floods = floods[floods['peak_time'].between(*eval_range)]
    if floods.empty:
        logging.warning("Skipping the stress test: no flood in the flood list peaks in the "
                        "validation period {} to {}".format(*eval_range))
        return
    changes = pd.read_csv(calibration_object.stress_test)
    forcings = list(changes['forcing'])
    runs = []
    for first in range(0, len(forcings), STRESS_RUNS_AT_ONCE):
        batch = forcings[first:first + STRESS_RUNS_AT_ONCE]
        logging.info("Stress test: runs {} to {} of {}".format(first + 1, first + len(batch), len(forcings)))
        _finish_storms(_start_storms(agent, batch))
        runs += _storm_outputs(calibration_object, agent, batch)
    tables = stress_reading(unchanged, runs, changes, floods, calibration_object.observed, eval_range)
    for name, table in zip(('floods', 'kge', 'summary'), tables):
        out = Path(agent.workdir) / '{}_stress_{}_{}.csv'.format(calibration_object.basinID, name, agent.run_name)
        table.to_csv(out, index=False)
