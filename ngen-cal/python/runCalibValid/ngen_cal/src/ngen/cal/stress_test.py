"""
The storm stress test: the best parameters' validation run repeated under changed storm forcings.

The changes are listed in a CSV named by objective_params' 'stress_test', one row per changed
forcing: its depth factor f, its concentration exponent γ and the forcing file (depth,
concentration, forcing). Each forcing runs at the validation run's parameters, each in its own
directory as the robust objectives' changed runs do, and the validation period's listed floods are
read under every change beside the unchanged run. The calibration itself is unchanged.
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

# The robust objectives' three runs at once, so a stress test needs no more cores than they do
STRESS_RUNS_AT_ONCE = 3


def _flood_summary(flood: pd.DataFrame) -> dict:
    """One flood's peak error under the unchanged storm, its largest under any change and where,
    and its response, the slope of ln Q̂ on ln f over the changes of depth alone."""
    unchanged = flood[flood['change'] == 0].iloc[0]
    worst = flood.loc[flood['peak_error'].idxmax()]
    sweep = flood[flood['concentration'] == 1.0]
    response = float('nan')
    if sweep['depth'].nunique() > 1 and (sweep['sim_peak_cms'] > 0).all():
        response = float(np.polyfit(np.log(sweep['depth']), np.log(sweep['sim_peak_cms']), 1)[0])
    return {'peak_time': unchanged['peak_time'], 'obs_peak_cms': unchanged['obs_peak_cms'],
            'peak_error': unchanged['peak_error'], 'largest_peak_error': worst['peak_error'],
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
    """Read the floods and KGE under each change.

    Parameters
    ----------
    unchanged : The validation run's hydrograph
    runs : The hydrograph under each row of ``changes``, in order
    changes : depth and concentration of each changed forcing
    floods : peak_time and obs_peak_cms of each flood read
    observed : Observed hydrograph
    eval_range : The period KGE is scored over

    Returns
    ----------
    Each flood under each change, change 0 the unchanged run and change k the k-th row of
    ``changes``, with its simulated peak, peak error and timing error in hours; KGE under each
    change; and each flood's summary (``_flood_summary``)
    """
    runs_read = [(0, 1.0, 1.0, unchanged)] + [
        (k + 1, depth, concentration, run)
        for k, (depth, concentration, run) in enumerate(zip(changes['depth'], changes['concentration'], runs))]
    rows, kge = [], []
    for change, depth, concentration, run in runs_read:
        peaks = flood_peaks(pd.DataFrame(run)['sim_flow'], floods)
        rows.append(pd.DataFrame({'change': change, 'depth': depth, 'concentration': concentration,
                                  'peak_time': floods['peak_time'].to_numpy(),
                                  'obs_peak_cms': floods['obs_peak_cms'].to_numpy(),
                                  'sim_peak_cms': peaks['sim_peak_cms'].to_numpy(),
                                  'sim_peak_time': peaks['sim_peak_time'].to_numpy()}))
        kge.append({'change': change, 'depth': depth, 'concentration': concentration,
                    'KGE': _calc_metrics(run, observed, eval_range)['KGE']})
    table = pd.concat(rows, ignore_index=True)
    table['peak_error'] = (table['sim_peak_cms'] - table['obs_peak_cms']).abs() / table['obs_peak_cms']
    table['timing_h'] = (table['sim_peak_time'] - table['peak_time']) / pd.Timedelta(hours=1)
    summary = pd.DataFrame([_flood_summary(flood) for _, flood in table.groupby('peak_time', sort=False)])
    return table, pd.DataFrame(kge), summary


def run_stress_test(agent: 'Agent', calibration_object: 'Evaluatable') -> None:
    """Rerun the worker's parameters under each changed forcing listed under objective_params'
    'stress_test', STRESS_RUNS_AT_ONCE at a time, and write the reading (``stress_reading``) of
    the validation period's floods beside the validation metrics, as
    <basin>_stress_floods_<run>.csv, <basin>_stress_kge_<run>.csv and
    <basin>_stress_summary_<run>.csv. Without the list it does nothing."""
    params = calibration_object.objective_params or {}
    if not params.get('stress_test'):
        return
    eval_range = calibration_object.valid_evaluation_range
    floods = read_floods(str(params['floods']))
    floods = floods[(floods['peak_time'] >= eval_range[0]) & (floods['peak_time'] <= eval_range[1])]
    if floods.empty:
        logging.warning("The stress test has no listed flood in the validation period {} to {}".format(*eval_range))
        return
    changes = pd.read_csv(params['stress_test'])
    forcings = list(changes['forcing'])
    runs = []
    for at in range(0, len(forcings), STRESS_RUNS_AT_ONCE):
        batch = forcings[at:at + STRESS_RUNS_AT_ONCE]
        logging.info("Stress test: runs {} to {} of {}".format(at + 1, at + len(batch), len(forcings)))
        _finish_storms(_start_storms(agent, batch), agent.cmd)
        runs += _storm_outputs(calibration_object, agent, batch)
    tables = stress_reading(calibration_object.output, runs, changes, floods,
                            calibration_object.observed, eval_range)
    for name, table in zip(('floods', 'kge', 'summary'), tables):
        out = Path(agent.workdir) / '{}_stress_{}_{}.csv'.format(calibration_object.basinID, name, agent.run_name)
        table.to_csv(out, index=False)
