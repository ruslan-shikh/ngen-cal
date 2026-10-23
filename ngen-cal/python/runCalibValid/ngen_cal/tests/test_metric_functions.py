import numpy as np
import pandas as pd
import pytest

from ngen.cal.metric_functions import NSE, Weighted_NSE

observed = pd.Series(np.linspace(0.5, 20.0, 200))
simulated = observed * 0.8 + 0.5


def test_log_nse_of_a_perfect_simulation_is_one():
    nselog = NSE(observed, observed, fun="log", epsilon="Pushpalatha2012")
    assert nselog == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_weighted_nse_weighs_nse_and_log_nse(weight):
    nse = NSE(observed, simulated)
    nselog = NSE(observed, simulated, fun="log", epsilon="Pushpalatha2012")
    assert nse != pytest.approx(nselog)
    assert Weighted_NSE(observed, simulated, weight=weight) == pytest.approx(
        weight*nse + (1 - weight)*nselog)


def test_weighted_nse_of_a_perfect_simulation_is_one():
    assert Weighted_NSE(observed, observed) == pytest.approx(1.0, abs=1e-12)
