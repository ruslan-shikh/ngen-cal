import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_where = str(Path(__file__).parent)
global_config = {
    "global": {
        "formulations": [
            {
                "name": "bmi_c",
                "params": {
                    "name": "bmi_c",
                    "model_type_name": "CFE",
                    "main_output_variable": "Q_OUT",
                    "init_config": "/Users/nels.frazier/workspace/ngen//data/bmi/c/cfe/cat-27_bmi_config.ini",
                    "allow_exceed_end_time": False,
                    "fixed_time_step": False,
                    "uses_forcing_file": False,
                    "variables_names_map": {
                        "atmosphere_water__liquid_equivalent_precipitation_rate": "precip_rate",
                        "water_potential_evaporation_flux": "potential_evapotranspiration",
                        "atmosphere_air_water~vapor__relative_saturation": "SPFH_2maboveground",
                        "land_surface_air__temperature": "TMP_2maboveground",
                        "land_surface_wind__x_component_of_velocity": "UGRD_10maboveground",
                        "land_surface_wind__y_component_of_velocity": "VGRD_10maboveground",
                        "land_surface_radiation~incoming~longwave__energy_flux": "DLWRF_surface",
                        "land_surface_radiation~incoming~shortwave__energy_flux": "DSWRF_surface",
                        "land_surface_air__pressure": "PRES_surface"
                    },
                    "model_params": {
                        "maxsmc": 0.21470105463393196,
                        "satdk": 0.0003343056064723208,
                        "slope": 0.5836411296916055,
                        "multiplier": 465.8926732378259,
                        "expon": 7.813285220525254
                    },
                    "library_file": "/Users/nels.frazier/workspace/ngen/extern/cfe/cmake_build/libcfebmi.dylib",
                    "registration_function": "register_bmi_cfe"
                }
            }
        ],
        "forcing": {
            #FIXME regex not working???
            #"file_pattern": ".*{{ID}}.*csv",
            "path":_where+"/data/cat-87_2015-12-01 00_00_00_2015-12-30 23_00_00.csv",
            "start_time": "2015-12-01 00:00:00",
            "end_time": "2015-12-30 23:00:00"
        }
    }
}

time = {
    "time": {
        "start_time": "2015-12-01 00:00:00",
        "end_time": "2015-12-30 23:00:00",
        "output_interval": 3600
    }
}

catchment = {
        "tst-1": {
            "formulations": [
            {
                "name": "bmi_c",
                "params": {
                    "name": "bmi_c",
                    "model_type_name": "CFE",
                    "main_output_variable": "Q_OUT",
                    "init_config": "/Users/nels.frazier/workspace/ngen//data/bmi/c/cfe/cat-27_bmi_config.ini",
                    "allow_exceed_end_time": False,
                    "fixed_time_step": False,
                    "uses_forcing_file": False,
                    "variables_names_map": {
                        "atmosphere_water__liquid_equivalent_precipitation_rate": "precip_rate",
                        "water_potential_evaporation_flux": "potential_evapotranspiration",
                        "atmosphere_air_water~vapor__relative_saturation": "SPFH_2maboveground",
                        "land_surface_air__temperature": "TMP_2maboveground",
                        "land_surface_wind__x_component_of_velocity": "UGRD_10maboveground",
                        "land_surface_wind__y_component_of_velocity": "VGRD_10maboveground",
                        "land_surface_radiation~incoming~longwave__energy_flux": "DLWRF_surface",
                        "land_surface_radiation~incoming~shortwave__energy_flux": "DSWRF_surface",
                        "land_surface_air__pressure": "PRES_surface"
                    },
                    "model_params": {
                        "maxsmc": 0.21470105463393196,
                        "satdk": 0.0003343056064723208,
                        "slope": 0.5836411296916055,
                        "multiplier": 465.8926732378259,
                        "expon": 7.813285220525254
                    },
                    "library_file": "/Users/nels.frazier/workspace/ngen/extern/cfe/cmake_build/libcfebmi.dylib",
                    "registration_function": "register_bmi_cfe"
                }
            }
        ],
            "forcing": {
                "path": _where+"/data/cat-87_2015-12-01 00_00_00_2015-12-30 23_00_00.csv",
                "start_time": "2015-12-01 00:00:00",
                "end_time": "2015-12-30 23:00:00"
            },
            "calibration": {"CFE": [
                {
                    "param": "some_param",
                    "min": 0.0,
                    "max": 1.0,
                    "init": 0.5
                },
                {
                    "param": "maxsmc",
                    "min": 0.2,
                    "max": 1.0,
                    "init": 0.439
                }
            ]}
        }
}


one_catchment = {
    "catchments": {**catchment}
}

two_catchment = {
    "catchments": {**catchment, **catchment}
}

config = {**global_config, **time, **one_catchment}

algorithm_good = {"algorithm": "dds"}
algorithm_bad = {"algorithm": "foo"}

strategy_estimation = {"type": "estimation", **algorithm_good}
strategy_sensitivity = {"type": "sensitivity"}

general_no_defaults = {"strategy":strategy_estimation, 
                    "iterations": 10,
                    "restart": True,
                    "start_iteration": 5,
                    "workdir": "/tmp",
                    "log_file": "test_log",
                    "parameter_log_file": "test_param_log",
                    "objectective_log_file": "test_obj_log"}

general_w_defaults = {"strategy":strategy_estimation, 
                    "iterations": 10,
                    "evaluation_start": "2015-12-01 00:00:00", 
                    "evaluation_stop": "2015-12-30 23:00:00"}

evaluation_options = {"eval_params":{
                        "evaluation_start": "2015-12-01 00:00:00", 
                        "evaluation_stop": "2015-12-30 23:00:00"}
                    }

model_params = {"binary":"echo", "args":"ngen args", **evaluation_options}


HOURS = pd.date_range("2020-01-01 01:00", periods=2000, freq="h")


def hydrograph(*peaks, name="sim_flow", base=1.0) -> pd.Series:
    """Return an hourly flow with a triangular peak at each (hour, height)."""
    q = np.full(len(HOURS), base)
    for at, height in peaks:
        for k in range(-12, 13):
            q[at + k] = max(q[at + k], base + (height - base) * (1 - abs(k) / 12))
    return pd.Series(q, index=HOURS, name=name)


def flood_list(tmp_path, *floods) -> str:
    """Write a flood list of (hour, observed peak) pairs and return its path."""
    path = tmp_path / "floods.csv"
    pd.DataFrame({"peak_time": [HOURS[at] for at, _ in floods],
                  "obs_peak_cms": [q for _, q in floods]}).to_csv(path, index=False)
    return str(path)


# a model run that writes the forcing path of its realization to ran.txt
SHOW = (f'{sys.executable} -c "import json, pathlib; pathlib.Path(\'ran.txt\').write_text('
        'json.load(open(\'realization.json\'))[\'global\'][\'forcing\'][\'path\'])"')


def worker(tmp_path, cmd=SHOW, run_name="calib") -> SimpleNamespace:
    """Write a realization into tmp_path and return an agent that runs cmd there."""
    (tmp_path / "realization.json").write_text(json.dumps(
        {"global": {"forcing": {"path": "/data/forcings.nc", "provider": "NetCDF"}},
         "time": {"start_time": "2014-10-01 00:00:00"}}, indent=4))
    return SimpleNamespace(job=SimpleNamespace(workdir=tmp_path, log_file=None), cmd=cmd,
                           realization_file=Path("/somewhere/else/realization.json"),
                           run_name=run_name, workdir=tmp_path)
