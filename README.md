# ngiab-cal Docker Image

## What is this?

This repository contains a modified version of ngen-cal designed to work with NextGen In A Box (NGIAB). It provides calibration capabilities for hydrologic models through a containerized environment.

## Limitations
Understanding Docker containers and how to use `docker exec` is recommended if you need to troubleshoot.
1. **Limited Calibration Algorithms**: Currently only configured to use the Dynamically Dimensioned Search (DDS) algorithm. Other algorithms available in the main ngen-cal branch are not supported.

2. **Single-Gage Calibration**: Designed for calibrating single-basin models using one USGS streamgage. Multi-gage or multi-objective calibration is not supported *_by this branch_*. The [flood objective](#flood-objective) combines KGE and flood peaks into one weighted score.

4. **Custom ngen-cal Branch**: Uses a modified version of ngen-cal from the `ngiab_cal` branch, which differs from the main branch. Features from newer releases of ngen-cal may not be available.

5. **Compatibility Concerns**: The customized ngen-cal implementation may not be compatible with current and future changes to ngen-cal's main branch.

## How to use this

1. **Prepare your data directory** using the NGIAB data preprocessor:
   ```bash
   # Example command to create a basin configuration
   uvx --from ngiab_data_preprocess cli -i gage-10154200 -sfr --start 2010-01-01 --end 2015-02-28
   ```

   The preprocessor will create a directory structure with all the necessary files for model simulation.

2. **Run ngiab-cal** to set up the calibration environment:
   ```bash
   # Create calibration configuration
   uvx ngiab-cal ~/ngiab_preprocess_output/gage-10154200 -g 10154200

   # Or create and automatically run calibration
   uvx ngiab-cal ~/ngiab_preprocess_output/gage-10154200 -g 10154200 --run
   ```

3. **Manually run the Docker image** (if not using the `--run` flag):
   ```bash
   docker run -it -v "~/ngiab_preprocess_output/gage-10154200/:/ngen/ngen/data" joshcu/ngiab-cal:demo
   ```

4. **Check results** in the `calibration/Output` directory after running.

## Container Structure

- The Docker image is based on the NGIAB container with additional files for calibration
- `/calibration/` contains the ngen-cal code and scripts
- `/calibration/run.sh` is the entry point for running calibration
- Files from your data directory are mounted to `/ngen/ngen/data/`
- The NGIAB container's existing directories and tools are preserved

## The mpi-ngen Script

In this container, `mpi-ngen` is a script that replaces the NextGen binary to allow ngen serial commands to be run in parallel. It wraps `ngen-serial` with the necessary steps to run simulations with MPI, modified from the [NGIAB guide.sh script](https://github.com/CIROH-UA/NGIAB-CloudInfra/blob/main/guide.sh).

## Modifying Calibration Parameters

To modify the calibration configuration parameters, you should:

1. Edit the configuration file in your calibration directory:
   ```
   ~/ngiab_preprocess_output/gage-XXXX/calibration/ngen_cal_conf.yaml
   ```

2. Be aware that running ngiab-cal with the `-f` flag will overwrite any manual changes.

## Flood objective

The objective `kge_flood` adds the error in flood peaks to KGE. Set it in `ngen_cal_conf.yaml`:

```yaml
model:
  eval_params:
    objective: kge_flood
    floods: /ngen/ngen/data/calibration/floods.csv
    flood_weight: 0.5
```

The calibration minimizes `(1 - w)(1 - KGE) + w*E`, where `w` is `flood_weight` (0 to 1) and `E` is the mean peak error `|sim - obs| / obs` of the floods that peak in the calibration period. `obs` is a flood's observed peak and `sim` is the largest simulated flow within seven days of it. The floods CSV has two columns: `peak_time`, the hour of the observed peak, and `obs_peak_cms`, the observed peak in m3/s. The config fails to load if no flood's peak falls between `evaluation_start` and `evaluation_stop`.

If `floods` is set, the metrics files get four more columns, whatever the objective:

- `FLOOD_MEAN_PEAK_ERR`: `E`
- `FLOOD_MAX_PEAK_ERR`: the largest peak error
- `FLOOD_MEAN_SHIFT_H`: the mean number of hours between the simulated and observed peaks
- `KGE_FLOOD`: one minus the loss above

They are NaN for a period with no flood.

### Changed storms

Two settings rerun the model with the storms in the forcing changed. Both need the `uniform` strategy, and each run uses as many cores as the main run.

- `objective: kge_flood_robust` with `storm_forcings`, a list of forcing files, runs every iteration once under each of them as well. It replaces `E` with the mean, over the floods, of each flood's largest `|ln(sim / obs)|` across all the runs, and reports it as `FLOOD_ROBUST_LOG_ERR`. KGE still comes from the original forcing.
- `stress_test`, a CSV with the columns `depth`, `concentration` and `forcing`, reruns the best parameters under each forcing after the validation run. It needs `floods`. For the floods in the validation period it writes `<basin>_stress_floods_<run>.csv` (each flood's peak, peak error and timing under every forcing), `<basin>_stress_kge_<run>.csv` (KGE under every forcing) and `<basin>_stress_summary_<run>.csv` (each flood's largest `|ln(sim / obs)|` over the forcings, and how its peak responds to storm depth).

## Custom Building

If you need to modify the container:

```bash
git clone https://github.com/YOUR-REPO/ngiab-cal.git
cd ngiab-cal
docker build -t ngiab-cal:custom .
```

Then use your custom image in place of `joshcu/ngiab-cal:demo`.
