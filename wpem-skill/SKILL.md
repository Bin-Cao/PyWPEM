---
name: wpem-skill
description: Run, validate, and tune reproducible PyWPEM whole-pattern XRD refinement workflows. Use when an LLM must prepare or execute PyWPEM calls such as BackgroundFit, CIFpreprocess, XRDfit, Plot_Components, XRDSimulation, or amorphous analysis; determine their required execution order; diagnose failed or poor fits; or adjust fitting parameters without changing unrelated inputs.
---

# WPEM Workflow

Execute PyWPEM from the repository root. Treat `work_dir` as the only output workspace; use absolute paths and never mix files from different workspaces.

## Select the workflow

Use this decision order:

1. Use `CIFpreprocess` and `XRDSimulation` for a structure/CIF-only task.
2. Use the XRD refinement workflow below for crystalline XRD data and one or more candidate CIFs.
3. Use `Amorphous_fit` and optionally `AmorphousRDFun` only after an amorphous component has been isolated.
4. Do not invoke XPS or EXAFS functions for XRD requests.

Read [references/xrd-refinement.md](references/xrd-refinement.md) before writing an XRD refinement call or changing its parameters.

## XRD refinement execution order

1. Create an empty, dedicated `work_dir`; copy or reference the two-column experimental `intensity.csv` there. Use a monotonically increasing 2θ column and non-negative intensity values.
2. Run the preflight check. For an existing refinement workspace:

   ```bash
   python3 wpem-skill/scripts/validate_xrd_workspace.py /absolute/path/to/work_dir --phases N --require-refinement
   ```

   Before background fitting, omit `--require-refinement`. Do not run `XRDfit` when preflight reports an error.
3. Run `WPEM.BackgroundFit` once, passing the experimental data and `work_dir`. Use its generated `ConvertedDocuments/no_bac_intensity.csv` and `ConvertedDocuments/bac.csv` as the next inputs.
4. For every phase, run `WPEM.CIFpreprocess` with the same wavelength and 2θ range. Confirm that the initial peak list is available as `work_dir/peak0.csv`, `peak1.csv`, and so on. Do not silently fabricate peak positions.
5. Assemble phase-aligned lists: `Lattice_constants`, `density_list`, and the `peakN.csv` files must have exactly the same phase order and count.
6. Run `WPEM.XRDfit` using explicit keyword arguments, including `work_dir`, `two_theta_range`, `cpu`, and the three background/data file paths.
7. Inspect `WPEMFittingResults/`, `DecomposedComponents/`, convergence text, and fit plots before claiming success. Run `WPEM.Plot_Components` only after a successful or diagnostically useful fit.

## Invocation pattern

Use this pattern; replace only values supported by the request and the inspected input data:

```python
from src import WPEM

work_dir = "/absolute/path/to/run"
WPEM.BackgroundFit(
    intensity_csv="/absolute/path/to/run/intensity.csv",
    work_dir=work_dir,
    window_length=17,
    polyorder=3,
    bac_split=5,
)

# Run once per phase and preserve the identical phase ordering below.
latt0, _, density0 = WPEM.CIFpreprocess(
    "/absolute/path/to/phase-0.cif",
    wavelength="CuKa",
    two_theta_range=(20, 80),
    work_dir=work_dir,
)

duration, initial_lattices = WPEM.XRDfit(
    wavelength=[1.54056, 1.54439],
    Var=50.0,
    Lattice_constants=[latt0],
    no_bac_intensity_file=f"{work_dir}/ConvertedDocuments/no_bac_intensity.csv",
    original_file=f"{work_dir}/intensity.csv",
    bacground_file=f"{work_dir}/ConvertedDocuments/bac.csv",
    density_list=[density0],
    two_theta_range=(20, 80),
    iter_max=40,
    cpu=4,
    work_dir=work_dir,
)
```

Use the actual instrument wavelength. Do not assume Cu Kα for non-Cu data. Set `cpu` no higher than the available CPU allocation.

## Parameter adjustment protocol

Make a baseline run first. Change one parameter group per rerun, record the old/new value and observed effect, and keep the better result only when the fit diagnostics and physical plausibility both improve.

- For background errors, adjust `segment`, `bac_split`, `window_length`, `polyorder`, `poly_n`, or `bac_var_type` in `BackgroundFit`; then rerun the entire downstream chain.
- For inadequate convergence, first verify phase order, wavelength, range, and peak lists. Then adjust `InitializationEpoch`, `iter_max`, `iter_limit`, or `lock_num` conservatively.
- For unstable or implausibly narrow peaks, adjust `limit`, `bta`, and `bta_threshold` conservatively. Preserve `0 <= bta_threshold <= bta <= 1`.
- For peak-position bias, revisit CIF lattice constants and `two_theta_range` before trying `ZeroShift`; enable `ZeroShift` only with a standard sample.
- Use `MODEL="REFINEMENT"` for lattice refinement and `MODEL="ANALYSIS"` for fixed-background component analysis. Do not change mode merely to force a lower residual.

Never change wavelength, CIF identity, phase count, and optimizer settings in the same experiment. Never report a lower R factor alone as a valid physical fit.

## Completion report

Report the workspace, function calls and non-default parameters, convergence flag/iteration/Rp/Rwp from the console, generated result paths, and every tuning change. State unresolved input or physical-model limitations explicitly.
