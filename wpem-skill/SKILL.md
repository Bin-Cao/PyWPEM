---
name: wpem-skill
description: Run, validate, and tune reproducible PyXplore WPEM whole-pattern XRD workflows. Use when an LLM must prepare or execute WPEM calls such as BackgroundFit, StructureSolve, CIFpreprocess, XRDfit, Plot_Components, XRDSimulation, or amorphous analysis; determine their required execution order; diagnose failed or poor fits; or adjust fitting parameters without changing unrelated inputs.
---

# WPEM Workflow

Use the PyXplore WPEM interface from an empty, dedicated run directory. Treat that directory as the output workspace: WPEM's relative output paths (including `ConvertedDocuments/`) are relative to the current working directory. Use absolute paths for inputs outside it, and do not mix artifacts from different runs.

## Select the workflow

Use this decision order:

1. Use `CIFpreprocess` and `XRDSimulation` for a structure/CIF-only task.
2. Use the XRD refinement workflow below for crystalline XRD data and one or more candidate CIFs.
3. Use `Amorphous_fit` and optionally `AmorphousRDFun` only after an amorphous component has been isolated.
4. Do not invoke XPS or EXAFS functions for XRD requests.

Read [references/xrd-refinement.md](references/xrd-refinement.md) before writing an XRD refinement call or changing its parameters.

## Environment check

Before every workflow, install PyXplore if absent and check PyPI for a newer release:

```bash
python3 -m pip install --upgrade PyXplore
python3 -c "from PyXplore import WPEM; print('PyXplore import passed')"
```

`--upgrade` leaves an installed current version unchanged and updates only when a newer version is available. Do not start a WPEM workflow if installation or import fails; report the command error and request a usable Python environment.

## XRD refinement execution order

1. Run the environment check above.
2. Require two input types before starting: one experimental XRD pattern and one CIF for each candidate phase. If either is absent, tell the user exactly which file is needed and stop; do not invent a pattern or structure.
3. Create an empty, dedicated run directory. The required pattern format is a headerless two-column CSV: column 1 is increasing 2θ and column 2 is non-negative intensity. If the supplied pattern is already compliant, place it there as `intensity.csv`. Otherwise, convert a parseable text/CSV/TSV file without changing the source:

   ```bash
   python3 /absolute/path/to/wpem-skill/scripts/normalize_xrd_csv.py \
     /absolute/path/to/supplied-pattern.txt /absolute/path/to/run/intensity.csv
   ```

   For files whose 2θ and intensity are not columns 0 and 1, inspect the file and pass `--angle-column` and `--intensity-column`. Stop and ask the user for column mapping when it cannot be determined safely.
4. Before background fitting, run the preflight check, listing every phase CIF:

   ```bash
   python3 /absolute/path/to/wpem-skill/scripts/validate_xrd_workspace.py \
     /absolute/path/to/run --phases N \
     --cif /absolute/path/to/phase-0.cif [--cif /absolute/path/to/phase-1.cif ...]
   ```

   Do not continue when it reports an error.
5. Read `intensity.csv` with `pandas.read_csv(..., header=None)`, then run `WPEM.BackgroundFit`. Preserve its result as `var`; it is an input to `XRDfit`. Use the generated `ConvertedDocuments/no_bac_intensity.csv` in all subsequent phase preparation.
6. For each candidate phase, run the `WPEM.StructureSolve(no_bac_intensity_file=..., cif_file=...)` **before** `CIFpreprocess`. It pre-optimizes that CIF for the later workflow. Keep the same CIF/phase ordering from this step onward.
7. Run `WPEM.CIFpreprocess` for every phase with the intended 2θ range and preserve its `latt` result in the corresponding position of `Lattice_constants`.
8. Assemble `Lattice_constants` in exactly the same order as the CIFs. Then run the refinement preflight check with `--require-refinement`. If `peak0.csv` through `peakN.csv` are absent, rerun `CIFpreprocess` for each affected CIF. It creates `output_xrd/<cif-stem>HKL.csv`; copy that file to the run-directory root as `peak<i>.csv`, where `i` is the phase's `Lattice_constants` index. Do not alter peak data or phase order. Rerun preflight and continue only after it passes.
9. Run `WPEM.XRDfit`, passing `var`, the phase-aligned lattice list, and the experimental, background, and background-subtracted file paths.
10. Inspect `WPEMFittingResults/`, `DecomposedComponents/`, convergence text, and fit plots before claiming success. Run `WPEM.Plot_Components` only after a successful or diagnostically useful fit.

## Invocation pattern

Run this script with the dedicated run directory as its current directory. Replace only values supported by the request and inspected data:

```python
from PyXplore import WPEM
import pandas as pd

intensity_csv = pd.read_csv("intensity.csv", header=None)
var = WPEM.BackgroundFit(
    intensity_csv,
    lowAngleRange=17,
    poly_n=13,
    bac_split=16,
    bac_num=300,
)

# Run once per phase, before CIFpreprocess.
WPEM.StructureSolve(
    no_bac_intensity_file="./ConvertedDocuments/no_bac_intensity.csv",
    cif_file="phase-0.cif",
)

latt0, _, _ = WPEM.CIFpreprocess(
    filepath="phase-0.cif",
    two_theta_range=(15, 75),
)

# When peak0.csv is absent, use the HKL list generated by CIFpreprocess.
from pathlib import Path
from shutil import copy2

generated_hkl = Path("output_xrd/phase-0HKL.csv")
if not Path("peak0.csv").is_file():
    if not generated_hkl.is_file():
        raise FileNotFoundError(f"CIFpreprocess did not generate {generated_hkl}")
    copy2(generated_hkl, "peak0.csv")

WPEM.XRDfit(
    wavelength=[1.540593, 1.544414],
    Var=var,
    Lattice_constants=[latt0],
    no_bac_intensity_file="./ConvertedDocuments/no_bac_intensity.csv",
    original_file="intensity.csv",
    bacground_file="./ConvertedDocuments/bac.csv",
    subset_number=11,
    low_bound=20,
    up_bound=70,
    bta=0.85,
    iter_max=5,
    asy_C=0,
    InitializationEpoch=0,
)
```

Use the actual instrument wavelength; do not assume a Cu Kα doublet for non-Cu data. For multiple phases, repeat `StructureSolve` and `CIFpreprocess` once per CIF, then pass all `latt` values to `Lattice_constants` in the identical order.

## Parameter adjustment protocol

Make a baseline run first. Change one parameter group per rerun, record the old/new value and observed effect, and keep the better result only when fit diagnostics and physical plausibility both improve.

- For background errors, adjust `lowAngleRange`, `poly_n`, `bac_split`, or `bac_num` in `BackgroundFit`; then rerun `StructureSolve`, `CIFpreprocess`, and `XRDfit`.
- For inadequate convergence, first verify CIF identity/order, wavelength, range, and peak lists. Then adjust `InitializationEpoch`, `iter_max`, `subset_number`, `low_bound`, or `up_bound` conservatively.
- For unstable or implausibly narrow peaks, adjust `bta` conservatively and keep it in `[0, 1]`.
- For peak-position bias, rerun `StructureSolve` and revisit the CIF lattice constants and 2θ range before trying any zero-shift option; enable zero shift only with a standard sample.

Never change wavelength, CIF identity, phase count, and optimizer settings in the same experiment. Never report a lower R factor alone as a valid physical fit.

## Completion report

Report the run directory, WPEM version, function calls and non-default parameters, convergence/iteration/Rp/Rwp information from the console, generated result paths, and every tuning change. State unresolved input or physical-model limitations explicitly.
