# PyXplore WPEM XRD refinement reference

## Required artifacts and phase order

Before each workflow, run `python3 -m pip install --upgrade PyXplore`. It installs PyXplore when missing, checks for a newer PyPI release, and upgrades only if necessary. Confirm the package is usable with `python3 -c "from PyXplore import WPEM"`; stop and report any installation or import failure.

Before any refinement, obtain an experimental XRD pattern and one CIF for each candidate phase. If either is missing, ask the user to provide it; never infer or fabricate either input.

`intensity.csv` is a headerless two-column CSV: increasing 2θ then non-negative intensity. `BackgroundFit` creates the background-subtracted pattern and fitted background under `ConvertedDocuments/`; `XRDfit` consumes all three files. To normalize a noncompliant but parseable text, CSV, or TSV input, run:

```bash
python3 /absolute/path/to/wpem-skill/scripts/normalize_xrd_csv.py INPUT OUTPUT
```

The converter preserves the source file and writes a headerless two-column CSV. Default columns are 0 (2θ) and 1 (intensity); specify `--angle-column N --intensity-column M` when needed. Do not guess an ambiguous column mapping.

For each phase, preserve this sequence and order:

| Phase index | Candidate CIF | `StructureSolve` | `CIFpreprocess` result | `Lattice_constants[i]` | Peak file |
| --- | --- | --- | --- | --- | --- |
| 0 | phase 0 | first | `latt0` | `latt0` | `peak0.csv` |
| 1 | phase 1 | first | `latt1` | `latt1` | `peak1.csv` |

`StructureSolve` is available in PyXplore WPEM v2026.8.17 and later. It accepts the background-subtracted pattern plus a CIF and must run before that CIF's `CIFpreprocess`; it pre-optimizes the input structure for the subsequent workflow.

`CIFpreprocess` generates `output_xrd/<cif-stem>HKL.csv`. Before calling `XRDfit`, each phase must have a root-level `peak<i>.csv`. If one is missing, copy the generated HKL file to `peak<i>.csv`; `i` must equal that phase's position in `Lattice_constants`. Never reorder, combine, or otherwise edit the generated peak list.

## Call contracts

```python
from PyXplore import WPEM
import pandas as pd

intensity_csv = pd.read_csv("intensity.csv", header=None)
var = WPEM.BackgroundFit(intensity_csv, lowAngleRange=17, poly_n=13,
                         bac_split=16, bac_num=300)

WPEM.StructureSolve(
    no_bac_intensity_file="./ConvertedDocuments/no_bac_intensity.csv",
    cif_file="phase.cif",
)
latt, atom_coordinates, description = WPEM.CIFpreprocess(
    filepath="phase.cif", two_theta_range=(15, 75)
)

from pathlib import Path
from shutil import copy2

# Required only when peak0.csv does not already exist.
if not Path("peak0.csv").is_file():
    copy2("output_xrd/phaseHKL.csv", "peak0.csv")

WPEM.XRDfit(
    wavelength=[1.540593, 1.544414], Var=var,
    Lattice_constants=[latt],
    no_bac_intensity_file="./ConvertedDocuments/no_bac_intensity.csv",
    original_file="intensity.csv",
    bacground_file="./ConvertedDocuments/bac.csv",
    subset_number=11, low_bound=20, up_bound=70,
    bta=0.85, iter_max=5, asy_C=0, InitializationEpoch=0,
)
```

The `bacground_file` spelling above matches the WPEM argument name. Retain it exactly unless the installed package's signature shows a different version-specific API.

## Safe tuning order

1. Correct input file schema, 2θ coverage, wavelength, phase order, and generated `peakN.csv` files.
2. Tune `BackgroundFit` parameters and regenerate all downstream artifacts, including `StructureSolve` results.
3. Make a baseline refinement.
4. Tune convergence/range controls: `InitializationEpoch`, `iter_max`, `subset_number`, `low_bound`, and `up_bound`.
5. Tune peak shape (`bta`) only if the data supports it.

Avoid accepting a refinement with negative or implausible phase quantities, lattice constants inconsistent with the selected phase, or residual structure indicating missing phases or a background mismatch.

## Outputs to inspect

- `ConvertedDocuments/`: background-processing outputs.
- `WPEMFittingResults/`: fit profiles and refinement diagnostics.
- `DecomposedComponents/`: phase-resolved profiles and updated background.
- `output_xrd/`: CIF-preprocessing outputs.

Inspect the actual run directory after each call; output directory names can vary by installed PyXplore release.
