# PyWPEM XRD refinement reference

## Required artifacts

`XRDfit` consumes three two-column CSV files: the original experimental pattern, the background fit, and the background-subtracted pattern. In a dedicated workspace, `BackgroundFit` creates the latter two under `ConvertedDocuments/`. The refinement also reads `peak0.csv` through `peakN.csv` directly from `work_dir`.

For phase index `i`, maintain this alignment:

| Phase index | CIF | `Lattice_constants[i]` | `density_list[i]` | Peak file |
| --- | --- | --- | --- | --- |
| 0 | phase 0 | phase 0 values | phase 0 density | `peak0.csv` |
| 1 | phase 1 | phase 1 values | phase 1 density | `peak1.csv` |

The XRD data files are headerless two-column CSVs. PyWPEM reads them as 2θ and intensity. Initial peak files use the expected WPEM peak-list schema, including a `2theta/TOF` column.

## Parameter defaults and constraints

| Function | Parameter | Default | Constraint / use |
| --- | --- | --- | --- |
| `BackgroundFit` | `window_length` | 17 | Positive odd integer; greater than `polyorder`; no larger than the data length for `mode="interp"`. |
| `BackgroundFit` | `polyorder` | 3 | Lower than `window_length`. |
| `BackgroundFit` | `bac_split` | 5 | Positive integer for automatic segmentation. |
| `XRDfit` | `bta` | 0.8 | Lorentzian fraction; keep in [0, 1]. |
| `XRDfit` | `bta_threshold` | 0.5 | Lower bound for `bta`; keep in [0, `bta`]. |
| `XRDfit` | `limit` | 0.0005 | Lower bound for peak sigma²; positive. |
| `XRDfit` | `iter_limit` | 0.05 | Likelihood-improvement threshold; positive. |
| `XRDfit` | `w_limit` | 1e-17 | Minimum peak weight; positive. |
| `XRDfit` | `iter_max` | 40 | Positive iteration cap. |
| `XRDfit` | `lock_num` | 2 | Stop after this many decreasing likelihood iterations. |
| `XRDfit` | `InitializationEpoch` | 2 | Initial epochs with peak locations frozen. |

`XRDfit` returns `(duration, initial_lattices)`. Its console reports convergence state and Rp/Rwp. A flag of 1 is convergence; 2 is epsilon/weight limit; 3 is iteration cap; 4 is the likelihood-decrease lock. Treat flags 2–4 as diagnostic stopping states, not automatic success.

## Output locations

- `ConvertedDocuments/`: background-processing outputs.
- `WPEMFittingResults/`: fit profiles, R-factor/log-likelihood plots, lattice and mass-fraction outputs.
- `DecomposedComponents/`: phase-resolved profiles and updated background.
- `output_xrd/`: CIF preprocessing outputs.

## Safe tuning order

1. Correct input file schema, range, wavelength, phase alignment, and `peakN.csv` files.
2. Tune background settings and regenerate background outputs.
3. Re-run with default optimizer values.
4. Tune initialization and convergence controls.
5. Tune peak-shape bounds only if the data supports it.

Avoid accepting any refinement that produces negative/implausible phase quantities, lattice constants inconsistent with the selected phase, or residual structure that indicates missing phases or background mismatch.
