# Version 1.1.0 verification

Run `python -B verification/v11/run_review_checks.py` from any directory. The driver creates a unique run directory and resolves input archives relative to this bundle. `--output-parent PATH` selects a results parent; `--workers 3` runs independent groups concurrently. Each subprocess uses one BLAS thread and a 600-second timeout. Dependencies: NumPy, SciPy, SymPy, mpmath and psutil. Python 3.13 on Windows was used.

The released suite contains 23 commands: fifteen checks of the bridge, exact-local models and height construction, plus four decoding and four rounding checks. Each script is copied into its run directory before execution. Assertions enforce the stated finite identities and numerical bounds.

Coverage includes magic extraction, sign leakage and mirrors, the two-projection cover, scalar repairs, exact paths and strips, band/row perturbations, cutoff arithmetic, bridge and compiler identities, exact height constants and phases, literal game structure, the S4 witness and renewal models. Finite checks support the written proofs; sampling does not establish universal theorems.

Exactly one passing run is included, produced from a relocated copy of this release. latest-verification.json identifies its results; build-summary.json and visual-qa.json describe the manuscript checks. The full literal boundary-set Bell contraction is outside the 23-command suite; retained finite evaluators and recorded outputs document its scope.

The package excludes version-control history and disposable build caches. SHA256SUMS uses LF. The original game and functional archives retain their exact bytes. Cross-platform execution has not been tested.
