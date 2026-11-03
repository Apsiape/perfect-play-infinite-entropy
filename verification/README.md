# Verification bundle

Version 1.1.0, November 2026. The original verification objects below are retained; full decoding/rounding proofs and their new checks are in the paper and `v11/`. The unrestricted logarithmic law is proved in the manuscript. Companion to *Perfect play in a finite nonlocal game requires infinite entanglement entropy*, by Seth Douglas and Nidhal Mghirbi.

## Objects and evidence

- `out5/functional-small.zip`: unchanged rational Bell functional. SHA-256 `bb5c0bf32e9a7175bef73d9b156a5c6f3e3a0487462c90a9138e82f809927aaf`.
- `out6/game.zip`: unchanged finite Boolean game. SHA-256 `f29237d3920afe1b4ff9a3749b68934169d8109bbaa0fa19dbbcc34f334824d2`.
- `out5/evaluate.py`: the construction's Bell evaluator. Reads every coefficient, reconstructs effects independently of the producer, and uses exact Q(i,sqrt(2),sqrt(3)) arithmetic.
- `reviews/bell/work/my_evaluate.py`: separate review evaluator. Its exact Q(zeta_24) arithmetic and measurement implementation share no code with the construction's evaluator.
- `out6/check_game.py`: the construction's game evaluator, independent of the game producer. It explicitly reuses the out5 matrix kernel for old measurements.
- `reviews/game/work/old_eval.py`, `new_eval.py`, `oldmodel.py`, `field24.py`, `gamespec.py`: independent review implementation of both the old measurements and new game rules. This is the second full measurement implementation for the game, not the reused construction's kernel.
- `out6/check_verifier.py` and `build_game.py`: public-verifier checks and the saved public predicate implementation. The latter is also embedded unchanged in game.zip.
- `reviews/game/work/brute.py`: direct matrix and changed-column controls for selected row gadgets.
- `analytic/checks.py`: scalar taper formulas, literal event audit, exact rank-one contractions, and asymptotic illustrations.
- `reviews/theoremD/work/eval_taper.py`: literal filtered-state evaluation of every Bell coefficient at M=1,2, with a reused cross-checked measurement builder and independent state/contraction code. Original outputs are in `recorded/theoremD/`; the archive path alone was made bundle-relative.
- `analytic/independent_d.py`: fresh independent same-family standard-library calculation of exact taper profiles, integrals and constants; no supplied checker is imported.
- `recorded/`: outputs supplied with the research handoff, separated from fresh runs.
- Root execution logs and `execution-manifest.json`: fresh packaged-suite results. JSON beside each evaluator is its latest execution output.

Scripts retain their computational logic. The filtered-state checker uses a bundle-relative archive path. Two Bell review scripts use the variable name `Q_bob`; one comment was clarified. Source and release hashes and adaptations are in `SOURCE-HASHES.json`. Extracted JSON/binary inputs sit in `reviews/bell/unz5/`, `reviews/game/unz5/` and `reviews/game/unz/`; these are exact archive members. They allow the review scripts' original relative layouts to run without machine-local paths. Independent means independent implementation and derivation, not an external referee or proof-assistant certification. The supplied game review was conditional on the separately reviewed Bell theorem.

## Environment

Python 3.13, NumPy and SciPy reproduce the review Bell evaluator; NumPy is needed by the other matrix evaluators. Optional psutil records review resource usage. The construction's resource checks use Windows ctypes. The supplied scripts were verified on Windows; platform-independent operation of every resource helper is not claimed. The scalar checks use the standard library. No optimizer, network, or service is required.

Use `python -B` and limit BLAS threads to one if running multiple scripts. Run these commands from the stated directories, sequentially.

### Bell, construction implementation (from out5/)

    python -B evaluate.py small

### Bell, independent review implementation (from reviews/bell/work/)

    python -B my_evaluate.py ../unz5 0 1 2

### Game, construction implementation (from out6/)

    python -B check_game.py 0 1 2
    python -B check_verifier.py

### Game, independent review implementation (from reviews/game/work/)

    python -B audit_interface.py
    python -B test_spec.py
    python -B old_eval.py 0
    python -B new_eval.py 0
    python -B old_eval.py 1
    python -B new_eval.py 1
    python -B old_eval.py 2
    python -B new_eval.py 2

The independent game runs take several minutes per cutoff and use approximately 0.6 GB each. The scripts write JSON next to themselves; the Bell review writes its result in the working directory. These are intended output locations inside this bundle.

### Literal filtered-state check (from reviews/theoremD/work/)

    python -B eval_taper.py 1
    python -B eval_taper.py 2

This is a separate contraction engine for (36), sharing the cross-checked `out5` measurement builder. It checks every literal coefficient for the taper and two mixed-sign rational filters at each cutoff. It is not a second independent measurement builder. Only M=1,2 are supported by the copied exact field implementation. The original review describes itself as a same-family review. The preparation review records a different-family check for every theorem and an additional reading of the Theorem D proof with independent scalar recomputation. The bundled `analytic/independent_d.py` reproduces shell/path formulas, elementary bounds and constants.

### Additional checks

From `analytic/`: `python -B checks.py` and `python -B independent_d.py`.

From `reviews/game/work/`: `python -B brute.py 0 R_isometry` and `python -B brute.py 0 --flip=3 offdiag_G12_0`. Recorded direct checks cover ten d=16 gadgets and a changed-column control. Additional commands from `reviews/bell/work/` are `python -B sos_check.py`, `python -B audit_core.py ../unz5/core.json`, and `python -B reconstruct_table.py ../unz5 reconstruct_small.json`. The optional original larger-presentation comparison in audit_core.py needs an additional historical input and is not required for this small-archive audit. Supplied recorded JSON is included.

Do not run the game builder merely to evaluate the predicate. Rebuilding may alter ZIP timestamps and archive hashes. The saved archives are the specified objects. The verifier functions do not invoke the builder.

## Expected values

At N=0,1,2 the Bell scores are exactly -2/75, -2/375, -2/825. The old local dimensions are 27,54,108. The only nonzero core row is R_isometry; every old direct forbidden event has zero probability.

At the same cutoffs the game local dimensions are 6912,13824,27648. Exactly one question pair loses. Uniform game losses are respectively 1/275680239360000, 1/1378401196800000, and 1/3032482632960000. The formula is t_N/(9600*169460^2), with t_N=1,1/5,1/11. These are imperfect finite strategies.

For the different, tapered family the Bell losses at N=1,2,3,5 are 33/241,121/2749,7149/349541,163641/22906777. Its game loss is the Bell loss divided by 169460^2; the sharp-cutoff conversion is not applicable. The leading Bell entropy constant is 0.7097931163542254.

## Scope

Exact finite checks cover coefficient streams, rule instances, effects, row identities and boundary normalizations. The game evaluators use justified structural zeros rather than dense enumeration of the entire question/answer square. The literal verifier suite exercises 10,076,128 graph/spectator answer cases. The independent review additionally reconstructs row slots from predicate data rather than trusting the serialized slot field.

The proofs establish normal infinite-dimensional attainment and pure-factor extraction. Finite computation alone does not establish those conclusions, asymptotic limits, or a universal entropy lower rate. Theorem D has independent scalar and literal filtered-state checks. The preparation review records that every theorem passed at least one different-family independent check with exact computation; the original Theorem D review retains its own same-family description. This provenance statement does not make the reused measurement builder independent. There is no claim of formal verification, robustness, smallest alphabets, or exhaustive bibliographic priority.

`SHA256SUMS` covers every file in this directory except itself and `__pycache__`. `SOURCE-HASHES.json` records the provenance hashes of the principal original inputs and copied scripts. To validate the current bundle, run `python -B verify_hashes.py` from this directory. Re-running scripts legitimately changes results containing timings; regenerate the manifest only when deliberately updating the bundle.
