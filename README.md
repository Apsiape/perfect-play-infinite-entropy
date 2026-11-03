# Perfect play in a finite nonlocal game requires infinite entanglement entropy

Seth Douglas and Nidhal Mghirbi. Version 1.0.0, 2 November 2026. DOI: [10.5281/zenodo.23902656](https://doi.org/10.5281/zenodo.23902656).

**This is a preprint. It has not been peer reviewed.**

## Abstract

How much infinity can a finite nonlocal game require? We construct a finite two-player game with classical questions, classical answers and a Boolean winning rule. It has a perfect strategy on a tensor product of separable Hilbert spaces. Every perfect normal spatial strategy contains, under local isometries, the same pure entangled state with infinite entropy. For mixed strategies, both marginal entropies are infinite and the pure state remains a tensor factor. Thus infinite dimension can be strengthened to infinite entropy without losing spatial attainment.

The proof first constructs a rational Bell functional that extracts a state whose Schmidt multiplicities are Catalan numbers. A finite compiler then turns rational linear identities among synchronized projections into winning conditions, using a finite maximally entangled ancilla. Near the optimum, the minimum marginal entropy diverges. Explicit finite-dimensional strategies give an upper bound of order the inverse fifth root of the loss. This exponent is optimal among scalar spectral filters of the canonical state, with the canonical measurements fixed; an unrestricted matching lower bound remains open.

## Contents and reproduction

- `paper/main.pdf`: manuscript.
- `paper/`: LaTeX source, bibliography, plain-text abstract and build driver. Run `python -B build.py` there with pdflatex and BibTeX installed.
- `verification/`: exact finite checks, specified archives, recorded outputs and reproduction commands. Start with its README. Validate exact bytes with `python -B verify_hashes.py` from that directory.

Computational checks support the written proofs; finite evaluations alone do not prove infinite-dimensional attainment or an unrestricted entropy lower rate.

## Related work

- *Perfect play in a binary-scoring nonlocal game requires infinite dimension but only finite energy*: https://doi.org/10.5281/zenodo.23772890
- *Finite games that read the type of a von Neumann algebra*: https://doi.org/10.5281/zenodo.23787124

## Citation and rights

See `CITATION.cff` for citation metadata.
Paper and documentation: CC BY 4.0. Scripts: MIT or Apache 2.0, at your option. See `RIGHTS.md`, the license files and `NOTICE`.
