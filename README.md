# Perfect play in a finite nonlocal game requires infinite entanglement entropy

Seth Douglas and Nidhal Mghirbi. Version 1.1.0, 3 November 2026. Concept DOI (all versions): [10.5281/zenodo.23902655](https://doi.org/10.5281/zenodo.23902655). Version 1.0.0: [10.5281/zenodo.23902656](https://doi.org/10.5281/zenodo.23902656). Version 1.1.0: [10.5281/zenodo.23932017](https://doi.org/10.5281/zenodo.23932017).

The minimum marginal entropy at loss epsilon is Theta(log(1/epsilon)) for all normal spatial strategies. The full decoding and rounding proofs are in the paper. The radial inverse-fifth-root bound remains valid and optimal within scalar spectral filters.

## Abstract

How much infinity can a finite nonlocal game require? We construct a finite two-player game with classical questions, classical answers and a Boolean winning rule. It has a perfect strategy on a tensor product of separable Hilbert spaces. Every perfect normal spatial strategy contains, under local isometries, the same pure entangled state with infinite entropy. For mixed strategies, both marginal entropies are infinite and the pure state remains a tensor factor. Thus infinite dimension can be strengthened to infinite entropy without losing spatial attainment.

The proof constructs a rational Bell functional whose perfect strategies contain a state with Catalan Schmidt multiplicities, and compiles it into Boolean tests. For either construction, the minimum marginal entropy at loss $\varepsilon$ is $\Theta(\log(1/\varepsilon))$. The lower bound holds for every normal spatial strategy, including mixed states and POVMs. It follows from polynomial decoding on the actual state and a finite-strip entropy estimate. Height and size cutoffs give the matching upper order. The sharp leading constant remains open.

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
