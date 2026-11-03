Perfect play in a finite nonlocal game requires infinite entanglement entropy
Seth Douglas and Nidhal Mghirbi
Version 1.1.0, November 2026

main.pdf is the manuscript. main.tex, sections/, references.bib and abstract.txt are the source.
Build: python -B build.py
Requires Python 3, pdflatex, BibTeX and the standard LaTeX packages named in main.tex.
The driver fails on final TeX warnings, undefined references/citations, or box warnings.
Each invocation uses a fresh empty build/clean-* directory. Shell escape is disabled on every LaTeX pass. Final warnings, citations and box diagnostics are checked.
Exclude build/ when preparing a public source upload; it is local build output.
The verification bundle is the sibling verification/ directory.
