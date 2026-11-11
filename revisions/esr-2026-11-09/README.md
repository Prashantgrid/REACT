# REACT — manuscript analysis snapshot, 9 November 2026

This directory freezes the processed inputs, analysis code, and archived numerical
outputs used by the revised Energy Strategy Reviews manuscript. It is separate
from the repository-root workflow, which uses a different bootstrap count and
M0 diagnostic specification.

The corrected storage dataset covers all 16 states. `results/r1/r1_copula.csv`
contains 30 year-block bootstrap replicates. The Table A.10 inputs and generated
LaTeX table are included, along with Tables 7, 8 and A.9 and all numerical macros.
See [code/README.md](code/README.md) for commands and data provenance.

Verification on 9 November 2026: the three headline scenarios reproduced their
archived classifications; all numerical macros and table bodies regenerated
byte for byte from the archived results. No new outcome validation or common-year
socio-economic analysis is asserted. The manuscript and reviewer correspondence
are not distributed in this code/data snapshot.
