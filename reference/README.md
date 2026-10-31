# Main manuscript reference audit

`classification_audit.csv` is the full 48-case audit for the archived **1 October
2025** MaStR battery panel, using its explicitly screened energy column for all
16 states. M=2000, diagnostic draws=10000, permutation audit R=5000, seed=12345.

The rule P_sel>=0.70 and T>0 identifies BY and BW in all three scenarios, BB and
SN under National Pool and Adjacency Flow: 10 cases. Sachsen is close to the
threshold and its National Pool classification varies across seeds. HH remains
below the main threshold. Permutation values are an audit, not part of this rule.
