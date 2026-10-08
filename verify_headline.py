#!/usr/bin/env python3
"""Headline verification for the REACT ESR repository.

Runs the three scenarios at the manuscript settings (M=2000, Mw=10000, seed 12345)
with a reduced permutation count. Identification depends only on P_sel and T, which
do not depend on the number of permutation replicates, so the identified cases are
exactly those reported in the manuscript (Table 4). Runtime: about one minute.
"""
import REACT_ESR_reproducible as react

EXPECTED = {
    "baseline": {("Bayern", "Climate"), ("Baden-Württemberg", "Climate")},
    "national_pool": {("Bayern", "Climate"), ("Baden-Württemberg", "Climate"),
                      ("Brandenburg", "Climate"), ("Hamburg", "Energy")},
    "adjacency_flow": {("Bayern", "Climate"), ("Baden-Württemberg", "Climate"),
                       ("Brandenburg", "Climate")},
}


def main():
    react.N_MONTE_CARLO = 2000
    react.N_DIAGNOSTIC = 10000
    react.RNG_SEED = 12345
    react.RESIDUAL_MODE = "press"
    react.ENERGY_CONSTRUCT = "load_relative"
    react.GRID_NORMALISATION = "area"

    hazard = react.core.load_hazard_panel()
    static = react._configured_static()
    fit = react.estimate_copula(
        hazard,
        ["hot_days_mean", "soil_moist_winter_mean", "p30_days_mean",
         "Rx1day_mm_mean", "Rx1day_mm_max", "Storm_Days_mean", "low_wind_days_mean"],
    )
    assert fit["model"] == "student_t", fit
    assert abs(float(fit["student_t_df"]) - 8.0) < 1e-12, fit

    for scenario, expected in EXPECTED.items():
        r = react.run_scenario(static, hazard, scenario, verbose=False, n_perm=20)
        got = {(s, r["dominant"][s]) for s in r["identified"].index if r["identified"][s]}
        assert got == expected, (scenario, got, expected)
        print(f"{scenario}: PASS ({len(got)} identified) -> {sorted(got)}")
    print("Headline verification: PASS")


if __name__ == "__main__":
    main()
