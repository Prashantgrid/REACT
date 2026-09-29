#!/usr/bin/env python3
"""Fast deterministic smoke test for the REACT ESR repository."""
import REACT_ESR_reproducible as react

EXPECTED = {
    "baseline": {"Bayern", "Baden-Württemberg", "Hamburg"},
    "national_pool": {"Bayern", "Baden-Württemberg", "Hamburg", "Mecklenburg-Vorpommern"},
    "adjacency_flow": {"Bayern", "Baden-Württemberg", "Hamburg", "Mecklenburg-Vorpommern"},
}

def main():
    react.N_MONTE_CARLO = 250
    react.N_DIAGNOSTIC = 1000
    react.N_PERMUTATION = 100
    react.RNG_SEED = 12345
    react.RESIDUAL_MODE = "press"

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
        result = react.run_scenario(static, hazard, scenario, verbose=False)
        got = set(result["calibrated"][result["calibrated"]].index)
        assert got == expected, (scenario, got, expected)
        print(f"{scenario}: PASS ({len(got)} calibrated) -> {sorted(got)}")

    print("Headline verification: PASS")

if __name__ == "__main__":
    main()
