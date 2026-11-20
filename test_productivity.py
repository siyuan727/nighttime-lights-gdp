# -*- coding: utf-8 -*-
"""
Synthetic-fixture test for src/productivity.py. Run from the repo root:
    python test_productivity.py

Builds a panel where, BY CONSTRUCTION, nighttime light tracks SIZE strongly
(light rises with employment/population) but PRODUCTIVITY only faintly (output
per worker has mostly its own variation). A correct module must:
  - find a high light-GDP (size) correlation and a low light-productivity one,
  - find a small within-county FE light coefficient for productivity,
  - find a tiny out-of-sample gain from adding light on top of size.
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
RNG = np.random.default_rng(7)


def make_panel(n=500, years=range(2016, 2023)):
    years = list(years)
    rows, emp_rows = [], []
    for i in range(n):
        a = RNG.normal(0, 1)          # size factor (big vs small county)
        q = RNG.normal(0, 1)          # productivity factor (own variation)
        state = f"{(i % 48) + 1:02d}"
        fips = f"{state}{i:04d}"[:5]
        ln_area = RNG.normal(0, 0.5)
        for y in years:
            t = y - 2016
            ln_emp = 8.0 + 1.0 * a + 0.02 * t + RNG.normal(0, 0.03)
            ln_pop = 8.3 + 1.0 * a + 0.01 * t + RNG.normal(0, 0.03)
            ln_gdp_pw = 10.6 + 0.15 * q + 0.03 * t + RNG.normal(0, 0.05)   # productivity ~ own
            ln_gdp = ln_gdp_pw + ln_emp                                     # accounting identity
            # brightness: mostly size, a whisper of productivity, plus noise
            ln_ntl = -1.0 + 1.0 * a + 0.04 * q + 0.02 * t + RNG.normal(0, 0.30)
            ntl_mean = max(np.exp(ln_ntl) - 0.01, 1e-4)
            ntl_sum = ntl_mean * np.exp(ln_area)
            rows.append(dict(fips=fips, year=y, state_fips=state,
                             gdp_thousands=np.exp(ln_gdp), ntl_mean=ntl_mean,
                             ntl_sum=ntl_sum, population=np.exp(ln_pop),
                             ln_gdp=ln_gdp, ln_ntl=ln_ntl, ln_pop=ln_pop))
            emp_rows.append(dict(fips=fips, year=y, employment=np.exp(ln_emp)))
    return pd.DataFrame(rows), pd.DataFrame(emp_rows)


base, emp = make_panel()
from src.productivity import build_productivity_panel, run_productivity, plot_productivity

prod = build_productivity_panel(base, emp_df=emp, save=False)
res = run_productivity(prod, save=False)

corr = res["corr"].set_index("outcome")["corr_with_log_light"]
size_r = corr["GDP (size)"]
pw_r = corr["GDP per worker (productivity)"]

reg = res["regressions"]
fe_pw = reg[(reg.outcome == "ln_gdp_pw") & (reg.spec == "Two-way FE (within county)")]["coef"].iloc[0]
fe_gdp = reg[(reg.outcome == "ln_gdp") & (reg.spec == "Two-way FE (within county)")]["coef"].iloc[0]

hr = res["horse_race"].set_index("outcome")
hr_pw = hr.loc["ln_gdp_pw", "delta_r2"]

ok = []
assert size_r > 0.7, f"size corr should be high, got {size_r:.3f}"
ok.append(f"size corr(light,GDP) = {size_r:.3f} (high, as designed)")
assert abs(pw_r) < 0.4, f"productivity corr should be low, got {pw_r:.3f}"
ok.append(f"productivity corr(light,GDP/worker) = {pw_r:.3f} (low, as designed)")
assert abs(fe_pw) < abs(fe_gdp), f"within light coef should be smaller for productivity ({fe_pw:.3f}) than GDP ({fe_gdp:.3f})"
ok.append(f"within-county light coef: GDP={fe_gdp:.3f} vs productivity={fe_pw:.3f} (smaller)")
assert hr_pw < 0.10, f"horse-race delta_r2 for productivity should be tiny, got {hr_pw:.3f}"
ok.append(f"out-of-sample gain from light on productivity: delta_r2={hr_pw:.4f} (tiny)")

plot_productivity(res, save_path="/tmp/_prod_test.png")
ok.append("figure rendered")

print("\n".join("PASS - " + o for o in ok))
print("\nVERDICT LINE: " + res["verdict"])
print("\nALL TESTS PASSED")
