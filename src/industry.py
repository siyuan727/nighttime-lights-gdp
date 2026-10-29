"""
Where the light-GDP proxy breaks: heterogeneity by industry.

The validation literature treats "lights proxy GDP" as one relationship, but it
should not be. A dollar of manufacturing or retail output lights up; a dollar of
farm output, finance, or government does not light up in proportion. So the
light-GDP elasticity should depend on what a county produces, and mapping that
dependence is the paper's novel contribution: it tells a user of satellite GDP
where the proxy can be trusted and where it quietly fails.

This module pulls county GDP by industry (BEA table CAGDP2), builds each county's
sector-composition shares, and estimates the within-county light-GDP elasticity
two ways:

  1. INTERACTIONS. Two-way fixed effects with ln_ntl interacted with (standardized,
     county-mean) sector shares. The interaction coefficients say how the
     elasticity shifts with a one-standard-deviation increase in each sector's
     share. Sector main effects are absorbed by county fixed effects.

  2. DOMINANT-SECTOR SPLIT. Assign each county its largest among a set of broad
     sectors, then estimate the within elasticity separately for each group, so
     the reader sees the elasticity fall or rise across sector types directly.

Hypothesis, stated so it can be rejected: the elasticity is higher where
manufacturing and trade shares are high (bright, measurable activity) and lower
where agriculture, finance, and government shares are high (dim, office-based, or
non-market activity).

Data: BEA CAGDP2 (GDP in current dollars by industry). Shares are ratios within a
county-year, so current vs chained dollars does not matter. Requires the same BEA
API key as CAGDP1.
"""

import logging

import numpy as np
import pandas as pd

from src.config import OUTPUT_TAB

log = logging.getLogger(__name__)

# Canonical broad sectors and the Description keywords that identify their BEA
# CAGDP2 line. Keyword matching is used so the code is robust to BEA renumbering
# its line codes; the first line whose description contains a keyword wins.
SECTOR_KEYWORDS = {
    "agriculture":   ["agriculture, forestry", "farm"],
    "mining":        ["mining, quarrying", "oil and gas"],
    "manufacturing": ["manufacturing"],
    "construction":  ["construction"],
    "trade":         ["retail trade"],          # retail; wholesale added below
    "wholesale":     ["wholesale trade"],
    "transport":     ["transportation and warehousing"],
    "information":   ["information"],
    "finance":       ["finance and insurance", "finance, insurance"],
    "realestate":    ["real estate"],
    "professional":  ["professional, scientific", "professional and business"],
    "government":    ["government and government enterprises", "government"],
}
TOTAL_KEYWORDS = ["all industry total"]

# The shares used as moderators in the interaction model and as groups in the
# split. Kept short so each group has enough counties and the table is readable.
KEY_SHARES = ["agriculture", "manufacturing", "trade", "finance",
              "government", "mining"]


# ── BEA CAGDP2 acquisition ───────────────────────────────────

def _bea_linecodes(table="CAGDP2"):
    """Discover {line_code: description} for a Regional table via the BEA API."""
    import requests
    from src.config import BEA_API_KEY
    from src.data_acquisition import BEA_ENDPOINT
    params = {
        "UserID": BEA_API_KEY, "method": "GetParameterValuesFiltered",
        "datasetname": "Regional", "TargetParameter": "LineCode",
        "TableName": table, "ResultFormat": "json",
    }
    r = requests.get(BEA_ENDPOINT, params=params, timeout=120)
    r.raise_for_status()
    vals = r.json()["BEAAPI"]["Results"]["ParamValue"]
    out = {}
    for v in vals:
        key = str(v.get("Key") or v.get("LineCode") or "").strip()
        desc = str(v.get("Desc") or v.get("Description") or "").strip()
        if key:
            out[key] = desc
    log.info(f"BEA {table}: discovered {len(out)} line codes.")
    return out


def _match_linecode(linecodes, keywords):
    """First (code, desc) whose description contains any keyword, else None."""
    for kw in keywords:
        for code, desc in linecodes.items():
            if kw in desc.lower():
                return code, desc
    return None, None


def fetch_bea_industry_gdp(use_cache=True):
    """
    County GDP by industry (BEA CAGDP2) for the study years, long format
    [fips, year, sector, gdp_ind] plus the all-industry total as sector='total'.
    Resolves line codes by description so it survives BEA renumbering. Cached.
    """
    import requests
    from src.config import (BEA_API_KEY, YEARS, EXCLUDE_FIPS_PREFIX,
                            START_YEAR, END_YEAR, DATA_RAW)
    from src.data_acquisition import BEA_ENDPOINT

    cache = DATA_RAW / "bea_county_gdp_by_industry.csv"
    if use_cache and cache.exists():
        log.info("Loading cached BEA industry GDP...")
        return pd.read_csv(cache, dtype={"fips": str})

    linecodes = _bea_linecodes("CAGDP2")
    wanted = {}
    tot_code, tot_desc = _match_linecode(linecodes, TOTAL_KEYWORDS)
    if tot_code:
        wanted["total"] = (tot_code, tot_desc)
    for sector, kws in SECTOR_KEYWORDS.items():
        code, desc = _match_linecode(linecodes, kws)
        if code:
            wanted[sector] = (code, desc)
    log.info("BEA CAGDP2 sector->line map: "
             + ", ".join(f"{s}={c}" for s, (c, _) in wanted.items()))

    rows = []
    for sector, (code, _desc) in wanted.items():
        params = {
            "UserID": BEA_API_KEY, "method": "GetData",
            "datasetname": "Regional", "TableName": "CAGDP2",
            "LineCode": code, "GeoFips": "COUNTY",
            "Year": ",".join(str(y) for y in YEARS), "ResultFormat": "json",
        }
        try:
            r = requests.get(BEA_ENDPOINT, params=params, timeout=180)
            r.raise_for_status()
            data = r.json()["BEAAPI"]["Results"]["Data"]
        except Exception as e:
            log.warning(f"BEA CAGDP2 line {code} ({sector}) failed: {e}")
            continue
        for rec in data:
            val = rec.get("DataValue", "")
            if val in ("(NA)", "(D)", "(L)", "", "--"):
                continue
            try:
                gdp = float(str(val).replace(",", ""))
            except ValueError:
                continue
            rows.append({"fips": str(rec.get("GeoFips", "")).zfill(5),
                         "year": int(rec.get("TimePeriod", 0)),
                         "sector": sector, "gdp_ind": gdp})

    df = pd.DataFrame(rows)
    df = df[~df["fips"].str[:2].isin(EXCLUDE_FIPS_PREFIX)]
    df = df[df["fips"].str.len() == 5]
    df = df[df["year"].between(START_YEAR, END_YEAR)]
    df.to_csv(cache, index=False)
    log.info(f"BEA industry GDP: {len(df)} rows, {df['fips'].nunique()} counties, "
             f"{df['sector'].nunique()} sectors.")
    return df


# ── Sector shares ────────────────────────────────────────────

def build_sector_shares(industry_df):
    """
    County sector shares = sector GDP / all-industry total, averaged over the
    sample years so each county has one time-invariant composition vector. The
    'trade' share combines retail and wholesale. Returns [fips, share_<sector>].
    """
    df = industry_df.copy()
    wide = df.pivot_table(index=["fips", "year"], columns="sector",
                          values="gdp_ind", aggfunc="sum")
    if "total" not in wide.columns:
        wide["total"] = wide.sum(axis=1)
    # Combine wholesale into trade if present.
    if "wholesale" in wide.columns:
        wide["trade"] = wide.get("trade", 0).fillna(0) + wide["wholesale"].fillna(0)
    denom = wide["total"].replace(0, np.nan)
    shares = pd.DataFrame(index=wide.index)
    for s in KEY_SHARES:
        if s in wide.columns:
            shares[f"share_{s}"] = (wide[s] / denom).clip(lower=0)
    shares = shares.reset_index()
    # County-mean composition (time-invariant moderator).
    county_shares = shares.groupby("fips").mean(numeric_only=True).drop(
        columns=["year"], errors="ignore").reset_index()
    log.info(f"Sector shares for {county_shares['fips'].nunique()} counties: "
             + ", ".join([c for c in county_shares.columns if c != "fips"]))
    return county_shares


# ── Estimation ───────────────────────────────────────────────

def _prep(df):
    d = df.copy()
    d["fips"] = d["fips"].astype(str)
    d["year"] = pd.to_datetime(d["year"], format="%Y")
    return d.dropna(subset=["ln_gdp", "ln_ntl"]).set_index(["fips", "year"])


def _full_rank_subset(frame, cols, always_keep):
    """
    Return the largest prefix-preserving subset of `cols` that keeps the design
    at full column rank, always retaining `always_keep`. Guards against the case
    where the chosen sector shares (nearly) partition a county, which makes the
    interactions collinear with ln_ntl. In real CAGDP2 data the key sectors are
    a subset and this drops nothing; it only triggers on degenerate inputs.
    """
    keep = list(always_keep)
    for c in cols:
        if c in keep:
            continue
        trial = keep + [c]
        M = frame[trial].to_numpy(dtype=float)
        if np.linalg.matrix_rank(M) == len(trial):
            keep.append(c)
        else:
            log.warning(f"Dropping collinear interaction term: {c}")
    return keep


def elasticity_interactions(panel, county_shares, save=True):
    """
    Two-way FE of ln_gdp on ln_ntl and ln_ntl x (standardized county sector
    share), for the key sectors. The interaction coefficient is how much the
    light elasticity moves per one-SD increase in that sector's share.
    """
    from linearmodels.panel import PanelOLS

    m = panel.merge(county_shares, on="fips", how="inner")
    share_cols = [c for c in county_shares.columns if c.startswith("share_")]
    # Standardize shares so interaction coefficients are per-SD and comparable.
    for c in share_cols:
        sd = m[c].std()
        m[c + "_z"] = (m[c] - m[c].mean()) / sd if sd and sd > 0 else 0.0

    d = _prep(m)
    regressors = ["ln_ntl"]
    for c in share_cols:
        name = f"ntlX_{c.replace('share_', '')}"
        d[name] = d["ln_ntl"] * d[c + "_z"]
        regressors.append(name)

    # Drop any interaction that is collinear with the rest (e.g. if the chosen
    # shares happen to sum to a constant), keeping ln_ntl itself.
    regressors = _full_rank_subset(d, regressors, always_keep=["ln_ntl"])

    res = PanelOLS(d["ln_gdp"], d[regressors],
                   entity_effects=True, time_effects=True,
                   drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for var in res.params.index:
        rows.append({"term": var,
                     "coef": round(float(res.params[var]), 4),
                     "se": round(float(res.std_errors[var]), 4),
                     "pval": round(float(res.pvalues[var]), 4)})
    table = pd.DataFrame(rows)
    log.info("\nLight-GDP elasticity, sector-share interactions "
             "(per 1 SD of share):\n" + table.to_string(index=False))
    if save:
        table.to_csv(OUTPUT_TAB / "industry_interactions.csv", index=False)
    return {"result": res, "table": table}


def elasticity_by_dominant_sector(panel, county_shares, min_counties=30, save=True):
    """
    Assign each county its largest key sector share, then estimate the within
    (two-way FE) light-GDP elasticity separately for each dominant-sector group.
    Groups with fewer than min_counties are skipped.
    """
    from linearmodels.panel import PanelOLS

    share_cols = [c for c in county_shares.columns if c.startswith("share_")]
    cs = county_shares.copy()
    cs["dominant"] = cs[share_cols].idxmax(axis=1).str.replace("share_", "", regex=False)
    m = panel.merge(cs[["fips", "dominant"]], on="fips", how="inner")

    rows = []
    for sector, sub in m.groupby("dominant"):
        if sub["fips"].nunique() < min_counties:
            log.info(f"  {sector}: only {sub['fips'].nunique()} counties, skipped.")
            continue
        d = _prep(sub)
        try:
            res = PanelOLS(d["ln_gdp"], d[["ln_ntl"]],
                           entity_effects=True, time_effects=True,
                           drop_absorbed=True).fit(cov_type="clustered",
                                                   cluster_entity=True)
        except Exception as e:
            log.warning(f"  {sector}: estimation failed ({e}).")
            continue
        rows.append({"dominant_sector": sector,
                     "elasticity": round(float(res.params["ln_ntl"]), 4),
                     "se": round(float(res.std_errors["ln_ntl"]), 4),
                     "counties": int(sub["fips"].nunique()),
                     "obs": int(res.nobs)})
        log.info(f"  {sector}: beta = {res.params['ln_ntl']:.4f} "
                 f"(SE {res.std_errors['ln_ntl']:.4f}), "
                 f"{sub['fips'].nunique()} counties")

    table = pd.DataFrame(rows).sort_values("elasticity", ascending=False)
    log.info("\nWithin light-GDP elasticity by dominant sector:\n"
             + table.to_string(index=False))
    if save:
        table.to_csv(OUTPUT_TAB / "industry_by_dominant_sector.csv", index=False)
    return table


def plot_elasticity_by_sector(table, save_path=None):
    """Bar chart of the within elasticity by dominant sector, with SE whiskers."""
    import matplotlib.pyplot as plt
    from src.config import OUTPUT_FIG, NAVY

    t = table.sort_values("elasticity")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(t["dominant_sector"], t["elasticity"], xerr=1.96 * t["se"],
            color=NAVY, alpha=0.85, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Within-county light-GDP elasticity")
    ax.set_title("Where the light-GDP proxy works and where it breaks\n"
                 "Two-way FE elasticity by county's dominant sector")
    fig.tight_layout()
    out = save_path or (OUTPUT_FIG / "industry_elasticity.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[viz] saved {out}")
    return out


def run_industry_analysis(panel, save=True):
    """End to end: pull CAGDP2, build shares, run both specifications, plot."""
    industry_df = fetch_bea_industry_gdp()
    county_shares = build_sector_shares(industry_df)
    inter = elasticity_interactions(panel, county_shares, save=save)
    by_sector = elasticity_by_dominant_sector(panel, county_shares, save=save)
    if save and not by_sector.empty:
        plot_elasticity_by_sector(by_sector)
    return {"shares": county_shares, "interactions": inter,
            "by_sector": by_sector}
