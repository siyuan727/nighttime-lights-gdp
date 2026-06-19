import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import logging
from scipy.ndimage import gaussian_filter
from src.config import (OUTPUT_FIG, PLOT_STYLE, FIGURE_DPI,
                        NAVY, MAROON, FOREST, AMBER, GREY)

log = logging.getLogger(__name__)
plt.style.use(PLOT_STYLE)


def plot_ntl_gdp_scatter(panel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: Levels ───────────────────────────────────────
    # Drop any inf or nan before plotting
    levels = panel[["ln_ntl", "ln_gdp"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    axes[0].scatter(levels["ln_ntl"], levels["ln_gdp"],
                    alpha=0.15, s=5, color=NAVY)

    # Only fit trend line if enough valid points
    if len(levels) > 10:
        try:
            m, b = np.polyfit(levels["ln_ntl"], levels["ln_gdp"], 1)
            x_range = np.linspace(levels["ln_ntl"].min(),
                                  levels["ln_ntl"].max(), 100)
            axes[0].plot(x_range, m * x_range + b,
                         color=MAROON, linewidth=2)
        except np.linalg.LinAlgError:
            pass   # skip trend line if fitting fails

    corr = levels.corr().iloc[0, 1]
    axes[0].set_xlabel("ln(NTL Mean Radiance)", fontsize=11)
    axes[0].set_ylabel("ln(GDP, thousands)", fontsize=11)
    axes[0].set_title(
        f"NTL vs GDP (Levels)\nCorrelation = {corr:.3f}", fontsize=12
    )

    # ── Panel 2: Growth rates ─────────────────────────────────
    diff = panel[["ntl_growth", "gdp_growth", "fips"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Clip to 1st–99th percentile to remove extreme outliers
    # that cause SVD non-convergence in polyfit
    ntl_lo, ntl_hi = diff["ntl_growth"].quantile([0.01, 0.99])
    gdp_lo, gdp_hi = diff["gdp_growth"].quantile([0.01, 0.99])
    diff_clipped = diff[
        diff["ntl_growth"].between(ntl_lo, ntl_hi) &
        diff["gdp_growth"].between(gdp_lo, gdp_hi)
    ]

    axes[1].scatter(diff_clipped["ntl_growth"],
                    diff_clipped["gdp_growth"],
                    alpha=0.15, s=5, color=FOREST)

    if len(diff_clipped) > 10:
        try:
            m2, b2 = np.polyfit(
                diff_clipped["ntl_growth"],
                diff_clipped["gdp_growth"], 1
            )
            x2 = np.linspace(ntl_lo, ntl_hi, 100)
            axes[1].plot(x2, m2 * x2 + b2,
                         color=MAROON, linewidth=2)
        except np.linalg.LinAlgError:
            pass

    corr2 = diff_clipped[["ntl_growth", "gdp_growth"]].corr().iloc[0, 1]
    axes[1].set_xlabel("NTL Growth (log diff)", fontsize=11)
    axes[1].set_ylabel("GDP Growth (log diff)", fontsize=11)
    axes[1].set_title(
        f"NTL vs GDP (Growth Rates)\nCorrelation = {corr2:.3f}",
        fontsize=12
    )
    axes[1].set_xlim(ntl_lo, ntl_hi)
    axes[1].set_ylim(gdp_lo, gdp_hi)

    plt.tight_layout()
    _save("ntl_gdp_scatter.png")

def plot_coefficient_comparison(ols_results, fe_results, hetero_results=None):
    models, coefs, ses, labels = [], [], [], []

    ols_map = {"ols_bivariate": "OLS (bivariate)",
               "ols_pop":       "OLS + Population",
               "ols_state_fe":  "OLS + State FE",
               "ols_fd":        "OLS First Diff"}
    for name, label in ols_map.items():
        if name not in ols_results:
            continue
        res = ols_results[name]
        var = "ntl_growth" if name == "ols_fd" else "ln_ntl"
        if var in res.params.index:
            models.append(label)
            coefs.append(float(res.params[var]))
            ses.append(float(res.bse[var]))
            labels.append("OLS")

    fe_map = {"fe_entity":     "County FE",
              "fe_twoway": "Two-Way FE (main spec)",
              "fe_twoway_pop": "Two-Way FE + Pop"}
    for name, label in fe_map.items():
        if name not in fe_results:
            continue
        res = fe_results[name]
        if "ln_ntl" in res.params.index:
            models.append(label)
            coefs.append(float(res.params["ln_ntl"]))
            ses.append(float(res.std_errors["ln_ntl"]))
            labels.append("Panel FE")

    if hetero_results:
        for name, res in hetero_results.items():
            if "ln_ntl" in res.params.index:
                models.append(f"Two-Way FE ({name})")
                coefs.append(float(res.params["ln_ntl"]))
                ses.append(float(res.std_errors["ln_ntl"]))
                labels.append("Subgroup")

    colors = {
        "OLS":      GREY,
        "Panel FE": NAVY,
        "Subgroup": FOREST,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(models))
    bar_colors = [colors[l] for l in labels]

    ax.barh(y_pos, coefs,
            xerr=[1.96 * s for s in ses],
            color=bar_colors, alpha=0.85, height=0.6,
            error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("NTL-GDP Elasticity (β)", fontsize=11)
    ax.set_title(
        "NTL-GDP Elasticity Across Specifications\n"
        "Grey = OLS  |  Navy = Panel FE  |  95% CI shown",
        fontsize=12)
    plt.tight_layout()
    _save("coefficient_comparison.png")


def plot_nowcast(nowcast_results):
    if not nowcast_results or "predictions" not in nowcast_results:
        log.warning("No nowcast results to plot.")
        return
    preds = nowcast_results["predictions"]
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly = preds.groupby("year")[["gdp_growth","gdp_growth_pred"]].mean()
    ax.plot(yearly.index, yearly["gdp_growth"],
            color=NAVY, linewidth=2, marker="o", label="Actual GDP Growth")
    ax.plot(yearly.index, yearly["gdp_growth_pred"],
            color=MAROON, linewidth=2, marker="s",
            linestyle="--", label="NTL Nowcast")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean County GDP Growth (log diff)", fontsize=11)
    ax.set_title(
        "NTL-Based GDP Nowcast vs Actual\n"
        f"Test period: {preds['year'].min()}–{preds['year'].max()}",
        fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save("nowcast_comparison.png")


def plot_urban_rural_ntl(panel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_map = {1: NAVY, 0: FOREST}
    labels_map  = {1: "Urban (pop > 100k)", 0: "Rural (pop ≤ 100k)"}

    for val in [1, 0]:
        sub = panel[panel["urban"] == val]
        axes[0].hist(sub["ln_ntl"], bins=50, alpha=0.6,
                     color=colors_map[val], label=labels_map[val], density=True)

    axes[0].set_xlabel("ln(NTL Mean Radiance)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("NTL Distribution by County Type", fontsize=12)
    axes[0].legend()

    yearly = (panel.groupby(["year","urban"])
              [["ntl_mean","gdp_thousands"]].mean().reset_index())
    for val in [1, 0]:
        sub = yearly[yearly["urban"] == val]
        axes[1].plot(sub["year"], sub["ntl_mean"],
                     color=colors_map[val], linewidth=2,
                     marker="o", label=labels_map[val])

    axes[1].set_xlabel("Year", fontsize=11)
    axes[1].set_ylabel("Mean NTL Radiance", fontsize=11)
    axes[1].set_title("NTL Trend by County Type", fontsize=12)
    axes[1].legend()
    plt.tight_layout()
    _save("urban_rural_ntl.png")


def plot_county_map(county_panel, county_gdf):
    try:
        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable
        from scipy.ndimage import gaussian_filter
        import numpy as np
    except ImportError as e:
        log.warning(f"Missing library: {e}")
        return

    latest_year = county_panel["year"].max()
    latest = county_panel[county_panel["year"] == latest_year].copy()

    gdf = county_gdf.copy()
    gdf["fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]

    from src.config import EXCLUDE_FIPS_PREFIX
    gdf = gdf[~gdf["STATEFP"].isin(EXCLUDE_FIPS_PREFIX)]
    gdf = gdf.to_crs("EPSG:5070")

    merged = gdf.merge(
        latest[["fips","ln_ntl","ln_gdp","ntl_mean","gdp_thousands"]],
        on="fips", how="left"
    )

    us_outline = merged.dissolve()

    xmin, ymin, xmax, ymax = merged.total_bounds
    x_pad = (xmax - xmin) * 0.01
    y_pad = (ymax - ymin) * 0.01

    # Map width/height ratio — critical for alignment
    map_w  = (xmax - xmin) + 2 * x_pad
    map_h  = (ymax - ymin) + 2 * y_pad
    aspect = map_w / map_h   # ~2.1 for contiguous US in Albers

    # ── Satellite colormap ────────────────────────────────────
    sat_cmap = LinearSegmentedColormap.from_list(
        "satellite_ntl",
        [(0.00, "#000000"),
         (0.08, "#0d0d0d"),
         (0.20, "#1a1408"),
         (0.35, "#3d2e10"),
         (0.50, "#7a5e28"),
         (0.65, "#c4a050"),
         (0.78, "#e8cc88"),
         (0.88, "#f5e8c0"),
         (0.95, "#fdf5e0"),
         (1.00, "#ffffff")],
        N=1024
    )

    ntl_vals = merged["ntl_mean"].fillna(0)
    vmin_sat  = ntl_vals.quantile(0.01)
    vmax_sat  = ntl_vals.quantile(0.999)
    norm_sat  = mcolors.PowerNorm(
        gamma=0.42, vmin=vmin_sat, vmax=vmax_sat
    )

    ntl_min  = merged["ln_ntl"].quantile(0.02)
    ntl_max  = merged["ln_ntl"].quantile(0.98)
    norm_ntl = mcolors.Normalize(vmin=ntl_min, vmax=ntl_max)

    gdp_min  = merged["ln_gdp"].quantile(0.02)
    gdp_max  = merged["ln_gdp"].quantile(0.98)
    norm_gdp = mcolors.Normalize(vmin=gdp_min, vmax=gdp_max)

    # ── Main figure ───────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 8), facecolor="black"
    )
    fig.subplots_adjust(
        left=0.01, right=0.99,
        top=0.88, bottom=0.10,
        wspace=0.04
    )
    ax1, ax2, ax3 = axes

    def set_extent(ax):
        ax.set_xlim(xmin - x_pad, xmax + x_pad)
        ax.set_ylim(ymin - y_pad, ymax + y_pad)
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_aspect("equal")

    def add_white_cbar(fig, ax, sm, label):
        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal",
            fraction=0.030, pad=0.02, shrink=0.80
        )
        cbar.set_label(label, color="white", fontsize=9, labelpad=4)
        cbar.ax.xaxis.set_tick_params(color="white", labelsize=8)
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")
        cbar.outline.set_edgecolor("white")
        cbar.ax.set_facecolor("black")
        return cbar

    # ══════════════════════════════════════════════════════════
    # PANEL 1 — Satellite view
    # Render to hidden figure whose dimensions exactly match
    # the map aspect ratio so pixels align with coordinates
    # ══════════════════════════════════════════════════════════
    ax1.set_facecolor("black")

    # Hidden figure sized to exact map aspect ratio
    render_h   = 8.0
    render_w   = render_h * aspect
    fig_tmp, ax_tmp = plt.subplots(
        figsize=(render_w, render_h),
        facecolor="black"
    )
    ax_tmp.set_facecolor("black")

    merged.plot(
        column="ntl_mean",
        ax=ax_tmp,
        cmap=sat_cmap,
        norm=norm_sat,
        missing_kwds={"color": "black"},
        linewidth=0,
        edgecolor="none",
    )

    # Set EXACTLY the same extent as the main panel
    ax_tmp.set_xlim(xmin - x_pad, xmax + x_pad)
    ax_tmp.set_ylim(ymin - y_pad, ymax + y_pad)
    ax_tmp.set_aspect("equal")
    ax_tmp.axis("off")

    # Remove ALL padding from hidden figure
    fig_tmp.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_tmp.canvas.draw()

    # Extract pixel buffer
    buf  = fig_tmp.canvas.buffer_rgba()
    w, h = fig_tmp.canvas.get_width_height()
    arr  = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    rgb  = arr[:, :, :3].astype(np.float32) / 255.0
    plt.close(fig_tmp)

    # Perceptual luminance
    lum = (0.2126 * rgb[:,:,0]
         + 0.7152 * rgb[:,:,1]
         + 0.0722 * rgb[:,:,2])

    # ── Three-pass glow ───────────────────────────────────────
    # Work at 3x resolution then downscale — removes pixelation
    # and makes county edges completely invisible after blurring
    from scipy.ndimage import zoom
    lum_up = zoom(lum, 3, order=1)      # upscale 3x (bilinear)

    # ── Four-pass glow at upscaled resolution ─────────────────
    # Each pass dissolves a different spatial scale of hard edges

    # Pass 1: ultra-wide atmospheric haze
    # Dissolves large county borders in the west where counties are huge
    g_atmos  = gaussian_filter(lum_up, sigma=55)

    # Pass 2: wide city glow — bleeds light well beyond county bounds
    g_wide   = gaussian_filter(lum_up, sigma=22)

    # Pass 3: medium bloom — fills gaps between neighbouring counties
    g_bloom  = gaussian_filter(lum_up, sigma=9)

    # Pass 4: tight detail — keeps brightest pixels visible but softened
    g_detail = gaussian_filter(lum_up, sigma=2.5)

    # Blend: atmospheric haze is faint background,
    # detail pass anchors bright city cores
    blended_up = np.clip(
        g_atmos  * 0.20
        + g_wide   * 0.35
        + g_bloom  * 0.50
        + g_detail * 0.70,
        0, 1
    )

    # Hard background stays black — no noise in empty ocean/border
    blended_up[lum_up < 0.008] = 0.0

    # Downscale back to original resolution
    blended = zoom(blended_up, 1/3, order=1)
    blended = np.clip(blended, 0, 1)

    # ── Build RGB glow image ──────────────────────────────────
    glow_rgba = sat_cmap(blended)
    glow_rgb  = glow_rgba[:, :, :3].copy()

    # Warm amber atmospheric tint on lit areas
    # Mimics sodium/LED streetlight scatter in atmosphere
    glow_mask = np.clip(
        zoom(gaussian_filter(lum_up, sigma=18), 1/3, order=1),
        0, 1
    )
    glow_rgb[:,:,0] = np.clip(glow_rgb[:,:,0] + glow_mask * 0.28, 0, 1)
    glow_rgb[:,:,1] = np.clip(glow_rgb[:,:,1] + glow_mask * 0.14, 0, 1)
    glow_rgb[:,:,2] = np.clip(glow_rgb[:,:,2] + glow_mask * 0.00, 0, 1)

    ax1.imshow(
        glow_rgb,
        origin="upper",
        extent=[xmin - x_pad, xmax + x_pad,
                ymin - y_pad, ymax + y_pad],
        interpolation="bilinear",
        aspect="auto",
    )

    us_outline.boundary.plot(
        ax=ax1, linewidth=1.2,
        color="white", alpha=0.85
    )
    set_extent(ax1)
    ax1.set_title(
        f"Simulated Satellite View -- {latest_year}\n"
        "Nighttime Light Intensity (VIIRS-style)",
        color="white", fontsize=11, pad=10
    )
    sm1 = ScalarMappable(cmap=sat_cmap, norm=norm_sat)
    sm1.set_array([])
    add_white_cbar(fig, ax1, sm1, "NTL Radiance (nW/cm2/sr)")

    # ══════════════════════════════════════════════════════════
    # PANEL 2 — ln(NTL) choropleth
    # ══════════════════════════════════════════════════════════
    merged.plot(
        column="ln_ntl", ax=ax2,
        cmap="inferno", norm=norm_ntl,
        missing_kwds={"color": "#333333"},
        linewidth=0.05, edgecolor="#111111",
    )
    us_outline.boundary.plot(
        ax=ax2, linewidth=1.2,
        color="white", alpha=0.9
    )
    set_extent(ax2)
    ax2.set_title(
        f"ln(NTL Radiance) by County -- {latest_year}\n"
        "Brighter = higher nighttime light intensity",
        color="white", fontsize=11, pad=10
    )
    sm2 = ScalarMappable(
        cmap=plt.get_cmap("inferno"), norm=norm_ntl
    )
    sm2.set_array([])
    add_white_cbar(fig, ax2, sm2, "ln(NTL Radiance)")

    # ══════════════════════════════════════════════════════════
    # PANEL 3 — ln(GDP) choropleth
    # ══════════════════════════════════════════════════════════
    merged.plot(
        column="ln_gdp", ax=ax3,
        cmap="Blues", norm=norm_gdp,
        missing_kwds={"color": "#333333"},
        linewidth=0.05, edgecolor="#111111",
    )
    us_outline.boundary.plot(
        ax=ax3, linewidth=1.2,
        color="white", alpha=0.9
    )
    set_extent(ax3)
    ax3.set_title(
        f"Real GDP by County -- {latest_year}\n"
        "Darker = higher economic output",
        color="white", fontsize=11, pad=10
    )
    sm3 = ScalarMappable(
        cmap=plt.get_cmap("Blues"), norm=norm_gdp
    )
    sm3.set_array([])
    add_white_cbar(fig, ax3, sm3, "ln(Real GDP, thousands $)")

    # ── Master title ──────────────────────────────────────────
    fig.suptitle(
        f"Nighttime Lights and Economic Activity -- Contiguous U.S. Counties\n"
        f"(N = {merged['ln_ntl'].notna().sum():,} counties  |  "
        f"Synthetic NTL -- real VIIRS at eogdata.mines.edu)",
        color="white", fontsize=13, y=0.97
    )

    plt.savefig(
        OUTPUT_FIG / "county_map.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none"
    )
    plt.close()
    log.info("Saved: county_map.png")

def _save(filename):
    path = OUTPUT_FIG / filename
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")