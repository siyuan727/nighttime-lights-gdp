# Nighttime Lights as a GDP Proxy: Panel Evidence from U.S. Counties

Nighttime satellite imagery has become a common proxy for economic output, especially in places where official accounts are unreliable or unavailable at a fine geographic scale. This project asks how far this proxy can be stretched in a setting where GDP is already accurately documented. U.S. counties have high-quality GDP figures that can serve as a benchmark, which makes them a good place to measure the light-to-output relationship carefully, separate real signal from confounding, and test whether light carries information early enough to be useful.

## Research questions

1. How strongly does nighttime light intensity track county GDP, and does the relationship survive once time-invariant county characteristics are removed through fixed effects?
2. Do changes in light predict GDP growth before the BEA releases official figures, that is, does light have nowcasting value?
3. Does the light-GDP relationship differ between urban and rural counties, and what does that reveal about what the satellite is actually measuring?

## Data

| Source                | Series                          | Coverage     |
| --------------------- | ------------------------------- | ------------ |
| BEA Regional Accounts | County real GDP (CAGDP1)        | 2013 to 2022 |
| VIIRS VCMSLCFG        | Nighttime radiance (nW/cm2/sr)  | 2013 to 2022 |
| Census TIGER          | County shapefiles               | 2022 vintage |
| Census PEP            | County population estimates     | 2013 to 2019 |

Annual VIIRS composites are published by the Earth Observation Group at the Colorado School of Mines (eogdata.mines.edu). This repository ships with a synthetic light generator calibrated to the empirical properties of the real series (levels correlation near 0.85, growth correlation near 0.35), so the pipeline runs end to end without downloading raster files. Set `USE_SYNTHETIC = False` in `src/config.py` to run on the real composites.

## Method

### Baseline and the omitted variable problem

A pooled OLS regression of log GDP on log light gives a coefficient near 1.15. That estimate is not credible as a structural relationship. Wealthier counties are brighter partly because they hold more people, more infrastructure, and a longer history of development, and the regression attributes all of it to light. The coefficient absorbs everything correlated with both brightness and output.

### Two-way fixed effects

Two-way fixed effects strip the confounding out:

```
ln(GDP_it) = b * ln(NTL_it) + a_i + g_t + e_it
```

The county term `a_i` absorbs time-invariant characteristics. The year term `g_t` absorbs national business cycles common to every county. What remains, `b`, is the within-county association between changes in light and changes in output. It falls to about 0.23, which is the more defensible figure.

### Spatial model

State GDP is spatially dependent, since activity in one state spills into its neighbors. A generalized-moments spatial lag model accounts for this:

```
ln(GDP_i) = rho * W * ln(GDP_i) + b * ln(NTL_i) + e_i
```

where `W` is a Queen contiguity weights matrix. The spatial parameter `rho` measures how much a state's output reflects its neighbors' output, independent of its own light.

### Nowcasting

County GDP from the BEA arrives with a long lag, while satellite light is available much sooner. The nowcasting test trains a Ridge regression on light growth and population controls through 2018, then evaluates it out of sample on 2019 through 2022, a window that includes the COVID contraction and the recovery that followed.

## Results

The comparison across specifications is the central result. Pooled OLS overstates the light-GDP elasticity by roughly a factor of five relative to the within-county estimate. The fixed effects coefficient near 0.23 is consistent with Henderson, Storeygard, and Weil (2012), who report elasticities of 0.28 to 0.35 from DMSP-OLS data in a cross-country panel.

The nowcast reproduces the direction of the 2020 contraction and the 2021 rebound but understates the size of both. This is expected. The pandemic broke the usual link between activity and light, as commercial districts went dark while residential areas stayed lit.

Urban counties are far brighter than rural ones, but their estimated elasticity is slightly lower. This points to saturation: in a dense metro, an additional unit of activity adds less incremental light than the same increase in a smaller place with a lower infrastructure base.

## Key figures

### Simulated satellite view vs. GDP

![County map](output/figures/county_map.png)

### NTL-GDP elasticity across specifications

![Coefficient comparison](output/figures/coefficient_comparison.png)

### Nowcast: light-based GDP prediction vs. actual

![Nowcast](output/figures/nowcast_comparison.png)

### NTL vs. GDP, levels and growth rates

![Scatter](output/figures/ntl_gdp_scatter.png)

### Urban vs. rural NTL

![Urban rural](output/figures/urban_rural_ntl.png)

## Limitations

The basic limitation of any light-based measure is that light tracks activity, not output. A county that shifts from manufacturing to finance can see its GDP rise while its light stays flat. The reverse happens too. Oil extraction in the Permian Basin produces intense flaring light, but much of the associated GDP accrues to firms headquartered elsewhere.

The shipped results also run on synthetic light rather than the real composites, so they demonstrate the method rather than establish empirical findings. The qualitative conclusions hold up against the real-data literature, including the direction of the OLS bias, the presence of a nowcasting signal in growth rates, and the urban-rural difference. The specific coefficient values, however, should not be cited as empirical estimates. Running the pipeline on the real VIIRS rasters is the natural next step.

## Replication

```
git clone https://github.com/siyuan727/nighttime-lights-gdp
cd nighttime-lights-gdp
pip install -r requirements.txt
```

Add a free BEA API key (apps.bea.gov) to `src/config.py`, then run:

```
python main.py
```

Figures are written to `output/figures/` and regression tables to `output/tables/`. The first run downloads shapefiles and GDP data automatically and takes a couple of minutes. Later runs read from cache and finish in under thirty seconds.

## References

Chen, X., and Nordhaus, W. (2011). Using luminosity data as a proxy for economic statistics. *PNAS*, 108(21), 8589 to 8594.

Elvidge, C. D., et al. (2017). VIIRS night-time lights. *International Journal of Remote Sensing*, 38(21), 5860 to 5879.

Henderson, J. V., Storeygard, A., and Weil, D. N. (2012). Measuring economic growth from outer space. *American Economic Review*, 102(2), 994 to 1028.
