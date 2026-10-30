# Nighttime Lights and Economic Output

This project started with a simple question that turned out to have a more
complicated answer than I expected: how much do satellite images of nighttime
light actually tell us about a place's GDP? I began it after reading Henderson,
Storeygard, and Weil's 2012 paper on measuring economic growth from space, and I
wanted to see for myself whether the relationship they documented held up once I
pushed on it with real data. The analysis runs on 3,049 U.S. counties from 2016
to 2022, a setting I chose because both GDP and population are measured well
enough here to tell apart, which turns out to be the whole point.

Everything in the results uses real data. County GDP comes from the Bureau of
Economic Analysis, population from the Census Bureau, and nighttime lights from
NASA's Black Marble VIIRS product. Nothing is simulated.

## The question

Nighttime lights are used all the time as a stand-in for GDP, especially in
places where official statistics are weak or slow. The easy version of that
claim, that brighter places are richer, is simple to show and not very
interesting. The harder question, and the one I actually cared about, is whether
light measures economic output or simply measures where people are. Brighter
places are also more crowded places, and more crowded places produce more almost
by definition, so a strong light-GDP correlation could just as easily be a
light-population correlation wearing a disguise. U.S. counties are a good place
to settle the matter, because here both GDP and population are measured carefully
enough that the two can be separated.

## What I found

On its own, light looked like an excellent proxy. The elasticity of GDP with
respect to light was 0.845, and light by itself explained about 71 percent of
the cross-sectional variation in county GDP. That is exactly the kind of result
that makes the method look convincing in the first place.

It did not survive contact with population. County population and GDP are
correlated at 0.95, and the moment I added population to the regression, the
light coefficient fell from 0.845 to 0.149. An out-of-sample test made the
problem concrete: population by itself predicted the level of county GDP with an
R-squared of 0.89, and adding light improved it by only 0.006. For year-to-year
changes, adding light bought another 0.017. Both gains were statistically real
but economically tiny. Put plainly, once you already know how many people live
in a county, satellite light tells you very little more about its GDP.

The one place light held its ground was industry composition. Its elasticity was
noticeably higher in trade-intensive counties and lower in agricultural ones,
which fits the intuition that commercial activity lights up and farmland does
not. So light is not uniformly redundant with population; it is more of a
selective proxy, most reliable where output is visible and commercial.

My reading of all this is that nighttime light is, to a first approximation, a
measure of where people are rather than of what they produce. It is a less
flattering conclusion than the raw correlation suggests, but I think it is the
honest one, and figuring out why the convincing version fell apart was the most
interesting part of the project.

## Data

| Source | Series | Coverage |
| --- | --- | --- |
| BEA Regional Accounts | County real GDP (CAGDP1) and GDP by industry (CAGDP2) | 2016 to 2022 |
| Census Population Estimates | County population | 2016 to 2022 |
| NASA Black Marble | VIIRS VNP46A4 annual radiance | 2016 to 2022 |
| Census TIGER | County boundaries | 2022 vintage |
| World Bank WDI | National GDP and structural covariates | 2013 to 2022 |

Black Marble tiles are pulled through the World Bank's blackmarblepy package and
aggregated to county polygons. The final panel covers 3,049 contiguous counties
and 21,343 county-year observations, after dropping Alaska, Hawaii, the
territories, counties with zero measured radiance, and the handful that fail to
merge after Connecticut reorganized its county equivalents in 2022.

## Method

The core of the analysis is a ladder of specifications that move from naive to
credible. A pooled regression of log GDP on log light gives the headline 0.845.
Adding log population drops it to 0.149, and adding state fixed effects on top of
population leaves it at 0.367. A two-way fixed-effects model, which uses only
within-county variation over time, gives 0.162, and first differences give
0.083. Every step that strips out the cross-sectional variation in size pulls the
light coefficient far below where it started.

Because a coefficient cannot by itself say whether a proxy is useful, the project
also runs the comparison as an out-of-sample forecast. Using cross-validation
that holds out whole counties, it pits population alone against population plus
light at predicting GDP, once in levels across counties and once in year-to-year
changes, and attaches a cluster bootstrap and a Diebold-Mariano test so the gains
come with real uncertainty rather than bare point estimates. On top of that, the
project reports Conley spatial-HAC standard errors for the cross-section, since
neighboring counties are clearly not independent; it interacts light with each
county's industry composition (from BEA CAGDP2) to find where the proxy holds up;
it instruments light with its own lag as a rough check on measurement-error
attenuation; and it runs a ridge nowcast of GDP growth, which confirms that light
carries little short-run signal and misses the 2020 downturn almost entirely.

There is also a cross-country module, built to ask the same question for a
representative panel of developing economies, where the method actually matters.
The pipeline is in place, but I have not yet run it on the full panel.

## Repository layout

```
src/
  config.py               settings, API keys, paths, years (not committed)
  data_acquisition.py     BEA GDP, Census population, shapefiles
  ntl_counties_bm.py      Black Marble aggregation to U.S. counties
  ntl_processing.py       nighttime-lights loading
  panel_construction.py   builds the county and state panels
  descriptives.py         summary-statistics and correlation tables
  benchmarks.py           pooled OLS ladder
  panel_fe.py             two-way and heterogeneous fixed effects
  spatial_se.py           Conley spatial-HAC standard errors
  spatial_model.py        state-level spatial lag model
  horse_race.py           out-of-sample horse race, with bootstrap and DM tests
  industry.py             light-GDP elasticity by sector composition (CAGDP2)
  measurement_error.py    errors-in-variables check on attenuation
  nowcasting.py           ridge nowcast
  visualizations.py       core figures
  us_maps.py              two-panel U.S. lights and GDP figure
  wb_data.py              World Bank GDP and boundaries
  country_real.py         single-country reconstruction (Ghana)
  country_panel.py        developing-economies external-validity panel
main.py                   runs the U.S. pipeline end to end
run_improvements.py       descriptives, industry, Conley, inference, attenuation
run_country_panel.py      the developing-economies analysis
```

## Running it

You will need three free credentials: a BEA API key
(https://apps.bea.gov/API/signup/), a Census API key
(https://api.census.gov/data/key_signup.html), and a NASA Earthdata token
(https://urs.earthdata.nasa.gov, Generate Token). Put the BEA and Census keys in
`src/config.py` and set the Earthdata token in the `BLACKMARBLE_TOKEN`
environment variable.

```
pip install pandas numpy statsmodels scikit-learn linearmodels geopandas wbgapi blackmarblepy

python main.py                 # builds the panel and runs the core analysis
python run_improvements.py     # descriptives, industry, Conley SEs, inference
python run_country_panel.py    # developing-economies analysis (large download)
```

The first run downloads several gigabytes of Black Marble tiles and takes a few
hours; everything caches afterward, so later runs are fast. One warning from
experience: if you ever switch data sources, delete the cached files in
`data/raw/` and `data/clean/` first, because the code will otherwise reload a
stale cache, and you will think you are on fresh data when you are not.

## Limitations and what's next

A few honest caveats. The GDP benchmark is itself a modeled series: BEA county
GDP is built partly from employment and earnings, which are correlated with both
light and population, so the comparison is really one estimate against another,
and the shared inputs may flatter the proxy somewhat. The panel is short, seven
years with the pandemic at its tail, when VIIRS actually reaches back to 2012.
Population barely moves within a county over seven years, so the fixed-effects
models cannot separate light from population as cleanly as the cross-section can.
The attenuation correction leans on a weak instrument and should be replaced with
two independent half-year radiance measures, which is on my list. And this is a
single country: whether light is just as redundant with population in low-income
economies, where census quality is lower and informal activity is larger, is the
question that actually matters, and running the developing-economies panel to
answer it is the clear next step.

## References

Chen, X., and Nordhaus, W. D. (2011). Using luminosity data as a proxy for
economic statistics. Proceedings of the National Academy of Sciences, 108(21),
8589 to 8594.

Donaldson, D., and Storeygard, A. (2016). The view from above: Applications of
satellite data in economics. Journal of Economic Perspectives, 30(4), 171 to 198.

Henderson, J. V., Storeygard, A., and Weil, D. N. (2012). Measuring economic
growth from outer space. American Economic Review, 102(2), 994 to 1028.

Roman, M. O., et al. (2018). NASA's Black Marble nighttime lights product suite.
Remote Sensing of Environment, 210, 113 to 143.
