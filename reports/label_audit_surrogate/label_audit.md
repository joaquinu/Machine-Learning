# Label audit: EMA-smeared knocks in the alert class

Source: `/tmp/claude-0/-home-user-Machine-Learning/57632400-8df8-5f99-8102-a05465361c65/scratchpad/field_surrogate.csv`. The target is recomputed from raw telemetry, once as KX-VAY-012 delivers it and once with a 3-sample median prefilter on raw severity ahead of the EMA. An **episode** is a run of consecutive rows at or above Y = 0.50; it is **knock-driven** when the raw (unsmoothed) severity clears the control limit for at most one sample while the smoothed score runs for three or more.

## Totals

- Rows: **81,600**
- Alert rows as delivered: **1,258**; with the prefilter: **1,255**
- Alert episodes: **63**, of which **0** are knock-driven
- Alert rows attributable to knocks: **0** (**0.0%** of the alert class)
- Rows whose tier changes under the prefilter: **36**

## Tier distribution, as delivered vs prefiltered

| tier | as delivered | prefiltered |
|---|---|---|
| normal | 98.39% | 98.39% |
| watch | 0.06% | 0.07% |
| alert | 0.16% | 0.16% |
| critical | 1.38% | 1.38% |

## Per group

| group | rows | alert rows | knock-driven alert rows | share |
|---|---|---|---|---|
| `C_CNT` | 20,400 | 0 | 0 | 0.0% |
| `C_LHS` | 20,400 | 0 | 0 | 0.0% |
| `C_RHS` | 20,400 | 1,258 | 0 | 0.0% |
| `R0` | 20,400 | 0 | 0 | 0.0% |
