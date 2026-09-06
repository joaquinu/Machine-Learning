# Multi-seed surrogate campaign

5 modelled failure modes x 6 seeds x 2 feature variants x 3 scoring variants = 180 scored runs. **All data is modelled**; this measures how much the surrogate's lead-time claim depends on the seed and the fault mode, not how a physical bearing behaves.

## Detection rate and lead time by failure mode

| fault mode | variant | detected | driver @ first | driver @ terminal | terminal lead (h) | % of life | pre-onset false alarm (h) |
|---|---|---|---|---|---|---|---|
| `outer_race` | single_bin | spec | 6/6 | 6/6 | 12.08 (10.17-12.17) | 31% |
| `outer_race` | single_bin | prefilter | 6/6 | 6/6 | 12.00 (10.17-12.17) | 30% |
| `outer_race` | single_bin | prefilter+specattr | 6/6 | 6/6 | 12.00 (10.17-12.17) | 30% |
| `outer_race` | sideband | spec | 6/6 | 6/6 | 12.00 (10.33-12.17) | 30% |
| `outer_race` | sideband | prefilter | 6/6 | 6/6 | 11.83 (10.17-12.17) | 30% |
| `outer_race` | sideband | prefilter+specattr | 6/6 | 6/6 | 11.83 (10.17-12.17) | 30% |
| `inner_race` | single_bin | spec | 6/6 | 2/6 | 10.42 (6.50-12.00) | 26% |
| `inner_race` | single_bin | prefilter | 6/6 | 3/6 | 10.25 (6.33-12.00) | 26% |
| `inner_race` | single_bin | prefilter+specattr | 6/6 | 4/6 | 10.25 (6.33-12.00) | 26% |
| `inner_race` | sideband | spec | 6/6 | 1/6 | 10.17 (6.17-11.00) | 26% |
| `inner_race` | sideband | prefilter | 6/6 | 2/6 | 10.00 (6.17-10.67) | 25% |
| `inner_race` | sideband | prefilter+specattr | 6/6 | 3/6 | 10.00 (6.17-10.67) | 25% |
| `ball_spin` | single_bin | spec | 6/6 | 2/6 | 9.83 (6.33-11.00) | 25% |
| `ball_spin` | single_bin | prefilter | 6/6 | 1/6 | 9.08 (6.00-10.67) | 23% |
| `ball_spin` | single_bin | prefilter+specattr | 6/6 | 2/6 | 9.08 (6.00-10.67) | 23% |
| `ball_spin` | sideband | spec | 6/6 | 1/6 | 9.83 (8.33-11.00) | 25% |
| `ball_spin` | sideband | prefilter | 6/6 | 1/6 | 9.17 (6.33-10.00) | 23% |
| `ball_spin` | sideband | prefilter+specattr | 6/6 | 3/6 | 9.17 (6.33-10.00) | 23% |
| `cage` | single_bin | spec | 6/6 | 6/6 | 12.92 (12.67-13.67) | 33% |
| `cage` | single_bin | prefilter | 6/6 | 6/6 | 12.92 (12.67-13.67) | 33% |
| `cage` | single_bin | prefilter+specattr | 6/6 | 6/6 | 12.92 (12.67-13.67) | 33% |
| `cage` | sideband | spec | 6/6 | 6/6 | 12.75 (12.67-13.50) | 32% |
| `cage` | sideband | prefilter | 6/6 | 6/6 | 12.83 (12.67-13.50) | 33% |
| `cage` | sideband | prefilter+specattr | 6/6 | 6/6 | 12.83 (12.67-13.50) | 33% |
| `thermal` | single_bin | spec | 6/6 | 6/6 | 11.75 (8.83-15.83) | 30% |
| `thermal` | single_bin | prefilter | 6/6 | 6/6 | 11.75 (7.17-15.67) | 30% |
| `thermal` | single_bin | prefilter+specattr | 6/6 | 5/6 | 11.75 (7.17-15.67) | 30% |
| `thermal` | sideband | spec | 6/6 | 6/6 | 11.75 (8.83-15.83) | 30% |
| `thermal` | sideband | prefilter | 6/6 | 6/6 | 11.75 (7.17-15.67) | 30% |
| `thermal` | sideband | prefilter+specattr | 6/6 | 6/6 | 11.75 (7.17-15.67) | 30% |
