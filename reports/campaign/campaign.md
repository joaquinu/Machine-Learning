# Multi-seed surrogate campaign

5 modelled failure modes x 6 seeds x 2 scoring variants = 60 scored runs. **All data is modelled**; this measures how much the surrogate's lead-time claim depends on the seed and the fault mode, not how a physical bearing behaves.

## Detection rate and lead time by failure mode

| fault mode | variant | detected | driver @ first | driver @ terminal | terminal lead (h) | % of life | pre-onset false alarm (h) |
|---|---|---|---|---|---|---|---|
| `outer_race` | spec | 6/6 | 1/6 | 6/6 | 11.75 (10.17-11.83) | 30% | 2.92 (max 5.17) |
| `outer_race` | prefilter | 6/6 | 3/6 | 6/6 | 11.67 (10.00-11.83) | 30% | 0.58 (max 3.17) |
| `inner_race` | spec | 6/6 | 1/6 | 1/6 | 10.00 (6.50-11.67) | 26% | 2.92 (max 5.17) |
| `inner_race` | prefilter | 6/6 | 2/6 | 2/6 | 9.75 (6.33-11.67) | 25% | 0.58 (max 3.17) |
| `ball_spin` | spec | 6/6 | 0/6 | 2/6 | 9.50 (7.17-10.67) | 24% | 2.92 (max 5.17) |
| `ball_spin` | prefilter | 6/6 | 1/6 | 1/6 | 8.75 (5.83-10.50) | 22% | 0.58 (max 3.17) |
| `cage` | spec | 6/6 | 1/6 | 6/6 | 12.58 (12.50-13.17) | 32% | 2.92 (max 5.17) |
| `cage` | prefilter | 6/6 | 3/6 | 6/6 | 12.67 (12.50-13.17) | 32% | 0.58 (max 3.17) |
| `thermal` | spec | 6/6 | 6/6 | 5/6 | 11.75 (9.00-16.00) | 30% | 2.92 (max 5.17) |
| `thermal` | prefilter | 6/6 | 6/6 | 6/6 | 11.75 (7.50-15.83) | 30% | 0.58 (max 3.17) |
