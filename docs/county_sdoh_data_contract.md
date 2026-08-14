# County SDOH data contract

## Geographic design

The pipeline uses county FIPS as its member enrichment key. It does not map
members to census tracts and does not derive a county from ZIP, name, or
coordinates. The raw Synthea `FIPS` value is retained as `fips`; a separate
format-normalized `county_fips` is provided for key comparison.

The intended, future flow is:

`Synthea member → county_fips → county ACS + county PLACES + county-aggregated SRAM → county SDOH profile → member enrichment → risk prioritization`.

No member enrichment or risk model is produced by this contract.

## Table contracts

| Table | Required key | Content | Current status |
| --- | --- | --- | --- |
| Member table | `member_id`, `county_fips` | Member demographics and clinical variables | Available from cleaned Synthea when source FIPS is present |
| County ACS | `county_fips` | Legitimate county-level ACS features | **NOT AVAILABLE**: current ACS rows use `0100000US` (national) |
| County PLACES | `county_fips` | County health prevalence measures | Available in `data/processed/places_features.csv` |
| County SRAM | `county_fips` | County aggregation of source SRAM tract measures | Available in `data/processed/sram_county_features.csv` after aggregation |
| County SDOH profile | `county_fips` | ACS + PLACES + SRAM county features | Blocked until county ACS is supplied |
| Member SDOH | `member_id`, `county_fips` | Member fields plus county SDOH profile | Not created |

## Key requirements

- County joins use only the normalized five-digit `county_fips` string.
- Leading zeroes are retained. Normalization validates format, not county
  existence; an authoritative county reference would be required for that.
- County names and state names are descriptive fields, never join keys.
- `0100000US` national ACS and `0400000US..` state ACS values are rejected as
  county ACS. A county ACS GEO_ID must match `0500000US` followed by a
  five-digit county FIPS.
- Missing or unmatched keys must be reported; they must not cause silent member
  loss.

## SRAM aggregation rules

The source SRAM table is tract-level and is not joined to members. It is first
grouped by its existing `county_fips` field. Source count measures are summed.
Binary tract flags become counts of tracts with the flag. `poverty_rate_pct`
and the two low-access percentage fields are omitted because the repository
does not establish a compatible county denominator or aggregation method.
The SRAM coverage report explicitly counts any missing county FIPS rows.

## Current blocker

County-level ACS data is still missing. The current national ACS files must not
be used as county data, copied to counties, or joined into the county profile.
