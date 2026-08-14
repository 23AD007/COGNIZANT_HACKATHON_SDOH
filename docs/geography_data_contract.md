# Geography Data Contract

## Current state

Synthea provides member-level county/ZIP/coordinate context, but no Census tract GEOID. The supplied ACS 2024 DP and Subject table extracts are national-level (`0100000US`), and SRAM is tract-level (`CensusTract20` / processed `tract_geoid`). No legitimate member-to-tract crosswalk is present. Geographic integration is therefore intentionally blocked.

No Census API, tract-boundary, ZIP-to-tract, geocoding, or crosswalk source is configured in this repository.

## A. Required Synthea input

The future mapping input must retain these source-backed fields:

| Field | Source field | Current level |
|---|---|---|
| `member_id` | `Id` | Member |
| `fips` | `FIPS` | County where present; not a tract key |
| `zip` | `ZIP` | ZIP; not a tract key |
| `city` | `CITY` | Place name; not a tract key |
| `state` | `STATE` | State name; not a tract key |
| `county` | `COUNTY` | County name; not a tract key |
| `lat` | `LAT` | Member coordinate; requires a documented boundary mapping resource |
| `lon` | `LON` | Member coordinate; requires a documented boundary mapping resource |

## B. Required member geography output

| Field | Requirement |
|---|---|
| `member_id` | Unique member identifier |
| `tract_geoid` | Supplied by a legitimate, documented member-to-tract mapping; exactly 11 digits |

No code may infer `tract_geoid` from county FIPS, ZIP, city, state, or a random assignment.

## C. Required ACS input

Future ACS input must contain census-tract records for the selected ACS tables: DP02, DP03, DP04, DP05, S0101, S1501, S1701, and S1901. Each record must carry a legitimate tract identifier and selected ACS SDOH variables.

The ingestion interface accepts only `GEO_ID` values in the tract form `1400000US` followed by an 11-digit tract GEOID, and rejects national `0100000US` and state `0400000US##` values. This format must be confirmed from the supplied tract-level ACS source when it is added; current repository ACS files do not provide such a record.

The canonical output key is the 11-digit suffix, `tract_geoid`.

## D. Required SRAM input

SRAM input must contain:

| Field | Requirement |
|---|---|
| `tract_geoid` | 11-digit tract identifier derived from documented `CensusTract20` |
| SRAM SDOH variables | Documented SRAM tract-level variables |

## E. Required final join

Only after both missing inputs are supplied and validated:

`member_id` → `tract_geoid` → ACS tract variables + SRAM tract variables

No ACS/SRAM or member-level geographic merge is authorized while either the tract ACS data or the member-to-tract mapping is absent.

## Required external inputs before integration

1. Legitimate tract-level 2024 ACS data for the selected tables, with tract `GEO_ID` records.
2. A documented, legitimate member-to-tract mapping/crosswalk keyed by `member_id` and `tract_geoid`.

The repository does not select a mapping method. Any future coordinate-to-tract boundary source, ZIP-to-tract crosswalk, or other Census geographic resource must be added with documentation and validated before use.
