# Local raw-data layout

Raw third-party datasets are intentionally **not committed** to Git. Keep provenance in `docs/REAL_DATA_AUDIT_V0.md` and `configs/data_sources.yaml`.

For the first R0 audit, place downloaded CSV files here:

```text
data/raw/
  wsg_cama/
    all_abundance_wide_format.csv
    DailyMaxTemperature.csv
  wsg_effort_geo/
    PAMA.Month.CPUE.csv
    Network_PAMA_CPUE.Map.csv
```

Sources:

- Washington Sea Grant / WDFW / UW Dryad dataset: DOI `10.5061/dryad.0rxwdbsdt`
- Washington Sea Grant / UW Dryad dataset: DOI `10.5061/dryad.sqv9s4ndp`

Important: the second dataset studies *Palaemon macrodactylus*. We reuse only shared Crab Team sampling metadata (`trap.sets`, site coordinates, protocol structure). PAMA outcomes are never treated as European green crab evidence.
