# RRUFF Sample Cache

Place curated RRUFF zip archives in this directory so the offline
ingestion scripts can operate without network access. For example:

```
data/raw/rruff/excellent_oriented.zip
```

`scripts/fetch_rruff_samples.py` reads `samples.yaml` to know which files
to extract and ingest into `raman_spectra.db`. Update `samples.yaml` if
you change the filenames or add new subsets.
