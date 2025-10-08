# Pharmaceutical Dataset Cache

Download the Springer Nature pharmaceutical Raman dataset from
`https://figshare.com/ndownloader/articles/27931131/versions/1` and place
the resulting `api_dataset.zip` file in this directory.

The curated ingestion script `scripts/fetch_pharma_samples.py` reads
`samples.yaml` to select a deterministic subset of spectra from the CSV
inside the archive. Update `samples.yaml` if you want to change the labels
or the number of samples imported.
