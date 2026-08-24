# Local data policy

Only synthetic fixtures are committed to Git. Real StreamSafe files are downloaded to
`data/raw/streamsafe/` and ignored by Git. Intermediate tables belong in `data/interim/`,
and normalized benchmark releases will later be written to `data/processed/` and published
through Hugging Face Hub.

The source dataset contains sensitive safety examples. Do not paste full responses into
issues, pull requests, logs, notebooks, or screenshots.
