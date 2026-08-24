# StreamSafe exploratory analysis

## Dataset version

- Repository: `Solitude0630/StreamSafe`
- Immutable revision: `16d0ff1f42e980bb99bd36125583361b15c664e3`
- License declared by the source: CC BY 4.0

## Files and schemas

The snapshot contains 5 data tables and 61,811 rows: 7,730 full-response rows, 52,881 partial-response rows, and 1,200 held-out test rows. Full and partial training tables share six fields; the held-out test table has a separate four-field binary-label schema. See `reports/tables/dataset_overview.csv` and `raw_schema_summary.csv`.

## Labels and harm categories

The source exposes three-way `answer` labels for training and validation and binary `label` values for test. Aggregate counts are {'safe': 29285, 'uncertain': 6544, 'unsafe': 25982}. Harm categories are multi-label; their distribution is recorded in `harm_category_distribution.csv`.

## Length and prefix structure

A total of 3,249 response traces could be reconstructed from uniquely matched prefixes. The unique-prefix alignment rate is 31.90%. Among reconstructed traces, 1,138 contain an observed unsafe prefix and 42 first become unsafe after at least 75% of the full response length. See `prefix_alignment_quality.csv` and `prefix_transition_summary.csv`.

## Data-quality findings

The raw validation found 206 rows participating in exact duplicate prompt-response groups across source tables. Cross-split leakage requires normalization. Prefixes without a unique full-response match are retained as unmatched or ambiguous rather than assigned heuristically.

## Recommendation for the first benchmark subset

A 500-trace baseline subset is feasible from the available label counts. Sampling should operate on normalized trace identifiers, stratify by final safety label and harm category, and explicitly include early-, middle-, and late-onset unsafe traces.

## Next decisions

- Normalize full and prefix records into a model-independent trace schema.
- Define whether unsafe onset is the first unsafe sentence boundary or the first unsafe character span.
- Build and manually audit the first balanced subset.
- Keep unmatched and ambiguous prefix groups out of leakage metrics until reviewed.
