# Molecule-FG Data Pipeline

This folder contains the end-to-end workflow that turns a list of SMILES strings into:

- a molecule-by-functional-group presence matrix
- compressed FG-pattern counts
- overlapping chemical clusters
- labeled cluster summaries
- model assignments based on domain coverage rules

The pipeline is designed to work from raw SMILES input, process it with AccFG, and then produce CSV artifacts for downstream analysis.

## Folder structure

- `smiles.json` — primary input file with the molecule SMILES list
- `pubchem_like_sample_120.csv` — fallback sample dataset if `smiles.json` is missing
- `build_sample_fg_dataset.py` — creates the FG presence table

This is the location that the scripts are written to read from by default.

- `build_pattern_count_dictionary.py` — builds compressed pattern counts
- `build_pattern_clusters.py` — clusters binary FG patterns
- `label_clusters.py` — decodes cluster patterns into functional-group labels
- `assign_clusters_to_models.py` — assigns cluster coverage to model buckets
- `patterns.py` — shared utilities for pattern generation/counting
- `bernoulli_mixture_clustering.py` — actual overlapping Bernoulli-mixture clustering logic
- `csv_outputs/` — all generated CSV outputs live here

## The overall workflow

The system is organized as a staged pipeline:

1. Load SMILES
2. Detect functional groups for each molecule
3. Convert each molecule into a binary FG vector
4. Compress identical patterns into counts
5. Cluster the compressed patterns
6. Label each cluster with meaningfully interpretable FG signatures
7. Assign cluster coverage to model target buckets

## Stage 1: create the FG presence matrix

Script:

- `build_sample_fg_dataset.py`

What it does:

- load molecule list (It is set to `smiles.json` if available)
- otherwise falls back to `pubchem_like_sample_120.csv`
- runs AccFG against each SMILES string
- converts each molecule into a binary vector of functional-group presence/absence
- writes a molecule-by-FG table

Output:

- `csv_outputs/fg_presence.csv`

Important idea:

- each molecule becomes a row
- each functional group becomes a column
- values are either 0 or 1

This is the raw chemistry matrix used for downstream pattern analysis.

## Stage 2: compress identical patterns

Script:

- `build_pattern_count_dictionary.py`

What it does:

- takes the molecule FG vectors
- converts each molecule into a binary FG pattern string
- groups molecules that share the exact same pattern
- counts how many molecules are in each pattern

Output:

- `csv_outputs/pubchem_like_pattern_counts.csv`

This step matters because many molecules often share the same FG pattern. Counting patterns reduces the dataset size and makes clustering more interpretable.

The key columns are typically:

- `pattern`
- `count`
- `pattern_index`

## Stage 3: discover clusters in the compressed pattern space

Script:

- `build_pattern_clusters.py`

What it does:

- loads the pattern-count table
- runs the Bernoulli-style overlapping clustering logic from `bernoulli_mixture_clustering.py`
- identifies groups of similar binary FG patterns
- allows one pattern to belong to multiple relevant clusters
- writes the clustered pattern table

Important detail about the Bernoulli model:

- `--max-components` (default 12; `--k` is an alias) is an upper bound on the number of clusters, not a forced count
- it fits a Bernoulli mixture to the binary FG data using EM
- each cluster has a vector of FG probabilities, where values near 1 mean “this FG is very likely in the cluster”
- weak components are pruned automatically using a small threshold
- the final cluster count is therefore discovered from the data, not hard-coded by the command

What "discovered from the data" does and does not mean:

- a component is removed only when its mixing weight ends below the prune threshold; no model-selection criterion (such as BIC) compares different cluster counts
- so with plenty of molecules you will usually get `--max-components` clusters back, and the count you see reflects that bound and the prune threshold rather than a statistical test
- `Discovered K` in the log reports how many components survived pruning

Output:

- `csv_outputs/pattern_clusters.csv`

This is the canonical clustering output: one row per unique pattern, with the overlapping cluster memberships kept in the same row rather than expanded into duplicate rows.

Key concepts:

- `cluster_id` identifies a cluster
- `pattern` is the FG signature of a specific pattern
- `count` tells how many molecules use that pattern
- `cluster_representative` stores the representative bitstring for the cluster
- `cluster_memberships` captures overlapping memberships for each pattern

This is the important modeling step: the data is binary, it uses a Bernoulli mixture and overlapping cluster assignment.

## Stage 4: label clusters with human-readable FG names

Script:

- `label_clusters.py`

What it does:

- reads the canonical cluster output
- normalizes the overlap memberships into the summaries used downstream
- converts binary strings back into actual FG names
- calculates cluster-level summaries such as:
  - centroid FG list
  - union of FGs across members
  - common FG set across cluster members

Outputs:

- `csv_outputs/pattern_clusters_labeled.csv`
- `csv_outputs/cluster_summary.csv`

The cluster summary is especially important because it turns a bit pattern into something chemically readable.

Example columns include:

- `cluster_id`
- `cluster_weight`
- `centroid_fgs`
- `centroid_n_fgs`
- `union_fgs`
- `common_fgs`

## Stage 5: assign clusters to model coverage buckets

Script:

- `assign_clusters_to_models.py`

What it does:

- reads the cluster summary
- reads the model rules from `model_specs.csv`
- chooses clusters that match each model’s FG-count and target-coverage constraints
- applies a reuse decay so the same cluster cannot be counted fully by every model
- writes assignment rows indicating which cluster belongs to which model

Inputs:

- `csv_outputs/cluster_summary.csv`
- `csv_outputs/model_specs.csv`

Outputs:

- `csv_outputs/model_assignments.csv`
- `csv_outputs/model_coverage_report.csv`

Model assignment logic:

- target coverage is expressed as a share of total cluster mass
- clusters are ranked by effective contribution
- a cluster reused by another model gets discounted via a reuse factor
- each model tries to reach its desired coverage without overusing the same cluster

Important assignment rule:

- every discovered cluster must still be assigned at least once
- this is enforced even when a cluster sits outside the min/max FG windows for all models
- in that case, the cluster is assigned to the closest compatible model by FG-count distance
- however, the fallback assignment is capped so it cannot push a model above its target coverage
- if a model has no remaining target budget, the cluster is not allowed to exceed the remaining allocation for that model

This means the pipeline enforces two constraints at the same time:

1. every cluster gets used at least once
2. no model exceeds its assigned domain coverage target

This is the key difference between a simple “closest model” fallback and a valid domain assignment policy.

## The model rule file

The model rules live in:

- `csv_outputs/model_specs.csv`

The file contains model definitions such as:

- model name
- target coverage
- min allowed FG count
- max allowed FG count

This is how the pipeline decides whether a certain cluster fits a model type.

Important modeling note:

- the base common list in `accfg/fgs_common.csv` contains 504 entries
- in the current full pipeline (`lite=False`), the runtime FG vocabulary is larger because heterocycle features are added on top of that common list
- in this repo, the active full-mode vocabulary is therefore 534 functional-group features, not 504
- those vocabulary entries are not a separate rule set for each model
- instead, each model defines a complexity window for the cluster centroids it is allowed to consider
- a cluster is only eligible for a model if its centroid FG count falls within that model’s min/max FG range

So the full FG list defines what features exist, while the model spec file defines which complexity bands each model is allowed to cover.

## Typical run order

From the project root, the intended order is:

```bash
python3 "molecule-fg-data/build_sample_fg_dataset.py"
python3 "molecule-fg-data/build_pattern_count_dictionary.py"
python3 "molecule-fg-data/build_pattern_clusters.py" --max-components 12 --seed 0
python3 "molecule-fg-data/label_clusters.py"
python3 "molecule-fg-data/assign_clusters_to_models.py"
```

That produces the full pipeline result.

On an LSF cluster, `cluster.lsf` runs stages 2–5 as one job (edit the paths block at the top first):

```bash
bsub < molecule-fg-data/cluster.lsf
```

## If `smiles.json` is missing

The scripts gracefully fall back to:

- `pubchem_like_sample_120.csv`

This makes it easy to test the pipeline without needing a full external dataset.

## How the outputs relate to each other

This is the most important conceptual link:

- `fg_presence.csv` = molecule-level binary table
- `pubchem_like_pattern_counts.csv` = compressed unique pattern table
- `pattern_clusters.csv` = canonical clustered pattern table
- `pattern_clusters_labeled.csv` = decoded FG labels for each pattern row
- `cluster_summary.csv` = chemical summary for each cluster
- `model_assignments.csv` = which cluster covers which model
- `model_coverage_report.csv` = model coverage summary and gaps

## Notes for future use

- Keep raw input data outside the generated CSV set when possible.
- Treat files in `csv_outputs/` as generated artifacts rather than source code.
- Re-run the pipeline whenever the input SMILES list changes.
- If you add new model rules, update `model_specs.csv` and then rerun `assign_clusters_to_models.py`.

For large datasets, place the main SMILES source in this folder as:

- `smiles.json` for JSON input, or
- a CSV file with a `smiles` column for CSV input

This folder is the expected input location for the pipeline. The loader checks for `smiles.json` first and falls back to `pubchem_like_sample_120.csv` only if the JSON file is absent.

For 100K+ molecules, the recommended pattern is:

- keep the raw SMILES file in this folder
- name it `smiles.json` if it is a JSON list or object, or
- name it something descriptive but keep a `smiles` column if it is CSV
- then rerun the pipeline scripts from the project root
