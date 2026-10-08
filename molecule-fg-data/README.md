# Molecule-FG Data Pipeline

This folder contains the end-to-end workflow that turns a list of SMILES strings into:

- a molecule-by-functional-group presence matrix
- compressed FG-pattern counts
- overlapping chemical clusters
- labeled cluster summaries
- model assignments based on domain coverage rules

The pipeline is designed to work from raw SMILES input, process it with AccFG, and then produce CSV artifacts for downstream analysis.

## Folder structure

- `smiles.json` — primary input file with the molecules. Each entry should be an object `{"cid": 176000001, "smiles": "..."}` so the source ID is carried through every stage. A plain list of SMILES strings still works, but then `cid` is only the molecule's position in the file, and stage 1 prints a warning
- `pubchem_like_sample_120.csv` — fallback sample dataset if `smiles.json` is missing
- `build_sample_fg_dataset.py` — creates the FG presence table

This is the location that the scripts are written to read from by default.

- `build_pattern_count_dictionary.py` — builds compressed pattern counts
- `build_pattern_clusters.py` — clusters binary FG patterns
- `label_clusters.py` — decodes cluster patterns into functional-group labels
- `assign_clusters_to_models.py` — assigns cluster coverage to model buckets
- `error_distributions.py` — samples one ground-truth error per molecule per model
- `plot_error_distributions.py` — plots every sampled error per model, coloured by FG cluster
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
- drops salts and mixtures (any SMILES containing `.`) and lists them, with their cid, in `csv_outputs/rejected_molecules.csv`; stages 2 and 3 drop the same molecules

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

- reads the FG matrix stage 1 already computed — `csv_outputs/fg_presence.csv`, or with `--fg-dir` the Parquet output of `stage1/fg_matrix.py` — so AccFG is not run again
- converts each molecule into a binary FG pattern string
- groups molecules that share the exact same pattern
- counts how many molecules are in each pattern

Outputs:

- `csv_outputs/pubchem_like_pattern_counts.csv`
- `csv_outputs/molecule_patterns.csv` — `cid, pattern_index`, one row per molecule
- `csv_outputs/fg_columns.json` — FG names in pattern-bit order
- `csv_outputs/molecule_status.csv` — `cid, status, detail` for every input molecule: `clustered`, `rejected_no_fg`, or the stage-1 reason it was dropped (`rejected_salt_or_mixture`, `rejected_unparseable`, `rejected_accfg_error`)

Stages 3 and 4 read these files instead of `smiles.json`, so every stage works from the same molecules and the same `pattern_index`.

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
- weak components are pruned automatically: a component whose mixing weight ends at or below `--prune-threshold` (default `0.001`, i.e. 0.1% of molecules), or that no pattern belongs to, is dropped and the remaining components are re-fitted; each drop is printed
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

- reads the cluster summary, the per-pattern cluster memberships (`pattern_clusters.csv`) and the model targets (`model_specs.csv`)
- assigns whole clusters to models so that every model's coverage is as close as possible to its target
- writes one row per (model, cluster), a per-model coverage report, and the pattern → cluster → model lineage map

Inputs:

- `csv_outputs/cluster_summary.csv`
- `csv_outputs/pattern_clusters.csv`
- `csv_outputs/model_specs.csv`

Outputs:

- `csv_outputs/model_assignments.csv`
- `csv_outputs/model_coverage_report.csv`
- `csv_outputs/pattern_cluster_model_map.csv`
- `csv_outputs/model_molecule_counts.csv` — per model: unique molecules (distinct cids, a molecule in two of the model's clusters counted once), coverage, how many are in that model only vs shared with other models, and the clusters and patterns behind it; the last row, `ALL MODELS`, counts distinct molecules across every model

Assignment rules:

1. every cluster is assigned to at least one model
2. clusters are assigned whole; a model never takes part of a cluster
3. a cluster may be assigned to several models (the targets may sum to more than 100%, in which case some clusters must be)
4. a model's coverage is the share of **unique clustered molecules** in its clusters; a molecule in two clusters of the same model counts once
5. every model keeps at least `--min-exclusive` (default `0.01`, i.e. 1%) of the clustered molecules that no other model covers; a `min_exclusive` column in `model_specs.csv` sets it per model, and `0` turns the rule off
6. subject to 1–5, the total distance from the targets, sum over models of |coverage − target|, is minimised

The minimisation is exact: a small integer program solved with `scipy.optimize.milp` (scipy ≥ 1.9). Because clusters are whole, a model can end above or below its target; `coverage_gap` in the report (target − coverage) shows by how much, and models that received no cluster are listed with coverage 0.

If no assignment can give every model its exclusive share, stage 5 writes `assignment_warnings.txt` and exits with status **3** as well. Exclusive molecules need clusters that only one model takes, so more clusters (a larger `--max-components`) make the rule easier to meet.

If any model ends more than `--tolerance` (default `0.02`, i.e. ±2 percentage points) from its target, the clusters are too coarse for the targets. All outputs are still written, but the script then prints a message naming each model that misses and the cluster sizes, saying to re-run `build_pattern_clusters.py` with a larger `--max-components`, saves it to `csv_outputs/assignment_warnings.txt`, and exits with status **3**. `within_tolerance` in the report shows which models pass.

## Stage 6: sample a ground-truth error per molecule per model

Script:

- `error_distributions.py`

What it does:

- gives every model two normal error distributions: in-domain, centred on `avg_error_in` with std `std_error_in`, and out-of-domain, centred 3 std higher (`avg_error_in + 3 * std_error_in`)
- a molecule is in-domain for a model when its pattern is in one of the model's clusters (`pattern_cluster_model_map.csv`)
- each FG cluster (a pattern's primary cluster) gets its own part of each model's distribution (`--cluster-weight`, default 0.6), molecules with similar FGs land close together (`--fg-weight`, default 0.3), and the rest is per-pattern noise, so molecules with the same FGs land in the same range but not on the same value
- in-domain: patterns are ranked by that score
- out-of-domain: patterns are ranked by their minimum Hamming distance to the model's in-domain patterns; the closest take the low end of the out-of-domain distribution (the score breaks ties, so clusters still separate at each distance)
- each molecule adds a small jitter of its own (`--jitter`, default 0.2 std); errors below 0 are clipped (`--no-clip` keeps them)
- prints the target vs sampled mean and std for every model and domain

Inputs:

- `csv_outputs/pubchem_like_pattern_counts.csv`
- `csv_outputs/molecule_patterns.csv`
- `csv_outputs/pattern_clusters.csv`
- `csv_outputs/pattern_cluster_model_map.csv`
- `csv_outputs/model_specs.csv` — needs `avg_error_in`; `std_error_in` is optional (`--std`, default 0.05)

Outputs:

- `csv_outputs/molecule_model_error_table.csv` — one row per (molecule, model): `cid`, `pattern_index`, `model_name`, `domain_flag`, `nearest_distance`, `nearest_reference_cluster`, `distance_rank`, `error_mean`, `error_std`, `sampled_error`
- `csv_outputs/pattern_model_error_table.csv` — one row per (pattern, model) with its distance, rank and position in the distribution

### Plotting the errors

`plot_error_distributions.py` reads `molecule_model_error_table.csv` and writes to `csv_outputs/figures/`:

- `error_points.png` — one row per model with in-domain and out-of-domain side by side, every panel on the same y axis (the sampled error itself, not centred): x = that panel's molecules grouped by FG cluster then pattern, one dot per molecule
- `error_points_<model>.png` — one model's two panels on their own, on the same axes
- `error_cluster_boxes.png` — per model and domain, one box per cluster (25–75%, whiskers 5–95%), to see whether the FG bunches sit in different ranges
- `error_histograms.png` — per model and domain, same bins, stacked by cluster
- `error_plot_summary.csv` — n, mean, std, min, max of the error per model, cluster and domain

A molecule's colour is its pattern's primary cluster (`pattern_clusters.csv`); the `--top-groups` largest clusters (default 7, the most colours that stay distinguishable) get their own colour and the rest are grey "Other". The legend names each cluster's common FGs from `cluster_summary.csv`. A dashed line marks each domain's mean. A panel says so when a model has no molecules in that domain. `--max-points N` plots N molecules per model (the same ones in every panel) when every point is too slow; the boxes, histograms and summary always use all of them.

The script first prints each model's in-domain and out-of-domain counts, and warns when a model has no out-of-domain molecules or when `error_mean` is 0 everywhere (a table made without `avg_error_in`, as the original `error_distributions.py` did).

```bash
python3 "molecule-fg-data/plot_error_distributions.py"
```

## The model rule file

The model rules live in:

- `csv_outputs/model_specs.csv`

The file contains one row per model:

- `model_name`
- `target_coverage` — the fraction of clustered molecules the model should cover
- `min_fgs` / `max_fgs` — optional, and ignored by the assignment
- `avg_error_in` — the model's average in-domain error (stage 6)
- `std_error_in` — optional std of both error distributions (stage 6; default 0.05)

Important modeling note:

- the base common list in `accfg/fgs_common.csv` contains 504 entries
- in the current full pipeline (`lite=False`), the runtime FG vocabulary is larger because heterocycle features are added on top of that common list
- in this repo, the active full-mode vocabulary is therefore 534 functional-group features, not 504
- those vocabulary entries are not a separate rule set for each model
- `model_specs.csv` needs `model_name` and `target_coverage` (a fraction of clustered molecules); `min_fgs` / `max_fgs` may be present but are ignored, so any cluster may go to any model

## Typical run order

From the project root, the intended order is:

```bash
python3 "molecule-fg-data/build_sample_fg_dataset.py"
python3 "molecule-fg-data/build_pattern_count_dictionary.py"
python3 "molecule-fg-data/build_pattern_clusters.py" --max-components 12 --seed 0
python3 "molecule-fg-data/label_clusters.py"
python3 "molecule-fg-data/assign_clusters_to_models.py"
python3 "molecule-fg-data/error_distributions.py"
python3 "molecule-fg-data/plot_error_distributions.py"
```

That produces the full pipeline result.

On an LSF cluster, `cluster.lsf` runs stages 2–5 as one job (edit the paths block at the top first):

```bash
bsub < molecule-fg-data/cluster.lsf
```

To run **everything as one job**, stage 1 included (sample 500,000 molecules from the
`processed_Compound_*.jsonl` files → FG matrix → patterns → clusters → labels → models), use
`pipeline.lsf` instead. Edit its paths block first:

```bash
bsub < molecule-fg-data/pipeline.lsf
```

It stops at the first stage that fails. Stages 3–5 are tried with each cluster count in
`MAX_COMPONENTS_TRY` (default `12 16 24 32 48`), smallest first, and the first clustering whose
model assignment meets every rule (coverage within `TOLERANCE`, `MIN_EXCLUSIVE` molecules of
its own for every model) is kept. If none does, the job exits with status 3; add larger values
and resubmit. Stage 1 resumes from its finished outputs, so only stages 2–5 run again.

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
