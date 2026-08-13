# Historical result provenance

These files are a byte-preserving archive of the result directories present at
Git commit `6e5aaec772f942573add5d211ab10edf683d69bd` before the rigorous pipeline
repair. They are regression references only. They are **not** confirmatory
outputs and must not be used to support cross-model or cross-treebank claims.

Known limitations at archival time:

- the data loader merged official UD train/dev/test files and resplit them;
- test-time all-head rankings were produced;
- configs, exact dataset/model revisions, manifests, exclusions, and raw
  per-instance outputs were incomplete or absent;
- the larger-model outputs did not establish the pre-unmask object-to-verb
  effect reported for the historical DiffuGPT experiment;
- unknown provenance fields remain explicitly unknown and are not inferred.

The archive is immutable by convention. New runs use the versioned run layout
under `results/<track>/<model>/<dataset>/<experiment>/<run_id>/`.
