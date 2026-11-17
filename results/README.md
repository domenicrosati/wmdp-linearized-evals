# Result provenance

`published-results.json` contains the currently auditable matched cells.

- The Unfiltered and Strong-Combined rows use the same WMDP-Bio population,
  seed-17 split, full-answer scorer, rank-512 answer-token feature interface,
  and validation-only selection rule.
- Synthetic DI's older four-letter-head result is retained for provenance but
  is explicitly marked incompatible. It must not be compared numerically with
  the full-answer rows.
- Accuracy and cross-entropy are attack outcomes. They are not universal
  resistance certificates.

The sealed reporting labels were used only after hyperparameter selection.
Future updates should retain the original row and add the exact model revision,
cache hash, result hash, and audit status for every new matched run.
