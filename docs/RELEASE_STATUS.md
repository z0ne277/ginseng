# Source-release scope and limitations

Source presence and unit-test success do not establish experimental reproduction.

Included: current structural model/training, ITA extraction, complete-ranking
evaluation, bootstrap, SSI analysis, controlled variants, self-supervised baseline
training, frozen-encoder adapters and tests.

Not included: datasets, exact author split CSVs, trained weights, feature caches,
per-query experiment results, or portable regeneration of every paper figure.
SimCLR and legacy MoCo V3 have model/extraction adapters only. Historical hybrid
code and illustrations are retained for compatibility, not as current results.

Remaining author/release actions:

1. Document access to the exact current data version and splits.
2. Supply matching weights/checksums if exact score reproduction is promised.
3. Add original SimCLR/MoCo V3 training drivers if claiming from-scratch
   reproduction for every baseline.
4. Verify checkpoint-to-table mapping before calling this an exact reproduction
   release. No end-to-end experiment was run during source preparation.

The existing MIT license is retained. Check rights to any subsequently added
dataset, weights or third-party source separately.
