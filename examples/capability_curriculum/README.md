# Capability curriculum policy template

`policy.template.json` is a **configuration skeleton**, not a training corpus,
prompt set, answer set, benchmark, or claim of model quality. Replace every
`replace-with-...` string with approved values from the operator's data-governance
and evaluation process before using it.

The compiler requires rich `vegai-training-record/v1` input with provenance and
metadata at the configured paths. It deliberately rejects bare TXT records for
this workflow because TXT has no metadata field in the repository's data
contract. After review, use the commands in
[`docs/CAPABILITY_CURRICULUM.md`](../../docs/CAPABILITY_CURRICULUM.md) to build,
verify, and hand the immutable stage ledger to the continuous pretraining
controller.

No source records are included here. Keep licensed data and held-out target-task
items in the governed data store, not in this example directory.
