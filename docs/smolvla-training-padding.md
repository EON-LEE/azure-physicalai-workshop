# Explicit TRAIN-only temporal-padding compatibility

The pinned LeRobot 0.4.4 source at
`8fff0fde7c79f23a93d845d1a50e985de01f8b8a` creates `action_is_pad` in its dataset
and preserves that key through the native processor pipeline. SmolVLA's native
`forward` instead reads `actions_id_pad`. This mismatch prevents its existing
temporal mask branch from excluding repeated end-of-episode target rows.
It is a demonstrated loss-semantics defect, not proof of the cause of the
historical validation rejection or evidence of better physical performance.

## Selection and exact behavior

Existing `physicalai.smolvla-checkpointing/v1` plans keep their old behavior.
The corrected recipe requires an explicitly reviewed **v2 checkpoint policy**:

```json
{
  "schema": "physicalai.smolvla-checkpointing/v2",
  "training_recipe": {
    "schema": "physicalai.smolvla-training-recipe/v1",
    "id": "native-temporal-padding-alias-v1",
    "boundary": "post-preprocessor-update-policy",
    "canonical_mask": "action_is_pad",
    "native_mask": "actions_id_pad"
  },
  "limits": {
    "max_files": 64,
    "max_file_bytes": 4294967296,
    "max_total_bytes": 8589934592,
    "max_json_bytes": 1048576,
    "max_header_bytes": 4194304,
    "max_checkpoints": 32,
    "publication_timeout_seconds": 180
  },
  "resume": null
}
```

The selector is closed: unknown recipe values, omitted v2 recipe, extra fields,
or a recipe attached to v1 fail before training. Limits are unchanged; the example
does not authorize an allocation, optimizer run or image build.

`checkpoint_runner.train_native` installs a scoped hook on the native trainer's
`update_policy` call. That call receives the **already preprocessed batch**.
The hook supplies a shallow copy with `actions_id_pad` pointing to the original
`action_is_pad` tensor, then delegates the original native update function.
An alias inserted before preprocessing is insufficient: the native converter
discards that noncanonical spelling.

Targets must be floating `B x 50 x 9` tensors; the canonical mask must be
`torch.bool`, exactly `B x 50`, on the action tensor's device. A present alias
must already agree in shape, dtype, device and values, or the update fails.
No tensor is cast, moved, clipped, resized or copied by the compatibility hook.
The canonical mask and every original batch value remain unchanged.

Native `DeviceProcessorStep` already moves complementary tensor data and applies
its configured dtype conversion only to floating tensors. The bool mask therefore
follows the action device without becoming a numeric mask. A mismatch fails
rather than silently transferring or coercing it.

Only the native temporal masking branch becomes reachable. Reduction is still
over the existing fixed horizon and **32 channels**, including its original
denominator. No valid-count renormalization, channel slicing, sampling change,
normalization/AMP change, new loss, policy override or homegrown optimizer loop
is introduced. Unpadded loss and gradients remain identical. Inference,
`predict_action_chunk`, `_load`, `reset`, all 50-row guards, and all 39 frozen
control files remain untouched.

## Provenance, checkpoints and processor compatibility

The internal child context is `physicalai.smolvla-checkpoint-run/v3`, which requires
the exact `training_recipe`. Legacy v2 contexts cannot implicitly enable it.
The selected recipe is included in the existing checkpoint training-config hash
and runtime hash. The runtime additionally verifies exact native dataset,
SmolVLA forward, processor, converter and device-transfer file hashes.
The complete training-source hash and newly reviewed source snapshot also change.
The native log records `PHYSICALAI_TRAINING_RECIPE` before entering the loop.

Candidate metadata, `training-context.json`, checkpoint manifest fields and native
processor files retain their existing shapes. In particular, the candidate's
`training.config_sha256` still hashes the original numeric `TrainOptions`; it is
not silently redefined. The recipe is recoverable from its archived, hash-bound
run configuration and native log, while the candidate's existing
`code_snapshot_sha256`, job identity and specification bindings identify the new
training execution. No fields are added for old closed model/import consumers
to ignore or reject.

The hook never wraps or registers a processor. Native
`make_pre_post_processors` can reload the saved standard processor JSON and
safetensors without a compatibility class; serialized processor names and tensor
contents are unchanged.

An old full-state checkpoint cannot resume under the new recipe: its original
config/source/runtime binding differs, and all those fields remain mandatory
for full-state continuation. Explicit weights-only initialization can select
the genuine prior model with a **fresh optimizer/scheduler/data state**, retaining
the original model and checkpoint provenance. New corrected full-state checkpoints
remain resumable only under their exact original corrected bindings and horizon.

The changed source requires a **new training image/static-source proof and fresh
approved job context**. Do not relabel `7ca/staticb56`, rebuild a historical archive,
or modify the independently released `abc48/9b0` audit. Config/import verification
for a new v2 recipe needs this updated non-frozen parser and an explicit verifier
rollout; an old parser should reject the new selector. No inference schema or
custom processor change is required to read the resulting ordinary candidate.
If a serving image is rebuilt with changed source files, its full-source runtime
descriptor must nevertheless be regenerated; old descriptors cannot attest
different bytes.

## Same-original-dataset P0 refinement

The existing standalone Command route can refine genuine P0 on its original
verified TRAIN20 without waiting for additional demonstrations. Use a **new**
job/config/source/output prefix, `parameters.resume_mode="weights_only"`,
`checkpointing.resume=null`, original raw manifest `ab4e65...`, P0 parent
`286e97...`, and **omit** `training_cohort`. That omission selects the original
TRAIN10001..10020; it does not weaken `p1_additional20`, which still requires its
distinct complete TRAIN11001..11020. Call the new run **P0 refinement**, not P1
or new demonstrations. Keep original jobs, failures, UUIDs and held-out splits.

This route does not make the earlier generic batch-8 proposal executable:
standalone Command currently requires one `Standard_NC24ads_A100_v4` LowPriority
compute, matching the options' tier, checkpoint interval at most 100, and at most
32 complete checkpoints. Twenty complete passes of 8,587 frames at batch 64 with
native `drop_last=false` would be 2,700 updates and 27 checkpoints at interval 100.
That is only a numerically compatible proposal: actual batch-64 memory,
zero-worker image decoding, 27 full-state transfers, disk space and the original
finite deadline must be qualified before approval. Neither a successful CPU mask
test nor this route approves training or proves physical learning quality.
