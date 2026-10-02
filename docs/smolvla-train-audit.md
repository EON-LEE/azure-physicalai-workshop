# TRAIN-only native SmolVLA audit

This is a non-motion diagnostic, not policy qualification, an Isaac trial, or a
new way to score failed trials. It uses the genuine CUDA checkpoint constructor,
saved processors and original `predict_action_chunk`. It does not create a
`SimulationApp`, actuator, live IPC request, grant or reference controller.
The historical zero-action validation remains incomplete; its missing prediction
and RGB packet cannot be reconstructed from these TRAIN observations.

## Evidence that motivates the audit

The retained P0 manifest identifies 20 TRAIN episodes, 8,587 frames and 1,000
optimizer updates. Its native log ends at `smpl:1K epch:0.12`, with learning rate
`2.5e-6`. The wrapper requested batch size 1, initial LR `1e-4`, 100 warmup steps
and cosine decay over all 1,000 steps. This is **0.11646 frame-anchor passes**,
not 50 times that many independent observations. Native frame sampling is
shuffled, without replacement within a pass, with `drop_last=false`; the
`ep:2` log field is not a record of which physical episodes were sampled.
Neither this low exposure nor final scalar loss proves the cause of the original
joint rejection.

The source audit pins eight exact LeRobot files at revision
[`8fff0fde7c79f23a93d845d1a50e985de01f8b8a`](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a).
The native trainer SHA is the same as the existing checkpoint-runner pin.

| Path | Established behavior |
| --- | --- |
| RAW-v3 capture and conversion | Nine joints in declared order: seven radians, two individual finger metres; action is the issued position target, not measured state, a delta, or summed gripper gap. Six real physics ticks per 10 Hz frame remain required. |
| Image conversion and inference | Both use RGB, PIL bilinear square resize to 256, then native processing. Audit compares actual converted tensors against original PNG inputs. |
| `prepare_seed` / saved processors | Dataset statistics initialize new nine-dimensional processors. Inference loads the saved checkpoint processors. The audit verifies their actual tensors against the original conversion statistics, not just JSON metadata. |
| Native dataset | Action offsets are 0..49; indexes past episode end repeat its final target and carry `action_is_pad`. The processor preserves that key. |
| Native SmolVLA loss | It reads **`actions_id_pad`**, so the dataset's temporal padding mask does not reach that branch in the pinned path. The audit observes this mismatch; it does not repair or hide it. |
| Native loss reduction | It averages 32 padded action channels. That scalar is not a physical nine-joint error; 32-channel averaging is not, by itself, declared a bug. |
| Native prediction | Flow matching produces an unbounded action tensor; saved mean/std unnormalization is not a bounds projection. Both finger TRAIN upper quantiles are near 0.04 m. Overshoot is plausible, but actual row, side, magnitude and dtype require inference evidence. |

The [exact-revision upstream guide](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/docs/source/smolvla.mdx)
uses a substantially larger fine-tuning example. Its episode counts, batch size
and A100 duration are guidance, not measured requirements or throughput for this
workcell.

## Immutable input contract

Use `learning/paused/train_audit.py` as an **external script**, outside the
qualified image's static source directory. Set `PYTHONPATH` to that directory,
not to a copied repository checkout. The script checks its own SHA, the static
manifest's complete inventory, imported `learning` location, native package
source hashes, all model/backbone files and the complete raw/conversion datasets.
No static image manifest, training archive or inference descriptor is rewritten.

The plan is a closed `physicalai.smolvla-train-audit-plan/v1` object containing
`audit_id`, original `scope`, the following SHA bindings, the exact
`environment_image`, fixed `random_seed`, `max_wall_seconds` (at most 1,800) and
`max_report_bytes` (at most 16 MiB):

| Binding | P0 value |
| --- | --- |
| `model_sha256` | `286e97cac7e84bdb84e2c4078e7e7e65d576d0c453473338daf99a6c946400ad` |
| `raw_manifest_sha256` | `ab4e65be3a6033c5d07146e70ce0e4ca5d6dce444c1ee9348ce5692c15b37215` |
| `conversion_sha256` | `1436fa2363f709490cf02bfb2b35a319fe0247ae1d3a8fbfcfc7ff5b651e18bf` |
| `backbone_sha256` | `a2f54260865839a32b21e38ba768f2c7cc43b5b151c40e3d2f4380930c7f011e` |
| `control_profile_sha256` | `851df47a362e4f62fcf0cbfa1b2761339ed5e346d1575123629c20355acb77dc` |
| `criteria_sha256` | `63b55c92a4d543c7b48d7c6ac0ed75f9526e5a8ec91852dac9f53b0c6804190a` |
| `frozen_plan_sha256` | `a63f0aac719ac5725f11b2728898f34faa97f4207f1982477aea1a3de4dfa130` |
| `static_source_sha256` | `b56fa0f10a5c15aaf51c7387793170810b48035cfcb6c6c2eb5384dc8561c1f9` |
| `audit_script_sha256` | SHA of the separately staged script's exact bytes, supplied with its handoff |

The preserved model copy may be staged from the reproducibility prefix
`learning/reproducibility/p0-d7a17854-7ae2-4fae-94c3-2b78d925c5f6/model/candidates/step-001000`.
Its six checkpoint files must still match the original manifest, including weights
`ac294f4b235bb52b18777ca745406b33fe453a7e2dd3cffa162c92fed9dbfce6`.
The audit does not need the ten optimizer-state checkpoints.

Stage the original RAW manifest plus **all** referenced frame streams/PNGs, the
original LeRobot conversion plus its complete inventory, the model directory and
the full backbone. `source-timing.jsonl`, parquet/image data, metadata and stats
cannot be replaced by summaries. If the original conversion is unavailable, stop;
do not silently regenerate it under a different hash.

## Execution contract for the cloud owner

Use the existing qualified dependency image:

```text
factory20n3ig3ttsxayp2.azurecr.io/physicalai-smolvla@sha256:7ca42cfbcc6a684d6cfe484605013f53fef7dce5ded6ab46e1fa6e529fa49e8d
```

It contains Python 3.11.14, Torch 2.7.1+cu126, LeRobot 0.4.4 and the locked
processors. Request **one A100 80 GB**, the proven P0 model class; CPU is supported
only for source/contract tests, never by bypassing the CUDA constructor. No new
package installation or image build is needed. Use task disk for the private
inputs, not a two-GiB simulator tmpfs. Record actual allocation, host/GPU memory,
image/command readback and source/input hashes in the executor receipt.

After staging, run once under the executor's original absolute job/allocation
deadline. Replace the local paths and plan SHA with the actual staged values:

```bash
PYTHONPATH=/opt/physicalai/source \
timeout --signal=TERM --kill-after=30s 1800s \
  /opt/smolvla-venv/bin/python -B /audit/train_audit.py \
  --plan /audit/plan.json --plan-sha256 <exact-plan-file-sha256> \
  --raw-root /audit/inputs/raw \
  --conversion-root /audit/inputs/converted \
  --model-root /audit/inputs/model \
  --backbone-root /audit/inputs/backbone \
  --source-root /opt/physicalai/source \
  --static-manifest /opt/physicalai/static-manifest.json \
  --output /audit/output
```

Staging and final artifact transfer must fit the **same finite parent-approved
window**; the 1,800 seconds above cover the audit process, not a renewed
allocation. The script checks elapsed time between bounded operations; the
external timeout is required to bound a blocked CUDA/native operation. Preserve
partial outputs and stderr on failure. Do not retry an ambiguous execution or
overwrite its output directory. This source work grants no cloud authority.

## Output and interpretation

For every original TRAIN episode, anchors are predeclared at the first, middle
and final frame: **60 anchors for P0**, at most 120 for the existing 40-episode
TRAIN union. RNG seed is plan seed plus anchor ordinal. These are new offline
audit requests (`physicalai.smolvla-train-audit-request/v1`), bound to the original
episode/frame/PNG hashes and checkpoint. They claim no original live request ID,
new publication, model command or motion authority.

Each create-only `sample-NNN.json` is at most 128 KiB and retains all **50 x 9**
normalized predictions, denormalized predictions and targets, original temporal
padding, actual inference timing, and physical nine-channel errors. Valid-target,
all-row, padded-target and first-row metrics are separate. Padding excludes
repeated targets only from the corresponding error summary: **all 50 rows still
go through the unchanged joint guard**. Nonfinite values are explicitly tagged;
no clipping, row dropping, tanh, fallback or precision adjustment occurs.

Actual normalized/denormalized tensor dtype/device is captured before reporting.
Saved safetensors, loaded normalization statistics and postprocessor statistics
after prediction also retain their dtypes. This distinguishes reduced-precision
rounding from larger policy errors without assuming either happened in the old
trial. Numeric JSON conversion does not call `.float()` on model outputs.

`report.json` is written last, bounded to 256 KiB within the 16 MiB total. It
contains input/source/package proofs, checkpoint config/processors, actual GPU
memory, sample file hashes and independent finite/bounds outcomes.
`status=completed` or exit zero means **the diagnostic ran**, not that the policy
is safe, usable, released, or successful at pick/place. Missing `report.json`
means incomplete audit evidence. An A100 audit also does not qualify A10 live
latency, render behavior or actual held-out policy quality.

## Prospective training, after audit review

The generated proposal uses **20 complete frame-anchor passes**, batch 8, a new
optimizer from the genuine P0 weights, initial LR `1e-4`, and ten full-state
checkpoints spaced every two passes. With 8,587 frames this is **21,480 updates,
171,740 actual anchors**, checkpoint interval 2,148. The existing native scheduler
then decays over this longer horizon, rather than reaching its floor at 0.12 pass.
For a complete approved P0+P1 union, recompute from its actual manifest; never add
unfinished collection or held-out data to the original manifest.

This is a prospective coverage target, **not a guarantee that 20 passes or 20
episodes suffice**. Before submission, the parent must review a separately
versioned training-only correction for the observed padding-mask mismatch.
Neither the audit nor this proposal silently modifies the pinned training loss,
vendor source, control bundle or existing checkpoint semantics.

The proposal bounds training to 12 hours inside a 16-hour overall job budget,
ten checkpoints, at most eight GiB each/eighty GiB total, and the existing
180-second publication bound per checkpoint. Staging, verification, ten transfers
and closure must fit the original remaining budget. Measure actual batch-8
peak memory and batch/decode/checkpoint throughput early; if the projection cannot
fit, stop and issue a separately reviewed job rather than extending its deadline.
No throughput or fit is claimed from the CPU tests.

Predeclare exact input/processor parity, finite tensors, full-horizon joint bounds
and separate physical nine-channel TRAIN error comparisons against P0. Keep every
checkpoint/outcome, not just the best-looking snapshot. Passing these checks only
permits consideration of a separately approved validation run; it does not waive
the frozen before/after held-out study or any physical watchdog.
