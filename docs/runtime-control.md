# Isaac runtime control and capture

The authenticated simulator bridge remains private. `POST /v1/commands` still
selects the existing reference controller; recording does not make that
controller a learned policy. CPU protocol and physics doubles are not Azure GPU
evidence.

## Nonblocking capture

`SimulatorRuntime.tick()` owns physics, cameras, command completion and the
actual engine heartbeat. It never finalizes or uploads a dataset. An approved
command snapshots its owner, environment revision, epoch and capture ID before
creating a single bounded `CaptureWorker`. Writer construction, all appends,
validation and managed-identity uploads execute on that same worker thread.
Motion waits for writer preparation without blocking the main loop.

The queue holds at most 64 pending frames and 64 MiB including a metadata
allowance. Overflow invalidates capture and stops the command; it never drops or
interpolates a sample. One persistence worker can exist at a time, including
finalization/upload. A second recorded command fails explicitly while it is
busy. Non-recorded physics and the engine heartbeat continue during upload.

`GET /v1/commands/{command_id}/capture` uses the existing authenticated
`X-Environment-Owner` lease and returns:

```json
{
  "capture_id": "<UUID>",
  "command_id": "<UUID>",
  "epoch": "<UUID>",
  "status": "recording",
  "receipt": null,
  "message": null
}
```

Capture phases are `recording`, `finalizing`, `uploading`, `ready` and `invalid`.
A ready receipt has the existing `DemonstrationResult` shape: `status=uploaded`,
private manifest URI, manifest SHA-256, episode UUID and frame count. Files are
validated before upload, and the manifest is uploaded last. Cancellation of an
invalid worker is checked again before each upload, including the manifest.

Physical `Execution.status`, completion time and final measured position are
committed independently, without waiting for storage. Only the matching
owner/epoch/command/capture can attach a receipt to that historical execution.
A stale callback cannot finish another command, restore readiness, play physics
or mutate a new scene. Reactivation invalidates unpublished captures. A failed
upload does not retroactively change physical success, and a successful upload
cannot turn a cancelled or timed-out command into success.

The original reference recording cadence remains 60 Hz with two actual
synchronized camera frames and effective issued nine-joint position targets.
It is not re-labelled as 10 Hz held-position training data.
