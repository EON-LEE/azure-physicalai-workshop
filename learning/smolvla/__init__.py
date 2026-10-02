"""Explicit Apache-licensed SmolVLA family. Never a relabeled or fallback GR00T checkpoint."""

POLICY_TYPE = "smolvla"
ACTION_HORIZON = 50
MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
BACKBONE_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SOURCE_COMMIT = "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"
UPSTREAM = {
    "source_commit": SOURCE_COMMIT,
    "package_version": "0.4.4",
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "backbone_id": BACKBONE_ID,
    "backbone_revision": BACKBONE_REVISION,
    "model_license": "Apache-2.0",
    "backbone_license": "Apache-2.0",
    "model_license_sha256": "a484071b8d7d0591452f795813fb54af9199fc1c48ef1ee995358d5ba07ee070",
    "backbone_license_sha256": "afc4bf5f519438b221a84292979a9eac3de4b9726e5baaa82033c55f59f9f7a5",
    "source_license_sha256": "0583375a0ec642ec79c447781f34322b0cfa66baff31258214d040f1f6d86d7c",
}
