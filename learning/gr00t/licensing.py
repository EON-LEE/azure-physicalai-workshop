"""Deployment policy derived from exact primary artifacts, not a legal-assent checkbox."""

from learning.common import require

MODEL_LICENSE_DECISIONS = {
    (
        "gr00t_n1_5",
        "869830fc749c35f34771aa5209f923ac57e4564e",
    ): "NVIDIA OneWay Noncommercial license; customer-facing commercial use is not approved",
    (
        "gr00t_n1_6",
        "d0814e7ecb19202e7c8468b46098b0b7ef3a6d61",
    ): "NVIDIA OneWay Noncommercial license; customer-facing commercial use is not approved",
    (
        "gr00t_n1_7",
        "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
    ): (
        "license conflict: README says commercial/Open Model "
        "but pinned LICENSE retains noncommercial section 3.3"
    ),
}

# Only a separately reviewed, unambiguous vendor grant/revision may change this source allowlist.
APPROVED_COMMERCIAL_MODELS: frozenset[tuple[str, str]] = frozenset()


def require_commercial_model(policy_type: str, model_revision: str) -> None:
    key = (policy_type, model_revision)
    require(
        key in APPROVED_COMMERCIAL_MODELS,
        "Model license blocks customer deployment: "
        + MODEL_LICENSE_DECISIONS.get(key, "unknown model/revision has no approved license"),
    )
