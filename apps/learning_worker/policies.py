import importlib

from apps.api.errors import Problem, unavailable

IMPLEMENTATIONS = {
    "smolvla": ("learning.smolvla", "PolicyJobs"),
    "gr00t_n1_5": ("learning.gr00t", "Gr00tJobs"),
    "gr00t_n1_7": ("learning.gr00t_n17", "Gr00tJobs"),
}


def implementation(policy_type: str, *, model_use: bool):
    if policy_type not in IMPLEMENTATIONS:
        raise Problem(
            503,
            "policy_implementation_missing",
            "The exact model implementation is not registered.",
        )
    if model_use and policy_type != "smolvla":
        raise Problem(
            503,
            "model_license_unapproved",
            "This GR00T model is not approved for the customer production use case.",
        )
    module_name, jobs_name = IMPLEMENTATIONS[policy_type]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise unavailable("Pinned versioned policy implementation") from exc
    if module.POLICY_TYPE != policy_type:
        raise Problem(
            503,
            "model_family_mismatch",
            "Model artifacts cannot be relabeled across implementations.",
        )
    return module_name, jobs_name
