from learning.gr00t.bootstrap import compare_bootstrap_trials as _compare_bootstrap
from learning.gr00t.bootstrap import evaluate_bootstrap as _bootstrap
from learning.gr00t.evaluation import compare_trials as _compare
from learning.gr00t.evaluation import evaluate_pair as _pair
from learning.smolvla import POLICY_TYPE
from learning.smolvla.artifacts import validate_model

PLAN_SCHEMA = "physicalai.smolvla-paired-plan/v1"
RESULT_SCHEMA = "physicalai.smolvla-paired-results/v2"
REPORT_SCHEMA = "physicalai.smolvla-paired-report/v1"
BOOTSTRAP_PLAN_SCHEMA = "physicalai.smolvla-bootstrap-plan/v1"
BOOTSTRAP_RESULT_SCHEMA = "physicalai.smolvla-bootstrap-results/v2"
BOOTSTRAP_REPORT_SCHEMA = "physicalai.smolvla-bootstrap-report/v1"


def compare_trials(plan, trials, *, live_gpu_verified):
    return _compare(
        plan,
        trials,
        live_gpu_verified=live_gpu_verified,
        plan_schema=PLAN_SCHEMA,
        report_schema=REPORT_SCHEMA,
        policy_type=POLICY_TYPE,
    )


def compare_bootstrap_trials(plan, trials, *, live_gpu_verified):
    return _compare_bootstrap(
        plan,
        trials,
        live_gpu_verified=live_gpu_verified,
        plan_schema=BOOTSTRAP_PLAN_SCHEMA,
        report_schema=BOOTSTRAP_REPORT_SCHEMA,
        policy_type=POLICY_TYPE,
    )


def evaluate_pair(plan, evidence_root, before_root, after_root, **kwargs):
    return _pair(
        plan,
        evidence_root,
        before_root,
        after_root,
        model_validator=validate_model,
        plan_schema=PLAN_SCHEMA,
        result_schema=RESULT_SCHEMA,
        report_schema=REPORT_SCHEMA,
        policy_type=POLICY_TYPE,
        **kwargs,
    )


def evaluate_bootstrap(plan, evidence_root, candidate_root, **kwargs):
    return _bootstrap(
        plan,
        evidence_root,
        candidate_root,
        model_validator=validate_model,
        plan_schema=BOOTSTRAP_PLAN_SCHEMA,
        result_schema=BOOTSTRAP_RESULT_SCHEMA,
        report_schema=BOOTSTRAP_REPORT_SCHEMA,
        policy_type=POLICY_TYPE,
        **kwargs,
    )
