from apps.api.errors import Problem
from apps.api.learning_models import BootstrapReport, PairedReport, TrainingRun
from apps.api.learning_ports import JobSpecification
from apps.api.models import Principal
from apps.api.simulation_reports import SimulationReport, validate_report_binding


def learning_publication(configuration, learning):
    empty = {"api_version": "public-learning-v1", "status": "not_published", "publication": None}
    if configuration.public_learning_evaluation_id is None:
        return empty
    actor = Principal(
        tenant_id=configuration.entra_tenant_id,
        object_id=configuration.public_learning_owner_id,
    )
    # Only this deployment-pinned project/evaluation may be projected, never a user query.
    project = learning.get(actor, "project", configuration.public_learning_project_id).value
    evaluation = learning.get(
        actor, "evaluation", configuration.public_learning_evaluation_id
    ).value
    if (
        evaluation.project_id != project.id
        or evaluation.status not in ("succeeded", "failed", "cancelled", "timed_out")
        or evaluation.report is None
    ):
        raise Problem(
            503, "learning_publication_unavailable", "The approved comparison is not verified."
        )
    candidate = learning.get(actor, "candidate", evaluation.candidate_id).value
    training = learning.get(actor, "training", candidate.training_run_id).value
    dataset = learning.get(actor, "dataset", candidate.dataset_id).value
    if (
        not isinstance(training, TrainingRun)
        or training.status != "succeeded"
        or training.candidate_id != candidate.id
        or candidate.project_id != project.id
        or dataset.project_id != project.id
        or training.dataset_id != dataset.id
        or training.pretrained_artifact_id != project.pretrained_artifact_id
    ):
        raise Problem(
            503,
            "learning_publication_unavailable",
            "The approved learning lineage is inconsistent.",
        )
    from apps.api.learning_service import validate_bootstrap_report, validate_paired_report

    report = evaluation.report
    if isinstance(report, SimulationReport):
        baseline = (
            None
            if report.comparison_kind == "reference_bootstrap"
            else learning._baseline(
                actor,
                evaluation.baseline_release_id,
                execution_timing="paused_simulation",
                control_profile_id=project.control_profile_id,
            )
        )
        if not all(
            record.matches_timing(project) for record in (evaluation, candidate, training, dataset)
        ):
            raise Problem(
                503, "learning_publication_unavailable", "Published timing lineage differs."
            )
        validate_report_binding(
            JobSpecification(
                owner_key=actor.owner_key,
                project=project,
                run=evaluation,
                candidate=candidate,
                baseline=baseline,
            ),
            report,
        )
    elif isinstance(report, BootstrapReport):
        validate_bootstrap_report(project, candidate, report)
    elif isinstance(report, PairedReport):
        baseline = learning._baseline(actor, evaluation.baseline_release_id)
        validate_paired_report(project, baseline, candidate, report)
    else:
        raise Problem(503, "learning_publication_unavailable", "Unknown comparison type.")
    return {
        "api_version": "public-learning-v1",
        "status": "published",
        "publication": {
            "title": project.display_name,
            "task": project.instruction,
            "policy_type": candidate.policy_type,
            "recorded_at": evaluation.updated_at.isoformat(),
            "evaluation_status": evaluation.status,
            "data_provenance": {
                "human_teleop": dataset.human_teleop_count,
                "reference_controller": dataset.reference_controller_count,
                "learned": dataset.learned_policy_count,
            },
            "training": {
                "optimizer_steps": candidate.optimizer_steps,
                "model_sha256": candidate.model_sha256,
                "parent_model_sha256": candidate.parent_model_sha256,
                "dataset_sha256": dataset.manifest_sha256,
                "created_at": candidate.created_at.isoformat(),
                "loss": training.metrics.loss,
            },
            "comparison": report.model_dump(mode="json", exclude={"artifact_id"}),
            "execution": "recorded_evaluation_not_live",
        },
    }
