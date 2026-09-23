from types import SimpleNamespace
from uuid import uuid4

import pytest

from agents.learning_coach import CoachContext, CoachProposal, extract_proposal
from apps.api.errors import Problem


def context():
    return CoachContext(
        project_id=uuid4(),
        task_id="part-kitting-v1",
        instruction="부품을 승인된 트레이에 놓습니다.",
        human_teleop_count=0,
        reference_controller_count=12,
        learned_policy_count=0,
        optimizer_step_limit=100,
        dataset_id=uuid4(),
        evaluation_conclusion="not_improved",
        approved_release_ids=(uuid4(),),
    )


def response(proposal):
    return SimpleNamespace(
        id="actual-test-foundry-response",
        output=[
            SimpleNamespace(
                type="function_call",
                name="propose_learning_step",
                arguments=proposal.model_dump_json(),
            )
        ],
    )


def test_coach_proposals_are_typed_advice_not_self_authorizing_paid_or_motor_commands():
    from pydantic import ValidationError

    owner = context()
    proposal = CoachProposal(
        project_id=owner.project_id,
        action="propose_training",
        summary="기준 제어기 시연 데이터이며 사람의 시연이 아닙니다.",
        dataset_id=owner.dataset_id,
        optimizer_steps=100,
        selected_release_id=None,
    )
    result, response_id = extract_proposal(response(proposal), owner)
    assert result.action == "propose_training"
    assert response_id == "actual-test-foundry-response"
    for extra in ("paid_approved", "motion_approved", "python", "joints", "model_uri"):
        with pytest.raises(ValidationError):
            CoachProposal.model_validate({**proposal.model_dump(), extra: True})


def test_coach_cannot_select_an_unreleased_policy_or_raise_an_approved_budget():
    owner = context()
    for changes in (
        {"action": "select_release", "selected_release_id": uuid4()},
        {"action": "propose_training", "optimizer_steps": 101},
        {"project_id": uuid4()},
        {"dataset_id": uuid4()},
    ):
        proposal = CoachProposal(
            **{
                "project_id": owner.project_id,
                "action": "explain_evaluation",
                "summary": "개선 미확인",
                "dataset_id": None,
                "optimizer_steps": None,
                "selected_release_id": None,
                **changes,
            }
        )
        with pytest.raises(Problem) as failure:
            extract_proposal(response(proposal), owner)
        assert failure.value.status == 422


def test_coach_requires_one_actual_response_id_and_one_registered_tool():
    owner = context()
    proposal = CoachProposal(
        project_id=owner.project_id,
        action="review_demonstrations",
        summary="검토 제안",
        dataset_id=owner.dataset_id,
        optimizer_steps=None,
        selected_release_id=None,
    )
    invalid = response(proposal)
    invalid.output *= 2
    with pytest.raises(Problem):
        extract_proposal(invalid, owner)
    invalid = response(proposal)
    invalid.id = None
    with pytest.raises(Problem):
        extract_proposal(invalid, owner)
