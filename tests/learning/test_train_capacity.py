from contextlib import nullcontext

import pytest

from learning.paused.train_capacity import (
    BackwardBoundary,
    ForbiddenOptimizer,
    StopBeforeOptimizer,
    batch_indices,
    parameter_inventory,
    training_schedule,
)


def episodes():
    return [
        {"episode_index": i, "frame_count": 400 + i, "split": "train", "seed": 10001 + i}
        for i in range(20)
    ]


def test_native_complete_pass_accounting():
    result = training_schedule(8587, 64)
    assert result["optimizer_steps"] == 2700
    assert result["actual_frame_anchors"] == 171740
    assert result["checkpoint_count"] == 27
    assert result["fits_existing_32_checkpoint_limit"] is True
    assert result["training_authorized"] is False
    assert training_schedule(8587, 8)["fits_existing_32_checkpoint_limit"] is False


def test_fixed_train_batch_includes_original_episode_boundaries():
    selected = batch_indices(episodes(), 64)
    assert len(selected) == 64
    assert selected[:3] == [0, 400, 801]
    assert selected[20] == 200
    assert selected[40] == 399
    assert selected[60] == 0
    assert all(0 <= index < 8190 for index in selected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("seed", 30001),
        ("frame_count", 0),
        ("episode_index", 1),
    ],
)
def test_rejects_rebound_or_invalid_train_inputs(field, value):
    rows = episodes()
    rows[0][field] = value
    with pytest.raises(ValueError):
        batch_indices(rows, 8)


def test_real_backward_delegation_stops_before_optimizer():
    events = []

    class Accelerator:
        def autocast(self):
            return nullcontext()

        def backward(self, value):
            events.append(("backward", value))

    boundary = BackwardBoundary(Accelerator(), lambda value: events.append(("observed", value)))
    with boundary.autocast():
        with pytest.raises(StopBeforeOptimizer):
            boundary.backward(123)
    assert boundary.reached is True
    assert events == [("backward", 123), ("observed", 123)]
    with pytest.raises(RuntimeError, match="never execute"):
        ForbiddenOptimizer().step()


def test_backward_error_is_not_successful_measurement():
    class Accelerator:
        def backward(self, value):
            raise RuntimeError("actual backward failed")

    boundary = BackwardBoundary(Accelerator(), lambda value: pytest.fail("must not observe"))
    with pytest.raises(RuntimeError, match="actual backward failed"):
        boundary.backward(1)
    assert boundary.reached is False


def test_native_mixed_trainable_dtypes_are_measured_not_coerced():
    class Parameter:
        requires_grad = True

        def __init__(self, dtype, elements, width):
            self.dtype, self.elements, self.width = dtype, elements, width

        def is_floating_point(self):
            return True

        def numel(self):
            return self.elements

        def element_size(self):
            return self.width

    fp32 = Parameter("torch.float32", 10, 4)
    bf16 = Parameter("torch.bfloat16", 20, 2)

    class Policy:
        def named_parameters(self):
            return iter((("projection", fp32), ("expert", bf16)))

    result = parameter_inventory(Policy())
    assert result["elements"] == 30
    assert result["bytes"] == 80
    assert result["by_dtype"]["torch.bfloat16"]["elements"] == 20
    assert result["dtype_changed"] is False
    assert fp32.dtype == "torch.float32" and bf16.dtype == "torch.bfloat16"


def test_no_trainable_parameters_is_an_explicit_failure():
    class FrozenPolicy:
        def named_parameters(self):
            return iter(())

    with pytest.raises(ValueError, match="no trainable"):
        parameter_inventory(FrozenPolicy())
