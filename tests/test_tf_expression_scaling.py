import numpy as np
import pytest
import torch

from scdori._core.config import TrainConfig
from scdori._core.train_grn import scale_topic_tf_expression


def test_explicit_within_topic_preserves_historical_operation():
    torch.manual_seed(53)
    expression = torch.rand(40, 300)
    expected = (expression - expression.min(1, keepdim=True).values) / (
        expression.max(1, keepdim=True).values - expression.min(1, keepdim=True).values + 1e-9
    )
    expected[expected < 0.1] = 0
    assert torch.equal(scale_topic_tf_expression(expression, 0.1, "within_topic"), expected)


def test_new_default_retains_low_expression_specific_tf():
    config = TrainConfig()
    assert (config.tf_expression_scaling, config.tf_expression_clamp) == ("mean_within_across", 0.2)
    values = np.array([[100, 8, 0], [120, 1, 0], [110, 1, 0]], dtype=np.float32)
    within = (values - values.min(1, keepdims=True)) / (np.ptp(values, axis=1, keepdims=True) + 1e-9)
    across = (values - values.min(0, keepdims=True)) / (np.ptp(values, axis=0, keepdims=True) + 1e-9)
    expected = (within + across) / 2
    expected[expected < 0.2] = 0
    actual = scale_topic_tf_expression(torch.from_numpy(values), config.tf_expression_clamp)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7)
    assert actual[0, 1] > 0.5
    assert torch.count_nonzero(actual[:, 2]) == 0


@pytest.mark.parametrize("cutoff", [0.0, 0.1, 0.2, 0.3, 1.0])
def test_mean_expression_scaling_matches_independent_numpy_calculation(cutoff):
    values = np.array([[100, 8, 0, 3], [120, 1, 0, 3], [110, 1, 0, 3]], dtype=np.float32)
    within = (values - values.min(1, keepdims=True)) / (np.ptp(values, axis=1, keepdims=True) + 1e-9)
    across = (values - values.min(0, keepdims=True)) / (np.ptp(values, axis=0, keepdims=True) + 1e-9)
    expected = (within + across) / 2
    expected[expected < cutoff] = 0
    expression = torch.from_numpy(values.copy())
    actual = scale_topic_tf_expression(expression, cutoff, "mean_within_across")
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(expression.numpy(), values)
    assert (actual[:, 2] == 0).all()  # no expression cannot be revived by specificity


def test_low_abundance_specific_tf_can_pass_mean_rule_and_cutoff_is_strict():
    expression = torch.tensor([[100.0, 8, 0], [120, 1, 0], [110, 1, 0]])
    assert scale_topic_tf_expression(expression, 0.1, "within_topic")[0, 1] == 0
    assert scale_topic_tf_expression(expression, 0.1, "mean_within_across")[0, 1] > 0.5
    boundary = torch.tensor([[0.0, 0.5, 1.0]])
    assert scale_topic_tf_expression(boundary, 0.5, "within_topic")[0, 1] == 0.5
    assert scale_topic_tf_expression(boundary, 0.25, "mean_within_across")[0, 1] == 0.25


@pytest.mark.parametrize("values", [[[1.0]], [[0.0, 0.0]], [[3.0, 3.0], [3.0, 3.0]], [[2.0], [3.0]]])
def test_expression_scaling_constant_and_singleton_axes_are_finite(values):
    result = scale_topic_tf_expression(torch.tensor(values), 0, "mean_within_across")
    assert torch.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    if len(set(sum(values, []))) == 1:
        assert (result == 0).all()


def test_expression_scaling_rejects_invalid_settings_and_counts():
    with pytest.raises(ValueError, match="Unknown tf_expression_scaling"):
        scale_topic_tf_expression(torch.ones(2, 2), 0.1, "typo")
    for cutoff in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="between 0 and 1"):
            scale_topic_tf_expression(torch.ones(2, 2), cutoff)
    for values in [torch.ones(3), torch.empty(0, 2)]:
        with pytest.raises(ValueError, match="nonempty topic-by-TF"):
            scale_topic_tf_expression(values, 0.1)
    for values in [torch.tensor([[-1.0, 0]]), torch.tensor([[float("nan"), 0]])]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            scale_topic_tf_expression(values, 0.1)
