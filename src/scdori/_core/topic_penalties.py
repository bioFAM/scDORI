"""Optional softmax topic-usage penalties."""

import math

import torch


def validate_topic_penalties(config):
    """Reject invalid weights without changing the zero-default objective."""
    for name in ("theta_entropy_weight", "usage_balance_weight"):
        value = getattr(config, name, 0.0)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")


def topic_usage_penalties(theta, config):
    """Return weighted terms for softmax proportions; omit every zero-weight term.

    Entropy is SUMMED over cells. Balance uses
    minibatch mean usage and is multiplied by batch size. It is not the entropy
    of a full-dataset estimate. Changing batch size changes the relative strength
    against reconstruction components that are averaged over cells.
    """
    terms = {}
    entropy_weight = getattr(config, "theta_entropy_weight", 0.0)
    balance_weight = getattr(config, "usage_balance_weight", 0.0)
    if entropy_weight:
        terms["theta_entropy"] = entropy_weight * -(theta * torch.log(theta + 1e-12)).sum()
    if balance_weight:
        usage = theta.mean(dim=0)
        terms["usage_balance"] = balance_weight * theta.shape[0] * (usage * torch.log(usage + 1e-12)).sum()
    return terms
