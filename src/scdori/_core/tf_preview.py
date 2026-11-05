"""Inspect TFs admitted by the phase-2 expression rule before fitting networks."""
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
import torch


@dataclass
class TFExpressionPreview:
    weights: pd.DataFrame
    excluded_topics: list
    # Measured-expression mode only: share of each topic's top cells with RNA, and scores before that rule.
    detection: pd.DataFrame | None = None
    scores_before_detection: pd.DataFrame | None = None
    min_detection: float = 0.0
    cells_per_topic: int = 0

    @property
    def removed(self):
        """One row per TF left out of a topic because its RNA was seen in too few of the top cells."""
        columns = ["Topic", "TF", "Top cells with RNA", "Share of top cells", "Score without this rule"]
        if self.detection is None:
            return pd.DataFrame(columns=columns)
        scores, detection = self.scores_before_detection.to_numpy(), self.detection.to_numpy()
        rows = [{"Topic": self.weights.index[k], "TF": self.weights.columns[j],
                 "Top cells with RNA": round(detection[k, j] * self.cells_per_topic),
                 "Share of top cells": detection[k, j], "Score without this rule": scores[k, j]}
                for k, j in np.argwhere((scores > 0) & (self.weights.to_numpy() == 0))]
        return pd.DataFrame(rows, columns=columns)

    def look_up(self, tf):
        """Show why one TF is kept or left out in each topic."""
        if tf not in self.weights.columns:
            raise ValueError(f"{tf} is not among the fitted TFs. Add it to tfs_user and repeat preprocessing and training.")
        table = pd.DataFrame({"Phase 2 score": self.weights[tf]})
        reason = pd.Series("Included", index=table.index)
        if self.detection is not None:
            table["Share of top cells with RNA"] = self.detection[tf]
            reason[(self.scores_before_detection[tf] > 0) & (self.weights[tf] == 0)] = "Left out: seen in too few top cells"
            reason[self.scores_before_detection[tf] == 0] = "Left out: below the expression cutoff"
        else:
            reason[self.weights[tf] == 0] = "Left out: below the expression cutoff"
        reason[reason.index.isin(self.excluded_topics)] = "Topic skipped in phase 2"
        table["Status"] = reason
        return table

    @property
    def summary(self):
        removed = self.removed["Topic"].value_counts()
        rows = []
        for topic, values in self.weights.iterrows():
            names = values[values > 0].sort_values(ascending=False, kind="stable").index
            row = {"Topic": topic, "TFs passing the rule": len(names),
                   "First 10 TFs by expression score": ", ".join(names[:10])}
            if self.detection is not None:
                row["Left out: seen in too few cells"] = int(removed.get(topic, 0))
            row["Phase 2"] = "Skipped topic" if topic in self.excluded_topics else "Included"
            rows.append(row)
        return pd.DataFrame(rows).set_index("Topic")

    def show(self):
        """Show a compact summary and expandable complete lists in Jupyter."""
        from IPython.display import HTML, display

        # Keep all ten names visible without changing global pandas display options.
        table = self.summary.to_html(escape=True)
        display(HTML('<div class="scdori-tf-preview">'
                     '<style>.scdori-tf-preview td {white-space: normal; '
                     'overflow-wrap: anywhere; max-width: 32em; text-align: left;}</style>'
                     + table + '</div>'))
        blocks = []
        for topic, values in self.weights.iterrows():
            names = values[values > 0].sort_values(ascending=False, kind="stable").index
            text = ", ".join(map(str, names)) or ("Topic skipped in phase 2." if topic in self.excluded_topics else "No TFs pass this rule.")
            blocks.append(f"<details><summary>{escape(str(topic))}: {len(names)} TFs — full list</summary><p>{escape(text)}</p></details>")
        display(HTML("\n".join(blocks)))
        removed = self.removed
        if self.min_detection > 0 and len(removed):
            lost = sorted(set(removed["TF"]) - set(self.weights.columns[(self.weights > 0).any()]))
            cells = int(np.ceil(self.min_detection * self.cells_per_topic - 1e-9))
            text = (f"Left out because their RNA was seen in fewer than {self.min_detection * 100:g}% of a topic's top "
                    f"{self.cells_per_topic} cells (fewer than {cells} cells): {len(removed)} TF–topic pairs, "
                    f"{removed['TF'].nunique()} TFs.")
            if lost:
                text += f" No longer used in any topic: {', '.join(lost)}."
            display(HTML(f"<p>{escape(text)}</p><details><summary>Every TF left out for low detection</summary>"
                         + removed.to_html(index=False, escape=True) + "</details>"))


def preview_phase2_tfs(model, device, train_loader, rna, atac, num_cells, tf_indices, onehot_batch, config):
    """Apply the configured expression rule and topic selection without altering the fit.

    Use the same training population, a deterministic loader and eval-mode encoder.
    Keep RNG state, module modes, weights and GRN selection unchanged. An unfrozen
    phase 2 can subsequently change the expression values; this is its starting preview.
    """
    from .train_grn import (get_latent_topics, scale_topic_tf_expression, topic_tf_detection_filter,
                            topic_tf_expression_means)
    from .topic_selection import resolve_grn_topics, sequential_loader

    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    modes = [(module, module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            loader = sequential_loader(train_loader)
            theta = ids = selected_cells = detection = None
            if config.tf_expression_mode in ("True", "actual") or config.grn_topic_selection == "live":
                theta, ids = get_latent_topics(model, device, loader, rna, atac, num_cells,
                                              tf_indices, onehot_batch, return_indices=True)
            if config.tf_expression_mode in ("True", "actual"):
                means, selected_cells = topic_tf_expression_means(theta, ids, rna, tf_indices, config.cells_per_topic)
                values = torch.as_tensor(means, device=device, dtype=model.topic_tf_decoder.dtype)
            elif config.tf_expression_mode == "latent":
                values = torch.softmax(model.topic_tf_decoder.detach(), dim=1)
            else:
                raise ValueError(f"Unknown tf_expression_mode: {config.tf_expression_mode!r}")
            values = scale_topic_tf_expression(values, config.tf_expression_clamp, config.tf_expression_scaling)
            scores = values.clone()
            if selected_cells is not None:
                detection, keep = topic_tf_detection_filter(selected_cells, rna, tf_indices, config)
                values = values * torch.as_tensor(keep, device=device, dtype=values.dtype)
            selected = resolve_grn_topics(model, config, theta)
            if selected is not None:
                selected = np.asarray(selected)
                if (selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected)
                        or len(np.unique(selected)) != len(selected) or selected.min() < 0
                        or selected.max() >= model.num_topics):
                    raise ValueError("Choose unique integer phase-2 topic IDs within the fitted range")
            excluded = [] if selected is None else sorted(set(range(model.num_topics)) - set(selected))
            values[excluded] = 0
            scores[excluded] = 0
            labels = {"index": [f"Topic_{k}" for k in range(model.num_topics)],
                      "columns": rna.var_names[np.asarray(tf_indices)]}
            frame = pd.DataFrame(values.cpu().numpy(), **labels)
            before = pd.DataFrame(scores.cpu().numpy(), **labels)
    finally:
        for module, training in modes:
            module.training = training
    if detection is None:
        return TFExpressionPreview(frame, [f"Topic_{k}" for k in excluded])
    return TFExpressionPreview(frame, [f"Topic_{k}" for k in excluded], pd.DataFrame(detection, **labels), before,
                               getattr(config, "tf_expression_min_detection", 0.0), selected_cells.shape[0])
