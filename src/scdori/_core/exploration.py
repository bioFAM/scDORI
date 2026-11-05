"""Choose what to show in example plots from the fitted results instead of fixed topic numbers.

Topic numbers change between fits, so the notebooks pick the topics, cell types, TFs and genes to plot
with these helpers. They only read results; nothing here changes a model or a network.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_palette(palette, key="celltype"):
    """Label → colour mapping from a dict or a JSON file.

    The JSON may be a flat ``{"label": "#rrggbb"}`` object or contain it under ``"<key>_colors"``.
    """
    if palette is None:
        return {}
    if not isinstance(palette, dict):
        palette = json.loads(Path(palette).read_text())
        palette = palette.get(f"{key}_colors", palette)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in palette.items()):
        raise ValueError("A palette must map each label to a colour string")
    return dict(palette)


def read_gene_sets(path, name_contains=None):
    """Gene sets as a long table with ``geneset`` and ``genesymbol`` columns.

    Reads a ``.gmt`` file, such as an MSigDB download, or a tab-separated table with those two columns.
    GMT fields are split on tabs only, because set names from some resources contain spaces.
    ``name_contains`` keeps the sets whose name contains that text; if none does, this raises an error.
    """
    path = Path(path)
    if path.suffix.lower() == ".gmt":
        with path.open() as handle:
            fields = [line.rstrip("\n").split("\t") for line in handle]
        rows = [(field[0], gene) for field in fields for gene in field[2:] if gene]
        sets = pd.DataFrame(rows, columns=["geneset", "genesymbol"])
    else:
        sets = pd.read_csv(path, sep="\t")
        if not {"geneset", "genesymbol"}.issubset(sets):
            raise ValueError("A gene-set table needs geneset and genesymbol columns")
        sets = sets[["geneset", "genesymbol"]]
    if name_contains is not None:
        sets = sets[sets["geneset"].str.contains(name_contains, regex=False)]
        if sets.empty:
            raise ValueError(f"No gene set in {path} has a name containing {name_contains!r}")
    return sets.drop_duplicates().reset_index(drop=True)


def apply_label_colors(adata, key="celltype", palette=None):
    """Set ``adata.uns[f"{key}_colors"]`` (used by scanpy plots) and return the label → colour mapping.

    Colours come from ``palette`` (dict or JSON file), else from colours already stored for every label in
    ``adata``, else a husl palette. Labels missing from the palette get husl colours and are reported.
    """
    import seaborn as sns

    column = adata.obs[key]
    labels = sorted(column.astype(str).unique())
    known = load_palette(palette, key)
    if not known and f"{key}_colors" in adata.uns and hasattr(column, "cat"):
        stored = list(adata.uns[f"{key}_colors"])
        if len(stored) == len(column.cat.categories):
            known = dict(zip(map(str, column.cat.categories), stored, strict=True))
    missing = [label for label in labels if label not in known]
    if palette is not None and missing:
        print(f"No palette colour for {len(missing)} label(s), using generated colours: {missing}")
    colors = {**dict(zip(missing, sns.color_palette("husl", len(missing)).as_hex(), strict=True)), **known}
    adata.obs[key] = pd.Categorical(column.astype(str), categories=labels)
    adata.uns[f"{key}_colors"] = [colors[label] for label in labels]
    return {label: colors[label] for label in labels}


def top_topics_for_group(topic_by_group, group, n=2):
    """The ``n`` topics (row names) with the highest mean use in one group (column), strongest first."""
    return topic_by_group[group].sort_values(ascending=False).head(n).index.tolist()


def top_groups_for_topics(topic_by_group, topics, n=2):
    """The ``n`` groups using each topic most, in topic order and without repeats."""
    groups = [g for t in topics for g in topic_by_group.loc[t].sort_values(ascending=False).head(n).index]
    return list(dict.fromkeys(groups))


def example_groups(topic_by_group, n=2):
    """``n`` groups whose strongest topic is used most, each with a different strongest topic."""
    chosen, used = [], set()
    for group in topic_by_group.max(axis=0).sort_values(ascending=False).index:
        topic = topic_by_group[group].idxmax()
        if topic not in used:
            chosen.append(group)
            used.add(topic)
        if len(chosen) == n:
            break
    return chosen


def tf_topic_strength(grn, tf_names, topics, sign=1):
    """Sum link magnitudes for one sign (topics × TFs); use sign=-1 for negative repressor links."""
    topics = list(topics)
    if sign not in (1, -1):
        raise ValueError("sign must be 1 for activators or -1 for signed repressors")
    return pd.DataFrame((sign * np.asarray(grn)[topics]).clip(min=0).sum(axis=2), index=topics, columns=list(tf_names))


def example_tf(grn, tf_names, topics, n_topics=2):
    """The TF with the strongest network in its ``n_topics``-th best topic.

    With ``n_topics=2`` this favours a TF that is strong in two topics, so its targets can be compared.
    """
    strength = tf_topic_strength(grn, tf_names, topics)
    n = min(n_topics, len(strength))
    nth = strength.apply(lambda s: s.nlargest(n).iloc[-1])
    if not (nth > 0).any():
        nth = strength.max(axis=0)
    if not (nth > 0).any():
        raise ValueError("No TF has links in these topics")
    return nth.idxmax()


def top_topics_for_tf(grn, tf_names, tf, topics, n=2, sign=1):
    """Topics with the largest link magnitude; sign=-1 selects negative repressor links."""
    strength = tf_topic_strength(grn, tf_names, topics, sign=sign)[tf]
    return strength[strength > 0].nlargest(n)


def top_targets(grn, tf_names, gene_names, tf, topic, n=30, sign=1):
    """Strongest target magnitudes; sign=-1 selects negative repressor links, separately."""
    if sign not in (1, -1):
        raise ValueError("sign must be 1 for activators or -1 for signed repressors")
    weights = pd.Series(sign * np.asarray(grn)[topic, list(tf_names).index(tf)], index=list(gene_names))
    return weights[weights > 0].nlargest(n)


def top_binding_tfs(topic_peak, binding, tf_names, n=1):
    """For each column of ``topic_peak`` (peaks × topics), the ``n`` TFs whose binding scores

    (peaks × TFs) follow that topic's peak weights most closely (Pearson correlation).
    """
    topic_peak, binding = pd.DataFrame(topic_peak), np.asarray(binding)
    b = (binding - binding.mean(0)) / np.where(binding.std(0) > 0, binding.std(0), np.inf)
    out = {}
    for topic in topic_peak:
        t = topic_peak[topic].to_numpy()
        t = (t - t.mean()) / (t.std() or np.inf)
        corr = pd.Series(t @ b / len(t), index=list(tf_names))
        out[topic] = corr.nlargest(n).index.tolist()
    return out


def top_linked_peaks(gene_peak, gene_names, peak_names, gene, n=20):
    """The ``n`` peaks most strongly linked to ``gene`` in a genes × peaks link matrix (links > 0)."""
    links = pd.Series(np.asarray(gene_peak)[list(gene_names).index(gene)], index=list(peak_names))
    return links[links > 0].nlargest(n)


def linked_peak_table(
    gene_peak, gene_names, peak_names, gene, accessibility, groups,
    n=20, min_link_weight=0.0, min_mean_accessibility=0.0, topic_peak_weights=None,
):
    """Rank linked regions with accessibility support in the requested cell types.

    ``accessibility`` is peaks × groups, containing nonnegative mean normalized
    counts, not z-scores. Each group has equal weight in the mean. Keep regions
    strictly above both cutoffs, then rank by link weight. These are descriptive
    filters, not significance tests or evidence of enhancer function.

    Optionally pass a peak-labelled Series from the normalized topic–peak
    matrix for one topic. The table then ranks positive ``link_weight *
    topic_peak_weight`` contributions, before gene-score standardization.
    Accessibility is still a filter, not a multiplier or a refitted link.
    Existing calls without a topic retain their original selection and columns.
    """
    groups = list(dict.fromkeys(groups))
    if not groups or n < 1:
        raise ValueError("Choose at least one cell type and a positive number of regions")
    if not np.isfinite([min_link_weight, min_mean_accessibility]).all() or min(
        min_link_weight, min_mean_accessibility
    ) < 0:
        raise ValueError("Region cutoffs must be finite and nonnegative")
    values = accessibility.loc[list(peak_names), groups]
    if not np.isfinite(values.to_numpy()).all() or (values.to_numpy() < 0).any():
        raise ValueError("Accessibility must contain nonnegative means, not centred scores")
    links = pd.Series(np.asarray(gene_peak)[list(gene_names).index(gene)], index=list(peak_names))
    table = pd.DataFrame({"link_weight": links, "mean_accessibility": values.mean(axis=1)})
    keep = (table.link_weight > min_link_weight) & (table.mean_accessibility > min_mean_accessibility)
    rank_by = "link_weight"
    if topic_peak_weights is not None:
        if not isinstance(topic_peak_weights, pd.Series) or not topic_peak_weights.index.is_unique:
            raise ValueError("Topic peak weights must be a Series with unique peak names")
        weights = topic_peak_weights.reindex(list(peak_names))
        if not np.isfinite(weights.to_numpy()).all() or (weights < 0).any() or (weights > 1).any():
            raise ValueError("Topic peak weights must cover all peaks with finite values between 0 and 1")
        table["topic_peak_weight"] = weights
        table["topic_contribution"] = links * weights
        keep &= table.topic_contribution > 0
        rank_by = "topic_contribution"
    return table.loc[keep].sort_values(rank_by, ascending=False, kind="stable").head(n)


def regions_for_link(model, tf, gene, topic, tf_names, gene_names, peak_names, sign="activator", n=None,
                     atac_grn=None, gate=None, final_grn=None):
    """Regions behind one TF→gene link in one topic, term by term.

    The ATAC-derived score summed by ``compute_atac_grn_*_with_significance`` is a link-weighted
    average over the gene's linked regions::

        activator: sum_p acc_p * link_p * binding_p / sum_p link_p
        repressor: sum_p (1 / acc_p) * link_p * binding_p / sum_p link_p   (binding_p <= 0)

    ``acc_p`` is the region's weight in the topic (softmax of the topic–peak decoder), ``link_p``
    the peak–gene weight and ``binding_p`` the predicted TF binding. Repressive support therefore
    comes mainly from regions that are closed in the topic. Each returned row is one linked region:
    ``contribution`` is its term in the sum and ``share`` its fraction of the score.

    Optional arrays add what happens to the link afterwards: ``atac_grn`` (the saved,
    background-filtered ATAC network for this sign) shows whether the link passed the
    permutation filter, ``gate`` (topics × TFs) the TF-expression multiplier and ``final_grn``
    the final signed network value. The phase-2 weight is read from the model.

    These numbers describe what the model used; they do not show that a region is a
    functional enhancer or that the TF binds it.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        Regions ranked by absolute contribution, and a one-link summary.
    """
    import torch

    if sign not in ("activator", "repressor"):
        raise ValueError("sign must be 'activator' or 'repressor'")
    tf_names, gene_names, peak_names = list(tf_names), list(gene_names), list(peak_names)
    if tf not in tf_names or gene not in gene_names:
        raise ValueError(f"{tf!r} must be a fitted TF and {gene!r} a fitted gene")
    if type(topic) is not int or not 0 <= topic < model.num_topics:
        raise ValueError(f"topic must be an integer between 0 and {model.num_topics - 1}")
    if n is not None and n < 1:
        raise ValueError("n must be positive or None")
    t, g = tf_names.index(tf), gene_names.index(gene)
    with torch.no_grad():
        acc = torch.softmax(model.topic_peak_decoder[topic].detach().float(), dim=0).cpu().numpy()
        link = (model.gene_peak_factor_learnt[g] * model.gene_peak_factor_fixed[g]).detach().float().cpu().numpy()
        matrix = model.tf_binding_matrix_activator if sign == "activator" else model.tf_binding_matrix_repressor
        binding = matrix[:, t].detach().float().cpu().numpy()
        weights = model.tf_gene_topic_activator_grn if sign == "activator" else model.tf_gene_topic_repressor_grn
        phase2_weight = float(torch.relu(weights[topic, t, g]).item())
    if len(peak_names) != acc.size:
        raise ValueError("peak_names must match the model's peaks")
    term = acc if sign == "activator" else 1 / (acc + 1e-20)
    linked = np.flatnonzero(link > 0)
    denominator = link.sum() + 1e-8
    contribution = term[linked] * link[linked] * binding[linked] / denominator
    score = contribution.sum()
    rank = np.searchsorted(np.sort(acc), acc[linked], side="left") / acc.size
    regions = pd.DataFrame({
        "link_weight": link[linked],
        "topic_accessibility": acc[linked],
        "accessibility_percentile_in_topic": 100 * rank,
        "tf_binding": binding[linked],
        "contribution": contribution,
        "share": contribution / score if score != 0 else 0.0,
    }, index=pd.Index(np.asarray(peak_names)[linked], name="region"))
    regions = regions.iloc[np.argsort(-np.abs(contribution), kind="stable")]
    summary = {"tf": tf, "gene": gene, "topic": topic, "sign": sign, "linked_regions": int(linked.size),
               "regions_with_binding": int((binding[linked] != 0).sum()), "atac_score": float(score),
               "phase2_weight": phase2_weight}
    if atac_grn is not None:
        value = float(np.asarray(atac_grn)[topic, t, g])
        summary.update(atac_network_value=value, passed_background_filter=value != 0)
    if gate is not None:
        gate = gate.detach().cpu().numpy() if hasattr(gate, "detach") else np.asarray(gate)
        summary["tf_expression_gate"] = float(gate[topic, t])
    if final_grn is not None:
        summary["final_network_value"] = float(np.asarray(final_grn)[topic, t, g])
    return regions.head(n) if n else regions, pd.Series(summary, name=f"{tf}→{gene}, Topic {topic}")


def normalized_counts_for_plot(adata, target_sum=None):
    """Return an independent normalized copy for plots; preserve X and all count layers.

    ``target_sum=None`` scales each cell to the median library size, as ``sc.pp.normalize_total`` does by default.
    """
    import scanpy as sc

    result = adata.copy()
    result.X = adata.layers.get("counts", adata.X).copy()
    sc.pp.normalize_total(result, target_sum=target_sum)
    return result


def choose_example_topics(topic_by_celltype, celltypes=None, topics=None, n=2):
    """Select worked examples without changing the table used to scale scores.

    Empty selections choose examples from the fitted data. Explicit topics must
    belong to the retained analysis topics; they are never substituted silently.
    """
    groups = list(dict.fromkeys(celltypes or example_groups(topic_by_celltype, n=2)))
    missing = [group for group in groups if group not in topic_by_celltype.columns]
    if missing:
        raise ValueError(f"Example cell types are absent from the data: {missing}")
    chosen = None
    if topics:
        if any(type(topic) is not int or topic < 0 for topic in topics) or len(set(topics)) != len(topics):
            raise ValueError("Example topics must be unique nonnegative integers")
        chosen = [f"Topic_{topic}" for topic in topics]
        missing = [topic for topic in chosen if topic not in topic_by_celltype.index]
        if missing:
            raise ValueError(
                f"Example topics are outside the retained analysis topics: {missing}. "
                "Check the fitted topic IDs and downstream topic_activity_threshold/exclude_topics."
            )
    return {group: list(chosen) if chosen is not None else top_topics_for_group(topic_by_celltype, group, n=n)
            for group in groups}
