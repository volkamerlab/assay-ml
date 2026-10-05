"""Regenerate the appendix tables of the revision.

End-to-end: reads the raw per-run ``predictions.csv`` files from ``data/hpc-data``,
computes the size-weighted Fisher-z aggregate correlations per (dataset, fold, method),
caches them as a CSV and renders three LaTeX tables:

A  per-fold aggregate correlations                 -> table_a_<group>.tex
B  paired differences with 95 % CI                 -> table_b_<group>.tex
C  pairwise significance tests (t, Wilcoxon, d)    -> table_c_<group>.tex

The aggregate correlations are computed once; on later runs they are loaded from
``--corrs`` unless ``--force`` is given.

Usage::

    python scripts/apendix_tables/generate.py
    python scripts/apendix_tables/generate.py --groups kinodata --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HPC_DATA = REPO_ROOT / "data" / "hpc-data"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

# --------------------------------------------------------------------------------------
# run inventory
# --------------------------------------------------------------------------------------

# The run directories that back the published numbers. Kept verbatim (including the
# random suffixes of the morgan runs) so the tables stay reproducible even though
# `data/hpc-data` also holds re-runs under the new dataset names.
RUNS = """
kinodata_small_good_allsets_0
kinodata_small_good_allsets_1
kinodata_small_good_allsets_2
kinodata_small_good_allsets_3
kinodata_small_good_allsets_4
kinodata_small_good_ic50_0
kinodata_small_good_ic50_1
kinodata_small_good_ic50_2
kinodata_small_good_ic50_3
kinodata_small_good_ic50_4
kinodata_small_good_sets_0
kinodata_small_good_sets_1
kinodata_small_good_sets_2
kinodata_small_good_sets_3
kinodata_small_good_sets_4
kinodata_good_allsets_0
kinodata_good_allsets_1
kinodata_good_allsets_2
kinodata_good_allsets_3
kinodata_good_allsets_4
kinodata_good_ic50_0
kinodata_good_ic50_1
kinodata_good_ic50_2
kinodata_good_ic50_3
kinodata_good_ic50_4
kinodata_good_sets_0
kinodata_good_sets_1
kinodata_good_sets_2
kinodata_good_sets_3
kinodata_good_sets_4
kinodata_morgan_0_allsets
kinodata_morgan_0_ic50_66f8
kinodata_morgan_0_sets_5c05
kinodata_morgan_1_allsets
kinodata_morgan_1_ic50_5f6e
kinodata_morgan_1_sets_b458
kinodata_morgan_2_allsets
kinodata_morgan_2_ic50_4523
kinodata_morgan_2_sets_0392
kinodata_morgan_3_allsets
kinodata_morgan_3_ic50_47d5
kinodata_morgan_3_sets_e67e
kinodata_morgan_4_allsets
kinodata_morgan_4_ic50_5e6b
kinodata_morgan_4_sets_9751
kinodata_small_allsets_0
kinodata_small_allsets_1
kinodata_small_allsets_2
kinodata_small_allsets_3
kinodata_small_allsets_4
kinodata_small_ic50_0
kinodata_small_ic50_1
kinodata_small_ic50_2
kinodata_small_ic50_3
kinodata_small_ic50_4
kinodata_small_sets_0
kinodata_small_sets_1
kinodata_small_sets_2
kinodata_small_sets_3
kinodata_small_sets_4
landrum_bad_allsets_0
landrum_bad_allsets_1
landrum_bad_allsets_2
landrum_bad_allsets_3
landrum_bad_allsets_4
landrum_bad_ic50_0
landrum_bad_ic50_1
landrum_bad_ic50_2
landrum_bad_ic50_3
landrum_bad_ic50_4
landrum_bad_sets_0
landrum_bad_sets_1
landrum_bad_sets_2
landrum_bad_sets_3
landrum_bad_sets_4
landrum_large_allsets_0
landrum_large_allsets_1
landrum_large_allsets_2
landrum_large_allsets_3
landrum_large_allsets_4
landrum_large_ic50_0
landrum_large_ic50_1
landrum_large_ic50_2
landrum_large_ic50_3
landrum_large_ic50_4
landrum_large_sets_0
landrum_large_sets_1
landrum_large_sets_2
landrum_large_sets_3
landrum_large_sets_4
landrum_morgan_0_allsets
landrum_morgan_0_ic50_402f
landrum_morgan_0_sets_6fb1
landrum_morgan_1_allsets
landrum_morgan_1_ic50_4643
landrum_morgan_1_sets_9ef8
landrum_morgan_2_allsets
landrum_morgan_2_ic50_0bde
landrum_morgan_2_sets_0651
landrum_morgan_3_allsets
landrum_morgan_3_ic50_db6b
landrum_morgan_3_sets_04c3
landrum_morgan_4_allsets
landrum_morgan_4_ic50_af10
landrum_morgan_4_sets_7497
omnivore_morgan_0_ic50_d9d1
omnivore_morgan_1_ic50_ce7d
omnivore_morgan_2_ic50_5bb3
omnivore_morgan_3_ic50_ab79
omnivore_morgan_4_ic50_f970
omnivore_morgan_0_allsets
omnivore_morgan_0_sets_1e02
omnivore_morgan_1_allsets
omnivore_morgan_1_sets_c799
omnivore_morgan_2_allsets
omnivore_morgan_2_sets_481a
omnivore_morgan_3_allsets
omnivore_morgan_3_sets_c87c
omnivore_morgan_4_allsets
omnivore_morgan_4_sets_334c
""".split()  # noqa: SIM905

DATASET_RENAMING = {
    "landrum": "chembl-curated",
    "landrum_large": "chembl-hc",
    "landrum_bad": "chembl-cs",
    "omnivore": "chembl-all",
    "kinodata": "kinodata",
    "kinodata_small": "kinodata-cs",
    "kinodata_good": "kinodata-hc",
    "kinodata_small_good": "kinodata-curated",
}

# Dataset groups, in the row order the tables use.
GROUPS = {
    "kinodata": ["kinodata", "kinodata-cs", "kinodata-hc", "kinodata-curated"],
    "chembl": ["chembl-all", "chembl-cs", "chembl-hc", "chembl-curated"],
}

METHOD_ORDER = ["ic50", "allsets", "sets"]
METHOD_LABELS = {
    "ic50": "direct",
    "allsets": "ranking (inter)",
    "sets": "ranking (intra)",
}

# (left, right) with the one-sided alternative "left < right".
COMPARISONS = [
    ("ic50", "allsets"),
    ("ic50", "sets"),
    ("allsets", "sets"),
]


def parse_run(run: str) -> tuple[str, str, str]:
    """Split a run directory name into (dataset, fold, method)."""
    parts = run.split("_")
    if run.startswith("kinodata_small_good"):
        dataset, fold, method = "_".join(parts[0:3]), parts[4], parts[3]
    else:
        dataset, fold, method = "_".join(parts[0:2]), parts[3], parts[2]
    if dataset.endswith("morgan"):
        dataset = dataset[: -len("_morgan")]
        if fold in {"sets", "allsets", "ic50"}:
            fold, method = method, fold
    return DATASET_RENAMING[dataset], fold, method


# --------------------------------------------------------------------------------------
# aggregate correlations
# --------------------------------------------------------------------------------------


def load_predictions(hpc_data: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for run in RUNS:
        dataset, fold, method = parse_run(run)
        path = hpc_data / run / "predictions.csv"
        if not path.exists():
            missing.append(run)
            continue
        d = pd.read_csv(path)
        d["dataset"] = dataset
        d["fold"] = fold
        d["method"] = method
        frames.append(d)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} run(s) without predictions.csv under {hpc_data}: "
            + ", ".join(missing)
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Size-weighted Fisher-z aggregate of the per-assay Pearson correlations.

    An assay that is degenerate for one (fold, method) combination is dropped for every
    combination of the same dataset that is visited afterwards, so that the three methods
    are compared on as similar an assay set as possible. The groups are iterated in
    sorted (dataset, fold, method) order, which is what the original notebook did and
    what the published numbers were produced with.
    """
    overall = []
    removed_assays: dict[str, set] = {ds: set() for ds in df["dataset"].unique()}

    for (dataset, fold, method), d in df.groupby(
        ["dataset", "fold", "method"], sort=True
    ):
        sizes = []
        pearsons = []
        for assay_id, g in d.groupby("assay_id"):
            if assay_id in removed_assays[dataset]:
                continue
            size = len(g)
            labels = g["label"].values
            predictions = g["prediction"].values
            if size <= 2 or np.std(labels) < 1e-9 or np.std(predictions) < 1e-9:
                removed_assays[dataset].add(assay_id)
                continue
            sizes.append(size)
            pearsons.append(pearsonr(labels, predictions)[0])

        pearsons = np.clip(pearsons, -1.0 + 1e-6, 1.0 - 1e-6)
        weights = np.maximum(np.array(sizes) - 3, 1)
        avg_z = np.average(np.arctanh(pearsons), weights=weights)

        overall.append(
            {
                "dataset": dataset,
                "fold": int(fold),
                "method": method,
                "pearson": np.tanh(avg_z),
            }
        )

    out = pd.DataFrame(overall)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["dataset", "fold", "method"]).reset_index(drop=True)


def get_aggregate_correlations(
    corrs_path: Path, hpc_data: Path, force: bool
) -> pd.DataFrame:
    if corrs_path.exists() and not force:
        print(f"loading aggregate correlations from {corrs_path}")
        out = pd.read_csv(corrs_path)
        out["method"] = pd.Categorical(
            out["method"], categories=METHOD_ORDER, ordered=True
        )
        return out

    print(f"computing aggregate correlations from {hpc_data}")
    out = aggregate_correlations(load_predictions(hpc_data))
    corrs_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(corrs_path, index=False)
    print(f"wrote {corrs_path}")
    return out


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------


def method_matrix(corrs: pd.DataFrame, dataset: str) -> dict[str, np.ndarray]:
    """Per-method vectors of fold-wise aggregate correlations, aligned by fold."""
    d = corrs[corrs["dataset"] == dataset]
    return {
        m: d[d["method"] == m].sort_values("fold")["pearson"].to_numpy()
        for m in METHOD_ORDER
    }


def paired_differences(corrs: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Mean fold-wise difference in aggregate correlation with a paired-t 95 % CI."""
    rows = []
    for dataset in datasets:
        by_method = method_matrix(corrs, dataset)
        for left, right in COMPARISONS:
            diff = by_method[right] - by_method[left]
            n = len(diff)
            mean = diff.mean()
            se = diff.std(ddof=1) / np.sqrt(n)
            half_width = stats.t.ppf(0.975, n - 1) * se
            rows.append(
                {
                    "dataset": dataset,
                    "left": left,
                    "right": right,
                    "n_folds": n,
                    "delta": mean,
                    "ci_lower": mean - half_width,
                    "ci_upper": mean + half_width,
                }
            )
    return pd.DataFrame(rows)


def cohen_d_pooled(x1: np.ndarray, x2: np.ndarray) -> float:
    n1, n2 = len(x1), len(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std


def significance_tests(corrs: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """One-sided paired tests on the Fisher-z transformed aggregate correlations.

    The p-values are FDR-adjusted (Benjamini-Hochberg) over all comparisons of the
    returned table, i.e. per dataset group.
    """
    rows = []
    for dataset in datasets:
        by_method = {k: np.arctanh(v) for k, v in method_matrix(corrs, dataset).items()}
        for left, right in COMPARISONS:
            v1, v2 = by_method[left], by_method[right]

            t_stat, p_t_two_sided = stats.ttest_rel(v1, v2)
            p_t = p_t_two_sided / 2 if t_stat < 0 else 1 - p_t_two_sided / 2
            w_stat, p_w = stats.wilcoxon(v1, v2, alternative="less")

            rows.append(
                {
                    "dataset": dataset,
                    "left": left,
                    "right": right,
                    "t_stat": t_stat,
                    "p_t_raw": p_t,
                    "w_stat": w_stat,
                    "p_w_raw": p_w,
                    "d": cohen_d_pooled(v1, v2),
                }
            )

    res = pd.DataFrame(rows)
    res["p_t_fdr"] = multipletests(res["p_t_raw"], method="fdr_bh")[1]
    res["p_w_fdr"] = multipletests(res["p_w_raw"], method="fdr_bh")[1]
    return res


# --------------------------------------------------------------------------------------
# LaTeX rendering
# --------------------------------------------------------------------------------------


def signed(value: float, decimals: int = 4) -> str:
    """Format a number so that positive values align with negative ones."""
    text = f"{value:.{decimals}f}"
    return text if text.startswith("-") else rf"\phantom{{-}}{text}"


def comparison_label(left: str, right: str) -> str:
    return f"{METHOD_LABELS[left]} $<$ {METHOD_LABELS[right]}"


def table_a(corrs: pd.DataFrame, datasets: list[str], group: str) -> str:
    caption = (
        rf"\caption{{\label{{tab:per-fold-correlations-{group}}}The complete results of "
        rf"aggregate correlations across all {group} data sets and folds.}}"
    )
    header = (
        r"{dataset} & {fold} & {direct} & \makecell[r]{ranking \\ (\textbf{inter}-assay)}"
        r" & \makecell[r]{ranking \\ (\textbf{intra}-assay)} \\"
    )
    lines = [
        r"\begin{table}",
        r"\centering",
        caption,
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for i, dataset in enumerate(datasets):
        d = corrs[corrs["dataset"] == dataset]
        folds = sorted(d["fold"].unique())
        if i:
            lines.append(r"\cline{1-5}")
        for j, fold in enumerate(folds):
            row = d[d["fold"] == fold].set_index("method")["pearson"]
            values = " & ".join(f"{row[m]:.4f}" for m in METHOD_ORDER)
            head = rf"\multirow[t]{{{len(folds)}}}{{*}}{{{dataset}}}" if j == 0 else ""
            lines.append(rf"{head} & {fold} & {values} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_b(diffs: pd.DataFrame, datasets: list[str], group: str) -> str:
    n_folds = int(diffs["n_folds"].iloc[0])
    caption = (
        rf"\caption{{\label{{tab:pairwise-differences-{group}}}Paired differences in "
        "aggregate intra-assay Pearson\n"
        r"correlation between modeling strategies. $\Delta\rho$ is the mean difference "
        f"across the {n_folds}\n"
        r"cross-validation folds; the 95\,\% confidence interval is obtained from a "
        "paired $t$-distribution\n"
        rf"with ${n_folds - 1}$ degrees of freedom.}}"
    )
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        caption,
        r"\begin{tabular}{llrc}",
        r"\toprule",
        r"{dataset} & {comparison} & {$\Delta\rho$} & {95\,\% CI} \\",
        r"\midrule",
    ]

    for i, dataset in enumerate(datasets):
        d = diffs[diffs["dataset"] == dataset]
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multirow[t]{{{len(d)}}}{{*}}{{{dataset}}}")
        for _, r in d.iterrows():
            label = f"{METHOD_LABELS[r['right']]} $-$ {METHOD_LABELS[r['left']]}"
            ci = rf"$[{signed(r['ci_lower'])},\ {r['ci_upper']:.4f}]$"
            lines.append(rf" & {label} & {r['delta']:.4f} & {ci} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_c(tests: pd.DataFrame, datasets: list[str], group: str) -> str:
    caption = (
        rf"\caption{{\label{{tab:significance-{group}}}Pairwise comparisons of modeling "
        r"strategies on Fisher-aggregate Pearson correlations. $p$-values are "
        r"FDR-adjusted; effect sizes are reported via Cohen's $d$.}"
    )
    header = (
        r"{dataset} & {comparison} & {statistic} & {$p$-value} & {statistic} & "
        r"{$p$-value} & {Cohen's $d$} \\"
    )
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        caption,
        r"\begin{tabular}{ll@{}S@{}@{}S@{}@{}S@{}@{}S@{}@{}S@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{t-test} & \multicolumn{2}{c}{Wilcoxon} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        header,
        r"\midrule",
    ]

    for i, dataset in enumerate(datasets):
        d = tests[tests["dataset"] == dataset]
        if i:
            lines.append(r"\midrule")
        for j, (_, r) in enumerate(d.iterrows()):
            head = rf"\multirow[t]{{{len(d)}}}{{*}}{{{dataset}}}" if j == 0 else ""
            values = " & ".join(
                [
                    f"{r['t_stat']:.4f}",
                    f"{r['p_t_fdr']:.4f}",
                    f"{r['w_stat']:.4f}",
                    f"{r['p_w_fdr']:.4f}",
                    f"{r['d']:.4f}",
                ]
            )
            lines.append(
                rf"{head} & {comparison_label(r['left'], r['right'])} & {values} \\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hpc-data",
        type=Path,
        default=DEFAULT_HPC_DATA,
        help="directory holding the <run>/predictions.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="where to write the outputs",
    )
    parser.add_argument(
        "--corrs",
        type=Path,
        default=None,
        help="aggregate correlations CSV (default: <out-dir>/aggregate_correlations.csv)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[*GROUPS, "all"],
        default=list(GROUPS),
        help="dataset groups to emit one set of tables for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute the aggregate correlations even if the CSV exists",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    corrs_path = args.corrs or out_dir / "aggregate_correlations.csv"

    corrs = get_aggregate_correlations(corrs_path, args.hpc_data, args.force)

    for group in args.groups:
        datasets = (
            [ds for g in GROUPS.values() for ds in g]
            if group == "all"
            else GROUPS[group]
        )
        missing = [ds for ds in datasets if ds not in set(corrs["dataset"])]
        if missing:
            raise ValueError(f"group {group!r} is missing datasets: {missing}")

        diffs = paired_differences(corrs, datasets)
        tests = significance_tests(corrs, datasets)

        for name, source in [
            ("a", table_a(corrs, datasets, group)),
            ("b", table_b(diffs, datasets, group)),
            ("c", table_c(tests, datasets, group)),
        ]:
            path = out_dir / f"table_{name}_{group}.tex"
            path.write_text(source)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
