"""
Auto-scan results/ and produce:
  - comparison.md  (markdown table)
  - LaTeX snippet  (printed to stdout, ready to paste into a paper)

Groups JSONL files by stripping trailing _offNNN shards, merges records
(dedup by instance_id, last shard wins), and computes stats per dataset.

Usage:
    python bench/compare_results.py
"""

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_MD   = RESULTS_DIR / "comparison.md"

TIMEOUT_S = 120.0

METHOD_LABELS = {
    "lave":              "LAVE",
    "dgrammar_v2_async": "Dgrammar",
    "dgrammar_dp":       "DPGrammar",
}
METHOD_ORDER = ["lave", "dgrammar_v2_async", "dgrammar_dp"]


# ── helpers ────────────────────────────────────────────────────────────────────

def _base_name(stem: str) -> str:
    return re.sub(r"_off\d+$", "", stem)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_all() -> dict[str, dict[str, dict]]:
    """Returns { base_name -> { instance_id -> record } }"""
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        base = _base_name(path.stem)
        for rec in load_jsonl(path):
            iid = rec.get("instance_id")
            if iid:
                groups[base][iid] = rec
    return groups


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def compute_stats(records: list[dict], benchmark_total: int) -> dict:
    n = len(records)
    skipped = benchmark_total - n

    times     = [r["time_taken"] for r in records if "time_taken" in r]
    resamples = [r["resamples"]  for r in records if "resamples"  in r]

    valid_count  = sum(1 for r in records if r.get("valid"))
    timeout_count = sum(1 for t in times if t > TIMEOUT_S)

    def _pct(x, total=n):
        return x / total if total else float("nan")

    def _stat(lst, fn):
        clean = [x for x in lst if x is not None and not math.isnan(x)]
        return fn(clean) if clean else float("nan")

    def _percentile(lst, p):
        clean = sorted(x for x in lst if x is not None and not math.isnan(x))
        if not clean:
            return float("nan")
        idx = (len(clean) - 1) * p / 100
        lo, hi = int(idx), min(int(idx) + 1, len(clean) - 1)
        return clean[lo] + (clean[hi] - clean[lo]) * (idx - lo)

    return {
        "benchmark_total": benchmark_total,
        "skipped":         skipped,
        "n":               n,
        "valid_count":     valid_count,
        "valid_rate":      _pct(valid_count),
        "timeout_count":   timeout_count,
        "mean_resamples":  _stat(resamples, lambda l: sum(l) / len(l)),
        "mean_time_s":     _stat(times, lambda l: sum(l) / len(l)),
        "median_time_s":   _stat(times, statistics.median),
        "p95_time_s":      _percentile(times, 95),
        "max_time_s":      _stat(times, max),
    }


# ── formatting ─────────────────────────────────────────────────────────────────

def _fmt(val, fmt: str, fallback: str = "—") -> str:
    v = _safe(val)
    if v is None:
        return fallback
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return fallback


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
            for r in rows]
    return "\n".join([head, sep] + body)


def latex_value_row(label: str, values: list[str], bold_idx: int | None = None) -> str:
    """Return one LaTeX table row with optional bolding of the best column."""
    cells = []
    for i, v in enumerate(values):
        cells.append(f"\\textbf{{{v}}}" if i == bold_idx else v)
    return f"{label:<40} & {' & '.join(cells)} \\\\"


# ── per-dataset comparison ─────────────────────────────────────────────────────

def compare_dataset(
    dataset: str,
    bases: list[str],
    groups: dict[str, dict[str, dict]],
    group_method: dict[str, str],
) -> tuple[list[str], list[str]]:
    """
    Returns (md_lines, latex_lines) for one dataset.
    `bases` is already sorted in display order.
    """
    labels = [METHOD_LABELS.get(group_method[b], group_method[b]) for b in bases]

    # benchmark_total = union of all instance IDs loaded for this dataset's methods
    all_ids_union: set[str] = set()
    for b in bases:
        all_ids_union |= groups[b].keys()
    benchmark_total = len(all_ids_union)

    # evaluated = intersection (fair comparison set)
    matched_ids: set[str] = set(groups[bases[0]].keys())
    for b in bases[1:]:
        matched_ids &= groups[b].keys()

    print(f"  [{dataset}]  benchmark={benchmark_total}  evaluated={len(matched_ids)}"
          f"  skipped={benchmark_total - len(matched_ids)}  groups={len(bases)}")

    stats: dict[str, dict] = {}
    for b in bases:
        recs = [r for r in groups[b].values() if r["instance_id"] in matched_ids]
        stats[b] = compute_stats(recs, benchmark_total)

    # ── markdown ──────────────────────────────────────────────────────────────
    md: list[str] = []
    md.append(f"## Dataset: {dataset}")
    md.append("")

    metric_rows = [
        ("Benchmark total",         "benchmark_total", "{:.0f}"),
        ("Skipped (grammar-invalid)","skipped",         "{:.0f}"),
        ("Evaluated (n)",            "n",               "{:.0f}"),
        ("Valid (count)",            "valid_count",     "{:.0f}"),
        ("Validity (%)",             "valid_rate",      "{:.1%}"),
        (f"Timeouts (>{TIMEOUT_S:.0f}s)", "timeout_count", "{:.0f}"),
        ("Mean resamples",           "mean_resamples",  "{:.2f}"),
        ("Mean time (s)",            "mean_time_s",     "{:.2f}"),
        ("Median time (s)",          "median_time_s",   "{:.2f}"),
        ("P95 time (s)",             "p95_time_s",      "{:.2f}"),
        ("Max time (s)",             "max_time_s",      "{:.2f}"),
    ]

    headers = ["Metric"] + labels
    rows = []
    for display, key, fmt in metric_rows:
        row = [display] + [_fmt(stats[b].get(key), fmt) for b in bases]
        rows.append(row)
    md.append(md_table(headers, rows))
    md.append("")

    # ── per-instance agreement ────────────────────────────────────────────────
    rec_lookup = {b: {r["instance_id"]: r for r in groups[b].values()
                      if r["instance_id"] in matched_ids}
                  for b in bases}

    def sym(v):
        return "✓" if v is True else ("✗" if v is False else "—")

    agreement: dict[tuple, list[str]] = defaultdict(list)
    for iid in sorted(matched_ids):
        pat = tuple(rec_lookup[b].get(iid, {}).get("valid") for b in bases)
        agreement[pat].append(iid)

    md.append("### Per-instance Validity Agreement")
    md.append("")
    md.append(f"Each row shows a validity pattern across methods ({', '.join(labels)}).")
    md.append("")
    agree_rows = []
    for pat, ids in sorted(agreement.items(), key=lambda x: -len(x[1])):
        examples = ", ".join(ids[:3]) + ("…" if len(ids) > 3 else "")
        agree_rows.append([
            "  ".join(sym(v) for v in pat),
            str(len(ids)),
            f"{len(ids) / len(matched_ids):.1%}" if matched_ids else "—",
            examples,
        ])
    md.append(md_table(["Pattern (" + " / ".join(labels) + ")", "Count", "%", "Example IDs"],
                       agree_rows))
    md.append("")

    # Disagreements
    if len(bases) > 1:
        md.append("### Disagreement Cases")
        md.append("")
        dis_rows = []
        for iid in sorted(matched_ids):
            valids = [rec_lookup[b].get(iid, {}).get("valid") for b in bases]
            truthy = [v for v in valids if v is not None]
            if truthy and not all(v == truthy[0] for v in truthy):
                dis_rows.append([iid] + [sym(v) for v in valids])
        if dis_rows:
            md.append(md_table(["Instance ID"] + labels, dis_rows))
        else:
            md.append("_No disagreements found._")
        md.append("")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    n_cols = len(bases)

    def best_idx(key, higher_is_better=True):
        """Return index of best column, or None if all equal / missing."""
        vals = [_safe(stats[b].get(key)) for b in bases]
        clean = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not clean:
            return None
        best_i, best_v = max(clean, key=lambda x: x[1]) if higher_is_better \
            else min(clean, key=lambda x: x[1])
        ties = [i for i, v in clean if v == best_v]
        return best_i if len(ties) == 1 else None

    def row(label, key, fmt, higher_is_better=True, bold=True):
        bi = best_idx(key, higher_is_better) if bold else None
        vals = [_fmt(stats[b].get(key), fmt) for b in bases]
        cells = [f"\\textbf{{{v}}}" if i == bi else v for i, v in enumerate(vals)]
        return f"  {label:<44} & {' & '.join(cells)} \\\\"

    def midrule():
        return "  \\midrule"

    def col_headers():
        return "  " + " & ".join([""] + labels) + " \\\\"

    lx: list[str] = []
    lx.append(f"% ── {dataset} ──")
    lx.append(f"\\begin{{tabular}}{{l{'c' * n_cols}}}")
    lx.append("  \\toprule")
    lx.append(col_headers())
    lx.append(midrule())

    # Skipped / evaluated block (no bolding — these are just counts)
    skipped_vals = [str(stats[b]["skipped"]) for b in bases]
    eval_vals    = [str(stats[b]["n"])       for b in bases]
    lx.append(f"  {'Skipped (grammar-invalid)':<44} & {' & '.join(skipped_vals)} \\\\")
    lx.append(f"  {'Evaluated ($n$)':<44} & {' & '.join(eval_vals)} \\\\")
    lx.append(midrule())

    # Validity block
    valid_vals = [f"{stats[b]['valid_count']} / {stats[b]['n']}" for b in bases]
    bi_v = best_idx("valid_count", higher_is_better=True)
    valid_cells = [f"\\textbf{{{v}}}" if i == bi_v else v for i, v in enumerate(valid_vals)]
    lx.append(f"  {'Valid':<44} & {' & '.join(valid_cells)} \\\\")

    lx.append(row("Validity (\\%)", "valid_rate",     "{:.1%}", higher_is_better=True))
    lx.append(row(f"Timeouts ($>${TIMEOUT_S:.0f}\\,s)", "timeout_count", "{:.0f}",
                  higher_is_better=False))
    lx.append(row("Mean resamples\\,$^\\S$", "mean_resamples", "{:.2f}",
                  higher_is_better=False))
    lx.append(midrule())

    lx.append(row("Mean time (s)",   "mean_time_s",   "{:.2f}", higher_is_better=False))
    lx.append(row("Median time (s)", "median_time_s", "{:.2f}", higher_is_better=False))
    lx.append(row("P95 time (s)",    "p95_time_s",    "{:.2f}", higher_is_better=False))
    lx.append(row("Max time (s)",    "max_time_s",    "{:.2f}", higher_is_better=False))

    lx.append("  \\bottomrule")
    lx.append("\\end{tabular}")

    return md, lx


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    groups = load_all()
    if not groups:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    _DATASET_RE = re.compile(r"(jsb_hard|jsb_medium|jsb_easy|jsonschema)")

    def infer_dataset(base: str, rec: dict) -> str:
        ds = rec.get("dataset")
        if ds:
            return ds
        m = _DATASET_RE.search(base)
        return m.group(1) if m else "unknown"

    group_method:  dict[str, str] = {}
    group_dataset: dict[str, str] = {}
    for base, recs_by_id in groups.items():
        first = next(iter(recs_by_id.values()), {})
        group_method[base]  = first.get("method", base)
        group_dataset[base] = infer_dataset(base, first)

    def sort_key(base):
        m = group_method[base]
        try:
            return (METHOD_ORDER.index(m), base)
        except ValueError:
            return (len(METHOD_ORDER), base)

    sorted_bases = sorted(groups.keys(), key=sort_key)

    # Pick the largest-record group per (dataset, recognized-method).
    representative: dict[tuple[str, str], str] = {}
    for base in sorted_bases:
        m = group_method[base]
        if m not in METHOD_ORDER:
            continue
        key = (group_dataset[base], m)
        cur = representative.get(key)
        if cur is None or len(groups[base]) > len(groups[cur]):
            representative[key] = base

    primary_bases: set[str] = set(representative.values())

    by_dataset: dict[str, list[str]] = defaultdict(list)
    for base in sorted_bases:
        if base in primary_bases:
            by_dataset[group_dataset[base]].append(base)

    # ── build outputs ──────────────────────────────────────────────────────────
    all_md:    list[str] = ["# Results Comparison", ""]
    all_latex: list[str] = []

    print("Per-dataset summary:")
    for dataset in sorted(by_dataset.keys()):
        md_lines, lx_lines = compare_dataset(
            dataset, by_dataset[dataset], groups, group_method
        )
        all_md.extend(md_lines)
        all_latex.extend(lx_lines)
        all_latex.append("")

    # File inventory
    all_md.append("## File Inventory")
    all_md.append("")
    inv_headers = ["Group (base name)", "Method", "Dataset", "Shards", "Records"]
    inv_rows = []
    for base in sorted_bases:
        shards = sorted(p for p in RESULTS_DIR.glob("*.jsonl") if _base_name(p.stem) == base)
        n_total = sum(sum(1 for _ in open(s)) for s in shards)
        inv_rows.append([base, group_method[base], group_dataset[base],
                         str(len(shards)), str(n_total)])
    all_md.append(md_table(inv_headers, inv_rows))
    all_md.append("")

    OUTPUT_MD.write_text("\n".join(all_md))
    print(f"\nWritten: {OUTPUT_MD}")

    print("\n" + "─" * 72)
    print("LaTeX snippet:")
    print("─" * 72)
    print("\n".join(all_latex))


if __name__ == "__main__":
    main()
