#!/usr/bin/env python3
"""Score results on validity the runner does not actually measure.

`result["valid"]` is the generator's `is_complete` flag -- EOS placed and no
MASK before it.  That is completion, not correctness: an unconstrained run can
finish cleanly on malformed JSON and still be counted valid, which flatters any
arm without a grammar and understates what the DP layer buys.  This scores the
three distinct questions separately:

    completes   what the runner recorded (kept, it is a real quantity)
    parses      json.loads succeeds
    schema-OK   validates against the schema stored alongside the output

Only runs written by a runner that saves `schema` can be scored; the older
no-DP baselines cannot and must be re-run.

Also reports the DP repair-site outcomes, since dead-ended DP calls fall
through to a remask and are the mechanism a window-ladder change acts on.

Usage:
    python bench/measure_valid.py --method descend 'results/..._descend*.jsonl' \
                                  --method full    'results/..._spanfull*.jsonl'
"""
import argparse, glob, json, statistics, sys

try:
    import jsonschema
except ImportError:
    sys.exit("needs jsonschema:  uv pip install jsonschema")


def load(patterns, name=""):
    """Load result shards, keyed by instance_id, first file wins.

    Prints the files it matched.  A tag like `_minedit` also matches
    `_full_minedit` from an older experiment, and because the older name sorts
    first its rows were the ones kept -- an entire comparison was read off
    three-day-old files with different instances and no instrumentation before
    the mismatched infra-failure count gave it away.
    """
    rows, matched = {}, []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            matched.append(path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        rows.setdefault(r["instance_id"], r)
    if matched:
        import os, time
        days = {int((time.time() - os.path.getmtime(p)) // 86400) for p in matched}
        stamp = f"  [ages: {sorted(days)} days]" if len(days) > 1 else ""
        print(f"  {name or 'files'}: {len(matched)} shard(s){stamp}")
        for p in matched:
            print(f"      {os.path.basename(p)}")
        if len(days) > 1:
            print("      ^ shards differ in age -- check the glob is not catching "
                  "an older experiment")
    return rows


def as_tree(row):
    """The output as a Python object, or None if it is not JSON.

    `extracted` is sometimes already a parsed dict/list.  Note this does NOT go
    through measure_degeneracy.text_of, which strips whitespace *inside* string
    values ("John Doe" -> "JohnDoe") -- harmless for counting empty brackets,
    but it rewrites the data being validated.
    """
    ex = row.get("extracted")
    if isinstance(ex, (dict, list)):
        return ex
    if not isinstance(ex, str) or not ex.strip():
        return None
    try:
        return json.loads(ex)
    except json.JSONDecodeError:
        return None


def leaf_count(row) -> int | None:
    """Number of scalar leaves in the output, or None if it is not JSON.

    Schema validity on its own rewards giving up: `{"objects": []}` scores a
    win with zero leaves while an 851-character document of real objects that
    does not quite close scores a loss.  Both were observed on the same
    instance, from the two arms being compared.  Pairing validity with a
    content floor stops a method being rewarded for emitting the smallest
    document its schema permits.
    """
    tree = as_tree(row)
    if tree is None:
        return None
    n = [0]

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            n[0] += 1

    walk(tree)
    return n[0]


def raw_len(row) -> int:
    ex = row.get("extracted")
    if isinstance(ex, str):
        return len(ex)
    return len(json.dumps(ex)) if ex is not None else 0


def smiles_ok(text) -> bool:
    """Does RDKit accept this as a molecule?

    Stricter than grammar conformance, which is all the decoder enforces: a
    string can parse as SMILES and still be chemically impossible (RDKit
    rejects `CC(C)(C)(C)C` on valence).  An empty string is a valid empty
    molecule to RDKit and is counted as a failure here.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        sys.exit("--lang smiles needs rdkit:  uv pip install rdkit")
    try:
        return Chem.MolFromSmiles(text.strip()) is not None
    except Exception:
        return False


def score(row, lang="json"):
    """(completes, parses, schema_ok) -- schema_ok is None when unscoreable."""
    if lang == "smiles":
        ok = smiles_ok(row.get("extracted"))
        return bool(row.get("valid")), ok, ok
    completes = bool(row.get("valid"))
    tree = as_tree(row)
    if tree is None:
        return completes, False, False
    schema = row.get("schema")
    if isinstance(schema, str):          # stored as a JSON string
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return completes, True, None
    if not isinstance(schema, dict):
        return completes, True, None
    try:
        jsonschema.validate(tree, schema)
        return completes, True, True
    except jsonschema.ValidationError:
        return completes, True, False
    except Exception:                    # malformed schema -- not the output's fault
        return completes, True, None


#: Harness failures, not decoding failures.  `checker_init_error` means
#: llguidance could not compile the schema into a grammar, so no method ever
#: got to run -- the same 7 jsb_medium instances fail this way in every arm.
#: Counting them punishes every arm equally but drags the absolute numbers
#: down by ~11% and measures llguidance's schema coverage, not decoding.
#: `modal_deadline` is different: the run really did not finish in time, and
#: that IS the method's fault, so it stays in.
INFRA_REASONS = ("checker_init_error", "no_schema", "prompt_error")


def is_infra_failure(row) -> bool:
    reason = row.get("timed_out_reason")
    return isinstance(reason, str) and reason.startswith(INFRA_REASONS)


def leaf_rates(rows, keys):
    """Mean over outputs of (empty strings / string leaves, 1-char / string leaves).

    Reported separately because the combined "vacuous leaf" figure hid that the
    two move in opposite directions: widening the repair window cut one-char
    strings hard and left empty strings alone.
    """
    empties, onechars = [], []
    for k in keys:
        tree = as_tree(rows[k])
        if tree is None:
            continue
        acc = dict(containers=0, empty_containers=0, empty_objects=0,
                   empty_arrays=0, leaves=0, str_leaves=0, empty_strs=0,
                   onechar_strs=0)
        _walk(tree, acc)
        if acc["str_leaves"]:
            empties.append(acc["empty_strs"] / acc["str_leaves"])
            onechars.append(acc["onechar_strs"] / acc["str_leaves"])
    if not empties:
        return 0.0, 0.0
    return statistics.mean(empties), statistics.mean(onechars)


def _walk(node, acc):
    if isinstance(node, dict):
        acc["containers"] += 1
        for v in node.values():
            _walk(v, acc)
    elif isinstance(node, list):
        acc["containers"] += 1
        for v in node:
            _walk(v, acc)
    else:
        acc["leaves"] += 1
        if isinstance(node, str):
            acc["str_leaves"] += 1
            if len(node) == 0:
                acc["empty_strs"] += 1
            elif len(node) == 1:
                acc["onechar_strs"] += 1


def sites_of(rows):
    return [s for r in rows.values() for s in r.get("span_sites_detail", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", nargs="+", metavar=("NAME", "GLOB"),
                    required=True)
    ap.add_argument("--common", action="store_true",
                    help="restrict to instances every method answered")
    ap.add_argument("--lang", default="json", choices=["json", "smiles"],
                    help="json: parse + validate against the stored schema. "
                         "smiles: RDKit MolFromSmiles (no schema is stored).")
    ap.add_argument("--latency", action="store_true",
                    help="print the mean/p50/p90/p95/p99/timeout latency profile")
    ap.add_argument("--timeout-s", type=float, default=120.0,
                    help="instances at or above this wall time count as timeouts")
    ap.add_argument("--min-leaves", type=int, default=5,
                    help="content floor: an output counts as a win only if it "
                         "validates AND carries at least this many leaves")
    ap.add_argument("--keep-infra", action="store_true",
                    help="keep instances whose schema llguidance cannot compile "
                         "(excluded by default; they fail identically in every arm)")
    a = ap.parse_args()

    arms = {spec[0]: load(spec[1:], spec[0]) for spec in a.method}
    ids = sorted(set.intersection(*(set(r) for r in arms.values()))) if a.common \
        else None

    # Drop harness failures from every arm at once, so the denominators stay
    # identical and the exclusion is visible rather than silent.
    infra = sorted({k for rows in arms.values() for k, r in rows.items()
                    if is_infra_failure(r)})
    if infra and not a.keep_infra:
        shown = ", ".join(infra[:8]) + (" ..." if len(infra) > 8 else "")
        print(f"excluding {len(infra)} instances whose schema llguidance cannot "
              f"compile (no method runs on them): {shown}\n")
        if ids is not None:
            ids = [i for i in ids if i not in set(infra)]
        else:
            drop = set(infra)
            arms = {n: {k: v for k, v in rows.items() if k not in drop}
                    for n, rows in arms.items()}

    if a.latency:
        # Latency profile in the shape the JSB difficulty tables use: the tail
        # is the part that matters, because a method that is fast on average and
        # occasionally hits a wall-clock cap is not usable, and a mean hides that.
        print(f"{'method':<14}{'n':>5}{'valid':>8}{'mean':>8}{'p50':>8}{'p90':>8}"
              f"{'p95':>8}{'p99':>8}{'timeout':>9}")
        print("-" * 76)
        for name, rows in arms.items():
            keys = ids if ids is not None else sorted(rows)
            sc = [score(rows[k], a.lang) for k in keys]
            ok = [s_ for _, _, s_ in sc if s_ is not None]
            t = sorted(float(rows[k]["time_taken"]) for k in keys
                       if rows[k].get("time_taken") is not None)
            if not t:
                print(f"{name:<14}{len(keys):>5}{'--':>8}"); continue
            q = lambda f: t[min(len(t) - 1, int(f * len(t)))]
            to = sum(1 for k in keys
                     if (rows[k].get("time_taken") or 0) >= a.timeout_s)
            print(f"{name:<14}{len(keys):>5}"
                  f"{100*sum(ok)/len(ok) if ok else float('nan'):>7.1f}%"
                  f"{statistics.mean(t):>8.2f}{statistics.median(t):>8.2f}"
                  f"{q(.90):>8.2f}{q(.95):>8.2f}{q(.99):>8.2f}{to:>9}")
        print()

    ok_label = "schema-OK" if a.lang == "json" else "rdkit-OK"
    parse_label = "parses" if a.lang == "json" else "nonempty"
    cont_label = f"ok&>={a.min_leaves}leaf"
    print(f"{'method':<14}{'n':>5}{'completes':>11}{parse_label:>9}{ok_label:>11}"
          f"{cont_label:>13}{'medLeaf|ok':>12}"
          f"{'emptyStr':>10}{'1char':>8}{'medChar':>9}{'medTime':>9}{'resample':>10}")
    print("-" * 96)
    for name, rows in arms.items():
        keys = ids if ids is not None else sorted(rows)
        n = len(keys)
        sc = [score(rows[k], a.lang) for k in keys]
        ok = [s for _, _, s in sc if s is not None]
        times = [float(rows[k]["time_taken"]) for k in keys
                 if rows[k].get("time_taken") is not None]
        res = sum(rows[k].get("resamples", 0) if isinstance(rows[k].get("resamples"), int)
                  else len(rows[k].get("resamples") or []) for k in keys)
        es, oc = leaf_rates(rows, keys) if a.lang == "json" else (None, None)
        leaf_cols = (f"{100*es:9.1f}%{100*oc:7.1f}%" if es is not None
                     else f"{'-':>10}{'-':>8}")
        ok_keys = [k for k, (_, _, sk) in zip(keys, sc) if sk is True]
        lc = [leaf_count(rows[k]) or 0 for k in ok_keys]
        cont = sum(1 for x in lc if x >= a.min_leaves)
        cont_cols = (f"{100*cont/n:12.1f}%"
                     + f"{statistics.median(lc):>12.0f}" if lc
                     else f"{'-':>13}{'-':>12}")
        print(f"{name:<14}{n:5d}"
              f"{100*sum(c for c,_,_ in sc)/n:10.1f}%"
              f"{100*sum(p for _,p,_ in sc)/n:8.1f}%"
              f"{100*sum(ok)/len(ok) if ok else float('nan'):10.1f}%"
              + cont_cols
              + leaf_cols
              + f"{statistics.median(raw_len(rows[k]) for k in keys):9.0f}"
              + f"{statistics.median(times) if times else 0:8.1f}s"
              + f"{res:10d}")

    print(f"\n{'method':<14}{'DP calls':>10}{'dead-end':>10}{'rate':>8}"
          f"{'span(ok)':>10}{'edits(ok)':>11}")
    print("-" * 63)
    for name, rows in arms.items():
        keys = ids if ids is not None else sorted(rows)
        sites = [s for k in keys for s in rows[k].get("span_sites_detail", [])]
        if not sites:
            print(f"{name:<14}{'-- no span instrumentation in these files --':>49}")
            continue
        dead = [s for s in sites if s["n_fixes"] is None]
        good = [s for s in sites if s["n_fixes"] is not None]
        print(f"{name:<14}{len(sites):10d}{len(dead):10d}"
              f"{100*len(dead)/len(sites):7.1f}%"
              f"{statistics.mean(s['span'] for s in good) if good else 0:10.2f}"
              f"{statistics.mean(s['n_fixes'] for s in good) if good else 0:11.2f}")

    # Where the remasks come from.  Totals alone hid the split: 75 DP calls and
    # 32 dead-ends against 139 resamples left two thirds unexplained.
    reasons = ["single_token", "dp_span_replay", "dp_suffix_replay", "dp_dead_end"]
    have = any(rows[k].get("resample_reasons")
               for rows in arms.values()
               for k in (ids if ids is not None else rows))
    if have:
        print(f"\n{'method':<14}" + "".join(f"{r:>18}" for r in reasons) + f"{'total':>8}")
        print("-" * (14 + 18 * len(reasons) + 8))
        for name, rows in arms.items():
            keys = ids if ids is not None else sorted(rows)
            agg = {r: 0 for r in reasons}
            other = 0
            for k in keys:
                for r, c in (rows[k].get("resample_reasons") or {}).items():
                    if r in agg:
                        agg[r] += c
                    else:
                        other += c
            tot = sum(agg.values()) + other
            print(f"{name:<14}" + "".join(f"{agg[r]:>18}" for r in reasons)
                  + f"{tot:>8}")

    # How much work the all-or-nothing DP contract discards.  `dead_step` is the
    # layer at which no candidate was accepted; every layer before it already
    # had a valid assignment that `return None` throws away.
    if any(s.get("dead_step") is not None
           for rows in arms.values()
           for k in (ids if ids is not None else rows)
           for s in rows[k].get("span_sites_detail", [])):
        print(f"\n{'method':<14}{'dead ends':>11}{'at layer 0':>12}{'past layer 0':>14}"
              f"{'positions recoverable':>23}")
        print("-" * 74)
        for name, rows in arms.items():
            keys = ids if ids is not None else sorted(rows)
            dead = [s for k in keys for s in rows[k].get("span_sites_detail", [])
                    if s.get("dead_step") is not None]
            if not dead:
                print(f"{name:<14}{'-- none recorded --':>60}")
                continue
            at0 = sum(1 for s in dead if s["dead_step"] == 0)
            rec = sum(s["dead_step"] for s in dead)
            print(f"{name:<14}{len(dead):>11}{at0:>12}{len(dead)-at0:>14}{rec:>23}")
            hist = {}
            for s in dead:
                b = s["dead_step"]
                key = str(b) if b <= 4 else "5-9" if b <= 9 else "10+"
                hist[key] = hist.get(key, 0) + 1
            order = ["0", "1", "2", "3", "4", "5-9", "10+"]
            line = "  ".join(f"{k}:{hist[k]}" for k in order if k in hist)
            spans = [s.get("dp_positions") for s in dead if s.get("dp_positions")]
            print(f"{'':<14}dead_step histogram   {line}"
                  + (f"   (span given to DP: mean {statistics.mean(spans):.1f})"
                     if spans else ""))

    # Where the DP's candidates came from.  `exact` is the share of
    # (state, position) pairs whose entire legal edge set fit inside top_k, so
    # the DP enumerated it rather than sampling from the vocabulary ranking.
    if any(s.get("legal_calls") for rows in arms.values()
           for k in (ids if ids is not None else rows)
           for s in rows[k].get("span_sites_detail", [])):
        print(f"\n{'method':<14}{'state-positions':>16}{'legal set mean':>16}"
              f"{'min':>7}{'enumerated exactly':>20}{'bias disagreements':>20}")
        print("-" * 93)
        for name, rows in arms.items():
            keys = ids if ids is not None else sorted(rows)
            sites = [s for k in keys for s in rows[k].get("span_sites_detail", [])
                     if s.get("legal_calls")]
            if not sites:
                print(f"{name:<14}{'-- not recorded --':>40}"); continue
            calls = sum(s["legal_calls"] for s in sites)
            mean = sum(s["legal_mean"] * s["legal_calls"] for s in sites) / calls
            mn = min(s["legal_min"] for s in sites if s.get("legal_min") is not None)
            ex = sum(s.get("legal_exact", 0) for s in sites)
            bad = sum(s.get("bias_disagreements", 0) for s in sites)
            print(f"{name:<14}{calls:>16}{mean:>16.0f}{mn:>7}"
                  f"{100*ex/calls:>19.0f}%{bad:>20}")

    # Which branch ended generation, and how much of the sequence was still
    # masked when it did.  A stop that leaves masks behind truncated the
    # document; one that leaves none merely padded it.
    if any(rows[k].get("stop") for rows in arms.values()
           for k in (ids if ids is not None else rows)):
        print(f"\n{'method':<14}{'stop branch':<22}{'count':>7}{'left masked':>14}")
        print("-" * 57)
        for name, rows in arms.items():
            keys = ids if ids is not None else sorted(rows)
            agg: dict[str, list] = {}
            for k in keys:
                st = rows[k].get("stop") or {}
                if st.get("reason"):
                    agg.setdefault(st["reason"], []).append(st.get("masks_left", 0))
            if not agg:
                print(f"{name:<14}{'-- not recorded --':<22}"); continue
            for r, ms in sorted(agg.items(), key=lambda kv: -len(kv[1])):
                trunc = sum(1 for m in ms if m > 0)
                print(f"{name:<14}{r:<22}{len(ms):>7}"
                      f"{f'{trunc} of {len(ms)}':>14}")

    # `resample_count` is not comparable across generators: the unconstrained
    # loop counts every grammar-rejected candidate, the DP loop only counts a
    # position actually handed back.  These two are defined the same in both.
    print(f"\n{'method':<14}{'rejections':>13}{'handbacks':>12}"
          "   (rejections = candidates the grammar refused;")
    print(f"{'':<14}{'':>13}{'':>12}"
          "    handbacks  = positions returned to the sampler)")
    print("-" * 39)
    for name, rows in arms.items():
        keys = ids if ids is not None else sorted(rows)
        rej = sum((rows[k].get("timing") or {}).get("rejections", 0) for k in keys)
        hb = sum((rows[k].get("timing") or {}).get("handbacks", 0) for k in keys)
        print(f"{name:<14}{rej:>13}{hb:>12}")

    # Which repair path writes the token at a violator, and what it writes.
    print(f"\n{'method':<14}{'path':<10}  most-written tokens at a violator")
    print("-" * 78)
    for name, rows in arms.items():
        keys = ids if ids is not None else sorted(rows)
        merged: dict[str, dict[str, int]] = {}
        for k in keys:
            for path, d in ((rows[k].get("timing") or {}).get("writes") or {}).items():
                m = merged.setdefault(path, {})
                for tok, c in d.items():
                    m[tok] = m.get(tok, 0) + c
        if not merged:
            print(f"{name:<14}{'--':<10}  no write instrumentation in these files")
            continue
        for path in sorted(merged, key=lambda p: -sum(merged[p].values())):
            top = sorted(merged[path].items(), key=lambda kv: -kv[1])[:8]
            total = sum(merged[path].values())
            shown = "  ".join(f"{tok!r}x{c}" for tok, c in top)
            print(f"{name:<14}{path:<10}  n={total:<6} {shown}")


if __name__ == "__main__":
    main()
