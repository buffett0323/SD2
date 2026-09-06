#!/usr/bin/env python3
"""How many grammar violations are property-ordering artifacts?

llguidance compiles a JSON schema into a grammar that fixes the order of an
object's properties to their order of declaration -- `{"a":1,"b":2}` is
rejected when the schema declares `b` first, although both orders satisfy the
schema and jsonschema accepts either.  A model that writes the properties in a
different order therefore produces a violation that is not an error.

Those violations cannot be repaired in place: the fix is to move a block, which
a positional DP cannot express, so the repair writes the grammar's demanded
property name over a value intended for another field.  This counts how large
that class is, using the token the model wrote at each violator against the
tokens the grammar would allow there.

Usage:
    python bench/measure_ordering.py --method run 'results/..._ordering*.jsonl'
"""
import argparse, glob, json, re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9_]+")


def property_names(schema) -> set[str]:
    """Every property name anywhere in the schema."""
    names: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                names.update(props.keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return names


def name_matches(tok: str, names: set[str]) -> set[str]:
    """Property names this token could be a piece of.

    Tokens are subword pieces and the piece is often mid-word, not a prefix:
    `ds` comes from `fields`, `ive` from `active`.  Prefix matching missed
    those and misfiled them as non-property forced positions, so this matches
    anywhere in the name.  Pieces shorter than two characters are ignored --
    they match too much to be evidence.
    """
    core = tok.strip().strip('"').strip()
    if len(core) < 2 or not _WORD.fullmatch(core):
        return set()
    return {n for n in names if core in n}


def classify(v: dict, names: set[str]) -> str:
    n_legal = v.get("n_legal", 0)
    want_hits = name_matches(v.get("want", ""), names)
    allow_hits: set[str] = set()
    for a in v.get("allow", []):
        allow_hits |= name_matches(a, names)

    if n_legal == 0:
        return "terminal: nothing consumable left"
    # Both sides name properties, and not the same one: the model is writing a
    # different key than the one the declaration order demands here.
    if want_hits and allow_hits and (want_hits - allow_hits):
        return "ordering: model wants a different property"
    if want_hits and allow_hits:
        return "same property, different piece"
    if want_hits and not allow_hits:
        return "model wrote a property name where none is allowed"
    if n_legal == 1:
        return "forced, and neither side is a property name"
    return "genuine local error (grammar offered alternatives)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", nargs="+", metavar=("NAME", "GLOB"),
                    required=True)
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()

    for spec in a.method:
        rows, seen = [], set()
        for pat in spec[1:]:
            for path in sorted(glob.glob(pat)):
                for line in open(path):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["instance_id"] in seen:
                        continue
                    seen.add(r["instance_id"])
                    rows.append(r)

        counts, samples, n_inst = Counter(), {}, 0
        for r in rows:
            vs = r.get("violations_detail") or []
            if not vs:
                continue
            try:
                names = property_names(json.loads(r["schema"]))
            except Exception:
                continue
            n_inst += 1
            for v in vs:
                k = classify(v, names)
                counts[k] += 1
                samples.setdefault(k, []).append((r["instance_id"], v))

        total = sum(counts.values())
        print(f"\n=== {spec[0]} ===")
        if not total:
            print("  no violation records -- rerun with the instrumented build")
            continue
        print(f"  {total} violations across {n_inst} instances\n")
        for k, c in counts.most_common():
            print(f"  {c:>5}  {100*c/total:>5.1f}%   {k}")
        print("\n  examples:")
        for k, _ in counts.most_common():
            for iid, v in samples[k][:a.examples // 2 or 1]:
                print(f"    [{k[:34]:<34}] {iid}  model wrote {v['want']!r}, "
                      f"grammar allows {v['n_legal']} e.g. {v['allow'][:3]}")


if __name__ == "__main__":
    main()
