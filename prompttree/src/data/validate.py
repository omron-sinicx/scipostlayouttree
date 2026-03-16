from __future__ import annotations

from typing import Any, Dict, List, Set
from collections import Counter

from data.parse import PredictedStructure


def validate_predicted_structure(
    pred: PredictedStructure,
    n_bboxes: int,
    root_parent: int = -1,
) -> List[str]:
    errors: List[str] = []

    # ---- reading_order checks ----
    ro = pred.reading_order
    if not isinstance(ro, list):
        return ["reading_order is not a list"]

    expected = list(range(1, n_bboxes + 1))
    if len(ro) != n_bboxes:
        errors.append(f"reading_order length {len(ro)} != n_bboxes {n_bboxes}")

    # elements must be int
    non_int = [x for x in ro if not isinstance(x, int)]
    if non_int:
        errors.append(f"reading_order contains non-int values: {non_int[:5]}")

    # range check
    out_of_range = [x for x in ro if isinstance(x, int) and not (1 <= x <= n_bboxes)]
    if out_of_range:
        errors.append(f"reading_order contains out-of-range values: {out_of_range[:5]}")

    # permutation check (only if ints in range)
    if not non_int:
        c = Counter(ro)
        dups = [k for k, v in c.items() if v > 1]
        if dups:
            errors.append(f"reading_order contains duplicates: {dups[:10]}")
        missing = sorted(set(expected) - set(ro))
        if missing:
            errors.append(f"reading_order missing values: {missing[:10]}")

    # ---- tree checks ----
    tree = pred.tree
    if not isinstance(tree, list):
        return errors + ["tree is not a list"]

    # each entry must be dict with keys bbox_number, parent
    nodes: List[int] = []
    parent_map: Dict[int, int] = {}

    for i, e in enumerate(tree):
        if not isinstance(e, dict):
            errors.append(f"tree[{i}] is not a dict")
            continue
        if "bbox_number" not in e or "parent" not in e:
            errors.append(f"tree[{i}] missing keys (bbox_number/parent)")
            continue
        bn = e["bbox_number"]
        p = e["parent"]
        if not isinstance(bn, int) or not isinstance(p, int):
            errors.append(f"tree[{i}] bbox_number/parent must be int (got {type(bn)}, {type(p)})")
            continue

        nodes.append(bn)
        # if duplicate bbox_number appears, keep first, but record error
        if bn in parent_map:
            errors.append(f"tree contains duplicate bbox_number: {bn}")
        else:
            parent_map[bn] = p

        # bbox_number range
        if not (1 <= bn <= n_bboxes):
            errors.append(f"tree[{i}] bbox_number out of range: {bn}")

        # parent range
        if not (p == root_parent or (1 <= p <= n_bboxes)):
            errors.append(f"tree[{i}] parent out of range: {p} (bbox_number={bn})")

        # self-parent
        if p == bn:
            errors.append(f"self-parent detected: bbox_number={bn}")

    # require coverage of all nodes 1..N exactly once
    if len(parent_map) != n_bboxes:
        errors.append(f"tree unique node count {len(parent_map)} != n_bboxes {n_bboxes}")
    missing_nodes = sorted(set(range(1, n_bboxes + 1)) - set(parent_map.keys()))
    if missing_nodes:
        errors.append(f"tree missing bbox_numbers: {missing_nodes[:10]}")

    extra_nodes = sorted(set(parent_map.keys()) - set(range(1, n_bboxes + 1)))
    if extra_nodes:
        errors.append(f"tree has extra bbox_numbers: {extra_nodes[:10]}")

    # ---- cycle check ----
    # Only attempt cycle check if parent_map has all nodes in range (otherwise it is noisy)
    if not missing_nodes and not extra_nodes:
        # detect cycles in a functional graph (each node has 1 parent)
        # parent can be root_parent; treat it as terminal
        state: Dict[int, int] = {}  # 0=unvisited, 1=visiting, 2=done

        def dfs(u: int) -> None:
            st = state.get(u, 0)
            if st == 1:
                errors.append(f"cycle detected (revisiting node {u})")
                return
            if st == 2:
                return
            state[u] = 1
            p = parent_map.get(u, root_parent)
            if p != root_parent:
                dfs(p)
            state[u] = 2

        for u in range(1, n_bboxes + 1):
            if state.get(u, 0) == 0:
                dfs(u)

    # ---- root reachability (Root must be the apex) ----
    # 1) At least one root child must exist
    root_children = [u for u, p in parent_map.items() if p == root_parent]
    if len(root_children) == 0:
        errors.append("no root child: there must be at least one node with parent=-1")

    # 2) Every node must reach root_parent by following parents
    # (This also rejects disconnected components that do not connect to root.)
    def reaches_root(u: int) -> bool:
        seen: Set[int] = set()
        cur = u
        while True:
            if cur in seen:
                # cycle should already be caught, but keep it safe
                return False
            seen.add(cur)
            p = parent_map.get(cur, root_parent)
            if p == root_parent:
                return True
            cur = p

    not_reaching = [u for u in range(1, n_bboxes + 1) if not reaches_root(u)]
    if not_reaching:
        errors.append(f"some nodes do not reach root (-1): {not_reaching[:10]}")

    return errors
