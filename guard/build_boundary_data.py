#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build turn-localized training rows from judged unsafe trajectories (boundary construction).

For every trajectory judged unsafe, locate the first unsafe Assistant turn i from the
judge evidence, then emit two prefix examples:
  - the prefix through turn i-1 as a safe example (only when i > 1)
  - the prefix through turn i as an unsafe example
Turns after i are discarded; trajectories that fail at turn 1 contribute only the unsafe
example; trajectories judged safe are not used. Each row carries audit fields
(source_guard_id / boundary_role / prefix_turn). The train/test split lives in
split_train_test.py.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_unsafe_turn(item: dict[str, Any]) -> int | None:
    evidence = item.get("judge_evidence")
    if not isinstance(evidence, list):
        judge_result = item.get("judge_result")
        evidence = judge_result.get("evidence") if isinstance(judge_result, dict) else None
    if not isinstance(evidence, list):
        return None
    turns = []
    for entry in evidence:
        if not isinstance(entry, dict):
            continue
        turn = entry.get("assistant_turn")
        if isinstance(turn, int) and turn >= 1:
            turns.append(turn)
        elif isinstance(turn, str) and turn.isdigit() and int(turn) >= 1:
            turns.append(int(turn))
    return min(turns) if turns else None


def prefix_through_assistant_turn(trace: list[dict[str, Any]], target_turn: int) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    assistant_turn = 0
    for message in trace:
        prefix.append(copy.deepcopy(message))
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if assistant_turn == target_turn:
            return prefix
    raise ValueError(f"Trace has no Assistant turn {target_turn} (only {assistant_turn} Assistant turns)")


def make_boundary_row(source: dict[str, Any], *, label: str, boundary_role: str,
                      first_unsafe: int, prefix_turn: int) -> dict[str, Any]:
    source_id = str(source["guard_id"])
    derived_key = f"companionguard-final|{source_id}|{boundary_role}|{prefix_turn}"
    row = copy.deepcopy(source)
    row["guard_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, derived_key))
    row["source_guard_id"] = source_id
    row["source_label"] = source.get("label")
    row["source_dialogue_turn"] = source.get("dialogue_turn")
    row["source_judge_evidence"] = copy.deepcopy(source.get("judge_evidence", []))
    row["source_first_unsafe_turn"] = first_unsafe
    row["prefix_turn"] = prefix_turn
    row["boundary_role"] = boundary_role
    row["label"] = label
    row["dialogue_turn"] = prefix_turn
    if boundary_role == "safe_prefix":
        row["judge_evidence"] = []
    else:
        source_evidence = row.get("judge_evidence", [])
        row["judge_evidence"] = [
            entry for entry in source_evidence
            if isinstance(entry, dict)
            and (isinstance(entry.get("assistant_turn"), int) and entry["assistant_turn"] <= prefix_turn
                 or isinstance(entry.get("assistant_turn"), str)
                 and entry["assistant_turn"].isdigit() and int(entry["assistant_turn"]) <= prefix_turn)
        ]
    row["dialogue_trace"] = prefix_through_assistant_turn(source["dialogue_trace"], prefix_turn)
    return row


def construct(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    first_turn_distribution: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    paired_sources = 0
    first_turn_only = 0

    for item in items:
        if item.get("label") != "unsafe":
            skipped["source_not_unsafe"] += 1
            continue
        if not item.get("dialogue_trace"):
            skipped["empty_dialogue_trace"] += 1
            continue
        first_unsafe = first_unsafe_turn(item)
        if first_unsafe is None:
            skipped["unsafe_without_evidence_turn"] += 1
            continue
        first_turn_distribution[str(first_unsafe)] += 1
        if first_unsafe > 1:
            output.append(make_boundary_row(item, label="safe", boundary_role="safe_prefix",
                                            first_unsafe=first_unsafe, prefix_turn=first_unsafe - 1))
            paired_sources += 1
        else:
            first_turn_only += 1
        output.append(make_boundary_row(item, label="unsafe", boundary_role="unsafe_boundary",
                                        first_unsafe=first_unsafe, prefix_turn=first_unsafe))

    stats = {
        "input_rows": len(items),
        "source_unsafe_rows": sum(1 for item in items if item.get("label") == "unsafe"),
        "output_rows": len(output),
        "label_distribution": dict(sorted(Counter(str(r.get("label")) for r in output).items())),
        "paired_source_rows": paired_sources,
        "first_turn_unsafe_source_rows": first_turn_only,
        "first_unsafe_turn_distribution": dict(sorted(first_turn_distribution.items(), key=lambda x: int(x[0]))),
        "skipped": dict(skipped),
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build turn-localized boundary rows from judged unsafe trajectories")
    parser.add_argument("--input", type=Path, default=Path("data/trajectories.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/boundary_selected.jsonl"))
    parser.add_argument("--stats_output", type=Path, default=Path("data/boundary_stats.json"))
    args = parser.parse_args()

    items = load_jsonl(str(args.input))
    if not items:
        raise ValueError(f"Empty input: {args.input}")
    selected, stats = construct(items)
    if stats["skipped"].get("unsafe_without_evidence_turn", 0):
        raise ValueError(f"Unsafe rows without a Claude evidence turn: {stats['skipped']['unsafe_without_evidence_turn']}")
    if not selected:
        raise ValueError("No boundary rows were constructed")

    stats["input"] = str(args.input)
    stats["output"] = str(args.output)
    stats["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    stats["unique_source_guard_ids"] = len({str(r["source_guard_id"]) for r in selected})
    save_jsonl(selected, str(args.output))
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[boundary] {stats['source_unsafe_rows']} unsafe sources → {stats['output_rows']} boundary rows "
          f"({stats['label_distribution']}) → {args.output}")
    print(f"[boundary] paired {paired if (paired := stats['paired_source_rows']) else 0}; "
          f"first-turn-only {stats['first_turn_unsafe_source_rows']}; stats → {args.stats_output}")


if __name__ == "__main__":
    main()
