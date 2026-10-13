#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Group-level train/test split (95% / 5%) of the boundary rows.

Rows are split by source_guard_id: the paired safe/unsafe prefixes of one source share all
of their context text, so a random row-level split would leak test prefixes into training
verbatim. By default 5% of groups form the held-out test set.

Outputs:
  data/train_selected.jsonl   95% rows (raw boundary rows)
  data/train_sft.jsonl        assembled SFT training rows (answer-first contract)
  data/test.jsonl             held-out test set for evaluate_companionguard.py
  data/split_stats.json       split statistics
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from utils import load_jsonl, save_jsonl


def render_dialogue_lines(trace: list[dict]) -> str:
    """dialogue_trace → 'User:/Assistant:' 行式文本；末条消息压平成单物理行（loader 取末行作待判句）。"""
    lines = []
    for idx, msg in enumerate(trace):
        prefix = "User: " if msg["role"] == "user" else "Assistant: "
        content = msg["content"]
        if idx == len(trace) - 1:
            content = " ".join(content.split())
        lines.append(prefix + content)
    return "\n".join(lines)


def render_turn_marked(trace: list[dict]) -> str:
    """dialogue_trace → '[user_turn=N]/[assistant_turn=N]' 格式（与 loader rebuild 同构）。"""
    parts, turn = [], 0
    for msg in trace:
        if msg["role"] == "user":
            turn += 1
            parts.append(f"[user_turn={turn}]\n{msg['content']}")
        else:
            parts.append(f"[assistant_turn={turn}]\n{msg['content']}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="95/5 group-level train/test split of boundary rows")
    parser.add_argument("--input", type=str, default="data/boundary_selected.jsonl")
    parser.add_argument("--train_output", type=str, default="data/train_selected.jsonl")
    parser.add_argument("--train_jsonl", type=str, default="data/train_sft.jsonl")
    parser.add_argument("--test_output", type=str, default="data/test.jsonl")
    parser.add_argument("--stats_output", type=str, default="data/split_stats.json")
    parser.add_argument("--holdout_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_tokens", type=int, default=16000)
    args = parser.parse_args()

    items = load_jsonl(args.input)
    if not items:
        raise ValueError(f"Empty input: {args.input}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[str(it["source_guard_id"])].append(it)
    group_ids = sorted(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)
    n_holdout = round(len(group_ids) * args.holdout_ratio)
    holdout_ids = set(group_ids[:n_holdout])

    train_rows = [it for gid in group_ids[n_holdout:] for it in groups[gid]]
    test_rows = [it for gid in sorted(holdout_ids) for it in groups[gid]]

    eval_rows = [{
        "uuid": str(it["guard_id"]),
        "dialogue": render_dialogue_lines(it["dialogue_trace"]),
        "label": it["label"],
        "category": it["dimension_name"] if it["label"] == "unsafe" else None,
    } for it in test_rows]

    dropped_overlength = 0
    try:
        from transformers import AutoTokenizer
        from guard_prompt import GUARD_USER_PROMPT_EN, render_guard_system_prompt
        tok = AutoTokenizer.from_pretrained(os.environ.get("QWEN3_PATH", "Qwen/Qwen3-8B"))
        system_prompt = render_guard_system_prompt("data/safety_taxonomy.json")
        kept = []
        for it, r in zip(test_rows, eval_rows):
            user_prompt = GUARD_USER_PROMPT_EN.format(dialogue=render_turn_marked(it["dialogue_trace"]))
            if len(tok(system_prompt + user_prompt)["input_ids"]) > args.max_prompt_tokens:
                dropped_overlength += 1
            else:
                kept.append(r)
        eval_rows = kept
        print(f"[split] token check: kept {len(kept)}, dropped {dropped_overlength} (> {args.max_prompt_tokens})")
    except Exception as exc:
        print(f"[split] token check skipped: {exc}")

    from guard_prompt import GUARD_USER_PROMPT_EN, render_guard_system_prompt
    system_prompt = render_guard_system_prompt("data/safety_taxonomy.json")
    train_assembled = []
    for it in train_rows:
        target = ("<answer>safe</answer>" if it["label"] == "safe"
                  else f"<answer>unsafe</answer>\n<category>{it['dimension_id']}</category>")
        train_assembled.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": GUARD_USER_PROMPT_EN.format(dialogue=render_turn_marked(it["dialogue_trace"]))},
            {"role": "assistant", "content": target},
        ]})

    save_jsonl(train_rows, args.train_output)
    save_jsonl(train_assembled, args.train_jsonl)
    test_path = Path(args.test_output)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "input": args.input,
        "input_rows": len(items),
        "input_groups": len(groups),
        "seed": args.seed,
        "holdout_ratio": args.holdout_ratio,
        "holdout_groups": n_holdout,
        "train_rows": len(train_rows),
        "train_unsafe": sum(1 for r in train_rows if r["label"] == "unsafe"),
        "holdout_rows_total": len(test_rows),
        "holdout_rows_eval": len(eval_rows),
        "dropped_overlength": dropped_overlength,
        "holdout_labels": dict(Counter(r["label"] for r in eval_rows)),
        "holdout_boundary_role": dict(Counter(r.get("boundary_role") for r in test_rows)),
        "holdout_turns": dict(sorted(Counter(r.get("dialogue_turn") for r in test_rows).items(),
                                     key=lambda x: (x[0] is None, x[0]))),
        "holdout_unsafe_by_dim": dict(Counter(r["dimension_id"] for r in test_rows if r["label"] == "unsafe")),
        "split_unit": "source_guard_id (paired prefixes stay on the same side of the split)",
    }
    Path(args.stats_output).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[split] groups {len(groups)} → holdout {n_holdout} ({args.holdout_ratio:.0%})")
    print(f"[split] train {len(train_rows)} rows (unsafe {stats['train_unsafe']}) → {args.train_jsonl}")
    print(f"[split] test  {len(eval_rows)} rows, labels {stats['holdout_labels']} → {args.test_output}")


if __name__ == "__main__":
    main()
