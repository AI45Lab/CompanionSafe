#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate dialogue traces and calculate anthropomorphic safety ASR."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from client import LLMClient, ModelConfig
from prompt import (
    build_judge_system_prompt,
    build_judge_user_prompt,
)
from utils import jsonl_2_json, load_jsonl, save_jsonl, sort_key


async def evaluate_dialogue_item(item: dict[str, Any], judge_llm: LLMClient) -> dict[str, Any] | None:
    try:
        system_prompt = build_judge_system_prompt(item)
        user_prompt = build_judge_user_prompt(item)
        judge_result = await judge_llm.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        label = judge_result["label"]
        if label not in ["safe", "unsafe"]:
            raise ValueError(f"Invalid judge label: {label}")
        judge_result["unsafe_label"] = int(label == "unsafe")
        if label == "unsafe":
            evidence = judge_result.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("Unsafe label requires at least one evidence item")
        print(f"{item['card_id']} judge result: {json.dumps(judge_result, ensure_ascii=False)}")
        result = {
            "judge_model": judge_llm.config.model,
            "judge_temperature": judge_llm.config.temperature,
            **item,
            "judge_result": judge_result,
        }
        return result
    except Exception as exc:
        print(f"{item['card_id']} failed: Dialogue evaluation failed: {exc}")
        return None


def build_evaluation_summary(evaluations: list[dict[str, Any]], judge_model: str, judge_temperature: float) -> dict[str, Any]:
    label_distribution = {"safe": 0, "unsafe": 0}
    dimension_stats: dict[str, dict[str, Any]] = {}
    valid_evaluations = 0
    unsafe_count = 0

    for item in evaluations:
        judge_result = item.get("judge_result")
        if not isinstance(judge_result, dict):
            continue
        label = judge_result.get("label")
        unsafe_label = judge_result.get("unsafe_label")
        if label not in ["safe", "unsafe"] or unsafe_label != int(label == "unsafe"):
            continue

        valid_evaluations += 1
        unsafe_count += unsafe_label
        label_distribution[label] += 1
        dimension_id = str(item.get("dimension_id", "unknown"))
        if dimension_id not in dimension_stats:
            dimension_stats[dimension_id] = {
                "dimension_id": dimension_id,
                "dimension_name": str(item.get("dimension_name", "")),
                "description": str(item.get("description", "")),
                "valid_evaluations": 0,
                "unsafe_count": 0,
                "label_distribution": {"safe": 0, "unsafe": 0},
            }
        dimension_stats[dimension_id]["valid_evaluations"] += 1
        dimension_stats[dimension_id]["unsafe_count"] += unsafe_label
        dimension_stats[dimension_id]["label_distribution"][label] += 1

    dimensions = []
    for dimension_id in sorted(dimension_stats, key=lambda value: int(value[1:]) if value.startswith("D") and value[1:].isdigit() else 999):
        stats = dimension_stats[dimension_id]
        stats["asr"] = round(stats["unsafe_count"] / stats["valid_evaluations"], 4)
        dimensions.append(stats)

    return {
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,
        "total_output_rows": len(evaluations),
        "valid_evaluations": valid_evaluations,
        "unsafe_count": unsafe_count,
        "asr": round(unsafe_count / valid_evaluations, 4) if valid_evaluations else 0.0,
        "label_distribution": label_distribution,
        "dimensions": dimensions,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dialogue traces and calculate anthropomorphic safety ASR")
    # input: dialogue trace JSONL file path
    parser.add_argument("--input", type=str, required=True, help="Input dialogue trace JSONL file path")
    parser.add_argument("--dimension_id", type=str, default=None, help="Only evaluate dialogues for one safety dimension, such as D1")

    # judge model: the model used for evaluating Character LLM responses
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model for dialogue safety evaluation")

    # api parameters: base_url and api_key are used for Judge LLM
    parser.add_argument("--base_url", type=str, default=None, help="API Base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--max_concurrent", type=int, default=1, help="Maximum concurrent evaluations")
    parser.add_argument("--judge_temperature", type=float, default=0.0, help="Judge LLM generation temperature")

    # test mode: evaluate the first remaining item
    parser.add_argument("--test", action="store_true", help="Only evaluate the first remaining dialogue")
    args = parser.parse_args()

    # model path
    input_path = Path(args.input)
    judge_model_path = args.judge_model.replace("/", "_")
    output_path = input_path.with_name(f"evaluation_judge_model={judge_model_path}.jsonl")
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"Output path: {output_path}")

    # resume download
    dialogue_items = load_jsonl(args.input)
    if args.dimension_id:
        dialogue_items = [item for item in dialogue_items if item["dimension_id"] == args.dimension_id]
        print(f"Selected dimension: {args.dimension_id}, rows={len(dialogue_items)}")

    evaluations = load_jsonl(str(output_path)) if output_path.exists() else []
    processed_ids = {evaluation["card_id"] for evaluation in evaluations}
    if evaluations:
        print(f"Loaded existing evaluation rows: {len(evaluations)}")

    remaining_items = [item for item in dialogue_items if item["card_id"] not in processed_ids]
    if not remaining_items:
        summary = build_evaluation_summary(evaluations, args.judge_model, args.judge_temperature)
        with open(summary_path, "w", encoding="utf-8") as summary_f:
            json.dump(summary, summary_f, ensure_ascii=False, indent=2)
        print("All dialogues are already evaluated")
        print(f"Evaluation JSON saved: {jsonl_2_json(str(output_path))}")
        print(f"Evaluation summary saved: {summary_path}")
        return
    if args.test:
        remaining_items = remaining_items[:1]
        print("Test only one remaining dialogue")

    # call api client
    judge_llm = LLMClient(
        ModelConfig(
            model=args.judge_model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_concurrent=args.max_concurrent,
            temperature=args.judge_temperature,
            max_tokens=16384,
        )
    )

    # evaluate dialogues
    print(f"Start evaluating dialogues: rows={len(remaining_items)}")
    with open(output_path, "a", encoding="utf-8") as output_f:
        tasks = [evaluate_dialogue_item(item, judge_llm) for item in remaining_items]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="evaluating dialogues"):
            result = await task
            if result is None:
                continue
            evaluations.append(result)
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()

    # save and sort evaluations
    evaluations.sort(key=sort_key)
    save_jsonl(evaluations, str(output_path))
    summary = build_evaluation_summary(evaluations, args.judge_model, args.judge_temperature)
    with open(summary_path, "w", encoding="utf-8") as summary_f:
        json.dump(summary, summary_f, ensure_ascii=False, indent=2)

    print(f"Evaluation results saved: {output_path}")
    print(f"Evaluation JSON saved: {jsonl_2_json(str(output_path))}")
    print(f"Evaluation summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
