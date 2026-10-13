#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate goal-aware Persona LLM and Character LLM dialogue traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from client import LLMClient, ModelConfig
from prompt import (
    CHARACTER_USER_PROMPT_EN,
    CHARACTER_USER_PROMPT_ZH,
    TRANSLATE_SYSTEM_PROMPT_ZH,
    TRANSLATE_USER_PROMPT_ZH,
    build_character_system_prompt,
    build_persona_system_prompt,
    build_persona_user_prompt,
)
from utils import jsonl_2_json, load_jsonl, save_jsonl, sort_key


def build_dialogue_prompts(item: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue_turn = item["dialogue_turn"]
    dialogue_trace = item["dialogue_trace"]
    prompts: list[dict[str, Any]] = []

    persona_system_prompt = build_persona_system_prompt(item, language="zh")
    character_system_prompt = build_character_system_prompt(item, language="zh")
    persona_user_prompt = build_persona_user_prompt(language="zh")
    dialogue_history: list[dict[str, str]] = []

    for turn in range(1, dialogue_turn + 1):

        prompts.append(
            {
                "call": "persona_llm",
                "turn": turn,
                "system_prompt": persona_system_prompt,
                "messages": [message.copy() for message in dialogue_history],
                "user_prompt": persona_user_prompt,
            }
        )
        persona_message = dialogue_trace[(turn - 1) * 2]["content"]
        dialogue_history.append({"role": "user", "content": persona_message})

        prompts.append(
            {
                "call": "character_llm",
                "turn": turn,
                "system_prompt": character_system_prompt,
                "messages": [message.copy() for message in dialogue_history],
                "user_prompt": CHARACTER_USER_PROMPT_ZH,
            }
        )
        character_message = dialogue_trace[(turn - 1) * 2 + 1]["content"]
        dialogue_history.append({"role": "assistant", "content": character_message})

    return prompts


async def generate_dialogue_item(
    item: dict[str, Any],
    dialogue_turn: int,
    persona_llm: LLMClient,
    character_llm: LLMClient,
) -> dict[str, Any] | None:
    try:
        # Static per-item prompts: persona system (target-aware) and character system (target-blind).
        persona_system_prompt = build_persona_system_prompt(item)
        character_system_prompt = build_character_system_prompt(item)
        persona_user_prompt = build_persona_user_prompt()
        dialogue_trace: list[dict[str, str]] = []

        for turn in range(1, dialogue_turn + 1):
            # Generate Persona LLM message and write it into the shared history.
            persona_message = (await persona_llm.chat(
                system_prompt=persona_system_prompt,
                messages=dialogue_trace,
                user_prompt=persona_user_prompt,
            )).strip()
            dialogue_trace.append({"role": "user", "content": persona_message})
            print(f"{item['card_id']} turn {turn} persona_llm: {persona_message}")

            # Generate Character LLM message and write it into the shared history.
            character_message = (await character_llm.chat(
                system_prompt=character_system_prompt,
                messages=dialogue_trace,
                user_prompt=CHARACTER_USER_PROMPT_EN,
            )).strip()
            dialogue_trace.append({"role": "assistant", "content": character_message})
            print(f"{item['card_id']} turn {turn} character_llm: {character_message}")

        return {
            "persona_model": persona_llm.config.model,
            "persona_model_temperature": persona_llm.config.temperature,
            "character_model": character_llm.config.model,
            "character_model_temperature": character_llm.config.temperature,
            "dialogue_turn": dialogue_turn,
            "generation_turn_budget": dialogue_turn,
            **item,
            "dialogue_trace": dialogue_trace,
        }
    except Exception as exc:
        print(f"{item['card_id']} failed: Dialogue generation failed: {exc}")
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate goal-aware Persona LLM and Character LLM dialogue traces")
    # input: persona-character-scenario JSONL file path
    parser.add_argument("--input", type=str, default="data/persona_character_scenarios.jsonl", help="Input persona-character-scenario JSONL file path")

    # dialogue turn: turn budget of the goal-aware dialogue; the main experiment runs 1/3/5/7/9 as independent tiers
    parser.add_argument("--dialogue_turn", type=int, choices=[1, 3, 5, 7, 9], default=5, help="Dialogue turn budget")

    # persona model: the model used for dialogue planning and Persona LLM messages
    # character model: the model used for generating Character LLM messages
    # translate model: the model used for translating the test item into Chinese
    parser.add_argument("--persona_model", type=str, default="grok-4.3", help="Model for dialogue planning and Persona LLM messages")
    parser.add_argument("--character_model", type=str, default="gpt-4o-mini", help="Character LLM model to evaluate")
    parser.add_argument("--translate_model", type=str, default="deepseek-v4.1-flash", help="Model for translating the test item into Chinese")

    # api parameters: base_url and api_key are used for all LLMs; max_concurrent is used for Persona LLM and Character LLM; translate model always uses max_concurrent=1
    parser.add_argument("--base_url", type=str, default=None, help="API Base URL")
    parser.add_argument("--api_key", type=str, default=None, help="API Key")
    parser.add_argument("--max_concurrent", type=int, default=1, help="Maximum concurrent dialogue items")
    parser.add_argument("--persona_temperature", type=float, default=0.7, help="Persona LLM generation temperature")
    parser.add_argument("--character_temperature", type=float, default=0.7, help="Character LLM generation temperature")
    parser.add_argument("--translate_temperature", type=float, default=0.0, help="Translation LLM generation temperature")

    # test mode: process one random remaining item and save its Chinese translation with reconstructed prompts
    parser.add_argument("--test", action="store_true", help="Only process one random remaining item")
    parser.add_argument("--dimension_id", type=str, default=None, help="Only generate dialogues for one safety dimension, such as D1")
    # output root: main-table runs stay in result/; ablation runs pass their own root (e.g. ablation_judge) to stay isolated
    parser.add_argument("--output_root", type=str, default="result", help="Root directory for dialogue trace output")
    args = parser.parse_args()

    # model path
    persona_model_path = args.persona_model.replace("/", "_")
    character_model_path = args.character_model.replace("/", "_")
    output_path = Path(
        f"{args.output_root}/persona_model={persona_model_path}/"
        f"character_model={character_model_path}/dialogue_turn={args.dialogue_turn}/dialogue_traces.jsonl"
    )
    translation_path = output_path.with_name("dialogue_traces_test_zh.json")
    os.makedirs(output_path.parent, exist_ok=True)
    print(f"Output path: {output_path}")
    if args.test:
        print(f"Test translation path: {translation_path}")

    # resume download
    input_items = load_jsonl(args.input)
    if args.dimension_id:
        input_items = [item for item in input_items if item["dimension_id"] == args.dimension_id]
        print(f"Selected dimension: {args.dimension_id}, rows={len(input_items)}")
    dialogues = load_jsonl(str(output_path)) if output_path.exists() else []
    processed_ids = {dialogue["card_id"] for dialogue in dialogues}
    if dialogues:
        print(f"Loaded existing dialogue rows: {len(dialogues)}")
    remaining_items = [item for item in input_items if item["card_id"] not in processed_ids]
    if not remaining_items:
        print("All rows are already generated")
        print(f"Dialogue JSON saved: {jsonl_2_json(str(output_path))}")
        return
    if args.test:
        # remaining_items = remaining_items[:1]
        # print("Test only one remaining item")
        remaining_items = [random.choice(remaining_items)]
        print(f"Test one random item: card_id={remaining_items[0]['card_id']}, dimension={remaining_items[0]['dimension_id']}")

    # call api client
    # max_tokens=16384 leaves headroom for thinking models whose reasoning tokens count against the output budget.
    persona_llm = LLMClient(
        ModelConfig(
            model=args.persona_model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_concurrent=args.max_concurrent,
            temperature=args.persona_temperature,
            max_tokens=16384,
        )
    )
    character_llm = LLMClient(
        ModelConfig(
            model=args.character_model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_concurrent=args.max_concurrent,
            temperature=args.character_temperature,
            max_tokens=16384,
        )
    )
    if args.test:
        translate_llm = LLMClient(
            ModelConfig(
                model=args.translate_model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_concurrent=1,
                temperature=args.translate_temperature,
                max_tokens=16384,
            )
        )
        translated_items = json.loads(translation_path.read_text(encoding="utf-8")) if translation_path.exists() else []

    # generate dialogues
    print(f"Start generating goal-aware dialogues: rows={len(remaining_items)}, turns={args.dialogue_turn}")
    with open(output_path, "a", encoding="utf-8") as output_f:
        tasks = [
            generate_dialogue_item(
                item,
                args.dialogue_turn,
                persona_llm,
                character_llm,
            )
            for item in remaining_items
        ]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="generating goal-aware dialogues"):
            result = await task
            if result is None:
                continue
            dialogues.append(result)
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()
            if args.test:
                try:
                    translated_item = await translate_llm.chat_json(
                        system_prompt=TRANSLATE_SYSTEM_PROMPT_ZH,
                        user_prompt=TRANSLATE_USER_PROMPT_ZH.format(json_item=json.dumps(result, ensure_ascii=False, indent=2)),
                    )
                    translated_item["dialogue_turn"] = result["dialogue_turn"]
                    translated_item["prompts"] = build_dialogue_prompts(translated_item)
                    translated_items.append(translated_item)
                    with open(translation_path, "w", encoding="utf-8") as translation_f:
                        json.dump(translated_items, translation_f, ensure_ascii=False, indent=2)
                    print(f"Translated test item saved: {translation_path}")
                except Exception as exc:
                    print(f"{result['card_id']} failed: Test item translation failed: {exc}")

    # save and sort dialogues
    dialogues.sort(key=sort_key)
    save_jsonl(dialogues, str(output_path))
    print(f"Dialogue traces saved: {output_path}")
    print(f"Dialogue JSON saved: {jsonl_2_json(str(output_path))}")


if __name__ == "__main__":
    asyncio.run(main())
