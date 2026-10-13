import json
from typing import Dict, List, Any, Optional, Set, Tuple
import random
import torch
import os
import numpy as np

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cuda_capability = torch.cuda.get_device_capability()
        if cuda_capability[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_SDP_ATTENTION"] = "never"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_jsonl(filepath):
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """保存为JSONL格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def sort_key(item: dict[str, Any]) -> tuple[int, str]:
    dimension_id = str(item.get("dimension_id", ""))
    dimension_number = int(dimension_id[1:]) if dimension_id.startswith("D") and dimension_id[1:].isdigit() else 999
    return dimension_number, str(item.get("card_id", ""))

def jsonl_2_json(jsonl_path: str) -> str:
    data = load_jsonl(jsonl_path)
    json_path = jsonl_path.replace(".jsonl", "_2_json.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return json_path
