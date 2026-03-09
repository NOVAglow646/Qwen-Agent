# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional


DatasetProcessor = Callable[[List[dict]], List[dict]]
DATASET_PROCESSOR_REGISTRY: Dict[str, DatasetProcessor] = {}


def register_dataset_processor(dataset_name: str):

    def decorator(fn: DatasetProcessor):
        DATASET_PROCESSOR_REGISTRY[dataset_name] = fn
        return fn

    return decorator


def _load_json(file_path: Path) -> List[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError('JSON list items must be objects (dict).')
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError('Unsupported JSON structure. Expect a dict or a list of dicts.')


def _load_jsonl(file_path: Path) -> List[dict]:
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f'Invalid JSONL at line {line_id}: {ex}') from ex
            if not isinstance(obj, dict):
                raise ValueError(f'JSONL line {line_id} must be a JSON object.')
            records.append(obj)
    return records


def _load_parquet(file_path: Path) -> List[dict]:
    try:
        import pandas as pd
    except ImportError as ex:
        raise ImportError('Reading parquet requires pandas and pyarrow. Please install them first.') from ex

    df = pd.read_parquet(file_path)
    return df.to_dict(orient='records')


def _select_samples_by_spec_ids(samples: List[dict], spec_ids: Optional[List[int]] = None) -> List[dict]:
    if spec_ids is None:
        return samples

    selected_samples = []
    for idx in spec_ids:
        if not isinstance(idx, int):
            raise TypeError(f'All spec_ids must be int, but got {type(idx).__name__}: {idx}')
        if idx < 0 or idx >= len(samples):
            raise IndexError(f'spec_id {idx} out of range for {len(samples)} samples.')
        selected_samples.append(samples[idx])
    return selected_samples


def load_raw_samples(file_path: str, spec_ids: Optional[List[int]] = None) -> List[dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {file_path}')

    suffix = path.suffix.lower()
    if suffix == '.json':
        samples = _load_json(path)
        return _select_samples_by_spec_ids(samples=samples, spec_ids=spec_ids)
    if suffix == '.jsonl':
        samples = _load_jsonl(path)
        return _select_samples_by_spec_ids(samples=samples, spec_ids=spec_ids)
    if suffix == '.parquet':
        samples = _load_parquet(path)
        return _select_samples_by_spec_ids(samples=samples, spec_ids=spec_ids)

    raise ValueError(f'Unsupported file extension: {suffix}. Only .json/.jsonl/.parquet are supported.')


def default_dataset_processor(samples: List[dict]) -> List[dict]:
    """Default dataset processor.

    TODO: Add dataset-specific normalization logic to obtain standardized fields
    such as image/question for each benchmark dataset.
    """
    return samples


@register_dataset_processor('AdaTool_SFT')
def process_adatool_sft(samples: List[dict]) -> List[dict]:
    processed = []

    for sample in samples:
        data = sample.get('data', sample) if isinstance(sample, dict) else {}

        images = data.get('images', []) if isinstance(data, dict) else []
        image = images[0] if isinstance(images, list) and images else None

        conversations = data.get('conversations', []) if isinstance(data, dict) else []

        question_raw = ''
        if isinstance(conversations, list) and len(conversations) > 1 and isinstance(conversations[1], dict):
            question_raw = str(conversations[1].get('value', ''))
        question = question_raw.split('\n\nGuidelines:')[0].strip()

        answer_raw = ''
        if isinstance(conversations, list) and conversations and isinstance(conversations[-1], dict):
            answer_raw = str(conversations[-1].get('value', ''))
        match = re.search(r'<answer>(.*?)</answer>', answer_raw, flags=re.DOTALL)
        answer = match.group(1).strip() if match else ''

        processed.append({
            'image': image,
            'question': question,
            'answer': answer,
        })

    return processed


def process_samples_by_dataset(samples: List[dict], dataset_name: Optional[str] = None) -> List[dict]:
    if not dataset_name:
        return default_dataset_processor(samples)

    processor = DATASET_PROCESSOR_REGISTRY.get(dataset_name, default_dataset_processor)
    return processor(samples)


def load_samples(file_path: str, dataset_name: Optional[str] = None, spec_ids: Optional[List[int]] = None) -> List[dict]:
    samples = load_raw_samples(file_path=file_path, spec_ids=spec_ids)
    return process_samples_by_dataset(samples=samples, dataset_name=dataset_name)
