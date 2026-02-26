"""Parallel sample-level inference demo with tool-calling agents.

Usage example:
python examples/parallel_inference.py \
  --input_path path/to/data.jsonl \
  --dataset_name my_dataset \
  --model_backend gemini \
  --workers 8 \
  --output_path outputs/predictions.json
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from qwen_agent.agents import FnCallAgent
from qwen_agent.utils.data_loader import load_samples


def _build_agent(model_backend: str) -> FnCallAgent:
	if model_backend == 'qwen3vl':
		llm_cfg = {
			'model_type': 'qwenvl_dashscope',
			'model': os.getenv('QWEN3_VL_MODEL', 'qwen3-vl-plus'),
			'api_key': os.getenv('DASHSCOPE_API_KEY'),
		}
	elif model_backend == 'gemini':
		llm_cfg = {
			'model_type': 'gemini',
			'model': os.getenv('GEMINI_MODEL', 'gemini-2.5-pro'),
			'location': os.getenv('GEMINI_LOCATION', 'global'),
			'credential_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
			'seed': int(os.getenv('GEMINI_SEED', '42')),
		}
	else:
		raise ValueError(f'Unsupported model_backend: {model_backend}')

	tools = [
		'image_zoom_in_tool',
		'image_search',
		'web_search',
	]
	return FnCallAgent(
		llm=llm_cfg,
		function_list=tools,
		name=f'ParallelInference-{model_backend}',
		system_message='',
	)


def _build_messages(sample: Dict) -> List[Dict]:
	image = sample.get('image') or sample.get('image_url') or sample.get('img_url')
	question = sample.get('question') or sample.get('query') or sample.get('prompt')
	if not question:
		raise ValueError('Sample must contain one of: question/query/prompt')

	content = []
	if image:
		content.append({'image': image})
	content.append({'text': question})
	return [{'role': 'user', 'content': content}]


def _infer_one(sample_idx: int, sample: Dict, model_backend: str, print_lock: threading.Lock) -> Dict:
	started_at = time.time()
	sample_id = sample.get('id', sample_idx)
	try:
		bot = _build_agent(model_backend=model_backend)
		messages = _build_messages(sample)
		response = list(bot.run(messages=messages))[-1]
		answer = response[-1].get('content', '') if response else ''

		result = {
			'index': sample_idx,
			'id': sample_id,
			'success': True,
			'answer': answer,
			'latency_sec': round(time.time() - started_at, 4),
		}
	except Exception as ex:
		result = {
			'index': sample_idx,
			'id': sample_id,
			'success': False,
			'error': str(ex),
			'latency_sec': round(time.time() - started_at, 4),
		}

	with print_lock:
		state = 'OK' if result['success'] else 'FAIL'
		print(f'[{state}] sample={sample_id} idx={sample_idx} time={result["latency_sec"]}s')
	return result


def run_parallel_inference(
	input_path: str,
	dataset_name: str,
	model_backend: str,
	workers: int,
	output_path: str,
	max_samples: int = -1,
) -> Dict:
	samples = load_samples(file_path=input_path, dataset_name=dataset_name)
	if max_samples and max_samples > 0:
		samples = samples[:max_samples]

	print(f'Loaded samples: {len(samples)}')
	print(f'Model backend: {model_backend}')
	print(f'Workers: {workers}')

	print_lock = threading.Lock()
	results = [None] * len(samples)
	started_at = time.time()

	with ThreadPoolExecutor(max_workers=workers) as executor:
		future_to_idx = {
			executor.submit(_infer_one, i, sample, model_backend, print_lock): i for i, sample in enumerate(samples)
		}
		for future in as_completed(future_to_idx):
			idx = future_to_idx[future]
			results[idx] = future.result()

	elapsed = round(time.time() - started_at, 4)
	success = sum(1 for x in results if x and x.get('success'))
	summary = {
		'total': len(results),
		'success': success,
		'failed': len(results) - success,
		'elapsed_sec': elapsed,
		'results': results,
	}

	if output_path:
		os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(summary, f, ensure_ascii=False, indent=2)
		print(f'Results saved to: {output_path}')

	print(f"Done: total={summary['total']} success={summary['success']} failed={summary['failed']} elapsed={elapsed}s")
	return summary


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, required=True, help='Input data path: json/jsonl/parquet')
	parser.add_argument('--dataset_name', type=str, default='generic', help='Dataset name for custom preprocessing')
	parser.add_argument('--model_backend', type=str, default='qwen3vl', choices=['qwen3vl', 'gemini'])
	parser.add_argument('--workers', type=int, default=8, help='Sample-level parallel workers')
	parser.add_argument('--max_samples', type=int, default=-1, help='Only run first N samples, -1 means all')
	parser.add_argument('--output_path', type=str, default='outputs/parallel_inference_results.json')
	return parser.parse_args()


def main():
	args = parse_args()
	run_parallel_inference(
		input_path=args.input_path,
		dataset_name=args.dataset_name,
		model_backend=args.model_backend,
		workers=args.workers,
		output_path=args.output_path,
		max_samples=args.max_samples,
	)


if __name__ == '__main__':
	main()

