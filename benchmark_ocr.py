#!/usr/bin/env python3
import time
import json
import base64
import boto3
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from io import BytesIO
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from eval_metrics.ocr_metrics import evaluate_sample
from common import DEFAULT_MODEL_ID, DEFAULT_REGION_ID

CONCURRENT_WORKERS = 15

class NovaOCRBenchmark:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime', region_name=DEFAULT_REGION_ID)
        self.model_id = DEFAULT_MODEL_ID
        
    def evaluate_prediction(self, prediction, answers):
        """Check if prediction matches any of the reference answers"""
        prediction = prediction.lower().strip()
        for answer in answers:
            if answer.lower().strip() in prediction or prediction in answer.lower().strip():
                return True
        return False
    
    def encode_image_from_pil(self, pil_image):
        """Encode PIL Image to base64"""
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def call_nova_ocr(self, pil_image, question):
        """Call Nova 2 Omni for OCR task"""
        start_time = time.time()
        
        try:
            image_b64 = self.encode_image_from_pil(pil_image)
            
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": image_b64}
                                }
                            },
                            {"text": question}
                        ]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 2048,
                    "temperature": 0.1
                }
            }
            
            ttft_time = time.time()
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            end_time = time.time()
            
            prediction = response_body['output']['message']['content'][0]['text']
            
            return {
                'prediction': prediction,
                'ttft': ttft_time - start_time,
                'end_to_end': end_time - start_time,
                'api_success': True
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                'prediction': '',
                'ttft': 0,
                'end_to_end': end_time - start_time,
                'api_success': False,
                'error': str(e)
            }
    
    def process_sample(self, sample):
        """Process a single sample"""
        pil_image = sample['image']
        question = sample['question']
        answers = sample['answers']
        
        result = self.call_nova_ocr(pil_image, question)
        
        # Create sample dict for evaluation
        eval_sample = {
            'predict': result['prediction'],
            'answers': answers,
            'type': sample.get('type', 'hf_dataset'),
            'bbox': sample.get('bbox', None),
            'bbox_list': sample.get('bbox_list', None),
            'content': sample.get('content', '')
        }
        
        # Evaluate with multiple metrics
        scores = evaluate_sample(eval_sample)
        
        return {
            'dataset_name': sample.get('dataset_name', 'ocrbench_v2'),
            'type': sample.get('type', 'hf_dataset'),
            'id': sample.get('id', 0),
            'question': question,
            'answers': answers,
            'predict': result['prediction'],
            'api_success': result['api_success'],
            'text_match': scores['text_match'] > 0,
            'teds_score': scores['teds'],
            'iou_score': scores['iou'],
            'vqa_anls': scores['vqa_anls'],
            'bleu_score': scores['bleu'],
            'f_measure': scores['f_measure'],
            'avg_anls': scores['avg_anls'],
            'error': result.get('error', ''),
            'ttft': result['ttft'],
            'end_to_end': result['end_to_end']
        }
    
    def run_benchmark(self, num_samples=100, output_file=None, task_filter=None):
        """Run OCR benchmark using HuggingFace dataset"""
        
        print(f"=== OCRBench v2 Configuration ===")
        print(f"Model ID: {self.model_id}")
        print(f"Region: {DEFAULT_REGION_ID}")
        print(f"Concurrent Workers: {CONCURRENT_WORKERS}")
        print(f"Sample Limit: {num_samples}")
        print(f"Task Filter: {task_filter if task_filter else 'All tasks'}")
        print(f"Output File: {output_file}")
        print()
        
        print(f"Loading OCRBench v2 dataset from HuggingFace (streaming)...")
        
        dataset = load_dataset("ling99/OCRBench_v2", streaming=True, split="test")
        
        samples = []
        task_counts = {}
        
        for sample in dataset:
            sample_type = sample.get('type', '')
            
            # Apply task filter if specified
            if task_filter:
                if task_filter.lower() not in sample_type.lower():
                    continue
            
            # Count task types
            task_counts[sample_type] = task_counts.get(sample_type, 0) + 1
            
            samples.append(sample)
            
            if len(samples) >= num_samples:
                break
        
        print(f"Collected {len(samples)} samples")
        print("Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")
        print()
        
        print(f"Processing {len(samples)} samples...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = {executor.submit(self.process_sample, sample): sample for sample in samples}
            
            for future in tqdm(as_completed(futures), total=len(samples), desc="Processing samples"):
                result = future.result()
                results.append(result)
        
        # Calculate statistics
        api_success = [r for r in results if r['api_success']]
        api_failed = [r for r in results if not r['api_success']]
        text_correct = [r for r in results if r.get('text_match', False)]
        
        # Calculate average scores with N/A handling based on task types
        teds_scores = [r['teds_score'] for r in results if r['api_success'] and r['teds_score'] > 0]
        iou_scores = [r['iou_score'] for r in results if r['api_success'] and r['iou_score'] > 0]
        vqa_scores = [r['vqa_anls'] for r in results if r['api_success'] and r['vqa_anls'] > 0]
        bleu_scores = [r['bleu_score'] for r in results if r['api_success'] and r['bleu_score'] > 0]
        f_scores = [r['f_measure'] for r in results if r['api_success'] and r['f_measure'] > 0]
        anls_scores = [r['avg_anls'] for r in results if r['api_success'] and r['avg_anls'] > 0]
        
        if api_success:
            ttft_times = [r['ttft'] for r in api_success]
            e2e_times = [r['end_to_end'] for r in api_success]
            
            stats = {
                'total_samples': len(results),
                'api_success': len(api_success),
                'api_failed': len(api_failed),
                'text_correct': len(text_correct),
                'api_success_rate': len(api_success) / len(results) * 100,
                'text_accuracy': len(text_correct) / len(results) * 100,
                'avg_teds': sum(teds_scores) / len(teds_scores) if teds_scores else "N/A",
                'avg_iou': sum(iou_scores) / len(iou_scores) if iou_scores else "N/A",
                'avg_vqa_anls': sum(vqa_scores) / len(vqa_scores) if vqa_scores else "N/A",
                'avg_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else "N/A",
                'avg_f_measure': sum(f_scores) / len(f_scores) if f_scores else "N/A",
                'avg_anls': sum(anls_scores) / len(anls_scores) if anls_scores else "N/A",
                'avg_ttft': sum(ttft_times) / len(ttft_times),
                'avg_e2e': sum(e2e_times) / len(e2e_times),
                'p50_e2e': sorted(e2e_times)[len(e2e_times)//2],
                'p95_e2e': sorted(e2e_times)[int(len(e2e_times)*0.95)],
                'p99_e2e': sorted(e2e_times)[int(len(e2e_times)*0.99)]
            }
        else:
            stats = {
                'total_samples': len(results),
                'api_success': 0,
                'api_failed': len(api_failed),
                'text_correct': 0,
                'api_success_rate': 0,
                'text_accuracy': 0,
                'avg_teds': "N/A",
                'avg_iou': "N/A",
                'avg_vqa_anls': "N/A",
                'avg_bleu': "N/A",
                'avg_f_measure': "N/A",
                'avg_anls': "N/A",
                'avg_ttft': 0,
                'avg_e2e': 0,
                'p50_e2e': 0,
                'p95_e2e': 0,
                'p99_e2e': 0
            }
        
        # Print results
        print("\n=== OCRBench v2 Results (All Metrics) ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"API Success: {stats['api_success']}")
        print(f"API Failed: {stats['api_failed']}")
        print(f"Text Correct: {stats['text_correct']}")
        print(f"API Success Rate: {stats['api_success_rate']:.1f}%")
        print(f"Text Accuracy: {stats['text_accuracy']:.1f}%")
        print(f"Average TEDS: {stats['avg_teds'] if stats['avg_teds'] != 'N/A' else 'N/A'}")
        print(f"Average IoU: {stats['avg_iou'] if stats['avg_iou'] != 'N/A' else 'N/A'}")
        print(f"Average VQA ANLS: {stats['avg_vqa_anls'] if stats['avg_vqa_anls'] != 'N/A' else 'N/A'}")
        print(f"Average BLEU: {stats['avg_bleu'] if stats['avg_bleu'] != 'N/A' else 'N/A'}")
        print(f"Average F-measure: {stats['avg_f_measure'] if stats['avg_f_measure'] != 'N/A' else 'N/A'}")
        print(f"Average ANLS: {stats['avg_anls'] if stats['avg_anls'] != 'N/A' else 'N/A'}")
        
        if api_success:
            print(f"Average E2E: {stats['avg_e2e']:.3f}s")
            print(f"P50 E2E: {stats['p50_e2e']:.3f}s")
            print(f"P95 E2E: {stats['p95_e2e']:.3f}s")
            print(f"P99 E2E: {stats['p99_e2e']:.3f}s")
        
        # Save results
        if output_file:
            output_data = {'results': results, 'statistics': stats}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        return results, stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='OCRBench v2 benchmark for Amazon Nova 2 Omni')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', default='benchmark/benchmark_ocr_results.json', help='Output file')
    parser.add_argument('--task_filter', type=str, help='Filter by task type (e.g., "table", "agent", "ocr")')
    
    args = parser.parse_args()
    
    os.makedirs('benchmark', exist_ok=True)
    
    benchmark = NovaOCRBenchmark()
    results, stats = benchmark.run_benchmark(args.num_samples, args.output, args.task_filter)

if __name__ == "__main__":
    main()
