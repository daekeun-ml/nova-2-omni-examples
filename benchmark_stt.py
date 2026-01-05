#!/usr/bin/env python3
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Audio
import boto3
import base64
import json
from jiwer import wer, cer
import numpy as np
from tqdm import tqdm
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from common import DEFAULT_MODEL_ID, DEFAULT_REGION_ID

CONCURRENT_WORKERS = 15

class NovaSTTBenchmark:
    def __init__(self):
        self.bedrock = boto3.client('bedrock-runtime', region_name=DEFAULT_REGION_ID)
        self.model_id = DEFAULT_MODEL_ID
        
    def call_nova_stt(self, audio_data):
        start_time = time.time()
        
        # Encode audio to base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "audio": {
                                "format": "wav",
                                "source": {"bytes": audio_b64}
                            }
                        },
                        {"text": "Please transcribe the Korean speech to text. Perform speech-to-text (STT) conversion and provide only the transcribed text without any additional commentary or formatting."}
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 1024,
                "temperature": 0.1
            }
        }
        
        ttft_time = None
        response = self.bedrock.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        result_text = ""
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            if 'contentBlockDelta' in chunk:
                if ttft_time is None:
                    ttft_time = time.time() - start_time
                result_text += chunk['contentBlockDelta']['delta']['text']
        
        end_time = time.time()
        return {
            'text': result_text.strip(),
            'ttft': ttft_time or (end_time - start_time),
            'end_to_end': end_time - start_time
        }

    def process_sample(self, sample):
        try:
            import soundfile as sf
            import io
            
            # Get raw audio bytes from the dataset
            audio_bytes = sample['audio']['bytes']
            
            # Load audio using soundfile
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            wav_bytes = buffer.getvalue()
            
            result = self.call_nova_stt(wav_bytes)
            
            reference = sample['text']
            hypothesis = result['text']
            
            return {
                'reference': reference,
                'hypothesis': hypothesis,
                'cer': cer(reference, hypothesis),
                'wer': wer(reference, hypothesis),
                'ttft': result['ttft'],
                'end_to_end': result['end_to_end'],
                'sample_id': sample.get('id', 'unknown')
            }
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    def run_benchmark(self, num_samples=100):
        print(f"=== STT Benchmark Configuration ===")
        print(f"Model ID: {self.model_id}")
        print(f"Region: {DEFAULT_REGION_ID}")
        print(f"Concurrent Workers: {CONCURRENT_WORKERS}")
        print(f"Sample Limit: {num_samples}")
        print(f"Dataset: kresnik/zeroth_korean (Korean STT)")
        print()
        
        print("Loading dataset...")
        test_dataset = load_dataset("kresnik/zeroth_korean", split="test")
        # Disable audio decoding to avoid torchcodec issues
        test_dataset = test_dataset.cast_column("audio", Audio(decode=False))
        samples = test_dataset.select(range(min(num_samples, len(test_dataset))))
        
        print(f"Running benchmark on {len(samples)} samples with {CONCURRENT_WORKERS} workers...")
        
        results = []
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            # Submit all tasks
            future_to_sample = {executor.submit(self.process_sample, sample): i 
                              for i, sample in enumerate(samples)}
            
            # Progress bar for completed tasks
            with tqdm(total=len(samples), desc="Processing samples", unit="sample") as pbar:
                for future in as_completed(future_to_sample):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
        
        if not results:
            print("No successful results!")
            return
        
        # Calculate metrics
        cer_scores = [r['cer'] for r in results]
        wer_scores = [r['wer'] for r in results]
        ttft_times = [r['ttft'] for r in results]
        e2e_times = [r['end_to_end'] for r in results]
        
        print("\n=== BENCHMARK RESULTS ===")
        print(f"Samples processed: {len(results)}")
        print(f"\nAccuracy Metrics:")
        print(f"CER: {np.mean(cer_scores):.4f} (±{np.std(cer_scores):.4f})")
        print(f"WER: {np.mean(wer_scores):.4f} (±{np.std(wer_scores):.4f})")
        
        print(f"\nLatency Metrics (seconds):")
        print(f"TTFT - P50: {np.percentile(ttft_times, 50):.3f}, P95: {np.percentile(ttft_times, 95):.3f}, P99: {np.percentile(ttft_times, 99):.3f}")
        print(f"E2E  - P50: {np.percentile(e2e_times, 50):.3f}, P95: {np.percentile(e2e_times, 95):.3f}, P99: {np.percentile(e2e_times, 99):.3f}")
        
        # Save detailed results
        detailed_results = []
        for r in results:
            detailed_results.append({
                'sample_id': r['sample_id'],
                'reference_text': r['reference'],
                'predicted_text': r['hypothesis'],
                'cer': r['cer'],
                'wer': r['wer'],
                'ttft_seconds': r['ttft'],
                'end_to_end_seconds': r['end_to_end']
            })
        
        # Create benchmark directory if it doesn't exist
        os.makedirs('benchmark', exist_ok=True)
        
        with open('benchmark/benchmark_stt_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'samples': len(results),
                    'cer_mean': float(np.mean(cer_scores)),
                    'cer_std': float(np.std(cer_scores)),
                    'wer_mean': float(np.mean(wer_scores)),
                    'wer_std': float(np.std(wer_scores)),
                    'ttft_p50': float(np.percentile(ttft_times, 50)),
                    'ttft_p95': float(np.percentile(ttft_times, 95)),
                    'ttft_p99': float(np.percentile(ttft_times, 99)),
                    'e2e_p50': float(np.percentile(e2e_times, 50)),
                    'e2e_p95': float(np.percentile(e2e_times, 95)),
                    'e2e_p99': float(np.percentile(e2e_times, 99))
                },
                'detailed_results': detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed results saved to benchmark/benchmark_stt_results.json")

def analyze_benchmark_results(json_file_path):
    """Analyze existing benchmark results from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'summary' in data:
            # Use summary data if available
            summary = data['summary']
            print("\n=== BENCHMARK ANALYSIS ===")
            print(f"Samples processed: {summary['samples']}")
            print(f"\nAccuracy Metrics:")
            print(f"CER: {summary['cer_mean']:.4f} (±{summary['cer_std']:.4f})")
            print(f"WER: {summary['wer_mean']:.4f} (±{summary['wer_std']:.4f})")
            
            print(f"\nLatency Metrics (seconds):")
            print(f"TTFT - P50: {summary['ttft_p50']:.3f}, P95: {summary['ttft_p95']:.3f}, P99: {summary['ttft_p99']:.3f}")
            print(f"E2E  - P50: {summary['e2e_p50']:.3f}, P95: {summary['e2e_p95']:.3f}, P99: {summary['e2e_p99']:.3f}")
        
        elif 'detailed_results' in data:
            # Calculate from detailed results if summary not available
            results = data['detailed_results']
            cer_scores = [r['cer'] for r in results]
            wer_scores = [r['wer'] for r in results]
            ttft_times = [r['ttft_seconds'] for r in results]
            e2e_times = [r['end_to_end_seconds'] for r in results]
            
            print("\n=== BENCHMARK ANALYSIS ===")
            print(f"Samples processed: {len(results)}")
            print(f"\nAccuracy Metrics:")
            print(f"CER: {np.mean(cer_scores):.4f} (±{np.std(cer_scores):.4f})")
            print(f"WER: {np.mean(wer_scores):.4f} (±{np.std(wer_scores):.4f})")
            
            print(f"\nLatency Metrics (seconds):")
            print(f"TTFT - P50: {np.percentile(ttft_times, 50):.3f}, P95: {np.percentile(ttft_times, 95):.3f}, P99: {np.percentile(ttft_times, 99):.3f}")
            print(f"E2E  - P50: {np.percentile(e2e_times, 50):.3f}, P95: {np.percentile(e2e_times, 95):.3f}, P99: {np.percentile(e2e_times, 99):.3f}")
        
        else:
            print("Error: Invalid JSON format. Expected 'summary' or 'detailed_results' key.")
            
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file_path}'.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Nova STT Benchmark')
    parser.add_argument('--analyze', type=str, help='Analyze existing results from JSON file')
    parser.add_argument('--num_samples', type=int, default=457, help='Number of samples to process')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_benchmark_results(args.analyze)
    else:
        benchmark = NovaSTTBenchmark()
        benchmark.run_benchmark(num_samples=args.num_samples)
