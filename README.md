# Amazon Nova 2 Omni (Preview) Multimodal Demo & Benchmark

A Streamlit-based demo application and comprehensive benchmarking suite to experience and evaluate the powerful multimodal AI capabilities of Amazon Nova 2 Omni.

[한국어 README](README_ko.md)

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher
- AWS account with Bedrock service access
- AWS credentials configured: `aws configure`

### Quick Start

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the project**
   ```bash
   git clone https://github.com/daekeun-ml/nova-2-omni-examples
   cd nova-2-omni-examples
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Run the demo**
   ```bash
   ./run_demo.sh
   ```
   
   Or run directly:
   ```bash
   uv run streamlit run main.py
   ```

5. **Access in browser**
   - Local: http://localhost:8501
   - Select desired features from the sidebar and explore

## 🧪 Benchmarking

### OCR Benchmarking (OCRBench v2)

Comprehensive OCR performance evaluation using the [OCRBench v2 dataset](https://huggingface.co/datasets/ling99/OCRBench_v2) with multiple metrics.

**Supported Metrics:**
- **Text Accuracy**: Basic text matching accuracy
- **TEDS**: Table Edit Distance based Similarity (for table parsing tasks)
- **IoU**: Intersection over Union (for localization tasks)
- **VQA ANLS**: VQA task evaluation with ANLS scoring
- **BLEU**: Machine translation quality metric (for OCR tasks)
- **F-measure**: Precision and recall harmonic mean (for OCR tasks)
- **ANLS**: Average Normalized Levenshtein Similarity

**Run OCR benchmark:**
```bash
# Install benchmark dependencies
uv sync --group benchmark

# Run with 100 samples (default)
uv run python benchmark_ocr.py

# Run with specific number of samples
uv run python benchmark_ocr.py --num_samples 200

# Filter by task type to test specific metrics
uv run python benchmark_ocr.py --num_samples 50 --task_filter "table"  # TEDS metric
uv run python benchmark_ocr.py --num_samples 50 --task_filter "ocr"    # BLEU, F-measure
uv run python benchmark_ocr.py --num_samples 50 --task_filter "agent"  # IoU metric
uv run python benchmark_ocr.py --num_samples 50 --task_filter "vqa"    # VQA ANLS
```

**Output:**
- Console: Real-time progress and comprehensive metrics
- `benchmark/benchmark_ocr_results.json`: Detailed per-sample results

### STT Benchmarking (Korean)

Benchmark Amazon Nova 2 Omni's Korean speech recognition performance using the [Zeroth-Korean dataset](https://huggingface.co/datasets/kresnik/zeroth_korean).

**Dataset Overview:**
- **Total Data**: 51.6 hours of training data and 1.2 hours of test data
- **Utterances**: 22,263 training utterances and 457 test utterances  
- **Speakers**: 105 training speakers and 10 test speakers
- **Sampling Rate**: 16kHz

**Features:**
- CER (Character Error Rate) and WER (Word Error Rate) metrics
- Latency measurements (TTFT, End-to-End) with P50/P95/P99 percentiles
- Parallel processing with 15 concurrent workers
- Progress tracking with real-time progress bars

**Run STT benchmark:**
```bash
# Run STT benchmark on all 457 test samples
uv run python benchmark_stt.py

# Run with specific number of samples
uv run python benchmark_stt.py --num_samples 100

# Analyze existing results
uv run python benchmark_stt.py --analyze benchmark/benchmark_stt_results.json
```

**Output:**
- Console: Summary statistics (CER/WER averages, latency percentiles)
- `benchmark/benchmark_stt_results.json`: Detailed per-sample results

## 📊 Benchmark Configuration

Both benchmarks support the following configuration:
- **Concurrent Workers**: 15 (for parallel processing)
- **Model**: us.amazon.nova-2-omni-v1:0
- **Region**: us-west-2
- **Metrics**: Task-specific evaluation metrics
- **Progress Tracking**: Real-time tqdm progress bars

## 🤖 About Amazon Nova 2 Omni

Amazon Nova 2 Omni is Amazon's next-generation multimodal reasoning and image generation model. It's a multimodal model that supports text, image, video, and speech inputs while generating both text and image outputs.

### 🌟 Key Features

#### Multimodal Understanding and Generation
- Unified processing of **text, image, video, and speech** inputs
- Native generation of **text and image** outputs
- Single model for diverse tasks without managing multiple AI models

#### Advanced Reasoning Capabilities
- **1M token context window** for large document processing
- Complex multi-step reasoning and long-term planning
- Flexible reasoning control for performance, accuracy, and cost optimization

#### Language and Speech Support
- Support for **200+ languages** in text processing
- **10 languages** supported for speech input (as of 2026.01 Preview)
- Multi-speaker conversation transcription, translation, and summarization

#### Image Generation and Editing
- High-quality image generation and editing using natural language
- Character consistency maintenance
- Text rendering within images
- Object and background modification capabilities

#### Speech Understanding
- Superior speech understanding through native reasoning
- Multi-speaker conversation transcription, translation, and summarization
- Real-time customer interaction support

### 🏢 Use Cases

- **Customer Service**: Multimodal chatbots and support systems
- **Content Creation**: Marketing materials and advertising creative production
- **Document Analysis**: Large document and video content analysis
- **Speech Processing**: Meeting transcription, translation, and summarization
- **Visual Search**: Image and video-based search systems

## 📁 Project Structure

```
nova-2-omni-examples/
├── main.py                    # Streamlit demo application
├── benchmark_ocr.py           # OCR benchmark (OCRBench v2)
├── benchmark_stt.py           # STT benchmark (Zeroth-Korean)
├── src/
│   ├── common.py             # Common utilities and configurations
│   └── eval_metrics/
│       └── ocr_metrics.py    # OCR evaluation metrics
├── benchmark/                # Benchmark results directory
└── README.md                 # This file
```

## 🔧 Development

### Adding New Benchmarks

1. Create a new benchmark script following the existing pattern
2. Add evaluation metrics to `src/eval_metrics/`
3. Update dependencies in `pyproject.toml`
4. Add documentation to README

### Configuration

- Model and region settings: `src/common.py`
- Benchmark parameters: Command-line arguments
- Dependencies: `pyproject.toml`

---

## References

For detailed information about Amazon Nova 2 Omni, please refer to the [AWS official documentation](https://aws.amazon.com/nova/).

**Note**: This demo is created for experiencing Amazon Nova 2 Omni's capabilities. Please perform appropriate security and performance optimization before using in production environments.
