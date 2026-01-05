# Amazon Nova 2 Omni (Preview) Multimodal Demo & Benchmark

A Streamlit-based demo application and benchmarking suite to experience and evaluate the powerful multimodal AI capabilities of Amazon Nova 2 Omni.

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

## 🧪 Audio Benchmarking

### STT (Speech-to-Text) Benchmark

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
- Detailed results saved to JSON

**Run benchmark:**
```bash
# Install benchmark dependencies
uv sync --group benchmark

# Run STT benchmark on all 457 test samples
uv run benchmark_stt.py

# Analyze existing results
uv run benchmark_stt.py --analyze benchmark/benchmark_stt_results.json
```

**Output:**
- Console: Summary statistics (CER/WER averages, latency percentiles)
- `benchmark/benchmark_stt_results.json`: Detailed per-sample results with reference/predicted text

---

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

---

## References

For detailed information about Amazon Nova 2 Omni, please refer to the [AWS official documentation](https://aws.amazon.com/nova/).

**Note**: This demo is created for experiencing Amazon Nova 2 Omni's capabilities. Please perform appropriate security and performance optimization before using in production environments.
