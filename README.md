# Voice Inference Benchmark

Benchmarks three self-deployed LLMs on [Baseten](https://baseten.co) against a sub-700ms p99 latency target for voice dictation transcript cleanup — a common real-time AI use case.

## The question

> A voice dictation app needs to clean up spoken transcripts in real time. Which model hits sub-700ms p99 total latency, and what does it cost?

## Models tested

| Model | GPU | Cost/min | Parameters |
|---|---|---|---|
| Qwen 2.5 3B Instruct | L4 | $0.01414 | 3B |
| Mistral 7B Instruct v0.3 | L4 | $0.01414 | 7B |
| Llama 3.3 70B Instruct | H100×4 | $0.43333 | 70B |

All models are deployed via [Baseten Engine Builder](https://docs.baseten.co/inference/engine-builder) (TRT-LLM, fp8 quantization, config-only — no custom code). Billing is per GPU-minute, not per token.

## Key findings

- **Llama 3.3 70B** is the only model that reliably meets the 700ms p99 target at every concurrency level tested. Qwen 2.5 3B comes close at low concurrency but degrades under load.
- **Mistral 7B** underperforms both smaller models on the same L4 GPU — model family and architecture matter as much as parameter count.
- **Cost tradeoff**: Llama costs ~30× more per request than Qwen. At scale, multiple Qwen replicas behind a load balancer could match Llama's consistency at a fraction of the GPU spend.

## Repo structure

```
models/
  qwen-2.5-3b/config.yaml       # Engine Builder configs (truss push)
  mistral-7b/config.yaml
  llama-3.3-70b/config.yaml
scripts/
  benchmark.py                  # Async benchmark — streams all 3 models
dashboard/
  index.html                    # Single-file dashboard (Chart.js + PapaParse)
results/
  benchmark_results.csv         # Pre-run results — open dashboard immediately
  samples/                      # Sample model outputs for quality review
```

## Run it yourself

### 1. Deploy the models

```bash
pip install truss
cd models/qwen-2.5-3b  && truss push
cd models/mistral-7b   && truss push
cd models/llama-3.3-70b && truss push
```

> Llama 3.3 70B is a gated model. You'll need to accept Meta's license on Hugging Face and add your HF token to Baseten secrets as `hf_access_token` before pushing.

### 2. Set environment variables

```bash
cp .env.example .env
# Fill in BASETEN_API_KEY and the three model URLs from the Baseten dashboard
```

### 3. Run the benchmark

```bash
python -m venv .venv && source .venv/bin/activate
pip install httpx numpy
python scripts/benchmark.py
```

Runs ~882 requests (3 models × 14 prompts × 7 repeats × 3 concurrency levels). Takes 10–20 minutes. Results are written to `results/benchmark_results.csv`.

### 4. View the dashboard

```bash
cd dashboard && python3 -m http.server 8080
# open http://localhost:8080
```

The dashboard reads the CSV and renders automatically. Reload after a new benchmark run to update.

## Cost methodology

Baseten self-deployed models bill by GPU-minute, not by token. Cost is derived as:

```
cost_per_request = (latency_s / 60) × instance_cost_per_min / concurrency
```

At higher concurrency the GPU serves multiple requests simultaneously, so cost is amortized across concurrent requests. This reflects real production economics — a model at 10% utilization costs 10× more per request than one running at capacity.
