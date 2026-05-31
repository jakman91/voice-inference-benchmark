# Voice Inference Benchmark

Benchmarks three self-deployed LLMs on [Baseten](https://baseten.co) against a sub-700ms p99 latency target for voice dictation transcript cleanup — inspired by [Wispr Flow's real-time dictation use case](https://www.baseten.co/resources/customers/wispr-flow/).

**[View live dashboard →](https://jakman91.github.io/voice-inference-benchmark/)**

## The question

> A voice dictation app needs to clean up spoken transcripts in real time. Which model hits sub-700ms p99 total latency, and what does it cost?

## Models tested

| Model | GPU | Cost/min | Parameters |
|---|---|---|---|
| Qwen 2.5 3B Instruct | L4 | $0.01414 | 3B |
| Mistral 7B Instruct v0.3 | L4 | $0.01414 | 7B |
| Llama 3.3 70B Instruct | H100×4 | $0.43333 | 70B |

All models are deployed via Baseten (TRT-LLM, fp8 quantization, config-only — no custom code). Billing is per GPU-minute, not per token.

## Key findings

- **Llama 3.3 70B** is the only model that approaches sub-700ms, with passes at c=5 (699ms p99) but edges over at c=1 (703ms) and c=10 (726ms). At ~26× the cost of Qwen per request, it's the most expensive option by a wide margin.
- **Cost Consideration:** At $15/month, Wispr Flow is priced at a premium in a market where dictation is increasingly commoditized. For example, Claude, Gemini, and ChatGPT now bundle voice input natively. If the SLA were relaxed from sub-700ms to sub-1s, Qwen becomes viable at ~26× lower cost, which could be a meaningful lever worth evaluating as the category matures given sub-1s p99 latency is still very impressive.

## Per-model results

- **Llama 3.3 70B** (H100×4): closest to the 700ms target — passes at c=5 (699ms p99) but edges over at c=1 (703ms) and c=10 (726ms). At ~26× the cost of Qwen per request, it's the most expensive option.
- **Qwen 2.5 3B** (L4): competitive at low concurrency (786ms p99 at c=1) but degrades under load (959ms at c=10). Never clears the 700ms target, but stays within striking distance — the cheapest model tested and the 1× cost baseline everything else is measured against. At ~26× lower cost, multiple replicas behind a load balancer could match Llama's throughput at a fraction of the GPU spend.
- **Mistral 7B** (L4): runs on the same GPU as Qwen at the same rate but costs ~3× more per request, because each request takes ~3× longer. P99 latency exceeds 6 seconds across all concurrency levels. Model architecture matters more than parameter count.

## Repo structure

```
models/
  qwen-2.5-3b/config.yaml       # Engine Builder configs (truss push)
  mistral-7b/config.yaml
  llama-3.3-70b/config.yaml
scripts/
  benchmark.py                  # Async benchmark — streams all 3 models
docs/
  index.html                    # Single-file dashboard (Chart.js + PapaParse)
  results/
    benchmark_results.csv       # Pre-run results — open dashboard immediately
    samples/                    # Sample model outputs for quality review
```

## Run it yourself

### 1. Sign up for Baseten and get your API key

Create an account at [baseten.co](https://baseten.co), then grab your API key from **Settings → API Keys**.

### 2. Set your API key

```bash
cp .env.example .env
# Add your BASETEN_API_KEY to .env
```

You need the API key before deploying — `truss push` authenticates with it.

### 3. Deploy the models

```bash
pip install truss
cd models/qwen-2.5-3b   && truss push
cd models/mistral-7b    && truss push
cd models/llama-3.3-70b && truss push
```

> Llama 3.3 70B is a gated model. You'll need to accept Meta's license on Hugging Face and add your HF token to Baseten secrets as `hf_access_token` before pushing.

After each push, copy the endpoint URL from the model page in the Baseten dashboard (**API → Endpoint**).

### 4. Add the model URLs to your .env

```bash
# In .env, fill in the three model URLs from the Baseten dashboard:
QWEN_MODEL_URL=https://model-<id>.api.baseten.co/environments/production/sync/v1
MISTRAL_MODEL_URL=https://model-<id>.api.baseten.co/environments/production/sync/v1
LLAMA_MODEL_URL=https://model-<id>.api.baseten.co/environments/production/sync/v1
```

### 5. Run the benchmark

```bash
python -m venv .venv && source .venv/bin/activate
pip install httpx numpy
python scripts/benchmark.py
```

Runs ~900 requests (33 short + 33 medium + 33 long prompts × 3 concurrency levels × 3 models). Takes 10–20 minutes. Results are written to `docs/results/benchmark_results.csv`.

### 6. View the dashboard

A pre-run version is live at **[jakman91.github.io/voice-inference-benchmark](https://jakman91.github.io/voice-inference-benchmark/)**.

To run locally against your own results:

```bash
cd docs && python3 -m http.server 8080
# open http://localhost:8080
```

The dashboard reads the CSV and renders automatically. Reload after a new benchmark run to update.

## Appendix

### Cost methodology

Baseten self-deployed models bill by GPU-minute, not by token. Cost is derived as:

```
cost_per_request = (latency_s / 60) × instance_cost_per_min / concurrency
```

At higher concurrency the GPU serves multiple requests simultaneously, so cost is amortized across concurrent requests. This reflects real production economics — a model at 10% utilization costs 10× more per request than one running at capacity.

### Assumptions

**Output quality is equivalent across models.** The benchmark measures latency and cost only — it does not score output quality. This is reasonable given the task (cleaning up short spoken transcripts) is straightforward enough that all three models produce acceptable results. For more complex or nuanced tasks, quality differences between a 3B and 70B model would need to be factored in separately.

**Cost is compared at equal concurrency, not equal throughput.** Because Baseten bills per GPU-minute regardless of utilization, the total infrastructure cost of keeping a server running is fixed — utilization doesn't change your bill, only the number of servers does. What latency *does* affect is cost per request: a faster model at the same concurrency level serves each request more cheaply. In this benchmark, Llama 3.3 70B has the better latency, which is reflected in its cost-per-request calculation — but its H100×4 GPU rate is so much higher that Qwen remains far cheaper per request overall. At production scale, a model's throughput characteristics (driven by latency) would also affect how many replicas are needed to hit a given requests-per-minute target, which could shift the effective cost ratio in either direction depending on your workload.
