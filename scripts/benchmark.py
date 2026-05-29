import asyncio
import csv
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

API_KEY = os.environ["BASETEN_API_KEY"]

# ---------------------------------------------------------------------------
# Model registry — add each model here after deploying it with:
#   cd models/<name> && truss push
# The model_id comes from the URL printed by truss push, or the dashboard.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "Qwen-2.5-3B",
        "base_url": os.environ["QWEN_MODEL_URL"],
        "model_slug": "model",
        "instance_cost_per_min": 0.01414,   # L4
    },
    {
        "name": "Mistral-7B-Instruct",
        "base_url": os.environ["MISTRAL_MODEL_URL"],
        "model_slug": "model",
        "instance_cost_per_min": 0.01414,   # L4
    },
    {
        "name": "Llama-3.3-70B",
        "base_url": os.environ["LLAMA_MODEL_URL"],
        "model_slug": "model",
        "instance_cost_per_min": 0.43333,   # H100:4
    },
]

SYSTEM_PROMPT = (
    "Clean up the following spoken transcript into clear, professional written text. "
    "Only output the cleaned text, nothing else."
)

PROMPTS = [
    # Short (~20 tokens)
    "um so yeah i need to uh schedule a meeting for thursday",
    "can you like send that email to john about the project update",
    "hey remind me to call sarah back uh sometime today or tomorrow",
    "so basically i want to move the deadline to next friday if that works",
    "yeah i was thinking we should probably update the doc with those changes",
    # Medium (~50 tokens)
    "so i had this call with the client earlier and they were saying that um they want to push back the launch date and also they had some concerns about the onboarding flow and wanted us to like simplify it a bit if we can",
    "hey so the thing is i've been looking at the analytics and it looks like our conversion rate dropped by like ten percent last week and i think it might be related to the new checkout changes we shipped on tuesday",
    "alright so for the standup today i finished the auth bug fix yesterday and um today i'm going to work on the dashboard component and i'm a bit blocked on getting the design specs from the team",
    "so basically what happened was the build failed in CI because someone forgot to update the environment variable in the production config and we didn't catch it until after the deploy went out",
    "i was thinking that we could like split the feature into two separate PRs one for the backend changes and one for the frontend so it's easier to review and we reduce the risk of something breaking",
    # Long (~100 tokens)
    "okay so i just got out of the product review meeting and there were a lot of takeaways um first thing is that the team really liked the new dashboard design but they had some feedback around the data visualization specifically the charts are a bit hard to read on smaller screens and we should probably look at making them responsive also the filters on the left sidebar are a bit confusing to users and we got some feedback from user testing that people don't know what some of the labels mean so we should probably work with the design team to simplify those",
    "hey so i wanted to give you a quick update on the infrastructure migration we've been working on so we've successfully moved about sixty percent of the services over to the new kubernetes cluster and things are looking pretty stable um we had one incident last wednesday where a memory leak caused one of the pods to restart a few times but we've identified the root cause and deployed a fix and we're monitoring it closely the remaining forty percent of services are scheduled to migrate over the next two weeks and we don't anticipate any major issues but we'll be doing it gradually to reduce risk",
    "so i had a long conversation with the design team today about the new onboarding flow and we went through a few different approaches the first option was to keep the current multi-step wizard but just simplify some of the steps and remove the ones that users tend to skip anyway the second option was to move to a more progressive disclosure model where we show users only the essential setup steps first and then surface advanced options later as they need them and the third option was basically a hybrid of those two and after discussing the tradeoffs the team felt like option two was probably the best balance between simplicity for new users and flexibility for power users",
    "alright so i've been reviewing the support tickets from this week and there are a few themes that keep coming up the biggest one is that users are confused about how billing works specifically around how proration is calculated when they upgrade or downgrade their plan mid-cycle and we're getting a lot of tickets asking why their invoice looks different than they expected the second common issue is around the export feature where users are saying that CSV exports are missing some columns that they expect to be there and the third thing is some users on the enterprise plan are reporting that the SSO integration is sometimes redirecting them to the wrong workspace after login",
    "so i wanted to talk through the Q3 roadmap priorities because we have a lot of things on the list and we need to make some decisions about what to cut or defer given our capacity the things that are definitely in are the mobile app performance improvements because we've been getting a lot of negative reviews in the app store about slow load times and that's affecting our ratings also the API rate limiting feature is committed to several enterprise customers so that has to ship and the new reporting dashboard has been promised to the sales team for their demos next quarter everything else is on the table and i think we should probably have a prioritization session with the full team to decide what makes the cut",
]

CONCURRENCY_LEVELS = [1, 5, 10]
REPEATS = 7
WARMUP_REQUESTS = 3
LATENCY_TARGET_S = 0.700  # sub-700ms p99 TTFT requirement

RESULTS_PATH = Path(__file__).parent.parent / "docs" / "results" / "benchmark_results.csv"
SAMPLES_DIR = Path(__file__).parent.parent / "docs" / "results" / "samples"

CSV_FIELDS = [
    "model_name", "concurrency", "repeat_index", "prompt_index",
    "prompt_length", "ttft_s", "total_latency_s", "tokens_per_second",
    "cost_per_request_usd", "cost_per_1k_tokens_usd",
]


async def stream_request(
    client: httpx.AsyncClient,
    prompt: str,
    base_url: str,
    model_slug: str,
    capture_output: bool = False,
) -> dict:
    payload = {
        "model": model_slug,
        "messages": [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}],
        "stream": True,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    ttft = None
    total_tokens = 0
    buffer = b""
    output_chunks: list[str] | None = [] if capture_output else None

    async with client.stream(
        "POST", f"{base_url}/chat/completions",
        json=payload, headers=headers, timeout=300.0,
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            if ttft is None:
                ttft = time.perf_counter() - start
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    usage = obj.get("usage") or {}
                    if usage.get("completion_tokens"):
                        total_tokens = usage["completion_tokens"]
                    else:
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            total_tokens += 1
                            if capture_output:
                                output_chunks.append(content)
                except (json.JSONDecodeError, IndexError):
                    pass

    total_latency = time.perf_counter() - start
    result = {
        "ttft_s": ttft or total_latency,
        "total_latency_s": total_latency,
        "tokens_per_second": total_tokens / total_latency if total_latency > 0 else 0,
    }
    if capture_output:
        result["output"] = "".join(output_chunks)
    return result


async def collect_samples(model: dict) -> None:
    """Run one short/medium/long request and save the cleaned outputs for quality review."""
    name = model["name"]
    sample_prompts = [
        ("short",  PROMPTS[0]),
        ("medium", PROMPTS[5]),
        ("long",   PROMPTS[10]),
    ]
    limits = httpx.Limits(max_connections=3, max_keepalive_connections=3)
    samples = []
    async with httpx.AsyncClient(limits=limits) as client:
        for label, prompt in sample_prompts:
            result = await stream_request(
                client, prompt, model["base_url"], model["model_slug"],
                capture_output=True,
            )
            samples.append((label, prompt, result.get("output", "")))

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"{name.lower().replace(' ', '-')}.txt"
    with open(path, "w") as f:
        for label, prompt, output in samples:
            f.write(f"[{label.upper()}]\n")
            f.write(f"INPUT:  {prompt}\n")
            f.write(f"OUTPUT: {output}\n\n")
    print(f"  Samples → {path}")


async def warmup(model: dict) -> None:
    print(f"Warming up {model['name']}...", flush=True)
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=5)
    async with httpx.AsyncClient(limits=limits) as client:
        for i in range(WARMUP_REQUESTS):
            await stream_request(
                client, PROMPTS[i % len(PROMPTS)],
                model["base_url"], model["model_slug"],
            )
    print(f"  Done ({WARMUP_REQUESTS} requests discarded)\n")


async def run_batch(model: dict, concurrency: int, repeats: int) -> list[dict]:
    name = model["name"]
    base_url = model["base_url"]
    model_slug = model["model_slug"]

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def bounded(repeat_idx: int, prompt_idx: int, prompt: str) -> None:
        async with semaphore:
            metrics = await stream_request(client, prompt, base_url, model_slug)
            ipm = model["instance_cost_per_min"]
            metrics["cost_per_request_usd"] = metrics["total_latency_s"] / 60 * ipm / concurrency
            metrics["cost_per_1k_tokens_usd"] = (
                (ipm / 60) / metrics["tokens_per_second"] * 1000 / concurrency
                if metrics["tokens_per_second"] > 0 else 0
            )
            metrics["model_name"] = name
            metrics["repeat_index"] = repeat_idx
            metrics["prompt_index"] = prompt_idx
            metrics["prompt_length"] = (
                "short" if prompt_idx < 5 else ("medium" if prompt_idx < 10 else "long")
            )
            metrics["concurrency"] = concurrency
            results.append(metrics)
            total = len(PROMPTS) * repeats
            print(
                f"  [{name} c={concurrency}] {len(results):>3}/{total} "
                f"r={repeat_idx} p={prompt_idx:02d} — "
                f"TTFT={metrics['ttft_s']:.3f}s  "
                f"lat={metrics['total_latency_s']:.3f}s  "
                f"tps={metrics['tokens_per_second']:.1f}"
            )

    tasks = [
        bounded(r, i, p)
        for r in range(repeats)
        for i, p in enumerate(PROMPTS)
    ]
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        await asyncio.gather(*tasks)

    return results


def percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def print_summary(all_results: list[dict]) -> None:
    model_names = list(dict.fromkeys(r["model_name"] for r in all_results))
    w = 110
    print("\n" + "=" * w)
    print(f"{'Model':<22} {'Conc':<6} {'Metric':<26} {'p50':>10} {'p90':>10} {'p99':>10}  {'Target'}")
    print("=" * w)
    for name in model_names:
        model_rows = [r for r in all_results if r["model_name"] == name]
        for c in CONCURRENCY_LEVELS:
            rows = [r for r in model_rows if r["concurrency"] == c]
            for metric, label, fmt in [
                ("ttft_s",              "TTFT (s)",           ".3f"),
                ("total_latency_s",     "Total Latency (s)",  ".3f"),
                ("tokens_per_second",   "Tokens/sec",         ".1f"),
                ("cost_per_request_usd",    "Cost/request ($)",   ".5f"),
                ("cost_per_1k_tokens_usd",  "Cost/1K tokens ($)", ".4f"),
            ]:
                vals = [r[metric] for r in rows]
                p99 = percentile(vals, 99)
                target_col = ""
                if metric == "ttft_s":
                    target_col = f"  {'PASS' if p99 <= LATENCY_TARGET_S else 'FAIL'} (p99 {'<=' if p99 <= LATENCY_TARGET_S else '>'} {LATENCY_TARGET_S*1000:.0f}ms)"
                print(
                    f"{name:<22} {c:<6} {label:<26} "
                    f"{percentile(vals, 50):>10{fmt}} "
                    f"{percentile(vals, 90):>10{fmt}} "
                    f"{p99:>10{fmt}}"
                    f"{target_col}"
                )
            print("-" * w)


def save_csv(all_results: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in CSV_FIELDS})
    print(f"\nResults saved to {RESULTS_PATH}")


async def main() -> None:
    all_results: list[dict] = []

    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"  {model['name']}")
        print(f"{'=' * 60}")

        await warmup(model)
        await collect_samples(model)

        for c in CONCURRENCY_LEVELS:
            total = len(PROMPTS) * REPEATS
            print(f"\nRunning concurrency={c} ({len(PROMPTS)} prompts × {REPEATS} repeats = {total} requests)...")
            batch = await run_batch(model, c, REPEATS)
            all_results.extend(batch)

    print_summary(all_results)
    save_csv(all_results)


if __name__ == "__main__":
    asyncio.run(main())
