import csv
from pathlib import Path

INSTANCE_COSTS = {
    "Qwen-2.5-3B":        0.01414,
    "Mistral-7B-Instruct": 0.01414,
    "Llama-3.3-70B":       0.43333,
}

CSV_PATH = Path(__file__).parent.parent / "docs" / "results" / "benchmark_results.csv"

rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        ipm = INSTANCE_COSTS[row["model_name"]]
        lat = float(row["total_latency_s"])
        tps = float(row["tokens_per_second"])
        c   = int(row["concurrency"])
        row["cost_per_request_usd"]   = lat / 60 * ipm / c
        row["cost_per_1k_tokens_usd"] = (ipm / 60) / tps * 1000 / c if tps > 0 else 0
        rows.append(row)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Fixed {len(rows)} rows. Sample cost checks:")
for name in INSTANCE_COSTS:
    for c in [1, 5, 10]:
        sample = next(r for r in rows if r["model_name"] == name and int(r["concurrency"]) == c)
        print(f"  {name} c={c}: ${float(sample['cost_per_request_usd']):.6f}/req")
