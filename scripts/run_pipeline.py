import json
from scripts.evaluate import evaluate_model
from scripts.train import train
from config import MODEL_NAME, OUTPUT_DIR, EVAL_K

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def calculate_improvement(baseline, finetuned):
    return {k: ((finetuned[k] - baseline[k]) / baseline[k] * 100) for k in baseline}

def main():
    print("Baseline evaluation...")
    baseline_metrics = evaluate_model(MODEL_NAME, k=EVAL_K)
    save_json("baseline_results.json", baseline_metrics)

    print("\nFine-tuning...")
    train()

    print("\nEvaluating fine-tuned model...")
    finetuned_metrics = evaluate_model(OUTPUT_DIR, k=EVAL_K)
    save_json("finetuned_results.json", finetuned_metrics)

    print("\nComparison:")
    for metric in baseline_metrics:
        b, f = baseline_metrics[metric], finetuned_metrics[metric]
        improvement = ((f - b) / b * 100) if b > 0 else 0
        print(f"{metric}: {b:.4f} -> {f:.4f} ({improvement:+.1f}%)")

    comparison = {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "improvement": calculate_improvement(baseline_metrics, finetuned_metrics)
    }

    save_json("comparison.json", comparison)

if __name__ == "__main__":
    main()
