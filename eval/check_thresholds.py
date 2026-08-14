import json
import sys

def main():
    try:
        with open("eval/last_run_scores.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: eval/last_run_scores.json not found. Run eval/run_ragas.py first.")
        sys.exit(1)

    scores = data.get("summary", {})
    
    # Baseline thresholds (intended as a starting quality gate; to be raised as the system improves)
    THRESHOLDS = {
        "faithfulness": 0.70,
        "answer_relevancy": 0.65
    }
    
    print("Checking evaluation metrics against baseline thresholds:")
    failed = []
    for metric, threshold in THRESHOLDS.items():
        actual = scores.get(metric, 0.0)
        # Handle cases where value is None
        if actual is None:
            actual = 0.0
        if actual < threshold:
            print(f"  ❌ {metric}: {actual:.4f} < {threshold:.2f} (FAILED)")
            failed.append(metric)
        else:
            print(f"  ✅ {metric}: {actual:.4f} >= {threshold:.2f} (PASSED)")
            
    # Also print other metrics for informational purposes
    for metric in ["context_precision", "context_recall"]:
        actual = scores.get(metric)
        if actual is not None:
            print(f"  ℹ️  {metric}: {actual:.4f} (Informational)")

    if failed:
        print(f"\nCI/CD Quality Gate Status: FAILED (Threshold violations: {', '.join(failed)})")
        sys.exit(1)
        
    print("\nCI/CD Quality Gate Status: PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
