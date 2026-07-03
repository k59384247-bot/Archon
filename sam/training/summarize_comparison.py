"""Print a table of pass rates across all comparison_runs/<model_slug>/eval_results.json.

Run last, after evaluate_comparison.py has been run for each candidate.
"""

import json
from pathlib import Path

RUNS_ROOT = Path("sam/training/comparison_runs")


def main():
    rows = []
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        results_path = run_dir / "eval_results.json"
        if not results_path.exists():
            continue
        rows.append(json.loads(results_path.read_text()))

    if not rows:
        print(f"No eval_results.json found under {RUNS_ROOT}")
        return

    rows.sort(key=lambda r: r["pass_rate"], reverse=True)
    width = max(len(r["base_model"]) for r in rows)
    print(f"{'base_model':<{width}}  pass_rate  passed/total")
    for r in rows:
        print(f"{r['base_model']:<{width}}  {r['pass_rate']:.3f}      {r['passed']}/{r['total']}")

    print(f"\nWinner: {rows[0]['base_model']} ({rows[0]['pass_rate']:.3f})")


if __name__ == "__main__":
    main()
