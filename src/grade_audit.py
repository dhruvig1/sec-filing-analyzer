# src/grade_audit.py: grade results/judge_audit.csv one passage at a time (blind to the judge)
import csv, sys, textwrap
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

path = RESULTS_DIR / "judge_audit.csv"
rows = list(csv.DictReader(open(path, newline="")))
fields = list(rows[0])
todo = [r for r in rows if not r["human_grade"].strip()]
print(f"{len(todo)} of {len(rows)} left.  2 = directly answers | 1 = on topic / partial | "
      "0 = not relevant, boilerplate, TOC or passing mention.  s = skip, q = quit\n")
for i, r in enumerate(todo, 1):
    print("=" * 90)
    print(f"[{i}/{len(todo)}] {r['ticker']}: {r['question']}\n")
    print(textwrap.fill(" ".join(r["passage"].split()), 90))
    while (g := input("\ngrade> ").strip().lower()) not in {"0", "1", "2", "s", "q"}:
        print("enter 0, 1, 2, s or q")
    if g == "q":
        break
    if g != "s":
        r["human_grade"] = g
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"\nSaved. Graded {sum(bool(r['human_grade'].strip()) for r in rows)} of {len(rows)}. "
      "Run: python src/eval_passages.py agreement")
