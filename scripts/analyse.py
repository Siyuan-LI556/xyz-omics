# scripts/analyse.py
# Overlay eps(t) comparison curves from the results_*.pt records saved by
# barseq_varifold.py.
#
#   PYTHONPATH=. python scripts/analyse.py [name] [results files...]
#
#   name           optional figure name -> output/eps_<name>.png/.pdf
#   results files  explicit .pt files; if omitted, all results_*.pt in output/
#
# Example (5.5.1, after runs 101-106):
#   PYTHONPATH=. python scripts/analyse.py init \
#       data/BARSeq/output/results_run10*_*_isotropic_lbfgs.pt
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.io.plot import plot_eps_comparison

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "BARSeq", "output")

args = sys.argv[1:]
name = args.pop(0) if args and not args[0].endswith(".pt") else "compare"
files = sorted(args) or sorted(glob.glob(os.path.join(OUTPUT_DIR, "results_*.pt")))

if not files:
    sys.exit(f"no results_*.pt found in {OUTPUT_DIR}; run barseq_varifold.py first")

print(f"comparing {len(files)} runs -> eps_{name}.png/.pdf")
for f in files:
    print(f"  {os.path.basename(f)}")
plot_eps_comparison(files, output_dir=OUTPUT_DIR, name=name)
