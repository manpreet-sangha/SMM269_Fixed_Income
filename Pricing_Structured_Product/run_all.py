"""
run_all.py
==========
Master runner — executes every q*_main.py in numerical order.
Place this file in the Pricing_Structured_Product/ folder and run:

    python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (folder_name, script_name) in question order
SCRIPTS = [
    ("Q1",  "q1_main.py"),
    ("Q3",  "q3_main.py"),
    ("Q5",  "q5_main.py"),
    ("Q6",  "q6_main.py"),
    ("Q8",  "q8_main.py"),
    ("Q9",  "q9_main.py"),
    ("Q10", "q10_main.py"),
    ("Q11", "q11_main.py"),
    ("Q12", "q12_main.py"),
    ("Q13", "q13_main.py"),
    ("Q14", "q14_main.py"),
    ("Q15", "q15_main.py"),
    ("Q16", "q16_main.py"),
    ("Q17", "q17_main.py"),
    ("Q18", "q18_main.py"),
    ("Q19", "q19_main.py"),
    ("Q20", "q20_main.py"),
    ("Q21", "q21_main.py"),
    ("Q22", "q22_main.py"),
]


def main() -> None:
    passed, failed = [], []
    t_total = time.perf_counter()

    for folder, script in SCRIPTS:
        script_path = ROOT / folder / script
        if not script_path.exists():
            print(f"  [SKIP]  {folder}/{script}  (not found)")
            continue

        label = f"{folder}/{script}"
        print(f"  [RUN]   {label} ...", end="", flush=True)
        t0 = time.perf_counter()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        elapsed = time.perf_counter() - t0

        if result.returncode == 0:
            print(f"  OK  ({elapsed:.1f}s)")
            passed.append(label)
        else:
            print(f"  FAIL  (exit {result.returncode}, {elapsed:.1f}s)")
            # Show last few lines of stderr for quick diagnosis
            err = result.stderr.strip().splitlines()
            for line in err[-5:]:
                print(f"         {line}")
            failed.append(label)

    elapsed_total = time.perf_counter() - t_total
    print()
    print(f"  Done: {len(passed)} passed, {len(failed)} failed  ({elapsed_total:.1f}s total)")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
