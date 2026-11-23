import subprocess
import sys
import pandas as pd

BINARY = "./build/matrix_app"
# Powers of two (2^4 to 2^11) plus 1.5x midpoints
SIZES = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]

rows = []
for n in SIZES:
    print(f"Running N={n}...")
    res = subprocess.run([BINARY, str(n)], capture_output=True, text=True, check=True)
    n_val, secs, gflops, pct = map(float, res.stdout.strip().split(","))
    rows.append({"N": int(n_val), "seconds": secs, "gflops": gflops, "pct_peak": pct})

df = pd.DataFrame(rows)
df.to_csv("naive_baseline.csv", index=False)
print(df)

subprocess.run([sys.executable, "plot_results.py"], check=True)
subprocess.run(["explorer.exe", "naive_baseline_report.pdf"])