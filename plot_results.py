import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("naive_baseline.csv")
df["eff_bw_gbs"] = (16.0 * (df["N"] ** 3)) / (df["seconds"] * 1e9)

cache_limits = [
    (36, "L1d (32 KB)", "#2ca02c"),
    (208, "L2 (1 MB)", "#ff7f0e"),
    (836, "L3 (16 MB)", "#d62728"),
]

with PdfPages("naive_baseline_report.pdf") as pdf:
    # Page 1: GFLOPS vs N
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(df["N"], df["gflops"], marker="o", linewidth=2, color="#1f77b4", label="Naive i-j-k GEMM (FP64)")
    for n_bound, label, color in cache_limits:
        ax.axvline(n_bound, linestyle="--", color=color, alpha=0.8, label=label)
    ax.set_title("Naive GEMM Throughput vs. Matrix Size (N)", fontweight="bold")
    ax.set_xlabel("Matrix Dimension (N x N)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: % of Theoretical Peak vs N
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(df["N"], df["pct_peak"], marker="s", linewidth=2, color="#9467bd", label="Naive % of Peak")
    ax.axhline(5.0, linestyle="--", color="#d62728", linewidth=1.5, label="5% Naive Ceiling Threshold")
    ax.set_title("Hardware Utilization (% of 81.6 GFLOPS FP64 Peak)", fontweight="bold")
    ax.set_xlabel("Matrix Dimension (N x N)")
    ax.set_ylabel("Theoretical Peak Utilization (%)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 3: Effective Memory Bandwidth vs N
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(df["N"], df["eff_bw_gbs"], marker="^", linewidth=2, color="#17becf", label="Effective Read Bandwidth (Naive i-j-k)")
    for n_bound, label, color in cache_limits:
        ax.axvline(n_bound, linestyle="--", color=color, alpha=0.8, label=label)
    ax.set_title("Effective Memory Bandwidth vs. Matrix Size (N)", fontweight="bold")
    ax.set_xlabel("Matrix Dimension (N x N)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print("Generated naive_baseline_report.pdf with 3 separate pages!")