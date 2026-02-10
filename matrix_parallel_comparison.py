import os
import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

# --- HARDWARE CONFIGURATION ---
TOTAL_CORES = multiprocessing.cpu_count()

# --- WORKLOAD CONFIGURATION ---
N_MATRICES = 32
MATRIX_SIZE = 2500

# --- SWEEP CONFIGURATION ---
# cores_per_unit values to benchmark:
#   1              = batched, 1 thread per worker  (max process parallelism)
#   intermediate   = hybrid (fewer workers x more threads each)
#   TOTAL_CORES    = single worker using all threads (max thread parallelism)
#
# NOTE: All configurations use BATCHED dispatch (data split into num_workers
# chunks, each processed with a single vectorized eigvals call).
CORES_PER_UNIT_SWEEP = [1, 2, 4, 8, 16]


# ============================================================================
# UTILITIES
# ============================================================================
def set_thread_limits(n_threads):
    """Set all BLAS/LAPACK thread environment variables."""
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = str(n_threads)


# ============================================================================
# WORKER FUNCTION
# ============================================================================
def computing_unit_task(seeds, n_threads):
    """
    Process a batch of matrices inside a computing unit.
    Generates matrices from seeds to avoid shipping huge arrays over IPC.
    """
    set_thread_limits(n_threads)
    results = []
    for seed in seeds:
        np.random.seed(seed)
        A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
        results.append(np.linalg.eigvals(A))
    return results


# ============================================================================
# METHOD 1: VANILLA (Implicit Parallel - Let NumPy use all cores)
# ============================================================================
def run_vanilla():
    """
    Vanilla approach: Single process, NumPy uses all available cores
    internally via BLAS/LAPACK threading. No explicit parallelization.
    """
    set_thread_limits(TOTAL_CORES)

    start = time.time()
    results = []
    for i in range(N_MATRICES):
        np.random.seed(i)
        A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
        results.append(np.linalg.eigvals(A))
    elapsed = time.time() - start

    return elapsed


# ============================================================================
# METHOD 2: BATCHED HYBRID SWEEP (parameterized by cores_per_unit)
# ============================================================================
def run_hybrid(cores_per_unit):
    """
    Batched hybrid approach using joblib:
      - num_workers = TOTAL_CORES // cores_per_unit  process-level parallelism
      - cores_per_unit                               thread-level parallelism per worker
      - Data is split into num_workers batches (each worker processes its batch)

    When cores_per_unit=1:  max workers, 1 thread each, batched.
    When cores_per_unit=TOTAL_CORES: single worker, all threads.
    """
    num_workers = TOTAL_CORES // cores_per_unit

    # Set env vars in parent (inherited by freshly spawned loky workers)
    set_thread_limits(cores_per_unit)

    # Split seeds into batches — one per worker
    all_seeds = list(range(N_MATRICES))
    batches = [all_seeds[i::num_workers] for i in range(num_workers)]

    start = time.time()

    results = Parallel(n_jobs=num_workers)(
        delayed(computing_unit_task)(batch, cores_per_unit)
        for batch in batches
    )

    elapsed = time.time() - start

    return elapsed


# ============================================================================
# MAIN — RUN ALL CONFIGURATIONS
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("MATRIX EIGENVALUE COMPUTATION — PARALLELIZATION SWEEP")
    print("=" * 70)
    print(f"Hardware:  {TOTAL_CORES} cores detected")
    print(f"Workload:  {N_MATRICES} x eigvals({MATRIX_SIZE}x{MATRIX_SIZE} float64)")
    print(f"Sweep:     cores_per_unit = {CORES_PER_UNIT_SWEEP}")
    print("=" * 70)

    results = {}

    # --- Vanilla baseline ---
    print("\n" + "-" * 70)
    print(f"[Vanilla] Sequential eigvals, {TOTAL_CORES} BLAS threads each")
    print("-" * 70)
    t = run_vanilla()
    label = f"Vanilla (seq, {TOTAL_CORES}T)"
    results[label] = t
    print(f"  Time: {t:.4f}s  |  Throughput: {N_MATRICES/t:.2f} matrices/s")

    # --- Sweep over cores_per_unit configurations ---
    for cpu in CORES_PER_UNIT_SWEEP:
        if TOTAL_CORES % cpu != 0:
            print(f"\n  Skipping cores_per_unit={cpu} "
                  f"(doesn't divide {TOTAL_CORES} evenly)")
            continue

        num_w = TOTAL_CORES // cpu
        print("\n" + "-" * 70)
        print(f"[Hybrid] {num_w} workers x {cpu} threads/worker "
              f"= {num_w * cpu} cores")
        print("-" * 70)

        t = run_hybrid(cpu)
        label = f"{num_w}W x {cpu}T"
        results[label] = t
        print(f"  Time: {t:.4f}s  |  Throughput: {N_MATRICES/t:.2f} matrices/s")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY — PERFORMANCE COMPARISON")
    print("=" * 70)

    fastest = min(results, key=results.get)
    fastest_time = results[fastest]

    print(f"\n{'Configuration':<30} {'Time (s)':>10} {'Throughput':>15} {'Relative':>10}")
    print("-" * 70)

    for label, elapsed in results.items():
        throughput = N_MATRICES / elapsed
        relative = fastest_time / elapsed
        marker = " <-- best" if label == fastest else ""
        print(f"{label:<30} {elapsed:>10.4f} {throughput:>11.2f}/s "
              f"{relative:>9.2f}x{marker}")

    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print(f"""
  Each worker receives a batch of seeds, generates matrices locally,
  and computes eigvals one at a time with its assigned BLAS thread count.
  This avoids shipping large matrices over IPC.

  cores_per_unit=1  -> {TOTAL_CORES} workers x 1 thread, batched.
                       Max process parallelism, no BLAS threading per eigvals.

  cores_per_unit=N  -> {TOTAL_CORES}//N workers x N BLAS threads, batched.
                       Fewer concurrent tasks, but each eigvals is BLAS-accelerated.

  Vanilla           -> Sequential: 1 eigvals at a time, all {TOTAL_CORES} BLAS threads.
                       No task parallelism at all.

  The sweet spot depends on how well BLAS scales eigvals({MATRIX_SIZE}x{MATRIX_SIZE})
  across threads vs. how many independent eigvals can run concurrently.
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
