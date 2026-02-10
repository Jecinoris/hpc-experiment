import os
import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

# --- HARDWARE CONFIGURATION ---
TOTAL_CORES = multiprocessing.cpu_count()

# --- WORKLOAD CONFIGURATION ---
N_MATRICES = 400000
MATRIX_SIZE = 10

# --- SWEEP CONFIGURATION ---
# cores_per_unit values to benchmark:
#   1              = batched, 1 thread per worker  (max process parallelism)
#   intermediate   = hybrid (fewer workers x more threads each)
#   TOTAL_CORES    = single worker using all threads (max thread parallelism)
#
# NOTE: All configurations use BATCHED dispatch (data split into num_workers
# chunks, each processed with a single vectorized eigvals call).
# This is NOT the same as true embarrassingly parallel, which would dispatch
# each matrix individually (400k IPC round-trips — dominated by overhead
# for tiny 10x10 matrices).
CORES_PER_UNIT_SWEEP = [1, 2, 4, 8, 16, 32]


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
def computing_unit_task(batch_of_matrices, n_threads):
    """
    Process a batch of matrices inside a computing unit.
    Sets thread limits inside the worker for robustness (in case the worker
    process was reused from a previous run with different settings).
    """
    set_thread_limits(n_threads)
    return np.linalg.eigvals(batch_of_matrices)


# ============================================================================
# METHOD 1: VANILLA (Implicit Parallel - Let NumPy use all cores)
# ============================================================================
def run_vanilla(data):
    """
    Vanilla approach: Single process, NumPy uses all available cores
    internally via BLAS/LAPACK threading. No explicit parallelization.
    """
    set_thread_limits(TOTAL_CORES)

    start = time.time()
    results = np.linalg.eigvals(data)
    elapsed = time.time() - start

    return elapsed


# ============================================================================
# METHOD 2: BATCHED HYBRID SWEEP (parameterized by cores_per_unit)
# ============================================================================
def run_hybrid(data, cores_per_unit):
    """
    Batched hybrid approach using joblib:
      - num_workers = TOTAL_CORES // cores_per_unit  process-level parallelism
      - cores_per_unit                               thread-level parallelism per worker
      - Data is split into num_workers batches (vectorized eigvals per batch)

    When cores_per_unit=1:  max workers, 1 thread each, but still BATCHED
                            (NOT fine-grained embarrassingly parallel).
    When cores_per_unit=TOTAL_CORES: single worker, all threads.
    """
    num_workers = TOTAL_CORES // cores_per_unit

    # Set env vars in parent (inherited by freshly spawned loky workers)
    set_thread_limits(cores_per_unit)

    # Split data into equal batches — one per worker
    batches = np.array_split(data, num_workers)

    start = time.time()

    results = Parallel(n_jobs=num_workers)(
        delayed(computing_unit_task)(batch, cores_per_unit)
        for batch in batches
    )

    final = np.concatenate(results)
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
    print(f"Workload:  {N_MATRICES:,} matrices ({MATRIX_SIZE}x{MATRIX_SIZE}) eigvals")
    print(f"Sweep:     cores_per_unit = {CORES_PER_UNIT_SWEEP}")
    print("=" * 70)

    # Generate data once and reuse across all runs
    print(f"\nGenerating {N_MATRICES:,} matrices ({MATRIX_SIZE}x{MATRIX_SIZE})...")
    data = np.random.rand(N_MATRICES, MATRIX_SIZE, MATRIX_SIZE)

    results = {}

    # --- Vanilla baseline ---
    print("\n" + "-" * 70)
    print(f"[Vanilla] Single process, NumPy implicit threading (~{TOTAL_CORES} threads)")
    print("-" * 70)
    t = run_vanilla(data)
    label = f"Vanilla ({TOTAL_CORES}T)"
    results[label] = t
    print(f"  Time: {t:.4f}s  |  Throughput: {N_MATRICES/t:,.0f} matrices/s")

    # --- Sweep over cores_per_unit configurations ---
    for cpu in CORES_PER_UNIT_SWEEP:
        if TOTAL_CORES % cpu != 0:
            print(f"\n  Skipping cores_per_unit={cpu} "
                  f"(doesn't divide {TOTAL_CORES} evenly)")
            continue

        num_w = TOTAL_CORES // cpu
        print("\n" + "-" * 70)
        print(f"[Hybrid] {num_w} workers x {cpu} threads/worker = {num_w * cpu} cores")
        print("-" * 70)

        t = run_hybrid(data, cpu)
        label = f"{num_w}W x {cpu}T"
        results[label] = t
        print(f"  Time: {t:.4f}s  |  Throughput: {N_MATRICES/t:,.0f} matrices/s")

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
        print(f"{label:<30} {elapsed:>10.4f} {throughput:>12,.0f}/s "
              f"{relative:>9.2f}x{marker}")

    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print(f"""
  All hybrid configs use BATCHED dispatch: data is split into num_workers
  chunks, each processed with one vectorized np.linalg.eigvals() call.

  cores_per_unit=1  -> {TOTAL_CORES} workers x 1 thread, batched.
                       Max process parallelism, no BLAS threading.

  cores_per_unit=N  -> Fewer workers x N BLAS threads, batched.
                       Lets BLAS exploit multi-threaded eigvals per batch.

  Vanilla           -> Single process, NumPy auto-threads everything.
                       Simplest code; good baseline.

  This is NOT the same as true embarrassingly parallel (1 matrix per dispatch),
  which would incur 400k IPC round-trips and be dominated by overhead for
  small matrices.

  The sweet spot depends on matrix size, BLAS implementation, and memory bandwidth.
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
