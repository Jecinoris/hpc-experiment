import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os

# --- CONFIGURATION ---
# We use a 2500x2500 matrix. 
# On Intel 6.0GHz: ~0.5s per task
# On EPYC 3.7GHz: ~1.0s per task
# BUT EPYC can do 192 of them at once. Intel can only do 32.
MATRIX_SIZE = 2500 
NUM_TASKS = 384  # Exactly 2 rounds of work for the EPYC (192 * 2)

def heavy_task(i):
    # Ensure isolation
    os.environ["OMP_NUM_THREADS"] = "1"
    
    np.random.seed(i)
    # Allocation (Memory Bandwidth Test)
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    # Compute (FPU Test) - SVD is very heavy
    U, S, V = np.linalg.svd(A)
    return S[0]

if __name__ == "__main__":
    total_cores = multiprocessing.cpu_count()
    print(f"--- HEAVYWEIGHT CHAMPIONSHIP ---")
    print(f"Cores Detected: {total_cores}")
    print(f"Workload: {NUM_TASKS} tasks of {MATRIX_SIZE}x{MATRIX_SIZE} SVD")
    print("-" * 40)

    start = time.time()
    
    # On EPYC: We cap at 192 to use physical cores only (avoids SMT jitter)
    # On Intel: We use all 32 threads
    if total_cores > 200:
        n_jobs = 192
        print("EPYC Mode: Limiting to 192 Physical Cores")
    else:
        n_jobs = -1
        print("Desktop Mode: Using all available threads")

    # Run the benchmark
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(heavy_task)(i) for i in range(NUM_TASKS)
    )
    
    end = time.time()
    duration = end - start
    
    print("-" * 40)
    print(f"Total Time:     {duration:.4f} seconds")
    print(f"Tasks Per Sec:  {NUM_TASKS / duration:.2f}")
    print("-" * 40)