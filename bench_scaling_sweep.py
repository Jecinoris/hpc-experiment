import time
import numpy as np
from joblib import Parallel, delayed
import os

# --- CONFIGURATION ---
# We use the same heavy matrix size to ensure the CPU is hungry for data
MATRIX_SIZE = 2500 
# A fixed pile of work. We want to see how fast different crew sizes finish it.
TOTAL_TASKS = 32 

def heavy_task(i):
    # Enforce strict single-threading per task
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    np.random.seed(i)
    # 1. Allocation (The Memory Bandwidth Test)
    A = np.random.rand(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
    
    # 2. Compute (The FPU Test)
    # SVD is heavy on both Compute and Memory
    U, S, V = np.linalg.svd(A)
    return S[0]

if __name__ == "__main__":
    print(f"--- MEMORY SATURATION TEST ---")
    print(f"Workload: {TOTAL_TASKS} tasks of {MATRIX_SIZE}x{MATRIX_SIZE} SVD")
    print(f"Goal: Find the peak 'Tasks Per Second' before memory chokes.")
    print("-" * 60)
    
    # The Suspects:
    # 16:  Matches your 1 stick per CPU (approx)
    # 32:  Slight oversaturation (common sweet spot)
    # 48:  Heavy traffic
    # 64:  Traffic jam
    core_counts = [4, 8, 16, 32]
    
    for n_workers in core_counts:
        print(f"Testing with {n_workers} Active Cores...", end="", flush=True)
        start = time.time()
        
        # Run the fixed pile of work
        Parallel(n_jobs=n_workers)(delayed(heavy_task)(i) for i in range(TOTAL_TASKS))
        
        duration = time.time() - start
        throughput = TOTAL_TASKS / duration
        
        print(f" Done!")
        print(f" -> Time:       {duration:.2f} seconds")
        print(f" -> Time:       {duration*6:.2f} seconds")
        print(f" -> Throughput: {throughput:.2f} tasks/sec")
        print("-" * 60)

