"""
Benchmark: B1 (DINGO O(N)) vs B2 (STATIC + DINGO O(K))

Evaluates computational efficiency gains from sparse indexing.
Metrics:
- Latency per decoding step (ms)
- Total time for full block decoding
- Token operations count (N vs K)
- Correctness verification (both produce identical output)
"""

import sys
import time
import statistics
from pathlib import Path
from typing import Callable
from dataclasses import dataclass, field

# Ensure src is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csr_dfa import (
    build_csr_from_dfa,
    build_csr_from_transition_dict,
    CSRTransitionMatrix,
)
from sparse_dingo import sparse_dingo_dp, compute_transition_costs_sparse
from baseline_dingo import baseline_dingo_dp, compute_transition_costs_naive


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    latency_ms: float
    latency_std_ms: float = 0.0
    tokens_decoded: list[int] = field(default_factory=list)
    probability: float = 0.0
    success: bool = False
    vocab_scans: int = 0  # N or K per (q',q) pair
    total_ops: int = 0


def make_transition_fn(transitions: dict[tuple[int, int], int]) -> Callable[[int, int], int | None]:
    def fn(q: int, t: int) -> int | None:
        return transitions.get((q, t))
    return fn


def run_benchmark(
    transitions: dict[tuple[int, int], int],
    num_states: int,
    vocab_size: int,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    num_warmup: int = 5,
    num_trials: int = 50,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run B1 and B2 benchmarks, return results.
    """
    transition_fn = make_transition_fn(transitions)
    csr = build_csr_from_transition_dict(transitions, num_states, vocab_size)
    
    # Verify both produce same output
    result_b1 = baseline_dingo_dp(
        num_states, vocab_size, transition_fn,
        prob_vectors, start_state, live_states
    )
    result_b2 = sparse_dingo_dp(
        csr, prob_vectors, start_state, live_states
    )
    
    assert result_b1.tokens == result_b2.tokens, "B1 and B2 must produce identical output"
    assert result_b1.success == result_b2.success
    
    # --- B1: O(N) timing ---
    latencies_b1: list[float] = []
    for _ in range(num_warmup + num_trials):
        start = time.perf_counter()
        baseline_dingo_dp(
            num_states, vocab_size, transition_fn,
            prob_vectors, start_state, live_states
        )
        elapsed = (time.perf_counter() - start) * 1000
        if _ >= num_warmup:
            latencies_b1.append(elapsed)
    
    avg_b1 = statistics.mean(latencies_b1)
    std_b1 = statistics.stdev(latencies_b1) if len(latencies_b1) > 1 else 0.0
    
    # Ops: d * |Q| * N for transition costs
    d = len(prob_vectors)
    ops_b1 = d * num_states * num_states * vocab_size
    
    res_b1 = BenchmarkResult(
        method="B1: DINGO O(N)",
        latency_ms=avg_b1,
        latency_std_ms=std_b1,
        tokens_decoded=result_b1.tokens,
        probability=result_b1.probability,
        success=result_b1.success,
        vocab_scans=vocab_size,
        total_ops=ops_b1,
    )
    
    # --- B2: O(K) timing ---
    latencies_b2: list[float] = []
    for _ in range(num_warmup + num_trials):
        start = time.perf_counter()
        sparse_dingo_dp(csr, prob_vectors, start_state, live_states)
        elapsed = (time.perf_counter() - start) * 1000
        if _ >= num_warmup:
            latencies_b2.append(elapsed)
    
    avg_b2 = statistics.mean(latencies_b2)
    std_b2 = statistics.stdev(latencies_b2) if len(latencies_b2) > 1 else 0.0
    
    max_k = csr.max_branch_factor()
    ops_b2 = d * num_states * max_k  # Approximate
    
    res_b2 = BenchmarkResult(
        method="B2: STATIC + DINGO O(K)",
        latency_ms=avg_b2,
        latency_std_ms=std_b2,
        tokens_decoded=result_b2.tokens,
        probability=result_b2.probability,
        success=result_b2.success,
        vocab_scans=max_k,
        total_ops=ops_b2,
    )
    
    return res_b1, res_b2


def print_benchmark_report(b1: BenchmarkResult, b2: BenchmarkResult, config: dict):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("BENCHMARK: B1 (DINGO O(N)) vs B2 (STATIC + DINGO O(K))")
    print("=" * 70)
    print(f"\nConfiguration: {config}")
    print("\n" + "-" * 70)
    print(f"{'Metric':<35} {'B1 (O(N))':<18} {'B2 (O(K))':<18}")
    print("-" * 70)
    
    print(f"{'Latency (ms)':<35} {b1.latency_ms:>10.3f} ± {b1.latency_std_ms:<5.2f} {b2.latency_ms:>10.3f} ± {b2.latency_std_ms:<5.2f}")
    
    speedup = b1.latency_ms / b2.latency_ms if b2.latency_ms > 0 else 0
    print(f"{'Speedup (B2 vs B1)':<35} {'—':<18} {speedup:>10.2f}x")
    
    vocab_label = "Vocab scans per (q,q')"
    print(f"{vocab_label:<35} {b1.vocab_scans:>18} {b2.vocab_scans:>18}")
    print(f"{'Total ops (approx)':<35} {b1.total_ops:>18} {b2.total_ops:>18}")
    
    print("-" * 70)
    print(f"Decoded tokens: {b1.tokens_decoded}")
    print(f"Success: {b1.success}, Probability: {b1.probability:.6f}")
    print("=" * 70 + "\n")


def benchmark_simple_dfa():
    """Benchmark with simple 3-state DFA."""
    transitions = {(0, 0): 1, (0, 1): 0, (1, 2): 2}
    num_states, start_state, live_states = 3, 0, {2}
    
    # Vary vocab size N to show O(N) vs O(K) scaling
    for vocab_size in [100, 500, 1000, 2000]:
        prob_vectors = [[0.0] * vocab_size for _ in range(3)]
        prob_vectors[0][0], prob_vectors[0][1] = 0.5, 0.5
        prob_vectors[1][0], prob_vectors[1][1] = 0.5, 0.5
        prob_vectors[2][2] = 1.0
        
        b1, b2 = run_benchmark(
            transitions, num_states, vocab_size,
            prob_vectors, start_state, live_states,
            num_warmup=2, num_trials=15,
        )
        config = {"N": vocab_size, "K": 2, "block_len": 3, "|Q|": num_states}
        print_benchmark_report(b1, b2, config)


def benchmark_json_like():
    """Benchmark with JSON-like DFA (K=1 per state)."""
    N = 5000
    transitions = {
        (0, 0): 1, (1, 1): 2, (2, 2): 3, (3, 1): 4, (4, 3): 5,
        (5, 1): 6, (6, 4): 7, (7, 1): 8, (8, 5): 9,
    }
    num_states, start_state, live_states = 10, 0, {9}
    
    prob_vectors = [[0.0] * N for _ in range(9)]
    for i, t in enumerate([0, 1, 2, 1, 3, 1, 4, 1, 5]):
        prob_vectors[i][t] = 1.0
    
    b1, b2 = run_benchmark(
        transitions, num_states, N,
        prob_vectors, start_state, live_states,
        num_warmup=3, num_trials=20,
    )
    config = {"N": N, "K": 1, "block_len": 9, "|Q|": num_states}
    print_benchmark_report(b1, b2, config)


def benchmark_transition_cost_only():
    """Isolated benchmark: transition cost computation only."""
    print("\n" + "=" * 70)
    print("ISOLATED: Transition Cost Computation (V_i) - O(N) vs O(K)")
    print("=" * 70)
    
    transitions = {(0, 0): 1, (0, 1): 0, (1, 2): 2}
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=100)
    transition_fn = make_transition_fn(transitions)
    
    num_trials = 1000
    for N in [100, 500, 1000, 2000]:
        prob = [0.0] * N
        prob[0], prob[1], prob[2] = 0.4, 0.4, 0.2
        
        # B1: O(N)
        start = time.perf_counter()
        for _ in range(num_trials):
            compute_transition_costs_naive(3, N, transition_fn, prob)
        t_b1 = (time.perf_counter() - start) * 1000
        
        # B2: O(K) - need to rebuild CSR for new N
        csr_n = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=N)
        start = time.perf_counter()
        for _ in range(num_trials):
            compute_transition_costs_sparse(csr_n, prob, N)
        t_b2 = (time.perf_counter() - start) * 1000
        
        speedup = t_b1 / t_b2 if t_b2 > 0 else 0
        print(f"  N={N:5d}, K=2:  B1={t_b1:6.2f}ms  B2={t_b2:6.2f}ms  Speedup={speedup:.1f}x")


def main():
    import io
    from pathlib import Path
    
    buf = io.StringIO()
    
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()
    
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, buf)
    
    try:
        print("\n" + "#" * 70, flush=True)
        print("#  B1 vs B2 Benchmark: DINGO O(N) vs STATIC + DINGO O(K)", flush=True)
        print("#" * 70, flush=True)
        
        print("\n--- 1. Simple DFA (varying N) ---", flush=True)
        benchmark_simple_dfa()
        
        print("\n--- 2. JSON-like DFA (N=5000, K=1) ---", flush=True)
        benchmark_json_like()
        
        print("\n--- 3. Isolated Transition Cost ---", flush=True)
        benchmark_transition_cost_only()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
    
    output = buf.getvalue() + "\n[Done]\n"
    
    # Write to file for inspection
    out_path = Path(__file__).parent.parent / "benchmark_results.txt"
    out_path.write_text(output, encoding="utf-8")
    print(output + f"Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
