"""
Test suite for Sparse Deterministic Speculative Decoding (SDSD).

Tests:
1. CSR conversion from DFA - sparse indexing
2. O(K) vs O(N) transition cost computation
3. Sparse DINGO DP correctness
4. End-to-end constrained decoding
"""

import math
import time
from csr_dfa import (
    CSRTransitionMatrix,
    build_csr_from_dfa,
    build_csr_from_transition_dict,
)
from sparse_dingo import (
    sparse_dingo_dp,
    compute_transition_costs_sparse,
    DINGOResult,
)
from baseline_dingo import baseline_dingo_dp
from herding import herding_decode, HerdingResult
from speculative_tree import speculative_decode, SpeculativeResult
from bidirectional_dingo import bidirectional_gap_dingo, states_compatible_with_suffix


def test_bidirectional_gap_dingo():
    """Bidirectional gap DP: left anchor + k masks + fixed suffix (meet on live states)."""
    print("=" * 60)
    print("Test: Bidirectional gap DINGO")
    print("=" * 60)
    # 0 --0--> 1 --1--> 2 (accepting)
    transitions = {(0, 0): 1, (1, 1): 2}
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=10)
    live = {2}
    # Empty suffix: same as forward DINGO for two steps
    p0 = [0.0] * 10
    p0[0] = 0.9
    p0[1] = 0.1
    p1 = [0.0] * 10
    p1[1] = 0.8
    p1[0] = 0.2
    r = bidirectional_gap_dingo(csr, 0, [p0, p1], [], live)
    assert r.success and r.tokens == [0, 1], r
    print(f"  k=2, no suffix: tokens={r.tokens}")

    # One mask + suffix token 1: must end gap at state 1 before suffix
    p_only = [[0.0] * 10]
    p_only[0][0] = 1.0
    good = states_compatible_with_suffix(csr, [1], live)
    assert 1 in good
    r2 = bidirectional_gap_dingo(csr, 0, p_only, [1], live)
    assert r2.success and r2.tokens == [0]
    print(f"  k=1, suffix=[1]: tokens={r2.tokens}")
    print("  ✓ Bidirectional gap DINGO OK\n")


def test_csr_conversion():
    """Test DFA to CSR conversion - verify sparse structure."""
    print("=" * 60)
    print("Test 1: CSR Conversion")
    print("=" * 60)
    
    # Simple DFA: states 0,1,2. Valid paths: 0->1 on token 0, 0->0 on token 1, 1->2 on token 2
    # Vocab size N=100, but only K=2 tokens valid from state 0, K=1 from state 1
    transitions = {
        (0, 0): 1,   # 0 --0--> 1
        (0, 1): 0,   # 0 --1--> 0
        (1, 2): 2,   # 1 --2--> 2
    }
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=100)
    
    assert csr.num_states == 3
    assert csr.vocab_size == 100
    
    # State 0: K=2 valid tokens
    tokens_0 = csr.get_valid_tokens(0)
    assert set(tokens_0) == {0, 1}, f"Expected {{0, 1}}, got {tokens_0}"
    print(f"  State 0: K={len(tokens_0)} valid tokens (vs N=100): {tokens_0}")
    
    # State 1: K=1 valid token
    tokens_1 = csr.get_valid_tokens(1)
    assert set(tokens_1) == {2}, f"Expected {{2}}, got {tokens_1}"
    print(f"  State 1: K={len(tokens_1)} valid tokens: {tokens_1}")
    
    # State 2: K=0
    tokens_2 = csr.get_valid_tokens(2)
    assert len(tokens_2) == 0
    print(f"  State 2: K={len(tokens_2)} valid tokens (terminal)")
    
    print("  ✓ CSR conversion correct\n")


def test_transition_costs_sparse():
    """Test O(K) transition cost computation."""
    print("=" * 60)
    print("Test 2: Sparse Transition Costs (O(K) vs O(N))")
    print("=" * 60)
    
    transitions = {
        (0, 0): 1,
        (0, 1): 0,
        (1, 2): 2,
    }
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=100)
    
    # Probability vector: token 0 has prob 0.1, token 1 has 0.8, token 2 has 0.05, rest ~0
    prob_vector = [0.0] * 100
    prob_vector[0] = 0.1
    prob_vector[1] = 0.8
    prob_vector[2] = 0.05
    
    Vi, Ti = compute_transition_costs_sparse(csr, prob_vector, vocab_size=100)
    
    # V(1, 0): from state 0, token 0 leads to 1. So max over tokens from 0 that go to 1 is 0.1
    assert (1, 0) in Vi
    assert abs(Vi[(1, 0)] - 0.1) < 1e-6
    assert Ti[(1, 0)] == 0
    
    # V(0, 0): from state 0, token 1 leads to 0. So max is 0.8
    assert (0, 0) in Vi
    assert abs(Vi[(0, 0)] - 0.8) < 1e-6
    assert Ti[(0, 0)] == 1
    
    # V(2, 1): from state 1, token 2 leads to 2. So max is 0.05
    assert (2, 1) in Vi
    assert abs(Vi[(2, 1)] - 0.05) < 1e-6
    
    print("  V_i(q, q') computed correctly using only K tokens per state")
    print("  ✓ Sparse transition costs correct\n")


def test_sparse_dingo_dp():
    """Test full Sparse DINGO DP - optimal constrained decoding."""
    print("=" * 60)
    print("Test 3: Sparse DINGO DP")
    print("=" * 60)
    
    # DFA: 0 --0--> 1, 0 --1--> 0, 1 --2--> 2
    # Valid sequences: 1* 0 2 (e.g. "0 2" or "1 0 2" or "1 1 0 2")
    transitions = {
        (0, 0): 1,
        (0, 1): 0,
        (1, 2): 2,
    }
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=10)
    
    # Block length 3. Valid: 0, 2 (path 0->1->2) or 1,0,2 etc.
    # Best path: maximize product of probs
    # v_1: prefer token 0 (go 0->1) or 1 (stay at 0)?
    # v_2: from 1, must use token 2 to reach 2
    # So valid: [0, 2] (len 2) or [1, 0, 2] (len 3)
    
    # For block length 2: need 0->1->2, so tokens [0, 2]
    prob_vectors = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    prob_vectors[0][0] = 0.9  # token 0
    prob_vectors[0][1] = 0.1  # token 1
    prob_vectors[1][2] = 1.0  # token 2
    
    live_states = {2}
    result = sparse_dingo_dp(csr, prob_vectors, start_state=0, live_states=live_states)
    
    assert result.success
    assert result.tokens == [0, 2]
    assert result.final_state == 2
    assert abs(result.probability - 0.9 * 1.0) < 1e-6
    print(f"  Block [0,2]: tokens={result.tokens}, prob={result.probability:.4f}")
    print("  ✓ Optimal path [0, 2] selected\n")
    
    # Block length 3: could do [1, 0, 2] - 0.1 * 0.9 * 1.0 = 0.09, or [0, 2, ?] - need 3 tokens
    # Actually from 2 we have no outgoing. So valid 3-step paths: 0->0->1->2 = [1, 0, 2]
    prob_vectors_3 = [
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # pos 0: 0 or 1
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    prob_vectors_3[0][1] = 0.5   # stay at 0
    prob_vectors_3[1][0] = 0.9   # go 0->1
    prob_vectors_3[2][2] = 1.0   # go 1->2
    
    result3 = sparse_dingo_dp(csr, prob_vectors_3, start_state=0, live_states=live_states)
    assert result3.success
    assert result3.tokens == [1, 0, 2]
    assert abs(result3.probability - 0.5 * 0.9 * 1.0) < 1e-6
    print(f"  Block [1,0,2]: tokens={result3.tokens}, prob={result3.probability:.4f}")
    print("  ✓ Optimal path [1, 0, 2] selected\n")


def test_json_like_dfa():
    """Test with a JSON-like constraint structure (small K per state)."""
    print("=" * 60)
    print("Test 4: JSON-like DFA (Structured Output)")
    print("=" * 60)
    
    # Simplified: states for { "key" : "value" }
    # Token IDs: 0={, 1=", 2=key, 3=:, 4=value, 5=}
    # States: 0 start, 1 after {, 2 after ", 3 after key, 4 after :, 5 after value, 6 accept
    N = 1000  # Large vocab to show K << N
    transitions = {
        (0, 0): 1,   # {
        (1, 1): 2,   # "
        (2, 2): 3,   # key
        (3, 1): 4,   # "
        (4, 3): 5,   # :
        (5, 1): 6,   # "
        (6, 4): 7,   # value
        (7, 1): 8,   # "
        (8, 5): 9,   # }
    }
    csr = build_csr_from_transition_dict(transitions, num_states=10, vocab_size=N)
    
    # Each state has K=1 except possibly more in real JSON
    max_k = csr.max_branch_factor()
    print(f"  Vocab size N={N}, Max branch factor K={max_k}")
    print(f"  Complexity: O(|Q|*K)={10 * max_k} vs O(|Q|*N)={10 * N}")
    print(f"  Speedup factor: ~{N // max(1, max_k)}x")
    
    # Decode a valid sequence
    prob_vectors = [[0.0] * N for _ in range(9)]
    for i, t in enumerate([0, 1, 2, 1, 3, 1, 4, 1, 5]):
        prob_vectors[i][t] = 1.0
    
    result = sparse_dingo_dp(csr, prob_vectors, start_state=0, live_states={9})
    assert result.success
    assert result.tokens == [0, 1, 2, 1, 3, 1, 4, 1, 5]
    print(f"  Decoded: {result.tokens}")
    print("  ✓ JSON-like constrained decoding correct\n")


def test_complexity_benchmark():
    """Benchmark: demonstrate O(K) vs O(N) scaling."""
    print("=" * 60)
    print("Test 5: Complexity Benchmark (O(K) vs O(N))")
    print("=" * 60)
    
    # Fixed K=5 valid tokens per state, varying N
    def make_csr(N: int):
        def transition_fn(q: int, t: int):
            if t < 5:  # Only first 5 tokens valid
                return (q + 1) % 3
            return None
        return build_csr_from_dfa(3, N, transition_fn)
    
    for N in [100, 1000, 10000]:
        csr = make_csr(N)
        prob_vector = [0.0] * N
        for i in range(5):
            prob_vector[i] = 0.2
        
        start = time.perf_counter()
        for _ in range(1000):
            compute_transition_costs_sparse(csr, prob_vector, N)
        elapsed = time.perf_counter() - start
        print(f"  N={N:5d}, K=5: {elapsed*1000:.2f} ms for 1000 iterations")
    
    print("  (Time should be ~constant as N grows, since we only scan K=5 tokens)")
    print("  ✓ Benchmark complete\n")


def test_b1_b2_equivalence():
    """Verify B1 (DINGO O(N)) and B2 (STATIC+DINGO O(K)) produce identical output."""
    print("=" * 60)
    print("Test 6: B1 vs B2 Equivalence")
    print("=" * 60)
    
    transitions = {(0, 0): 1, (0, 1): 0, (1, 2): 2}
    num_states, vocab_size = 3, 100
    prob_vectors = [[0.0] * vocab_size for _ in range(3)]
    prob_vectors[0][0], prob_vectors[0][1] = 0.9, 0.1
    prob_vectors[1][0], prob_vectors[1][1] = 0.5, 0.5
    prob_vectors[2][2] = 1.0
    
    def trans_fn(q, t):
        return transitions.get((q, t))
    
    csr = build_csr_from_transition_dict(transitions, num_states, vocab_size)
    r1 = baseline_dingo_dp(num_states, vocab_size, trans_fn, prob_vectors, 0, {2})
    r2 = sparse_dingo_dp(csr, prob_vectors, 0, {2})
    
    assert r1.tokens == r2.tokens, f"B1={r1.tokens} vs B2={r2.tokens}"
    assert abs(r1.probability - r2.probability) < 1e-6
    print(f"  B1 and B2 both decode: {r1.tokens}, prob={r1.probability:.4f}")
    print("  ✓ B1/B2 equivalence verified\n")


def test_herding_momentum():
    """Test Herding preserves blocked intentions via momentum."""
    print("=" * 60)
    print("Test 7: Herding Momentum")
    print("=" * 60)
    
    # DFA: 0 --0--> 1 (only 0 valid at start), 1 --2--> 2
    # Step 0: model wants token 1 (0.8) but grammar forces 0 - "sampling wall"
    # Herding: w += p - e_0, so w[1]=0.8 accumulates (blocked intention)
    transitions = {(0, 0): 1, (1, 2): 2}
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=10)
    
    prob_vectors = [[0.0] * 10 for _ in range(2)]
    prob_vectors[0][0], prob_vectors[0][1] = 0.2, 0.8  # forced to take 0, model wanted 1
    prob_vectors[1][2] = 1.0
    
    result = herding_decode(csr, prob_vectors, start_state=0, live_states={2})
    assert result.success
    assert result.tokens == [0, 2]
    # After step 0: w = p - e_0 => w[1]=0.8 (blocked mass for token 1)
    assert result.momentum_trace[1][1] > 0.5
    print(f"  Herding decode: {result.tokens}")
    print(f"  Momentum w[1]={result.momentum_trace[1][1]:.2f} (blocked mass preserved)")
    print("  ✓ Herding momentum verified\n")


def test_speculative_tree():
    """Test Self-Speculative Tree: one forward pass → multiple tokens."""
    print("=" * 60)
    print("Test 8: Self-Speculative Tree")
    print("=" * 60)

    # DFA: 0->1 on 0, 0->0 on 1, 1->2 on 2. Path [0,2] or [1,0,2]
    transitions = {(0, 0): 1, (0, 1): 0, (1, 2): 2}
    csr = build_csr_from_transition_dict(transitions, num_states=3, vocab_size=10)

    # One "forward pass" predicts 3 positions
    prob_vectors = [[0.0] * 10 for _ in range(3)]
    prob_vectors[0][0], prob_vectors[0][1] = 0.9, 0.1
    prob_vectors[1][2] = 1.0
    prob_vectors[2][2] = 1.0  # state 2 may have self-loop if we add it

    # Path 0->1->2 gives tokens [0, 2]. Need 2 prob vectors for 2 steps.
    prob_vectors = prob_vectors[:2]
    result = speculative_decode(csr, prob_vectors, start_state=0, live_states={2})
    assert result.success
    assert result.tokens == [0, 2]
    assert result.nfe_used == 1
    assert result.draft_length == 2
    print(f"  Draft length=2, tokens={result.tokens}, NFE={result.nfe_used}")
    print("  ✓ Bonus: 2 tokens from 1 forward pass\n")

    # Full tree with draft_length=3 (permissive DFA)
    from test_dllm_sdsd import build_permissive_dfa
    csr_p, start_p, live_p = build_permissive_dfa(100)
    probs_3 = [[0.0] * 100 for _ in range(3)]
    for i in range(3):
        probs_3[i][i % 50] = 0.5
        probs_3[i][(i + 1) % 50] = 0.5
    res3 = speculative_decode(csr_p, probs_3, start_p, live_p, draft_length=3)
    assert len(res3.tokens) == 3
    assert res3.nfe_used == 1
    print(f"  Draft length=3, got {len(res3.tokens)} tokens, NFE={res3.nfe_used}")
    print("  ✓ Self-Speculative Tree verified\n")


def run_all_tests():
    """Run all SDSD tests."""
    print("\n" + "=" * 60)
    print("SDSD: Sparse Deterministic Speculative Decoding - Test Suite")
    print("=" * 60 + "\n")
    
    test_csr_conversion()
    test_transition_costs_sparse()
    test_sparse_dingo_dp()
    test_json_like_dfa()
    test_complexity_benchmark()
    test_b1_b2_equivalence()
    test_herding_momentum()
    test_speculative_tree()
    test_bidirectional_gap_dingo()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
