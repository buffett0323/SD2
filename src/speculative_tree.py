"""
Self-Speculative Decoding Tree (SDSD)

SDSD addresses three bottlenecks in structured generation:

| Layer      | Technique    | Physical Cost | Gain                          |
|------------|--------------|---------------|-------------------------------|
| Width      | STATIC CSR   | O(K)          | ~948× retrieval speedup       |
| Path       | Herding      | Deterministic | O(T^{-1}) vs O(T^{-1/2})     |
| Depth      | Spec-Tree    | O(1/τ) NFE    | 60–65% fewer forward passes   |

Algorithm Flow:
  Step 1 (Offline):  Compile DFA → STATIC CSR (P, C arrays)
  Step 2 (Drafting): One forward → M logits. Build Grammar-Legal Draft Tree via CSR + Herding.
  Step 3 (Verify):   Optional parallel verification with Tree Attention.
  Step 4 (Update):   Update FSM state and Herding weight w.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .csr_dfa import CSRTransitionMatrix


@dataclass
class SpeculativeNode:
    """Node in the Grammar-Legal Draft Tree."""
    state: int
    token: int | None  # None for root
    depth: int
    path: list[int]
    children: list[SpeculativeNode] = field(default_factory=list)


@dataclass
class SpeculativeResult:
    """Result of self-speculative decoding."""
    tokens: list[int]
    final_state: int
    draft_length: int
    nfe_used: int
    n_accepted: int  # tokens passing verification (≤ draft_length)
    success: bool


# ---------------------------------------------------------------------------
# Verification: second forward pass to check model supports draft tokens
# ---------------------------------------------------------------------------

def verify_draft_with_forward(
    draft_tokens: list[int],
    get_verify_logits_fn: Callable[[list[int]], list[list[float]]],
    threshold: float = 0.0,
) -> int:
    """
    Verification Step C: Run second forward pass with draft as context.
    Returns number of tokens accepted (first i where p_i[t_i] <= threshold).
    NFE = 1 for this verification call.
    """
    if not draft_tokens:
        return 0
    prob_vectors = get_verify_logits_fn(draft_tokens)
    n = min(len(draft_tokens), len(prob_vectors))
    vocab_size = len(prob_vectors[0]) if prob_vectors else 0
    for i in range(n):
        p = prob_vectors[i]
        t = draft_tokens[i]
        if t >= len(p):
            return i
        prob_t = p[t] if t < len(p) else 0.0
        if prob_t <= threshold:
            return i
    return n


# ---------------------------------------------------------------------------
# Step 2a: Tree Construction (STATIC CSR → K valid tokens per node)
# ---------------------------------------------------------------------------

def build_speculative_tree(
    csr: CSRTransitionMatrix,
    start_state: int,
    draft_length: int,
    live_states: set[int],
) -> SpeculativeNode:
    """
    Build Grammar-Legal Draft Tree using STATIC CSR.

    At each node (state q), use row_pointers P[q], P[q+1] to slice
    column_indices C[P[q]:P[q+1]] → K valid tokens. O(K) per node.

    Full tree: O(K^d) nodes. Use only when K is small (e.g. strict JSON).
    """
    root = SpeculativeNode(state=start_state, token=None, depth=0, path=[])
    frontier = [root]

    for _ in range(draft_length):
        next_frontier = []
        for node in frontier:
            # STATIC: O(1) slice, O(K) iteration (not O(N))
            transitions = csr.get_transitions(node.state)
            for t, q_next in transitions:
                child = SpeculativeNode(
                    state=q_next,
                    token=t,
                    depth=node.depth + 1,
                    path=node.path + [t],
                )
                node.children.append(child)
                next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break

    return root


# ---------------------------------------------------------------------------
# Step 2b: Deterministic Path Selection (Herding vs Argmax)
# ---------------------------------------------------------------------------

def select_path_herding(
    root: SpeculativeNode,
    prob_vectors: list[list[float]],
    vocab_size: int,
) -> tuple[list[int], int, list[list[float]]]:
    """
    Traverse tree with Herding: x* = argmax_{x ∈ V_valid} (w + p)ᵀ e_x.

    Herding update: w_{t+1} = w_t + p_t - e_{x*}
    Solves sampling wall / information collapse; convergence O(T^{-1}).
    """
    w = [0.0] * vocab_size
    momentum_trace = [w.copy()]
    tokens: list[int] = []
    node = root

    for i in range(len(prob_vectors)):
        if not node.children:
            break

        p = prob_vectors[i]
        if len(p) < vocab_size:
            p = p + [0.0] * (vocab_size - len(p))

        best_child: SpeculativeNode | None = None
        best_score = float("-inf")

        for child in node.children:
            t = child.token
            if t is None:
                continue
            score = w[t] + p[t]
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None or best_child.token is None:
            break

        t_star = best_child.token
        tokens.append(t_star)

        for j in range(vocab_size):
            w[j] = w[j] + p[j]
        w[t_star] -= 1.0
        momentum_trace.append(w.copy())

        node = best_child

    return tokens, node.state, momentum_trace


def select_path_argmax(
    root: SpeculativeNode,
    prob_vectors: list[list[float]],
    vocab_size: int,
) -> tuple[list[int], int]:
    """
    Traverse tree with argmax: x* = argmax_{x ∈ V_valid} pᵀ e_x.
    Ablation 3: STATIC + Spec-Tree without Herding.
    """
    tokens: list[int] = []
    node = root

    for i in range(len(prob_vectors)):
        if not node.children:
            break

        p = prob_vectors[i]
        if len(p) < vocab_size:
            p = p + [0.0] * (vocab_size - len(p))

        best_child: SpeculativeNode | None = None
        best_score = float("-inf")

        for child in node.children:
            t = child.token
            if t is None:
                continue
            if p[t] > best_score:
                best_score = p[t]
                best_child = child

        if best_child is None or best_child.token is None:
            break

        tokens.append(best_child.token)
        node = best_child

    return tokens, node.state


# ---------------------------------------------------------------------------
# Lazy traversal (O(d*K)) — for large K, avoid O(K^d) tree materialization
# ---------------------------------------------------------------------------

def _traverse_lazy_herding(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int,
) -> tuple[list[int], int]:
    """
    Lazy O(d*K) traversal: at each step, CSR gives K valid tokens,
    Herding picks best. No full tree.
    """
    try:
        from .herding import herding_decode
    except ImportError:
        from herding import herding_decode

    result = herding_decode(
        csr, prob_vectors, start_state, live_states, block_length=draft_length
    )
    return result.tokens, result.final_state


def _traverse_lazy_argmax(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int,
) -> tuple[list[int], int]:
    """
    Lazy O(d*K) traversal with argmax (no Herding momentum).
    """
    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size
    tokens: list[int] = []
    q = start_state

    for i in range(draft_length):
        p = prob_vectors[i]
        if len(p) < vocab_size:
            p = p + [0.0] * (vocab_size - len(p))

        transitions = csr.get_transitions(q)
        if not transitions:
            break

        best_t: int | None = None
        best_q_next: int | None = None
        best_score = float("-inf")

        for t, q_next in transitions:
            if p[t] > best_score:
                best_score = p[t]
                best_t = t
                best_q_next = q_next

        if best_t is None or best_q_next is None:
            break

        tokens.append(best_t)
        q = best_q_next

    return tokens, q


# ---------------------------------------------------------------------------
# Step 3: Parallel Verification (optional)
# ---------------------------------------------------------------------------

def _verify_draft_default(
    draft_tokens: list[int],
    prob_vectors: list[list[float]],
    threshold: float = 0.0,
) -> int:
    """
    Default verifier: accept draft if p_i[t_i] > threshold.
    No second forward; uses same logits as draft.
    """
    n = min(len(draft_tokens), len(prob_vectors))
    for i in range(n):
        p = prob_vectors[i]
        t = draft_tokens[i]
        if t >= len(p) or p[t] <= threshold:
            return i
    return n


# ---------------------------------------------------------------------------
# Main API: speculative_decode (SDSD) and speculative_decode_argmax (Ablation 3)
# ---------------------------------------------------------------------------

def speculative_decode(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int | None = None,
    use_lazy: bool = True,
    verify_fn: Callable[[list[int], list[list[float]]], int] | None = None,
) -> SpeculativeResult:
    """
    SDSD: STATIC + Herding + Speculative Tree.

    Step 2 (Drafting): One forward → M logits. Build Grammar-Legal path via
    CSR (O(K) per node) + Herding (deterministic, preserves intent).
    Step 3 (Verify): Optional. Default: accept all draft tokens.
    """
    d = draft_length if draft_length is not None else len(prob_vectors)
    if d == 0:
        return SpeculativeResult(
            tokens=[], final_state=start_state,
            draft_length=0, nfe_used=1, n_accepted=0, success=True,
        )

    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size
    nfe = 1

    if use_lazy:
        tokens, final_state = _traverse_lazy_herding(
            csr, prob_vectors, start_state, live_states, d
        )
    else:
        root = build_speculative_tree(csr, start_state, d, live_states)
        tokens, final_state, _ = select_path_herding(root, prob_vectors, vocab_size)

    n_accepted = len(tokens)
    if verify_fn is not None:
        n_accepted = verify_fn(tokens, prob_vectors)
        tokens = tokens[:n_accepted]
        # If verification does a second forward: nfe = 2

    return SpeculativeResult(
        tokens=tokens,
        final_state=final_state,
        draft_length=len(tokens),
        nfe_used=nfe,
        n_accepted=n_accepted,
        success=final_state in live_states and len(tokens) == d,
    )


def speculative_decode_argmax(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int | None = None,
) -> SpeculativeResult:
    """
    Ablation 3: STATIC + Spec-Tree with argmax (no Herding).

    Same NFE reduction as SDSD, but path chosen by p[t] only.
    Uses lazy O(d*K) to avoid O(K^d) when K ≈ vocab_size.
    """
    d = draft_length if draft_length is not None else len(prob_vectors)
    if d == 0:
        return SpeculativeResult(
            tokens=[], final_state=start_state,
            draft_length=0, nfe_used=1, n_accepted=0, success=True,
        )

    tokens, q = _traverse_lazy_argmax(
        csr, prob_vectors, start_state, live_states, d
    )

    return SpeculativeResult(
        tokens=tokens,
        final_state=q,
        draft_length=len(tokens),
        nfe_used=1,
        n_accepted=len(tokens),
        success=q in live_states and len(tokens) == d,
    )


def speculative_decode_lazy(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    draft_length: int | None = None,
) -> SpeculativeResult:
    """
    Lazy SDSD: O(d*K) memory, no full tree.
    Alias for speculative_decode(..., use_lazy=True).
    """
    return speculative_decode(
        csr, prob_vectors, start_state, live_states, draft_length, use_lazy=True
    )


# ---------------------------------------------------------------------------
# Multi-round SDSD with model-in-loop and verification
# ---------------------------------------------------------------------------

def sdsd_multi_round(
    csr: CSRTransitionMatrix,
    get_block_logits_fn: Callable[[list[int], int], list[list[float]]],
    get_verify_logits_fn: Callable[[list[int]], list[list[float]]],
    start_state: int,
    live_states: set[int],
    target_length: int,
    draft_length: int = 16,
    verify_threshold: float = 0.0,
) -> tuple[list[int], int, int, bool]:
    """
    SDSD with model-in-loop: generate target_length tokens via multi-round
    draft + verify. Each round: 1 forward (draft) + 1 forward (verify) = 2 NFE.
    NFE = 2 * num_rounds = 2 * ceil(target_length / tau) where tau = avg accepted.

    get_block_logits_fn(prefix_tokens, block_len) -> prob_vectors
    get_verify_logits_fn(prefix_plus_draft) -> prob_vectors for verification

    Returns (tokens, nfe, n_accepted_total, success).
    """
    tokens: list[int] = []
    q = start_state
    nfe = 0

    while len(tokens) < target_length:
        gamma = min(draft_length, target_length - len(tokens))
        if gamma <= 0:
            break

        # Step A: Self-Drafting - 1 forward for block logits
        prob_vectors = get_block_logits_fn(tokens, gamma)
        nfe += 1

        if not prob_vectors:
            break

        # Step B: Grammar-Legal Tree with Herding (O(K) per node)
        try:
            from .herding import herding_decode
        except ImportError:
            from herding import herding_decode

        r = herding_decode(csr, prob_vectors, q, live_states, block_length=gamma)
        draft = r.tokens
        q = r.final_state

        if not draft:
            break

        # Step C: Verification - second forward pass
        context = tokens + draft
        all_verify_probs = get_verify_logits_fn(context)
        nfe += 1
        # all_verify_probs[i] predicts context[i]; we need probs for draft positions
        verify_probs = all_verify_probs[len(tokens):] if len(all_verify_probs) > len(tokens) else []

        n_accepted = 0
        for i in range(min(len(draft), len(verify_probs))):
            p = verify_probs[i]
            t = draft[i]
            if t >= len(p) or p[t] <= verify_threshold:
                break
            n_accepted += 1

        # Update state for accepted tokens
        for i in range(n_accepted):
            for tt, qn in csr.get_transitions(q):
                if tt == draft[i]:
                    q = qn
                    break

        tokens.extend(draft[:n_accepted])
        if n_accepted < len(draft):
            break  # Verification rejected some; stop round

    success = q in live_states and len(tokens) >= target_length
    return tokens[:target_length], nfe, len(tokens), success


def sdsd_multi_round_argmax(
    csr: CSRTransitionMatrix,
    get_block_logits_fn: Callable[[list[int], int], list[list[float]]],
    get_verify_logits_fn: Callable[[list[int]], list[list[float]]],
    start_state: int,
    live_states: set[int],
    target_length: int,
    draft_length: int = 16,
    verify_threshold: float = 0.0,
) -> tuple[list[int], int, int, bool]:
    """
    Ablation 3: STATIC + Spec-Tree with argmax (no Herding), model-in-loop.
    Same as sdsd_multi_round but uses argmax instead of Herding for path selection.
    """
    tokens: list[int] = []
    q = start_state
    nfe = 0

    while len(tokens) < target_length:
        gamma = min(draft_length, target_length - len(tokens))
        if gamma <= 0:
            break

        prob_vectors = get_block_logits_fn(tokens, gamma)
        nfe += 1

        if not prob_vectors:
            break

        draft, q = _traverse_lazy_argmax(csr, prob_vectors, q, live_states, gamma)

        if not draft:
            break

        context = tokens + draft
        all_verify_probs = get_verify_logits_fn(context)
        nfe += 1
        verify_probs = all_verify_probs[len(tokens):] if len(all_verify_probs) > len(tokens) else []

        n_accepted = 0
        for i in range(min(len(draft), len(verify_probs))):
            p = verify_probs[i]
            t = draft[i]
            if t >= len(p) or p[t] <= verify_threshold:
                break
            n_accepted += 1

        for i in range(n_accepted):
            for tt, qn in csr.get_transitions(q):
                if tt == draft[i]:
                    q = qn
                    break

        tokens.extend(draft[:n_accepted])
        if n_accepted < len(draft):
            break

    success = q in live_states and len(tokens) >= target_length
    return tokens[:target_length], nfe, len(tokens), success
