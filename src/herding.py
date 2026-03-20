"""
Herding for Constrained Decoding: Deterministic Path with Momentum

Implements the time-dependent herding algorithm from:
  Suzuki et al., "Deterministic Discrete Denoising" (ddd_herding.pdf)
  Eqs. (9)-(10): x* = argmax_{x∈V} (w_t + p_t)ᵀ x,  w_{t+1} = w_t + p_t - x*

When constraints force a token different from the model's top prediction (e.g.,
grammar requires " but model wanted "password"), the probability mass is lost
("evaporated" by the sampling wall). Herding preserves this via a residual
weight vector w that accumulates blocked intentions.

Update rule: x* = argmax_{x ∈ V_valid} (w_t + p_t)ᵀ e_x
Herding update: w_{t+1} = w_t + p_t - e_{x*}

When the grammar later allows string input, the accumulated momentum in w
deterministically favors tokens like "password" that were previously blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .csr_dfa import CSRTransitionMatrix


@dataclass
class HerdingMomentumState:
    """
    Persistent Herding momentum across many frontier picks (e.g. diffusion steps).

    w starts at zero; each constrained step does w <- w + p - e_{t*} on the full vocab.
    prev_token is the last chosen token (for optional delta tie-break).
    """

    w: list[float] | None = None
    prev_token: int | None = None

    def ensure_w(self, vocab_size: int) -> list[float]:
        if self.w is None or len(self.w) != vocab_size:
            self.w = [0.0] * vocab_size
        return self.w


@dataclass
class HerdingResult:
    """Result of Herding constrained decoding."""
    tokens: list[int]
    final_state: int
    momentum_trace: list[list[float]]  # w at each step (for analysis)
    success: bool


def herding_single_constrained_step(
    prob_vector: list[float],
    csr: "CSRTransitionMatrix",
    start_state: int,
    w: list[float],
    prev_token: int | None,
    delta: float = 0.0,
) -> tuple[int | None, bool]:
    """
    One step of constrained Herding with external momentum w (updated in place).

    score(t) = w[t] + p[t] + (delta if t == prev_token else 0)  for t in V_valid(start_state)
    Pick argmax; then w <- w + p - e_{t*}.

    Returns (chosen_token, True) or (None, False) if no valid transition.
    """
    vocab_size = len(w)
    if len(prob_vector) < vocab_size:
        prob_vector = list(prob_vector) + [0.0] * (vocab_size - len(prob_vector))

    transitions = csr.get_transitions(start_state)
    if not transitions:
        return None, False

    best_t: int | None = None
    best_q_next: int | None = None
    best_score = float("-inf")

    for t, q_next in transitions:
        if t < 0 or t >= vocab_size:
            continue
        score = w[t] + prob_vector[t]
        if delta > 0 and prev_token is not None and t == prev_token:
            score += delta
        if score > best_score:
            best_score = score
            best_t = t
            best_q_next = q_next

    if best_t is None or best_q_next is None:
        return None, False

    for j in range(vocab_size):
        w[j] = w[j] + prob_vector[j]
    w[best_t] -= 1.0

    return best_t, True


def herding_decode(
    csr: CSRTransitionMatrix,
    prob_vectors: list[list[float]],
    start_state: int,
    live_states: set[int],
    block_length: int | None = None,
    delta: float = 0.0,
    prev_token: int | None = None,
) -> HerdingResult:
    """
    Greedy constrained decoding with Herding momentum.

    Aligns with Suzuki et al. "Deterministic Discrete Denoising" (ddd_herding.pdf):
      Eq. (9): x* = argmax_{x∈V} (w_t + p_t)ᵀ x
      Eq. (10): w_{t+1} = w_t + p_t - x*
    We restrict argmax to V_valid(q) for grammar-constrained decoding.

    At each step i:
      score(t) = w_i[t] + p_i[t] + (δ if t == prev_token else 0)   for t ∈ V_valid(q)
      t* = argmax score over valid t
      w_{i+1} = w_i + p_i - e_{t*}
      q_{i+1} = δ(q_i, t*)

    Optional delayed-switching (Eq. 11): delta > 0 favors prev_token to reduce switching.
    O(K) per step via CSR sparse lookup.
    """
    d = block_length if block_length is not None else len(prob_vectors)
    if d == 0:
        return HerdingResult(
            tokens=[], final_state=start_state,
            momentum_trace=[], success=True,
        )
    
    vocab_size = len(prob_vectors[0]) if prob_vectors else csr.vocab_size
    w = [0.0] * vocab_size
    tokens: list[int] = []
    momentum_trace: list[list[float]] = [w.copy()]
    q = start_state
    
    for i in range(d):
        p = prob_vectors[i]
        if len(p) < vocab_size:
            p = p + [0.0] * (vocab_size - len(p))
        
        transitions = csr.get_transitions(q)
        if not transitions:
            return HerdingResult(
                tokens=tokens, final_state=q,
                momentum_trace=momentum_trace,
                success=False,
            )
        
        best_t: int | None = None
        best_q_next: int | None = None
        best_score = float("-inf")
        
        for t, q_next in transitions:
            score = w[t] + p[t]
            if delta > 0 and prev_token is not None and t == prev_token:
                score += delta
            if score > best_score:
                best_score = score
                best_t = t
                best_q_next = q_next
        
        if best_t is None or best_q_next is None:
            return HerdingResult(
                tokens=tokens, final_state=q,
                momentum_trace=momentum_trace,
                success=False,
            )
        
        tokens.append(best_t)
        prev_token = best_t

        # Herding update: w_new = w + p - e_{t*}  (Eq. 10)
        for j in range(vocab_size):
            w[j] = w[j] + p[j]
        w[best_t] -= 1.0
        
        q = best_q_next
        momentum_trace.append(w.copy())
    
    return HerdingResult(
        tokens=tokens,
        final_state=q,
        momentum_trace=momentum_trace,
        success=q in live_states,
    )
