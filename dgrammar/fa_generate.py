"""Proactive constrained decoding for diffusion LMs via exact FA inference.

This replaces the reactive repair pipeline in ``dp_generate.py``. There, the
grammar is enforced at a single frontier position, every other position is an
unconstrained argmax, violations are therefore inevitable, and six layers of
machinery (greedy retry, span DP, constraint-end heuristics, progressive window
expansion, resample budget, post-hoc bracket enrichment) exist to clean them up.

Here the constraint is a factor in the distribution itself.  At every denoising
step we condition the model's mean-field prediction on the whole constraint and
decode from the exact constrained posterior

    p(x^0 | x^t, G)  ∝  ( Π_i p_θ(x_i^0 | x^t) ) · 1[x^0 ∈ L(G)]

which a finite automaton makes tractable: the automaton is a chain-structured
graphical model over positions, and multiplying a chain by a fully-factorized
measure leaves a chain -- only the emission factors are reweighted.  Exact
inference is then HMM forward-backward.  Violations cannot occur, so none of
the repair machinery is needed.

Two axes, set independently, spanning the 2x2 the experiments need:

    reactive  vs  proactive   -- dp_generate.py  vs  this file
    mode      vs  mass        -- decoder="viterbi"  vs  "marginal" / "sample"

Both matter.  Moving to proactive alignment removes degeneracy caused by
*projecting* a bad proposal onto the constraint set (the nearest valid point to
garbage after ``{`` is ``}``).  It does not remove degeneracy caused by
*mode-seeking*: the highest-probability valid string is often the shortest one,
since closing the structure costs one token and EOS padding is nearly free,
while content pays entropy at every position.  Only mass-based decoding fixes
that, by summing over the combinatorially many content-bearing paths.

References
----------
Dang & Ermon, "Constrained Decoding for Diffusion Language Models via Efficient
Inference over Finite Automata" -- sections 3.2, 4.2, and appendix A.2 (the
state-space formulation used here).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. Constraint representation
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything below needs exactly four things from the grammar backend:
#
#     num_states, start, accept[num_states], delta[num_states, 256]
#
# with delta[s, b] = -1 for "no transition".  That is a byte-level DFA.  It is
# lifted to token level once per instance and then never touched again -- all
# per-step work is dense tensor arithmetic.


@dataclass
class ByteDFA:
    """Byte-level deterministic automaton.  ``delta[s, b] == -1`` means dead."""

    delta: np.ndarray          # [S, 256] int32
    start: int
    accept: np.ndarray         # [S] bool

    @property
    def num_states(self) -> int:
        return int(self.delta.shape[0])

    def __post_init__(self) -> None:
        assert self.delta.ndim == 2 and self.delta.shape[1] == 256
        assert self.accept.shape == (self.num_states,)
        assert 0 <= self.start < self.num_states


def seed_prefix(bdfa: ByteDFA, text: bytes) -> ByteDFA:
    """Advance the start state by consuming ``text``; raise if it is rejected.

    Needed whenever the model does not generate from the beginning of the
    language.  CPP-Bench puts the function declaration in the prompt via
    ``assistant_start_line()``, so the generation region is a function *body*
    and the automaton must already be inside the parse when decoding starts.
    """
    state = bdfa.start
    for i, b in enumerate(text):
        nxt = int(bdfa.delta[state, b])
        if nxt < 0:
            raise ValueError(
                f"grammar rejects the seed prefix at byte {i} ({text[:i+1]!r})"
            )
        state = nxt
    return ByteDFA(delta=bdfa.delta, start=state, accept=bdfa.accept)


def append_optional_suffix(bdfa: ByteDFA, suffix: bytes) -> ByteDFA:
    """Also accept ``L(G) · suffix``, keeping ``L(G)`` itself accepted.

    Used for the closing code fence.  The prompts in this harness ask for output
    inside ```` ```lang ... ``` ````, but whether the opening fence lands in the
    prompt or in the generation region depends on ``prepare_prompt``.  Accepting
    the closing fence *optionally* makes the automaton correct either way, which
    is safer than guessing and silently constraining the model into a language
    it was not asked to produce.
    """
    if not suffix:
        return bdfa

    S = bdfa.num_states
    extra = len(suffix)
    delta = np.full((S + extra, 256), -1, dtype=np.int32)
    delta[:S] = bdfa.delta

    # Chain of fresh states spelling out the suffix, entered from every state
    # that already accepts.
    for i, b in enumerate(suffix):
        src = S + i - 1
        dst = S + i
        if i == 0:
            for s in np.nonzero(bdfa.accept)[0]:
                if delta[s, b] >= 0:
                    raise ValueError(
                        f"suffix byte {b!r} already has a transition from accepting "
                        f"state {s}; the optional suffix would be ambiguous"
                    )
                delta[s, b] = dst
        else:
            delta[src, b] = dst

    accept = np.concatenate([bdfa.accept, np.zeros(extra, dtype=bool)])
    accept[S + extra - 1] = True
    return ByteDFA(delta=delta, start=bdfa.start, accept=accept)


def _terminal_states(num_states, ptr, tok, dst, eos_id) -> np.ndarray:
    """States from which the only remaining output is EOS padding.

    A state is terminal when every edge leaving it emits EOS and lands in a
    state that is itself terminal (or is the same state, i.e. the absorbing EOS
    self-loop).  Reaching one means the structure is finished and the rest of
    the window is padding.

    This is what makes "close now" separable from "keep writing" in
    :func:`decode`'s diagnostic: an edge into a terminal state is the decoder
    choosing to stop.
    """
    terminal = np.zeros(num_states, dtype=bool)
    # Seed: states whose every outgoing edge is EOS.
    all_eos = np.zeros(num_states, dtype=bool)
    for s in range(num_states):
        lo, hi = int(ptr[s]), int(ptr[s + 1])
        all_eos[s] = hi > lo and bool((tok[lo:hi] == eos_id).all())

    changed = True
    while changed:                       # converges in <= depth of the EOS tail
        changed = False
        for s in np.nonzero(all_eos & ~terminal)[0]:
            lo, hi = int(ptr[s]), int(ptr[s + 1])
            outs = dst[lo:hi]
            if bool(((outs == s) | terminal[outs]).all()):
                terminal[s] = True
                changed = True
    return terminal


@dataclass
class TokenDFA:
    """Token-level automaton as a flat edge list, ready for message passing.

    Edges are ``(src[e], tok[e], dst[e])``.  The flat form is what makes the
    per-position transition matrices a single scatter-add.
    """

    num_states: int
    start: int
    accept: torch.Tensor       # [S] bool
    src: torch.Tensor          # [E] int64, sorted ascending
    tok: torch.Tensor          # [E] int64
    dst: torch.Tensor          # [E] int64
    src_ptr: torch.Tensor      # [S+1] int64, CSR offsets into the edge arrays
    terminal: torch.Tensor     # [S] bool, "everything from here on is EOS padding"
    vocab_size: int
    eos_id: int

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())

    def to(self, device) -> "TokenDFA":
        return TokenDFA(
            num_states=self.num_states, start=self.start,
            accept=self.accept.to(device), src=self.src.to(device),
            tok=self.tok.to(device), dst=self.dst.to(device),
            src_ptr=self.src_ptr.to(device), terminal=self.terminal.to(device),
            vocab_size=self.vocab_size, eos_id=self.eos_id,
        )

    def out_edges(self, state: int) -> slice:
        """Edge range leaving ``state``.

        The decode walk needs this once per position.  Scanning ``src == state``
        instead would cost O(E) per position, i.e. another O(L*E) per denoising
        step -- around 100M comparisons on a median JSON-Schema automaton, for
        work that a CSR offset makes O(1).
        """
        return slice(int(self.src_ptr[state]), int(self.src_ptr[state + 1]))

    # ── construction ──────────────────────────────────────────────────────

    @staticmethod
    def _assemble(num_states, start, accept, src, tok, dst, vocab_size, eos_id,
                  device) -> "TokenDFA":
        """Sort edges by source state, build the CSR table, mark terminal states."""
        order = np.argsort(src, kind="stable")
        src, tok, dst = src[order], tok[order], dst[order]
        counts = np.bincount(src, minlength=num_states)
        ptr = np.concatenate([[0], np.cumsum(counts)])

        terminal = _terminal_states(num_states, ptr, tok, dst, eos_id)

        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
        return TokenDFA(
            num_states=int(num_states), start=int(start),
            accept=t(accept, torch.bool),
            src=t(src, torch.int64), tok=t(tok, torch.int64), dst=t(dst, torch.int64),
            src_ptr=t(ptr, torch.int64), terminal=t(terminal, torch.bool),
            vocab_size=int(vocab_size), eos_id=int(eos_id),
        )

    @staticmethod
    def from_outlines_index(
        index,
        vocab_size: int,
        eos_id: int,
        *,
        device: str | torch.device = "cpu",
    ) -> "TokenDFA":
        """Build from an ``outlines_core.Index``.

        Preferred over :meth:`from_byte_dfa`.  ``Index`` is constructed in Rust
        from a regex plus the real tokenizer vocabulary, so it already performs
        the byte-DFA-to-token lift correctly -- including tokens that are
        fragments of a multi-byte UTF-8 character, which the ``batch_decode``
        route in :func:`build_vocab_bytes` can only drop.  It also already
        installs the EOS self-loop on final states, so no padding tail is added
        here.

        ``index.get_transitions()`` is ``{state: {token_id: next_state}}``.
        """
        transitions = index.get_transitions()
        finals = set(index.get_final_states())

        states = sorted(set(transitions) | finals | {index.get_initial_state()})
        remap = {s: i for i, s in enumerate(states)}

        src_l, tok_l, dst_l = [], [], []
        for s, row in transitions.items():
            si = remap[s]
            for token, nxt in row.items():
                if nxt not in remap:          # transition into an unlisted sink
                    continue
                src_l.append(si)
                tok_l.append(token)
                dst_l.append(remap[nxt])

        accept = np.zeros(len(states), dtype=bool)
        for s in finals:
            accept[remap[s]] = True

        return TokenDFA._assemble(
            num_states=len(states),
            start=remap[index.get_initial_state()],
            accept=accept,
            src=np.asarray(src_l, dtype=np.int64),
            tok=np.asarray(tok_l, dtype=np.int64),
            dst=np.asarray(dst_l, dtype=np.int64),
            vocab_size=vocab_size, eos_id=eos_id, device=device,
        )

    @staticmethod
    def from_byte_dfa(
        bdfa: ByteDFA,
        vocab_bytes: list[Optional[bytes]],
        eos_id: int,
        *,
        state_chunk: int = 64,
        device: str | torch.device = "cpu",
    ) -> "TokenDFA":
        """Lift a byte DFA to token level, then add the EOS padding tail.

        For every (state, token) pair we walk the token's bytes through the byte
        automaton.  Done as ``state = delta[state, byte]`` gathers over the whole
        [S, V] grid at once, so the cost is ``max_token_len`` vectorized gathers
        rather than S*V Python loops.

        ``vocab_bytes[v] is None`` marks a token that must never be emitted
        (MASK, control tokens).  EOS is excluded here and reintroduced below as
        the padding alphabet.

        The EOS tail is what lets a fixed-length window hold a variable-length
        string: from any accepting state, EOS moves to an absorbing accepting
        state.  Closing the structure early therefore becomes a *modelled*
        decision that has to compete for posterior mass against every content
        continuation, instead of an external ``if is_accepting(): break``.
        """
        S, V = bdfa.num_states, len(vocab_bytes)
        max_len = max((len(b) for b in vocab_bytes if b), default=1)

        # Pad token bytes into a dense [V, max_len] grid.
        tok_bytes = np.zeros((V, max_len), dtype=np.int64)
        tok_lens = np.zeros(V, dtype=np.int64)
        for v, b in enumerate(vocab_bytes):
            if not b:
                continue                                  # length 0 => skipped
            tok_bytes[v, : len(b)] = np.frombuffer(b, dtype=np.uint8)
            tok_lens[v] = len(b)

        delta = bdfa.delta
        src_l, tok_l, dst_l = [], [], []

        for lo in range(0, S, state_chunk):
            hi = min(lo + state_chunk, S)
            cur = np.repeat(np.arange(lo, hi, dtype=np.int64)[:, None], V, axis=1)
            alive = np.repeat((tok_lens > 0)[None, :], hi - lo, axis=0)

            for j in range(max_len):
                step = alive & (tok_lens[None, :] > j)
                if not step.any():
                    break
                nxt = delta[cur, tok_bytes[None, :, j]]
                cur = np.where(step, nxt, cur)
                alive &= ~(step & (cur < 0))
                cur = np.where(cur < 0, 0, cur)           # park dead paths

            rows, cols = np.nonzero(alive)
            src_l.append(rows + lo)
            tok_l.append(cols)
            dst_l.append(cur[rows, cols])

        src = np.concatenate(src_l) if src_l else np.zeros(0, dtype=np.int64)
        tok = np.concatenate(tok_l) if tok_l else np.zeros(0, dtype=np.int64)
        dst = np.concatenate(dst_l) if dst_l else np.zeros(0, dtype=np.int64)

        # EOS tail: new absorbing accepting state reachable from every accepting
        # state (and from itself) on EOS.
        eos_state = S
        accept = np.concatenate([bdfa.accept, [True]])
        acc_states = np.nonzero(bdfa.accept)[0]
        src = np.concatenate([src, acc_states, [eos_state]])
        tok = np.concatenate([tok, np.full(len(acc_states) + 1, eos_id, dtype=np.int64)])
        dst = np.concatenate([dst, np.full(len(acc_states) + 1, eos_state, dtype=np.int64)])

        return TokenDFA._assemble(
            num_states=S + 1, start=bdfa.start, accept=accept,
            src=src.astype(np.int64), tok=tok.astype(np.int64), dst=dst.astype(np.int64),
            vocab_size=V, eos_id=eos_id, device=device,
        )


_REPLACEMENT = "�"


def build_vocab_bytes(
    tokenizer,
    vocab_size: int,
    *,
    blocked_ids: Optional[set[int]] = None,
    verbose: bool = True,
) -> list[Optional[bytes]]:
    """Byte string for each token id; ``None`` for non-emittable ids.

    Ids that decode to nothing, that appear in ``blocked_ids`` (MASK, EOS,
    control tokens), or that cannot be resolved to real bytes are marked
    non-emittable.  EOS re-enters via the padding tail added in
    :meth:`TokenDFA.from_byte_dfa`.

    Byte-level BPE caveat
    ---------------------
    Some ids in a byte-level BPE are *fragments* of a multi-byte UTF-8
    character.  Decoding such an id in isolation yields U+FFFD, and encoding
    that back produces the bytes of the replacement character rather than the
    token's real bytes -- which would install wrong transitions in the
    automaton, silently.

    We therefore prefer the tokenizer's byte-level view when one is exposed
    (``convert_ids_to_tokens`` plus the GPT-2 byte decoder), fall back to
    ``batch_decode``, and drop any id that still resolves to U+FFFD.  Dropping
    is the safe direction: those tokens simply become unemittable, which costs
    a little coverage but cannot make an invalid string reachable.  ``verbose``
    reports how many were dropped -- a large count means the byte-level path
    failed and the automaton is missing real vocabulary.
    """
    blocked = set(blocked_ids or ())
    out: list[Optional[bytes]] = [None] * vocab_size
    dropped = 0

    byte_decoder = None
    try:                                        # GPT-2 style byte-level BPE
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        pieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    except Exception:                           # noqa: BLE001
        byte_decoder = None
        pieces = tokenizer.batch_decode(
            [[i] for i in range(vocab_size)], skip_special_tokens=False
        )

    for i, piece in enumerate(pieces):
        if i in blocked or not piece:
            continue
        if byte_decoder is not None:
            try:
                out[i] = bytes(byte_decoder[c] for c in piece)
                continue
            except KeyError:
                pass                            # special token or unmapped glyph
        if _REPLACEMENT in piece:
            dropped += 1
            continue
        out[i] = piece.encode("utf-8")

    if verbose:
        emittable = sum(b is not None for b in out)
        mode = "byte-level" if byte_decoder is not None else "decode fallback"
        print(f"  vocab: {emittable}/{vocab_size} emittable via {mode}"
              + (f", {dropped} dropped as undecodable" if dropped else ""))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Exact inference: forward-backward over the constrained posterior
# ═══════════════════════════════════════════════════════════════════════════


_CHUNK_ELEMS = 4_000_000     # ~32 MB per float64 buffer


def _pos_chunk(num_edges: int) -> int:
    """Positions to process at once, sized by the automaton rather than fixed.

    Every chunked pass allocates a ``[chunk, E]`` buffer.  A fixed chunk is fine
    at the JSON-Bench median (E~390k) but fatal on JSONSchemaBench-medium, where
    E reaches 1.8M and beyond: 32 positions x 1.8M x 8 bytes is 460 MB *per
    buffer*, and there are several live at once.  That is what killed the first
    jsb_medium run with "Runner was terminated whilst exceeding its memory
    request".  Scaling the chunk down keeps the footprint flat as E grows.
    """
    return max(1, min(32, _CHUNK_ELEMS // max(num_edges, 1)))


def _emission(
    probs: torch.Tensor,
    committed: torch.Tensor,
    dfa: TokenDFA,
    mask_id: int,
    lo: int,
    hi: int,
) -> torch.Tensor:
    """Emission weights ``w[t, e]`` for positions ``[lo, hi)`` only.

    Masked positions take the model's mean-field probability for that edge's
    token.  Committed positions take a delta on the token already there, which
    is how the posterior conditions on what has been decided so far.

    Never materialise this for the whole sequence.  Real JSON-Schema automata
    reach ~400k edges at the median and ~20M in the tail, so a full ``[L, E]``
    tensor is 800 MB at the median and tens of GB in the tail.  Every caller
    works in position chunks and discards each chunk.
    """
    w = probs[lo:hi][:, dfa.tok]                                   # [hi-lo, E]
    part = committed[lo:hi]
    fixed = part != mask_id
    if fixed.any():
        hit = dfa.tok.unsqueeze(0) == part.unsqueeze(1)
        w = torch.where(fixed.unsqueeze(1), hit.to(w.dtype), w)
    return w


def _transition_matrices(
    probs: torch.Tensor,
    committed: torch.Tensor,
    dfa: TokenDFA,
    mask_id: int,
    *,
    reduce: str = "sum",
    pos_chunk: Optional[int] = None,
) -> torch.Tensor:
    """Per-position transition matrices ``[L, S, S]`` aggregated over tokens.

    ``reduce="sum"``   M[t, s, s'] = Σ over edges s -> s' of w[t, edge]
    ``reduce="amax"``  M[t, s, s'] = max over edges s -> s' of w[t, edge]

    The reduction must match the semiring: sum-product marginalizes over which
    token took the transition, max-product maximizes over it.  Using ``sum``
    under max-product silently inflates any state pair joined by several tokens
    (a digit class, a character class) by roughly its size, which is enough to
    flip the decision.

    One scatter per position chunk.  This is the only place the edge list is
    touched during a step; the recursions below run on the small [S, S] blocks.
    """
    L = probs.shape[0]
    S = dfa.num_states
    pos_chunk = pos_chunk or _pos_chunk(dfa.num_edges)
    flat_idx = dfa.src * S + dfa.dst                          # [E]
    M = torch.zeros(L, S * S, dtype=probs.dtype, device=probs.device)
    for lo in range(0, L, pos_chunk):
        hi = min(lo + pos_chunk, L)
        w = _emission(probs, committed, dfa, mask_id, lo, hi)  # [hi-lo, E]
        idx = flat_idx.expand(hi - lo, -1)
        if reduce == "sum":
            M[lo:hi].scatter_add_(1, idx, w)
        else:
            # weights are non-negative, so a zero-initialised amax is safe
            M[lo:hi] = M[lo:hi].scatter_reduce(
                1, idx, w, reduce="amax", include_self=True
            )
        del w
    return M.view(L, S, S)


def _backward(M: torch.Tensor, dfa: TokenDFA, *, mode: str = "sum"):
    """Backward messages over the chain.

    ``beta[t, s]`` is the (rescaled) partition function of positions t..L-1 given
    state s at t, restricted to paths ending in an accepting state.  ``mode``
    selects sum-product (mass) or max-product (mode).

    Returns ``(beta, log_scale)`` with ``beta[L] = accept``.  Per-position
    rescaling keeps everything in float range over long windows; the log scales
    are carried so partition functions remain comparable across positions.

    Use float64.  Rescaling divides each ``beta[t]`` by its max over states, so
    it bounds the *largest* entry -- it does nothing about the spread *within*
    the vector, and that spread is enormous: on a real JSON-Schema automaton the
    start state's mass-to-go runs ~1e-107 below the best state's, because the
    start state can emit one bracket while a state inside a free string has
    thousands of continuations. In float32 that is under the 1e-38 floor, so
    ``beta[0, start]`` flushes to zero and :func:`decode` reports "no accepting
    path" on inputs that are perfectly satisfiable. float64 leaves ~1e-200 of
    headroom and the figure does not compound with sequence length.
    """
    L, S = M.shape[0], M.shape[1]
    beta = torch.zeros(L + 1, S, dtype=M.dtype, device=M.device)
    log_scale = torch.zeros(L + 1, dtype=M.dtype, device=M.device)
    beta[L] = dfa.accept.to(M.dtype)

    for t in range(L - 1, -1, -1):
        if mode == "sum":
            b = M[t] @ beta[t + 1]
        else:
            b = (M[t] * beta[t + 1].unsqueeze(0)).max(dim=1).values
        m = b.max()
        if m > 0:
            b = b / m
            log_scale[t] = log_scale[t + 1] + torch.log(m)
        else:
            log_scale[t] = log_scale[t + 1]
        beta[t] = b
    return beta, log_scale


def _forward(M: torch.Tensor, dfa: TokenDFA):
    """Forward messages ``alpha[t, s]`` over states, with per-position rescaling."""
    L, S = M.shape[0], M.shape[1]
    alpha = torch.zeros(L + 1, S, dtype=M.dtype, device=M.device)
    alpha[0, dfa.start] = 1.0
    for t in range(L):
        a = alpha[t] @ M[t]
        m = a.max()
        alpha[t + 1] = a / m if m > 0 else a
    return alpha


def _token_marginals(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    w_t: torch.Tensor,
    dfa: TokenDFA,
    t: int,
) -> torch.Tensor:
    """Exact constrained marginal over the vocabulary at position ``t``.

        p(x_t = v | x^t, G)  ∝  Σ_{e: tok(e)=v} α[t, src(e)] · w[t, e] · β[t+1, dst(e)]

    Note the sum is over *all* source states, not just the one the decode walk
    currently occupies -- that global aggregation is exactly what separates
    max-marginal from Viterbi.  A single closing token is one path; content is
    combinatorially many, each individually unlikely.
    """
    contrib = alpha[t][dfa.src] * w_t * beta[t + 1][dfa.dst]    # [E]
    out = torch.zeros(dfa.vocab_size, dtype=w_t.dtype, device=w_t.device)
    out.scatter_add_(0, dfa.tok, contrib)
    return out


def _all_token_marginals(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    probs: torch.Tensor,
    committed: torch.Tensor,
    dfa: TokenDFA,
    mask_id: int,
    *,
    pos_chunk: Optional[int] = None,
) -> torch.Tensor:
    """Constrained token marginals for every position -> ``[L, V]``.

    Same quantity as :func:`_token_marginals`, batched over positions.  Doing it
    one position at a time inside the decode walk cost an E-sized gather plus a
    host synchronisation per position; at L=256 that was the dominant term in
    the whole denoising step.
    """
    L = probs.shape[0]
    pos_chunk = pos_chunk or _pos_chunk(dfa.num_edges)
    out = torch.zeros(L, dfa.vocab_size, dtype=probs.dtype, device=probs.device)
    tok_idx = dfa.tok.expand(1, -1)
    for lo in range(0, L, pos_chunk):
        hi = min(lo + pos_chunk, L)
        w = _emission(probs, committed, dfa, mask_id, lo, hi)          # [chunk, E]
        contrib = alpha[lo:hi][:, dfa.src] * w * beta[lo + 1:hi + 1][:, dfa.dst]
        out[lo:hi].scatter_add_(1, tok_idx.expand(hi - lo, -1), contrib)
        del w, contrib
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. Decoders -- all three read the same messages
# ═══════════════════════════════════════════════════════════════════════════


def decode(
    probs: torch.Tensor,
    committed: torch.Tensor,
    dfa: TokenDFA,
    *,
    mask_id: int,
    decoder: str = "marginal",
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    stats: Optional["FAStats"] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one length-L sequence from the constrained posterior.

    ``decoder``:
      ``viterbi``   max-product; the single most probable valid string (mode).
      ``sample``    ancestral sampling from the exact posterior (mass, stochastic).
      ``marginal``  constrained max-marginal walk (mass, deterministic): at each
                    position take the token with the largest *global* marginal
                    among those reachable from the current state, then advance.
                    Staying on the automaton keeps the result valid by
                    construction, which independent per-position argmax would not.

    Returns ``(tokens[L], confidence[L])`` where confidence is the normalized
    constrained marginal of the chosen token -- the remasking signal.  This is
    Dang & Ermon's "Mar" confidence; using the base mean-field instead ("Mf") is
    the ablation.
    """
    L = probs.shape[0]
    is_mode = decoder == "viterbi"
    args = (probs, committed, dfa, mask_id)

    # Max-product needs its own reduction over tokens; see _transition_matrices.
    M = _transition_matrices(*args, reduce="amax" if is_mode else "sum")
    beta, _ = _backward(M, dfa, mode="max" if is_mode else "sum")

    if beta[0, dfa.start] <= 0:
        raise RuntimeError(
            "no accepting path: the committed prefix cannot be completed within "
            "the window. Under proactive alignment this is unreachable unless the "
            "window is too short or the DFA is wrong."
        )

    # Marginals are always sum-product, whatever the decoder, so that the
    # remasking confidence stays a probability.  Computed for every position up
    # front, in chunks: doing it inside the walk costs one E-sized gather plus a
    # host sync per position, which dominated the whole step.
    marg = None
    if decoder == "marginal":
        alpha = _forward(M if not is_mode else _transition_matrices(*args), dfa)
        marg = _all_token_marginals(alpha, beta, *args)               # [L, V]
    del M

    # The walk is inherently sequential and needs the current state on the host
    # to index the CSR table.  Run it entirely on CPU so no step forces a
    # device sync; the tensors moved here are small next to a [L, V] of logits.
    beta_c = beta.cpu()
    committed_c = committed.cpu()
    marg_c = marg.float().cpu() if marg is not None else None

    # Every decoder now ranks by edge score, so all of them need the emission
    # probabilities host-side.  `marg` stays for the remasking confidence only.
    probs_c = probs.cpu()
    tok_c, dst_c, ptr_c = dfa.tok.cpu(), dfa.dst.cpu(), dfa.src_ptr.cpu()

    tokens = torch.zeros(L, dtype=torch.long)
    conf = torch.zeros(L, dtype=probs.dtype)
    state = dfa.start

    for t in range(L):
        lo, hi = int(ptr_c[state]), int(ptr_c[state + 1])
        if hi <= lo:
            raise RuntimeError(f"dead state {state} at position {t}")
        edge_tok = tok_c[lo:hi]
        edge_dst = dst_c[lo:hi]

        # Emission for this state's out-edges only: O(deg), not O(E).
        fixed = int(committed_c[t]) != mask_id
        w_e = (edge_tok == committed_c[t]).to(probs.dtype) if fixed \
            else probs_c[t, edge_tok]
        score = w_e * beta_c[t + 1][edge_dst]

        total = score.sum()
        if float(total) <= 0:
            raise RuntimeError(f"no continuation from state {state} at position {t}")

        if decoder == "marginal":
            # Conditional marginal given the prefix:
            #     p(x_t = v | x_<t, C)  ∝  p_θ(v) · β_sum[t+1][δ(s_t, v)]
            # which is exactly `score` under the sum-product semiring.
            #
            # Ranking by the *global* marginal instead is incoherent with the
            # walk and actively harmful: marg[v] aggregates over every source
            # state, so a token filling many structural roles across the
            # automaton -- `]`, `}`, `,` in JSON -- accumulates mass from all of
            # them, while a content token appears in few. That systematically
            # steers the walk into closers. It also strands the walk, since a
            # token can be globally popular while *this* edge leads nowhere
            # accepting. Measured on JSONSchemaBench-medium o10217, the global
            # rule emitted `{"months":[]}` where Viterbi wrote a full document.
            #
            # With the conditional rule, mode vs mass reduces to exactly the
            # right thing: β_max (best single completion) vs β_sum (total mass
            # over completions). Feasibility is automatic -- score > 0.
            pick = int(torch.argmax(score))
            # Remasking confidence stays the position marginal p(x_t = v | x^t, C)
            # -- Dang & Ermon's "Mar" signal -- not the conditional used to pick.
            row = marg_c[t]
            row_total = row.sum()
            conf[t] = row[edge_tok[pick]] / row_total if row_total > 0 else 0.0
        elif decoder == "viterbi":
            pick = int(torch.argmax(score))
            conf[t] = score[pick] / total
        elif decoder == "sample":
            p = score / total
            if temperature != 1.0:
                p = p.pow(1.0 / max(temperature, 1e-6))
                p = p / p.sum()
            pick = int(torch.multinomial(p, 1, generator=generator))
            conf[t] = p[pick]
        else:
            raise ValueError(f"unknown decoder {decoder!r}")

        tokens[t] = edge_tok[pick]
        state = int(edge_dst[pick])

    # ── content diagnostic ────────────────────────────────────────────────
    # How much content does the constrained posterior *want*, versus how much
    # this decoder actually emitted?  Both counted in non-EOS tokens.
    #
    #   expected = sum_t P(x_t != EOS | x^t, G)   -- the posterior's own answer
    #   decoded  = non-EOS tokens in this output
    #
    # This is the go/no-go for mass-based decoding. If expected is far above
    # what Viterbi decodes, the posterior does carry the content and a
    # mass-based decoder can recover it. If expected is itself low, the
    # posterior prefers to stop, no decoder that reads it will help, and the
    # fix has to change the objective (a length penalty) rather than the search.
    #
    # Unlike counting "close" decisions at terminal states, this captures inner
    # emptiness too -- an empty string or an empty nested array shortens the
    # output without ever finishing it, and that is most of the degeneracy.
    if stats is not None:
        stats.decoded_content.append(int((tokens != dfa.eos_id).sum()))
        if marg_c is not None:                    # sum-product marginals only
            totals = marg_c.sum(dim=1)
            p_eos = torch.where(totals > 0, marg_c[:, dfa.eos_id] / totals,
                                torch.zeros_like(totals))
            stats.expected_content.append(float((1.0 - p_eos).sum()))

    return tokens.to(probs.device), conf.to(probs.device)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Generation loop
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FAStats:
    forward_times: list = None
    infer_times: list = None
    build_time: float = 0.0
    num_states: int = 0
    num_edges: int = 0
    steps: int = 0

    # Content the posterior wants vs content the decoder emitted, per step.
    expected_content: list = None
    decoded_content: list = None

    def __post_init__(self):
        self.forward_times = self.forward_times or []
        self.infer_times = self.infer_times or []
        self.expected_content = self.expected_content or []
        self.decoded_content = self.decoded_content or []

    def summary(self) -> dict:
        f, i = self.forward_times, self.infer_times
        out = {
            "build_time_s": self.build_time,
            "num_states": self.num_states,
            "num_edges": self.num_edges,
            "steps": self.steps,
            "forward_total_s": sum(f),
            "forward_mean_ms": 1000 * sum(f) / len(f) if f else 0.0,
            "infer_total_s": sum(i),
            "infer_mean_ms": 1000 * sum(i) / len(i) if i else 0.0,
            "constraint_overhead_pct": 100 * sum(i) / (sum(f) + sum(i)) if (f or i) else 0.0,
        }
        out.update(self.mass_summary())
        return out

    def mass_summary(self) -> dict:
        """Whether a mass-based decoder can fix the brevity bias at all.

        ``content_gap`` is the decisive number: how many non-EOS tokens the
        constrained posterior expects, minus how many this decoder emitted.

            large positive  ->  the posterior does carry the content and the
                                decoder is throwing it away; max-marginal and
                                sampling can recover it.
            near zero       ->  the posterior itself prefers to stop. No decoder
                                reading that posterior will help, and the fix has
                                to change the objective (a length penalty)
                                rather than the search.

        ``expected`` needs sum-product marginals, so it is absent on the Viterbi
        arm; compare Viterbi's ``decoded`` against the marginal arm's
        ``expected`` from the same sweep.
        """
        dec = self.decoded_content
        exp = self.expected_content
        out: dict = {}
        if dec:
            out["decoded_content_mean"] = sum(dec) / len(dec)
        if exp:
            out["expected_content_mean"] = sum(exp) / len(exp)
        if dec and exp:
            n = min(len(dec), len(exp))
            out["content_gap"] = (sum(exp[:n]) - sum(dec[:n])) / n
        return out


@torch.no_grad()
def generate_fa(
    model,
    prompt: torch.Tensor,
    dfa: TokenDFA,
    *,
    steps: int = 128,
    gen_length: int = 256,
    decoder: str = "marginal",
    temperature: float = 1.0,
    mask_id: int = 126336,
    eos_id: int = 126081,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    stats: Optional[FAStats] = None,
) -> Iterator[tuple[torch.Tensor, bool]]:
    """Proactively aligned diffusion decoding.

    Each step: one forward pass, exact inference over the constrained posterior,
    decode a fully valid ``x^0``, commit the positions the constrained marginals
    are most confident about, remask the rest.

    There is no violator detection, no resampling, no repair, and no completion
    fallback: every intermediate ``x`` is a prefix of some accepted string, and
    the final ``x`` is accepted outright.

    Yields ``(x, done)`` after each step so callers can stream or time it.
    """
    device = model.device
    dfa = dfa.to(device)
    gen_start = prompt.shape[1]
    L = gen_length

    x = torch.full((1, gen_start + L), mask_id, dtype=torch.long, device=device)
    x[:, :gen_start] = prompt

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)

    if stats is not None:
        stats.num_states, stats.num_edges = dfa.num_states, dfa.num_edges

    # Positions to commit per step, spread as evenly as the schedule allows.
    per_step = [L // steps + (1 if i < L % steps else 0) for i in range(steps)]

    for step, k in enumerate(per_step):
        masked = x[0, gen_start:] == mask_id
        if not masked.any():
            break
        if k == 0:
            continue

        t0 = time.perf_counter()
        logits = model(x).logits[0, gen_start:]
        if stats is not None:
            stats.forward_times.append(time.perf_counter() - t0)

        probs = F.softmax(logits.to(dtype), dim=-1)
        if probs.shape[-1] < dfa.vocab_size:
            probs = F.pad(probs, (0, dfa.vocab_size - probs.shape[-1]))
        else:
            probs = probs[:, : dfa.vocab_size]

        t1 = time.perf_counter()
        tokens, conf = decode(
            probs, x[0, gen_start:], dfa, mask_id=mask_id,
            decoder=decoder, temperature=temperature, generator=gen, stats=stats,
        )
        if stats is not None:
            stats.infer_times.append(time.perf_counter() - t1)
            stats.steps += 1

        # Commit the k masked positions with the highest constrained marginal.
        conf = torch.where(masked, conf, torch.full_like(conf, -math.inf))
        k = min(k, int(masked.sum()))
        chosen = torch.topk(conf, k=k).indices
        x[0, gen_start + chosen] = tokens[chosen]

        yield x, False

    yield x, True


def finalize(x: torch.Tensor, gen_start: int, eos_id: int, tokenizer) -> str:
    """Decoded generation region with the EOS padding tail removed."""
    ids = x[0, gen_start:].tolist()
    cut = next((i for i, t in enumerate(ids) if t == eos_id), len(ids))
    return tokenizer.decode(ids[:cut], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Self-test -- runs without a model, rustformlang, or a GPU
# ═══════════════════════════════════════════════════════════════════════════


def _toy_dfa() -> ByteDFA:
    """Accepts ``{}`` or ``{"a":DDD}`` for three digits D.

    Small enough to enumerate by hand, and it isolates the exact structure that
    produces the degeneracy: one short accepting path (``{}``) competing against
    1000 longer content-bearing paths, each individually far less probable than
    the short one.  Viterbi compares the short path against the *best single*
    content path and takes the short one; the marginal compares it against their
    *sum*.  Which wins is decided by multiplicity, not by the decoder.
    """
    delta = np.full((10, 256), -1, dtype=np.int32)
    b = ord
    delta[0, b("{")] = 1
    delta[1, b("}")] = 9              # early accept: "{}"
    delta[1, b('"')] = 2
    delta[2, b("a")] = 3
    delta[3, b('"')] = 4
    delta[4, b(":")] = 5
    for src_state, dst_state in ((5, 6), (6, 7), (7, 8)):
        for d in "0123456789":
            delta[src_state, b(d)] = dst_state
    delta[8, b("}")] = 9
    accept = np.zeros(10, dtype=bool)
    accept[9] = True
    return ByteDFA(delta=delta, start=0, accept=accept)


def _self_test() -> None:
    bdfa = _toy_dfa()
    vocab = [b"{", b"}", b'"', b"a", b":", *[str(d).encode() for d in range(10)], None]
    EOS = len(vocab) - 1
    dfa = TokenDFA.from_byte_dfa(bdfa, vocab, eos_id=EOS)
    print(f"states={dfa.num_states} edges={dfa.num_edges}\n")

    L = 10                                    # '{"a":ddd}' is 9 tokens plus EOS
    V = dfa.vocab_size

    def run(p_close: float, decoder: str, seed: int = 0) -> str:
        """Mean-field where structure is near-forced and content mass is diffuse.

        Structural tokens sit at 0.9 because context makes them near-certain.
        The digits share ``1 - p_close``, so total content mass is high while
        every individual digit string is unlikely -- the realistic regime.
        """
        probs = torch.full((L, V), 1e-9, dtype=torch.float64)
        for tid in (0, 2, 3, 4):                      # '{'  '"'  'a'  ':'
            probs[:, tid] = 0.9
        probs[:, 1] = p_close                         # '}'
        probs[:, 5:15] = (1.0 - p_close) / 10.0       # the ten digits
        probs[:, EOS] = 0.9
        probs = probs / probs.sum(-1, keepdim=True)

        committed = torch.full((L,), -1, dtype=torch.long)
        g = torch.Generator().manual_seed(seed)
        toks, _ = decode(probs, committed, dfa, mask_id=-1,
                         decoder=decoder, generator=g)
        return "".join(vocab[t].decode() for t in toks.tolist() if vocab[t] is not None)

    for p_close in (0.50, 0.05):
        row = {d: run(p_close, d) for d in ("viterbi", "marginal", "sample")}
        print(f"p(close)={p_close:<5}  " + "  ".join(f"{k}={v!r}" for k, v in row.items()))

    # Closing is genuinely likely: every decoder should agree on the short string.
    assert run(0.50, "viterbi") == "{}"
    assert run(0.50, "marginal") == "{}"

    # Closing is unlikely but still the single most probable path, because the
    # content mass is spread over 1000 digit strings.  This is the degeneracy.
    assert run(0.05, "viterbi") == "{}", "mode should still collapse"
    assert run(0.05, "marginal") != "{}", "mass should recover content"

    # Committed tokens must be respected exactly, and must steer the posterior.
    probs = torch.full((L, V), 1e-9, dtype=torch.float64)
    for tid in (0, 2, 3, 4):
        probs[:, tid] = 0.9
    probs[:, 1] = 0.9                                  # '}' strongly preferred
    probs[:, 5:15] = 0.01
    probs[:, EOS] = 0.9
    probs = probs / probs.sum(-1, keepdim=True)
    committed = torch.full((L,), -1, dtype=torch.long)
    committed[1] = 2                                   # pin '"' at position 1
    toks, conf = decode(probs, committed, dfa, mask_id=-1, decoder="viterbi")
    got = "".join(vocab[t].decode() for t in toks.tolist() if vocab[t] is not None)
    assert got.startswith('{"a":') and got.endswith("}") and len(got) == 9, got
    assert conf.shape == (L,)

    _test_marginal_dead_end()
    print("\nok")


def _test_marginal_dead_end() -> None:
    """The max-marginal walk must not strand itself in a dead region.

    Regression for a failure seen on JSON-Bench (``no continuation from state
    222 at position 107``).  Global marginals aggregate over *all* source
    states, so a token can carry large mass because it is good from some other
    state while the edge leaving the *current* state reaches nowhere accepting.
    Ranking by marginal alone then walks into that edge and dies one position
    later.  Selection has to be restricted to edges with non-zero mass-to-go.

    The automaton below is the smallest shape where the two disagree: state 1
    must split its surviving mass over two continuations, so that a single
    high-marginal dead edge can outrank each of them individually.

        0 -a-> 1        1 -z-> 3   (dead)      4 -e-> 7 (accept)
        0 -b-> 2        1 -y1-> 4             5 -e-> 7
                        1 -y2-> 5             6 -e-> 7
                        2 -z-> 6
    """
    a, b, z, y1, y2, e = range(6)
    V = 7
    edges = [(0, a, 1), (0, b, 2),
             (1, z, 3), (1, y1, 4), (1, y2, 5),
             (2, z, 6),
             (4, e, 7), (5, e, 7), (6, e, 7)]
    accept = np.zeros(8, dtype=bool)
    accept[7] = True
    dfa = TokenDFA._assemble(
        num_states=8, start=0, accept=accept,
        src=np.array([s for s, _, _ in edges], dtype=np.int64),
        tok=np.array([t for _, t, _ in edges], dtype=np.int64),
        dst=np.array([d for _, _, d in edges], dtype=np.int64),
        vocab_size=V, eos_id=V - 1, device="cpu",
    )

    L = 3
    probs = torch.zeros(L, V, dtype=torch.float64)
    for tid, p in ((a, 1.0), (b, 1.0), (z, 0.7), (y1, 0.5), (y2, 0.5), (e, 1.0)):
        probs[:, tid] = p
    committed = torch.full((L,), -1, dtype=torch.long)

    toks, _ = decode(probs, committed, dfa, mask_id=-1, decoder="marginal")
    got = toks.tolist()

    # Position 0 must choose `a` (state 1 keeps more mass than state 2), and
    # position 1 must then reject `z` despite its higher global marginal.
    assert got[0] == a, got
    assert got[1] in (y1, y2), f"walked into the dead edge: {got}"
    assert got[2] == e, got

    # Sanity: the trap really is a trap. Ranking by bare marginal picks z.
    M = _transition_matrices(probs, committed, dfa, -1)
    beta, _ = _backward(M, dfa)
    alpha = _forward(M, dfa)
    marg = _all_token_marginals(alpha, beta, probs, committed, dfa, -1)
    assert marg[1][z] > max(marg[1][y1], marg[1][y2]), marg[1][:6]
    assert beta[2][3] == 0


if __name__ == "__main__":
    _self_test()
