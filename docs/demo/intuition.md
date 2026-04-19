# Intuition: Why Diffusion LLMs, and Why This Project Exists

## The Short Version

GPT-style language models generate text **left to right**.
Diffusion language models generate text by starting with many masked positions and then repeatedly **filling, revising, and refining** them.

This project asks:

> If diffusion LLMs can generate text in a more parallel and bidirectional way, how can we make them reliably produce structured outputs such as valid JSON?

The problem is that grammar constraints are easy for left-to-right models, but much harder for diffusion models.

---

## Diffusion Is Not Only for Images

Diffusion is most famous for image generation, such as Stable Diffusion.
In image diffusion, the model starts from noise and gradually denoises it into an image:

```text
random noise
  -> less noisy image
  -> clearer image
  -> final image
```

Diffusion language models use a similar idea, but the "noise" is usually masked tokens:

```text
Step 0:
[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]

Step 1:
{ [MASK] [MASK] : [MASK] [MASK] }

Step 2:
{ "method" : [MASK] , "id" : [MASK] }

Step 3:
{ "method" : "get" , "id" : "1a2b..." }
```

So the intuition is:

> A diffusion LLM does not write the answer strictly from left to right.
> It starts with a rough masked template and gradually fills in the whole sequence.

---

## How GPT Generates

GPT is an **autoregressive** language model.
That means it generates one token at a time from left to right:

```text
Step 1: {
Step 2: { "method"
Step 3: { "method" :
Step 4: { "method" : "get"
Step 5: { "method" : "get" ,
...
```

This is very natural for grammar checking.
A JSON parser or grammar checker also works left to right:

```text
generated prefix: {
grammar state: OK

generated prefix: { "method"
grammar state: OK

generated prefix: { "method" :
grammar state: OK
```

The model and the parser move in the same direction.

That is why constrained decoding is relatively straightforward for GPT-style models.

---

## How Diffusion LLMs Generate

A diffusion LLM may hold an incomplete sequence like this:

```text
{ "method" : [MASK] , "id" : [MASK] }
```

It can update several positions in parallel:

```text
{ "method" : "GET" , "id" : "grp_abc" }
```

This has two important properties:

1. The model can use both left and right context.
2. The model can potentially generate multiple tokens in fewer sequential steps.

This is why people study diffusion LLMs.

The goal is not necessarily that diffusion LLMs are already better than GPT today.
The goal is to explore a different generation paradigm that may be more parallel, more bidirectional, and useful for editing or structured infilling.

---

## Then Why Not Just Use GPT?

For many real products today, GPT-style models are the practical choice.
They are stronger, more mature, and already have good tooling for JSON mode, tool calling, and structured output.

So the point of this project is not:

```text
Diffusion LLMs are obviously better than GPT.
```

The point is:

```text
Diffusion LLMs have a promising generation style, but reliable structured generation is still hard.
This project studies how to fix that.
```

In research terms, the motivation is:

- GPT-style models are mature but sequential.
- Diffusion LLMs may offer more parallel generation.
- Structured generation is important for APIs, JSON, code, and tool use.
- Existing grammar-constrained methods do not transfer cleanly to diffusion generation.

---

## Why Grammar Constraints Are Easy for GPT

Suppose the schema says:

```text
method must be one of: "get", "post", "put"
id must be a 24-character hex string
```

A GPT-style model generates:

```text
{ "method" :
```

At this point, the grammar checker knows exactly what tokens are allowed next:

```text
"get"
"post"
"put"
```

The decoder can simply mask out invalid tokens.

This works because there is always a concrete prefix.

---

## Why Grammar Constraints Are Hard for Diffusion

Diffusion generation may produce an incomplete or partially wrong sequence:

```text
{ "method" : [MASK] , "id" : [MASK] }
```

or:

```text
{ "method" : "GET" , "id" : "grp_abc" }
```

Now the grammar checker has a harder job.

It wants to know whether the current prefix is valid, but the sequence may contain holes or correlated errors.
For example:

```text
"GET"     violates the lowercase enum
"grp_abc" violates the 24-character hex pattern
```

Fixing only one token may not fix the whole structure.
The valid answer requires several positions to become consistent together.

This is the key difficulty:

> Diffusion gives us parallel, bidirectional proposals, but grammar checkers usually assume a concrete left-to-right prefix.

---

## What LAVE Does

LAVE tries to handle the incomplete-prefix problem using random lookahead.

At a frontier position, it samples many possible completions:

```text
sample 1: { "method": "GET", "id": "abc" }       invalid
sample 2: { "method": "get", "id": "123" }       invalid
sample 3: { "method": "get", "id": "1a2b..." }   valid
```

If at least one sampled completion is valid, LAVE accepts the token.

This works when valid completions are easy to sample.
But with hard schemas, valid completions can be extremely rare.

For example, randomly producing a 24-character hex string is unlikely.
So LAVE can fail even when a valid completion exists.

The failure mode is:

```text
valid path exists
but random sampling does not find it
so the method rejects or times out
```

---

## What Dgrammar Changes

Dgrammar removes random suffix sampling.

Instead of asking:

```text
Can I randomly find a valid completion?
```

it asks:

```text
Given the current grammar state, which tokens are valid at the frontier?
```

Then it masks out invalid tokens directly.

The intuition:

> Do not guess whether a token can be valid.
> Use the grammar checker to deterministically restrict the next frontier token.

Dgrammar also keeps the parser state incrementally, remasks only the offending token, and overlaps grammar-mask computation with the next model forward pass.

This makes decoding more stable and removes timeout-heavy random trial-and-error.

---

## Is Dgrammar Still Left-to-Right?

Yes, partly.

Dgrammar does reintroduce a left-to-right **frontier** for grammar checking.
That is the main tradeoff.

But this does not mean the diffusion model itself becomes a GPT-style autoregressive model.

The important distinction is:

```text
model proposal:        still diffusion / parallel / bidirectional
grammar verification:  left-to-right frontier
```

The diffusion model can still run a forward pass over the whole sequence and produce logits for many masked positions at once:

```text
{ [MASK] [MASK] [MASK] [MASK] [MASK] }
       |      |      |      |      |
     logits logits logits logits logits
```

Dgrammar then decides which proposed tokens can be safely committed under the grammar.

So Dgrammar is not:

```text
generate token 1 with one model forward
generate token 2 with one model forward
generate token 3 with one model forward
```

Instead, it is closer to:

```text
run one diffusion forward
get proposals for many positions
use a left-to-right grammar frontier to verify and commit them safely
```

---

## Why Not Check Every Position in Parallel?

Because grammar validity often depends on earlier structure.

For example:

```json
{ "id": "abc" }
```

Whether `"abc"` is valid depends on what key it belongs to.

If the key is `"id"`, the schema may require a 24-character hex string:

```text
"abc" is invalid
```

But if the key is `"name"`, then the same string may be valid:

```text
"abc" is valid
```

Now consider an incomplete diffusion state:

```json
{ [MASK]: "abc" }
```

The grammar checker cannot reliably decide whether `"abc"` is valid until the key is known.

This is why a fully parallel grammar check is hard.
The legality of later tokens can depend on earlier choices.

So Dgrammar uses a frontier:

```text
verify the longest concrete prefix
find the first unverified position
apply grammar constraints there
move the frontier forward
```

---

## Is This Against the Purpose of Diffusion?

Partly yes, and partly no.

It is true that Dgrammar sacrifices some of the ideal parallelism of diffusion.
The grammar checker still has to move through the sequence from left to right.

But it does not fully collapse into GPT-style generation, because:

- the model forward is still diffusion-based;
- the model can still see both left and right context;
- the model can still propose many positions in one pass;
- Dgrammar can commit batches of tokens, not just one token per model forward;
- later DPGrammar repairs a whole span jointly rather than only one token at a time.

So the honest framing is:

> Dgrammar reintroduces a left-to-right frontier for grammar verification, but not for model generation.

This is a practical compromise.

Formal grammars are naturally prefix-based.
Diffusion models are naturally parallel and bidirectional.
Dgrammar combines them by using diffusion for proposing tokens and a frontier parser for safely validating them.

---

## Why DPGrammar Is Needed After Dgrammar

Dgrammar's frontier is useful for safety, but it can still be too local.

If several tokens are wrong together, greedy frontier repair may fix one token while leaving the rest inconsistent.

Example:

```json
{
  "method": "GET",
  "id": "grp_abc"
}
```

Dgrammar may do:

```text
fix "GET" -> "get"
then later discover "grp_abc" is still invalid
then resample again
```

DPGrammar does:

```text
repair ["GET", "grp_abc"] as one span
```

This is closer to the diffusion intuition because it treats multiple positions as a joint decision.

So the method progression becomes:

```text
LAVE:
diffusion proposal + random suffix verification

Dgrammar:
diffusion proposal + deterministic left-to-right frontier verification

DPGrammar:
diffusion proposal + frontier verification + joint span repair
```

The broader takeaway:

> Dgrammar gives correctness infrastructure.
> DPGrammar recovers some multi-token joint reasoning when greedy frontier repair is not enough.

---

## Why Dgrammar Is Still Not Enough

Dgrammar mostly repairs one offending position at a time.

But some schema failures are joint failures:

```json
{
  "method": "GET",
  "id": "grp_abc"
}
```

The correct repair is not just one local change.
The method and the ID must both become valid:

```json
{
  "method": "get",
  "id": "1a2b3c4d5e6f7890abcdef12"
}
```

Greedy token-by-token repair can be slow or brittle here.

---

## What DPGrammar Adds

DPGrammar treats a whole violated span as one decision.

Instead of fixing one token at a time, it runs a Viterbi-style dynamic program over grammar states:

```text
find the highest-probability token sequence
such that the entire span is grammar-valid
```

The intuition:

> When several tokens must be correct together, repair them together.

This is why DPGrammar improves validity so much.
It changes the repair problem from local trial-and-error into a small structured search.

---

## The Main Story of the Project

The project can be explained as this progression:

```text
LAVE:
randomly sample completions and hope one is valid

Dgrammar:
use the grammar checker to deterministically mask the frontier

DPGrammar:
when a span is wrong, repair the whole span jointly with dynamic programming
```

The high-level takeaway is:

> Diffusion LLMs are interesting because they may generate text in a parallel and bidirectional way.
> But that breaks the simple prefix assumption used by grammar-constrained decoding.
> This project rebuilds constrained decoding for that diffusion setting.

---

## One-Sentence Presentation Version

GPT-style models generate left to right, so grammar constraints are easy to enforce with a prefix parser.
Diffusion LLMs generate by repeatedly filling and revising masked positions, which is potentially more parallel and bidirectional, but breaks the prefix assumption.
Our project studies how to recover reliable grammar-constrained generation for diffusion LLMs using deterministic frontier masking and Viterbi span repair.
