# Method Comparison: LAVE vs DGrammar vs DP

**Dataset:** `jsb_medium`, seed=0, T=128, offsets=[0, 68, 136, 204] (272 LAVE / 241 DGrammar & DP instances)

## Summary Table

| Method   |  n  | Syntactic | Functional | Mean Time (s) | Median (s) | P95 (s) | Max (s) | Constraint % (median) |
|----------|:---:|:---------:|:----------:|:-------------:|:----------:|:-------:|:-------:|:---------------------:|
| LAVE     | 272 |   66.5%   |   66.5%    |     35.66     |   20.64    | 120.00  | 120.01  |         3.2%          |
| DGrammar | 241 |   86.7%   |   86.7%    |     13.76     |   13.94    |  20.91  |  33.15  |        30.2%          |
| DP       | 241 |   83.0%   |   83.0%    |     16.89     |   16.29    |  28.84  |  37.75  |         8.3%          |

> **Syntactic = Functional** (schema validity used as proxy for functional correctness).

## Notes

- **LAVE** hits the 120 s timeout on ~5% of instances (P95 ≈ Max), explaining the inflated mean vs median.
- **DGrammar** is the fastest method (mean 13.8 s) with the highest validity (86.7%), but carries a high constraint overhead (30.2% median) due to synchronous mask computation in the greedy retry loop.
- **DP** reduces constraint overhead to 8.3% (median) via async mask overlap — mask computation runs in a background thread during the GPU forward pass. Validity (83.0%) is slightly below DGrammar because DP searches only the top-100 tokens per position, while DGrammar exhausts the full vocabulary.
- DP uses an average of **1.0 resamples/sample** vs DGrammar's **28.2**, at the cost of ~17 extra forward passes per sample (124 vs 107).

## Per-Offset Breakdown

| Method   | Offset |  n  | Syntactic | Mean Time (s) | Constraint % (median) |
|----------|--------|:---:|:---------:|:-------------:|:---------------------:|
| LAVE     | 0      |  68 |   55.9%   |     41.26     |         4.0%          |
| LAVE     | 68     |  68 |   69.1%   |     33.01     |         3.3%          |
| LAVE     | 136    |  68 |   76.5%   |     27.90     |         1.9%          |
| LAVE     | 204    |  68 |   64.7%   |     40.45     |         7.0%          |
| DGrammar | 0      |  61 |   80.3%   |     13.20     |        32.0%          |
| DGrammar | 68     |  59 |   88.1%   |     13.44     |        25.8%          |
| DGrammar | 136    |  63 |   90.5%   |     13.32     |        26.3%          |
| DGrammar | 204    |  58 |   87.9%   |     15.14     |        34.1%          |
| DP       | 0      |  61 |   73.8%   |     16.12     |         8.6%          |
| DP       | 68     |  59 |   84.7%   |     19.11     |         8.2%          |
| DP       | 136    |  63 |   90.5%   |     14.85     |         6.5%          |
| DP       | 204    |  58 |   82.8%   |     17.65     |         9.7%          |
