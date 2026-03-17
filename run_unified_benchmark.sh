#!/bin/bash
# Unified Benchmark: SDSD + Dgrammar + LAVE on JSON-Bench (jsonschema)
#
# Produces Dgrammar-style table:
#   | Method | Syntactic | Functional | Mean Time | Median | P95 | Max | Constraint % |
#
# Prerequisites:
#   - GPU (A100 recommended)
#   - vendor/CD4dLLM (for ETH checker)
#   - vendor/dgrammar (for Dgrammar/LAVE)
#
# Usage:
#   bash run_unified_benchmark.sh              # Run all
#   bash run_unified_benchmark.sh sdsd-only    # SDSD methods only
#   bash run_unified_benchmark.sh aggregate    # Aggregate existing results only

set -e
cd "$(dirname "$0")"
mkdir -p results/unified

MODE="${1:-all}"

if [ "$MODE" = "aggregate" ]; then
    echo "=== Aggregating existing results ==="
    python aggregate_unified_results.py results/unified vendor/dgrammar/results
    exit 0
fi

# 1. Run SDSD methods (Baseline, Ablation1, Ablation2, Ablation3, SDSD)
if [ "$MODE" = "all" ] || [ "$MODE" = "sdsd-only" ]; then
    echo "=== Running SDSD methods on jsonschema ==="
    python run_unified_benchmark.py \
        --methods baseline,ablation1,ablation2,ablation3,sdsd \
        --output results/unified
fi

# 2. Run Dgrammar (if vendor/dgrammar set up)
if [ "$MODE" = "all" ] && [ -d "vendor/dgrammar" ]; then
    echo "=== Running Dgrammar ==="
    (
        cd vendor/dgrammar
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi
        # Run Dgrammar v2+async+AC4 (seed 0, 272 instances, T=128)
        python bench/run_dgrammar_timed.py 0 272 jsonschema 128 0 2>/dev/null || true
    ) || echo "  Dgrammar run skipped (check vendor/dgrammar setup)"
fi

# 3. Run LAVE (if vendor/dgrammar set up)
if [ "$MODE" = "all" ] && [ -d "vendor/dgrammar" ]; then
    echo "=== Running LAVE ==="
    (
        cd vendor/dgrammar
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi
        python bench/run_lave_timed.py 0 272 jsonschema 128 0 2>/dev/null || true
    ) || echo "  LAVE run skipped (check vendor/dgrammar setup)"
fi

# 4. Aggregate all results into comparison table
echo ""
echo "=== Aggregating results ==="
python aggregate_unified_results.py results/unified vendor/dgrammar/results 2>/dev/null || \
python aggregate_unified_results.py results/unified

echo ""
echo "Done. See results/unified/unified_comparison.json"
