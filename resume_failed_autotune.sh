#!/usr/bin/env bash
# Resume failed auto-tuning experiments from iter011 and iter012
# Generated: 2026-03-16
#
# HOW IT WORKS:
#   - run_parallel_strict.py --resume checks is_experiment_finished()
#   - is_experiment_finished() only looks at progress.round >= N_ROUNDS
#   - Failed experiments with round < N_ROUNDS will be RE-ATTEMPTED automatically
#   - The pipeline uses checkpoint + trace files to resume from the last completed round
#   - NO need to delete status files or any data
#
# IMPORTANT NOTES:
#   - iter011 cand0 failed at R15 with "Only 68 valid selections, but 88 required"
#     This is a PARAMETER BUG, not infrastructure. It will likely fail again.
#     Consider skipping it (commented out below).
#   - iter011 cand2-5 failed on SiliconFlow LLM API DNS resolution error
#   - iter012 cand4-5 failed on I/O error at R9
#   - Make sure LLM API (SiliconFlow) is accessible before running!

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================"
echo "Resume Failed Auto-Tuning Experiments"
echo "Started: $(date)"
echo "============================================"

# ─────────────────────────────────────────────
# iter011: 5 failed experiments (LLM DNS + selection error)
# Config: N_ROUNDS=16, QUERY_SIZE=88, EPOCHS=10, SEED=42
# ─────────────────────────────────────────────
echo ""
echo ">>> Resuming iter011 (5 failed experiments)..."
echo "    cand0: failed@R15 (selection count bug - MAY RE-FAIL)"
echo "    cand2: failed@R9  (DNS)"
echo "    cand3: failed@R9  (DNS)"
echo "    cand4: failed@R7  (DNS)"
echo "    cand5: failed@R7  (DNS)"

python src/run_parallel_strict.py \
    --resume autotune_opt_iter011_20260316_0723 \
    --seed 42 \
    --n-rounds 16 \
    --epochs-per-round 10 \
    --include "auto_opt_iter11_cand0_00,auto_opt_iter11_cand2_02,auto_opt_iter11_cand3_03,auto_opt_iter11_cand4_04,auto_opt_iter11_cand5_05" \
    --agent-workers 3 \
    --non-agent-workers 0 || echo "[WARN] iter011 resume exited with non-zero status"

echo ""
echo ">>> iter011 resume done."

# ─────────────────────────────────────────────
# iter012: 2 failed experiments (I/O error)
# Config: N_ROUNDS=10, QUERY_SIZE=147, EPOCHS=8, SEED=42
# ─────────────────────────────────────────────
echo ""
echo ">>> Resuming iter012 (2 failed experiments)..."
echo "    cand4: failed@R9 (I/O Error)"
echo "    cand5: failed@R9 (I/O Error)"

python src/run_parallel_strict.py \
    --resume autotune_opt_iter012_20260316_1238 \
    --seed 42 \
    --n-rounds 10 \
    --epochs-per-round 8 \
    --include "auto_opt_iter12_cand4_04,auto_opt_iter12_cand5_05" \
    --agent-workers 2 \
    --non-agent-workers 0 || echo "[WARN] iter012 resume exited with non-zero status"

echo ""
echo "============================================"
echo "All resume tasks completed: $(date)"
echo "============================================"
echo ""
echo "Check results with:"
echo "  python src/monitor_auto_tuning.py --autotune-report"
